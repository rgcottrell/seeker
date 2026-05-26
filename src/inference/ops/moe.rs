//! MoE FFN dispatchers — top-k expert selection + per-expert matvec with
//! ID indirection + routing-weighted fused down step. Everything stays on
//! the GPU: the host never reads back expert ids between dispatches.
//!
//! Three op groups, matching the three shaders:
//!
//!   1. `record_topk_softmax` — wraps `topk_moe.slang` to turn router
//!      logits `[n_experts, n_tokens]` into top-k `ids[n_expert_used,
//!      n_tokens]` (u32) and softmax-routed `weights[n_expert_used,
//!      n_tokens]` (f32). One workgroup per 4 tokens; subgroup shuffle
//!      for the iterative argmax.
//!
//!   2. `record_matvec_q4k_id` — wraps the `mul_mat_vec_q4_k.id` SPV
//!      variant. Dispatches `[M.div_ceil(NUM_ROWS), n_expert_used, 1]`
//!      workgroups; gl_WorkGroupID.y resolves an `expert_id` via the
//!      ids buffer, which selects the weight slab in `data_a`. Use for
//!      both `ffn_gate_exps` and `ffn_up_exps`.
//!
//!   3. `record_moe_down_q5k` — wraps the new `moe_down_q5_k.slang`,
//!      which fuses the down matvec, routing-weight multiply, and
//!      cross-expert sum into a single dispatch. Output is the final
//!      `[n_embd, n_tokens]` MoE FFN contribution, no intermediate
//!      `[n_embd, n_expert_used, n_tokens]` tensor.
//!
//! For N>1 (prefill) the host dispatches once per token by sweeping
//! `expert_i1` and the ids buffer offset; the shaders themselves don't
//! batch.

use std::error::Error;

use crate::inference::buffer::BufferRange;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

// ── topk_moe ───────────────────────────────────────────────────────

pub const GATING_SOFTMAX: u32 = 0;
#[allow(dead_code)]
pub const GATING_SIGMOID: u32 = 1;
#[allow(dead_code)]
pub const GATING_SOFTMAX_WEIGHT: u32 = 2;

const TOPK_MOE_PUSH_BYTES: u32 = 10 * 4;

#[derive(Clone, Copy)]
pub struct TopkMoeParams {
    /// Total number of experts the router scores against (e.g. 256 for
    /// Qwen35MoE). Becomes the spec constant `n_experts_spec`.
    pub n_experts: u32,
    /// How many experts to keep per token (e.g. 8). Becomes the push
    /// constant `n_expert_used`.
    pub n_expert_used: u32,
    /// Gating function applied to the raw router logits.
    pub gating_func: u32,
    /// Whether to normalize the top-k routing weights so they sum to 1.
    /// Qwen35MoE uses `false` (softmax already sums to 1 over all experts,
    /// the top-k subset doesn't need rescaling).
    pub with_norm: bool,
}

/// Dispatch the top-k expert selection.
///
/// Bindings:
///   0 — logits  `[n_experts, n_tokens]` F32 (input)
///   1 — bias    (unused; bound to logits to keep the descriptor layout valid)
///   2 — weights `[n_expert_used, n_tokens]` F32 (output)
///   3 — ids     `[n_experts,    n_tokens]` u32 (output; only first
///       `n_expert_used` entries per row are written, but the shader's
///       `ids_offset = n_experts * row` math requires this slack)
pub fn record_topk_moe(
    ctx: &mut DispatchContext,
    gate_logits: TensorView,
    weights_out: BufferRange,
    ids_out: BufferRange,
    params: TopkMoeParams,
) -> Result<(), Box<dyn Error>> {
    let n_tokens = gate_logits.dims[1].max(1) as u32;

    let mut push = [0u8; TOPK_MOE_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, n_tokens);                     // n_rows
    put_u(&mut push, &mut w, params.n_experts);             // n_experts_push (only read when nexperts_use_push=true)
    put_u(&mut push, &mut w, params.n_expert_used);
    // clamp_min / clamp_max bound the weight SUM before normalization.
    // llama.cpp defaults to ±infinity (ggml-vulkan.cpp:11611), i.e.
    // effectively unbounded. Setting these to 0.0 force-divides by 0 and
    // produces NaN routed-expert output when with_norm=true.
    put_f(&mut push, &mut w, f32::NEG_INFINITY);            // clamp_min
    put_f(&mut push, &mut w, f32::INFINITY);                // clamp_max
    put_u(&mut push, &mut w, params.gating_func);
    put_u(&mut push, &mut w, 0);                            // has_bias
    put_u(&mut push, &mut w, if params.with_norm { 1 } else { 0 });
    put_f(&mut push, &mut w, 1.0);                          // output_scale
    put_f(&mut push, &mut w, 0.0);                          // output_bias

    // Spec constants: WARP_SIZE, n_experts_spec, nexperts_use_push.
    // Pinning n_experts_spec at the actual count avoids the runtime
    // `nexperts_use_push` branch and lets the inner loops unroll.
    let spec_constants = vec![32, params.n_experts, 0];

    let key = PipelineKey {
        name: format!("topk_moe_f32_n{}", params.n_experts),
        binding_indices: vec![0, 1, 2, 3],
        push_size: TOPK_MOE_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::TOPK_MOE_F32_SPV.as_bytes())?;

    let workgroups = [n_tokens.div_ceil(4), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3],
        &[
            gate_logits.range(),
            gate_logits.range(), // bias slot — dummy bind
            weights_out,
            ids_out,
        ],
        &push,
        workgroups,
    )?;
    // One coalesced barrier covering both outputs — downstream matvec_id
    // reads ids; the swiglu/moe_down later reads weights. Emit in a
    // single vkCmdPipelineBarrier instead of two separate ones.
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[ids_out, weights_out],
    );
    Ok(())
}

// ── mul_mat_vec_q4_k with MUL_MAT_ID ───────────────────────────────

/// Push-constant block for the MUL_MAT_ID branch of `mul_mat_vec_q4_k.slang`.
/// Layout is the shared `MulMatVecParams` struct from
/// `shaders/include/mul_mat_vec_head.slang` — 8 leading fields shared
/// with the non-id variant + 4 expert-routing fields.
const MULMATVEC_ID_PUSH_BYTES: u32 = 12 * 4;

/// Per-expert matvec for Q4_K weights. Computes, in one dispatch:
///
///   dst[m, k] = sum_d dequant_q4k(a[ids[k], m, d]) * b[d]   for k ∈ [0, n_expert_used)
///
/// where `a` is shape `[ncols, n_rows, n_experts]` and `b` is shape
/// `[ncols, 1]` (decode-only; for N>1 the caller sweeps `expert_i1`).
///
/// Used for `ffn_gate_exps` and `ffn_up_exps` in the Qwen35MoE FFN.
pub fn record_matvec_q4k_id(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    ids: BufferRange,
    dst: TensorView,
    n_expert_used: u32,
) -> Result<(), Box<dyn Error>> {
    record_matvec_kquant_id(
        ctx, a, b, ids, dst, n_expert_used,
        "mul_mat_vec_q4_k_id",
        shaders::MUL_MAT_VEC_Q4_K_ID_SPV.as_bytes(),
        /* b_alias_v4= */ true,
        /* fence= */ true,
    )
}

/// As [`record_matvec_q4k_id`] but skips the trailing barrier — caller fences.
pub fn record_matvec_q4k_id_nofence(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    ids: BufferRange,
    dst: TensorView,
    n_expert_used: u32,
) -> Result<(), Box<dyn Error>> {
    record_matvec_kquant_id(
        ctx, a, b, ids, dst, n_expert_used,
        "mul_mat_vec_q4_k_id",
        shaders::MUL_MAT_VEC_Q4_K_ID_SPV.as_bytes(),
        /* b_alias_v4= */ true,
        /* fence= */ false,
    )
}

/// Per-expert matvec for Q5_K weights. Same contract as the Q4_K
/// variant above. Used for the rare layers in the Q4_K_XL checkpoint
/// whose `ffn_gate_exps` / `ffn_up_exps` are Q5_K (blk.39 only on the
/// reference checkpoint).
pub fn record_matvec_q5k_id(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    ids: BufferRange,
    dst: TensorView,
    n_expert_used: u32,
) -> Result<(), Box<dyn Error>> {
    record_matvec_kquant_id(
        ctx, a, b, ids, dst, n_expert_used,
        "mul_mat_vec_q5_k_id",
        shaders::MUL_MAT_VEC_Q5_K_ID_SPV.as_bytes(),
        /* b_alias_v4= */ false, // Q5_K matvec uses B_TYPEV2 (binding 5)
        /* fence= */ true,
    )
}

/// As [`record_matvec_q5k_id`] but skips the trailing barrier — caller fences.
pub fn record_matvec_q5k_id_nofence(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    ids: BufferRange,
    dst: TensorView,
    n_expert_used: u32,
) -> Result<(), Box<dyn Error>> {
    record_matvec_kquant_id(
        ctx, a, b, ids, dst, n_expert_used,
        "mul_mat_vec_q5_k_id",
        shaders::MUL_MAT_VEC_Q5_K_ID_SPV.as_bytes(),
        /* b_alias_v4= */ false,
        /* fence= */ false,
    )
}

fn record_matvec_kquant_id(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    ids: BufferRange,
    dst: TensorView,
    n_expert_used: u32,
    name: &'static str,
    spv: &[u8],
    b_alias_v4: bool,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    let ncols = a.dims[0] as u32;
    let n_rows = a.dims[1] as u32;
    let n_tokens = b.dims[1] as u32;
    // llama.cpp reshapes `cur` to `[n_embd, 1, n_tokens]` before mul_mat_id —
    // i.e. ne11=1 and the token axis lives in ne12 / the batch index. The
    // shader's b_offset uses `expert_i0 % ne11`, so ne11 must be 1 for all
    // 8 active experts in a dispatch to read the SAME token's B vector;
    // otherwise expert_i0 leaks into the column stride and the per-expert
    // outputs read scrambled B slabs (manifests as ~24× under-summing in
    // the gate/up matmul vs llama.cpp on the same prompt).
    let stride_a = ncols;
    let stride_b = ncols;
    let stride_d = n_rows;
    let batch_stride_a = ncols * n_rows;
    let batch_stride_b = ncols; // = stride_b * ne11 with ne11=1
    let batch_stride_d = n_rows * n_expert_used;

    // Q4_K matvec binds B as both data_b and data_b_v4 (vec4) at slot 4;
    // Q5_K binds B as data_b and data_b_v2 (vec2) at slot 5. The
    // packed16 / packed32 A aliases sit at 3 / 6 either way.
    let b_alias_slot: u32 = if b_alias_v4 { 4 } else { 5 };
    let bindings: Vec<u32> = vec![0, 1, 2, 3, b_alias_slot, 6, 7];

    // Spec-const order on `mul_mat_vec_head.slang`: BLOCK_SIZE, NUM_ROWS,
    // ACCUMULATE. MoE matvec_id always writes a fresh dst slice, so
    // accumulate stays 0.
    let key = PipelineKey {
        name: name.to_string(),
        binding_indices: bindings.clone(),
        push_size: MULMATVEC_ID_PUSH_BYTES,
        spec_constants: vec![32, 2, 0],
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, spv)?;
    let workgroups = [n_rows.div_ceil(2), n_expert_used, 1];

    // Per-token sweep: dispatch once per token with `expert_i1` set to the
    // current token index. Mirrors llama.cpp ggml-vulkan.cpp:9020. ids
    // row stride is `n_experts` (the topk_moe shader writes one expert
    // pick array per token at `n_experts * row`), so pass that as nbi1.
    let n_experts = (ids.size / ((n_tokens.max(1) as u64) * 4)) as u32;
    for expert_i1 in 0..n_tokens {
        let mut push = [0u8; MULMATVEC_ID_PUSH_BYTES as usize];
        let mut w = 0;
        fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
            out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
            *w += 4;
        }
        put_u(&mut push, &mut w, ncols);
        put_u(&mut push, &mut w, stride_a);
        put_u(&mut push, &mut w, stride_b);
        put_u(&mut push, &mut w, stride_d);
        put_u(&mut push, &mut w, batch_stride_a);
        put_u(&mut push, &mut w, batch_stride_b);
        put_u(&mut push, &mut w, batch_stride_d);
        put_u(&mut push, &mut w, 0); // fusion_flags
        put_u(&mut push, &mut w, n_expert_used); // nei0
        put_u(&mut push, &mut w, 1); // ne11 = 1 (cur reshaped to [n_embd,1,n_tokens])
        put_u(&mut push, &mut w, expert_i1); // expert_i1 = current token
        put_u(&mut push, &mut w, n_experts); // nbi1 — ids row stride (= n_experts)

        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &bindings,
            &[
                a.range(),   // 0: data_a
                b.range(),   // 1: data_b
                dst.range(), // 2: data_d
                a.range(),   // 3: data_a_packed16 (alias of A)
                b.range(),   // 4 or 5: data_b_v4 / data_b_v2 (alias of B)
                a.range(),   // 6: data_a_packed32 (alias of A)
                ids,         // 7: data_ids
            ],
            &push,
            workgroups,
        )?;
    }
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    }
    Ok(())
}

// ── moe_down_q5_k (fused routing-weighted sum) ─────────────────────

/// Push-constant block for `moe_down_q5_k.slang::MoeDownParams`.
const MOEDOWN_PUSH_BYTES: u32 = 6 * 4;

/// Fused routing-weighted down matvec for Q5_K weights.
///
///   dst[m] = sum_k routing[k] * sum_d dequant_q5k(a[ids[k], m, d]) * ffn_h[d, k]
///
/// `a` has shape `[ff, n_embd, n_experts]` (Q5_K), `ffn_h` is
/// `[ff, n_expert_used]` F32, and `dst` is `[n_embd, 1]` F32. The kernel
/// loops over `k` internally so the per-(token, expert) intermediate
/// tensor never materializes.
pub fn record_moe_down_q5k(
    ctx: &mut DispatchContext,
    down_exps: TensorView,
    ffn_h: TensorView,
    ids: BufferRange,
    routing_weights: BufferRange,
    dst: TensorView,
    n_expert_used: u32,
) -> Result<(), Box<dyn Error>> {
    record_moe_down_impl(
        ctx,
        down_exps,
        ffn_h,
        ids,
        routing_weights,
        dst,
        n_expert_used,
        "moe_down_q5_k",
        shaders::MOE_DOWN_Q5_K_DEFAULT_SPV.as_bytes(),
        /* bindings_with_b_v4= */ false,
    )
}

/// Same shape contract as [`record_moe_down_q5k`] but for Q6_K weights.
/// Used on blocks where `ffn_down_exps` is Q6_K (the Unsloth UD-Q4_K_XL
/// checkpoint mixes Q5_K and Q6_K — blk.34, blk.38, blk.39 are Q6_K).
/// The kernel binds `data_b_v4` (float4 view of ffn_h) at slot 4 rather
/// than the float2 view at slot 5, matching the Q6_K matvec convention.
pub fn record_moe_down_q6k(
    ctx: &mut DispatchContext,
    down_exps: TensorView,
    ffn_h: TensorView,
    ids: BufferRange,
    routing_weights: BufferRange,
    dst: TensorView,
    n_expert_used: u32,
) -> Result<(), Box<dyn Error>> {
    record_moe_down_impl(
        ctx,
        down_exps,
        ffn_h,
        ids,
        routing_weights,
        dst,
        n_expert_used,
        "moe_down_q6_k",
        shaders::MOE_DOWN_Q6_K_DEFAULT_SPV.as_bytes(),
        /* bindings_with_b_v4= */ true,
    )
}

fn record_moe_down_impl(
    ctx: &mut DispatchContext,
    down_exps: TensorView,
    ffn_h: TensorView,
    ids: BufferRange,
    routing_weights: BufferRange,
    dst: TensorView,
    n_expert_used: u32,
    name: &'static str,
    spv: &[u8],
    bindings_with_b_v4: bool,
) -> Result<(), Box<dyn Error>> {
    let ncols = down_exps.dims[0] as u32; // ff
    let n_embd = down_exps.dims[1] as u32;
    let stride_a = ncols;
    let stride_b = ncols;
    let stride_d = n_embd;
    let batch_stride_a = ncols * n_embd;
    // ffn_h has shape [ff, n_expert_used, n_tokens]; dst is [n_embd, n_tokens];
    // ids/routing_weights are [n_expert_used, n_tokens].
    let n_tokens = ffn_h.dims[2].max(1) as u32;
    let ffn_h_per_token_bytes = (ncols as u64) * (n_expert_used as u64) * 4;
    // ids row stride = n_experts (topk_moe writes one expert-pick array
    // per token at offset `n_experts * row`); routing_weights row stride
    // = n_expert_used (topk_moe packs weights tightly per token).
    let n_experts = (ids.size / (n_tokens.max(1) as u64 * 4)) as u32;
    let ids_per_token_bytes = (n_experts as u64) * 4;
    let weights_per_token_bytes = (n_expert_used as u64) * 4;
    let dst_per_token_bytes = (n_embd as u64) * 4;

    let mut push = [0u8; MOEDOWN_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, ncols);
    put_u(&mut push, &mut w, stride_a);
    put_u(&mut push, &mut w, stride_b);
    put_u(&mut push, &mut w, stride_d);
    put_u(&mut push, &mut w, batch_stride_a);
    put_u(&mut push, &mut w, n_expert_used);

    let b_alias_slot = if bindings_with_b_v4 { 4u32 } else { 5u32 };
    let bind_idx: Vec<u32> = vec![0, 1, 2, 3, b_alias_slot, 7, 8];

    let key = PipelineKey {
        name: name.to_string(),
        binding_indices: bind_idx.clone(),
        push_size: MOEDOWN_PUSH_BYTES,
        spec_constants: vec![],
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, spv)?;

    let workgroups = [n_embd.div_ceil(2), 1, 1];

    // Per-token loop: shift ffn_h / ids / weights / dst by the token's
    // contiguous slot. The shader has no token coordinate — it processes
    // exactly one token's MoE down step per dispatch.
    for t in 0..(n_tokens as u64) {
        let ffn_h_t = ffn_h.range_with_offset(t * ffn_h_per_token_bytes);
        let dst_t = dst.range_with_offset(t * dst_per_token_bytes);
        let ids_t = BufferRange {
            buffer: ids.buffer,
            offset: ids.offset + t * ids_per_token_bytes,
            size: ids_per_token_bytes,
        };
        let weights_t = BufferRange {
            buffer: routing_weights.buffer,
            offset: routing_weights.offset + t * weights_per_token_bytes,
            size: weights_per_token_bytes,
        };
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &bind_idx,
            &[
                down_exps.range(), // 0: data_a (Q5_K or Q6_K)
                ffn_h_t,           // 1: data_b
                dst_t,             // 2: data_d
                down_exps.range(), // 3: data_a_packed16 (alias)
                ffn_h_t,           // 4 or 5: data_b_v4 / data_b_v2 (alias)
                ids_t,             // 7: data_ids
                weights_t,         // 8: data_weights
            ],
            &push,
            workgroups,
        )?;
    }
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}
