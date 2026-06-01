//! `flash_attn` dispatch — scalar reference variant from
//! shaders/compute/flash_attn.slang.
//!
//! Spec constants: Bc, HSK, HSV, MASK_ENABLE, Clamp (5 in declaration order).
//! Push constants follow llama.cpp's `vk_flash_attn_push_constants`
//! (ggml-vulkan.cpp:9396), 128 bytes.
//!
//! Bindings are sparse: indices [0, 1, 2, 3, 5] (skipping 4, which the
//! reference shader reserves for split-K output).
//!
//! The Slang port writes its output in `[hidden = HSV * n_head, L]`
//! contiguous layout (not llama.cpp's `[HSV, L, n_head]` post-permute
//! layout), so the next matmul can read it directly.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

const FA_PUSH_BYTES: u32 = 32 * 4;

/// Combine-pass push (`FaSplitKParams`): D, ne1, ne2, ne3, k_num, sinks.
const FA_SPLIT_K_PUSH_BYTES: u32 = 6 * 4;

/// KV block width — must match the `Bc` spec constant in flash_attn.slang,
/// and the alignment llama.cpp snaps split-K KV chunks to.
const FA_BC: u32 = 32;

/// Placeholder compute-unit count when the device doesn't advertise one,
/// matching llama.cpp's flash-attention fallback.
const FA_SPLIT_CORE_COUNT_FALLBACK: u32 = 16;

/// Upper bound on the split-K workgroup count for any single flash-attn
/// dispatch. Caps `pick_k_num` and sizes the per-call partials buffer
/// so its (offset, size) descriptor binding stays stable across decode
/// tokens — required for the persistent-decode-cmdbuf optimization,
/// where the host changes only the indirect-dispatch wg count between
/// submits. On qwen35moe Strix Halo decode the heuristic naturally
/// saturates at ~8 (target=80 / base_wgs=10); 16 leaves headroom.
pub const FA_MAX_K_NUM: u32 = 16;

/// Max KV keys a single flash-attn workgroup may walk in the VISION
/// full-attention path before we force a KV-split. The vision tower attends
/// bidirectionally over all `n_pos` patches, so the single-pass kernel has each
/// workgroup walk all `n_pos` keys serially; past ~14k keys that one dispatch
/// trips the RADV/Strix-Halo per-dispatch watchdog (the device is lost). We
/// split the KV (split-K, direct dispatch, partials sized at the actual small
/// k_num) so each split walks ≤ this many keys. EMPIRICAL: on Strix Halo the
/// scalar F32 vision FA faults when a workgroup walks ≳ a few-thousand keys
/// (walk 8052 → device lost; 3220 and 2684 → fine), so 3000 keeps each split's
/// walk safely short. Lower = more splits = shorter walks = safer, but larger
/// partials (≈ (hd+2)·n·heads·k_num) — 3000 fits the 40k-float/token scratch
/// budget up to the 4096-token image cap. Override with `SEEKER_FA_VISION_WALK`
/// (also used to force the split at small n_pos to validate against single-pass).
/// Only read on the vision path (mask=None, n>1), never on the decode hot path.
fn vision_fa_kv_walk() -> u32 {
    std::env::var("SEEKER_FA_VISION_WALK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&w| w > 0)
        .unwrap_or(3_000)
}

/// Same heuristic as `pick_k_num`, clamped to `FA_MAX_K_NUM`. Used by
/// the Engine to decide whether a cached decode cmdbuf can be replayed
/// for the current `kv` — the wg count is baked into the cmdbuf via
/// `cmd_update_buffer`, so we have to re-record when the heuristic
/// would now pick a different value.
pub fn pick_k_num_clamped(shader_core_count: u32, base_wgs: u32, kv: u32) -> (u32, u32) {
    let (k, bps) = pick_k_num(shader_core_count, base_wgs, kv);
    if k <= FA_MAX_K_NUM {
        (k, bps)
    } else {
        let num_blocks = kv.div_ceil(FA_BC).max(1);
        (FA_MAX_K_NUM, num_blocks.div_ceil(FA_MAX_K_NUM))
    }
}

#[derive(Clone, Copy)]
pub struct FlashAttnParams {
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub gqa_ratio: u32, // n_head / n_head_kv
    pub scale: f32,     // 1 / sqrt(head_dim)
}

/// Record flash attention.
///
/// `q` is `[head_dim, L, n_head]` (permuted view of the post-RoPE Q tensor),
/// `k` and `v` are `[head_dim, total_len, n_head_kv]`. `mask` is the F32
/// causal mask `[total_len, L]`, or `None` for single-token decode — there
/// every KV slot is causally visible, so the mask row is all-zeros and the
/// shader runs with `MASK_ENABLE=0`. `out` is `[hidden, L]` contiguous
/// (`[HSV, n_head, L]` post-permute).
///
/// For small `L` (decode) over a long KV cache, the KV dimension is split
/// across `k_num` workgroups per head and the partials merged by
/// `flash_attn_split_k_reduce`; otherwise a single workgroup per head walks
/// the whole cache serially, starving the GPU and making per-token latency
/// grow with context length.
pub fn record(
    ctx: &mut DispatchContext,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    mask: Option<TensorView>,
    out: TensorView,
    params: FlashAttnParams,
    // Actual KV length to iterate over. The caller passes this
    // explicitly instead of using `k.dims[1]` because the direct-cache
    // fast path binds the full cache layer (so `k.dims[1] = max_seq_len`)
    // and the shader bounds its iteration by `DecodeDyn::kv_len`.
    kv_actual: u32,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(q.dtype, GgmlType::F32);
    debug_assert_eq!(out.dtype, GgmlType::F32);
    if let Some(m) = mask {
        debug_assert_eq!(m.dtype, GgmlType::F32, "mask is always F32 now");
    }
    debug_assert_eq!(
        k.dtype, v.dtype,
        "flash_attn requires K and V to share a dtype (one variant per cache dtype, \
         not per K/V combo). Materialize the odd side to match if you need a \
         heterogeneous combo.",
    );

    // Length of the always-visible cached prefix preceding this batch's
    // query tokens (= position_offset). The host builds only the within-chunk
    // [L × L] causal mask and the scalar shader treats columns < prefix_len as
    // visible; `kv_actual = prefix_len + L`.
    let prefix_len = kv_actual.saturating_sub(q.dims[1] as u32);

    // Cooperative-matrix flash-attention path (`flash_attn_cm1.slang`).
    // Processes Br=16 query rows per workgroup using a 16×16 CoopMat
    // fragment for QK^T. F16 KV only — softmax + PV stay scalar. Gated
    // until it's been validated on more models; the shader had a Slang
    // Load API bug we fixed alongside `mul_mm_cm.slang`, but it's never
    // been exercised on real hardware before now.
    //
    // Single-token decode passes `mask = None` and goes through the scalar
    // path below, which owns the split-K decode optimization; cm1 only runs
    // for the masked (prefill) batches it was designed for.
    //
    // cm1 still expects the legacy full `[kv_len, L]` mask layout, which only
    // coincides with the new within-chunk `[L, L]` mask when there is no
    // prefix (`prefix_len == 0`, i.e. the first/only ubatch). Fall back to the
    // scalar path for chunked prefill at a non-zero offset until cm1 learns the
    // within-chunk + offset contract.
    if ctx.device.coop_matrix
        && k.dtype == GgmlType::F16
        && v.dtype == GgmlType::F16
        && *crate::runtime_flags::FA_CM
        && prefix_len == 0
    {
        if let Some(m) = mask {
            return record_cm1(ctx, q, k, v, m, out, params, kv_actual);
        }
    }

    let (variant_name, variant_spv) = match k.dtype {
        GgmlType::F32 => ("flash_attn_f32_f32", shaders::FLASH_ATTN_F32_F32_SPV.as_bytes()),
        GgmlType::F16 => ("flash_attn_f32_f16", shaders::FLASH_ATTN_F32_F16_SPV.as_bytes()),
        GgmlType::BF16 => ("flash_attn_f32_bf16", shaders::FLASH_ATTN_F32_BF16_SPV.as_bytes()),
        GgmlType::Q4_0 => ("flash_attn_f32_q4_0", shaders::FLASH_ATTN_F32_Q4_0_SPV.as_bytes()),
        GgmlType::Q4_1 => ("flash_attn_f32_q4_1", shaders::FLASH_ATTN_F32_Q4_1_SPV.as_bytes()),
        GgmlType::Q5_0 => ("flash_attn_f32_q5_0", shaders::FLASH_ATTN_F32_Q5_0_SPV.as_bytes()),
        GgmlType::Q5_1 => ("flash_attn_f32_q5_1", shaders::FLASH_ATTN_F32_Q5_1_SPV.as_bytes()),
        GgmlType::Q8_0 => ("flash_attn_f32_q8_0", shaders::FLASH_ATTN_F32_Q8_0_SPV.as_bytes()),
        GgmlType::IQ4_NL => (
            "flash_attn_f32_iq4_nl",
            shaders::FLASH_ATTN_F32_IQ4_NL_SPV.as_bytes(),
        ),
        other => {
            return Err(format!(
                "flash_attn: no shader variant for K/V dtype {other:?}"
            )
            .into());
        }
    };

    let n = q.dims[1] as u32; // L (rows of Q per head)
    // kv_actual is the caller-provided KV length to iterate over.
    // k.dims[1] may be max_seq_len (direct-cache fast path) — the
    // shader uses DecodeDyn::kv_len to bound its loops.
    let kv = kv_actual;
    let ne1 = n; // output rows per head
    let ne2 = q.dims[2] as u32; // n_head
    let ne3 = q.dims[3].max(1) as u32; // batch
    let neq2 = q.dims[2] as u32;
    let neq3 = q.dims[3].max(1) as u32;
    let nek2 = k.dims[2] as u32;
    let nek3 = k.dims[3].max(1) as u32;
    let nev2 = v.dims[2] as u32;
    let nev3 = v.dims[3].max(1) as u32;
    // Mask dims only matter when MASK_ENABLE != 0; supply 1s when disabled.
    let (nem1, nem2, nem3) = match mask {
        Some(m) => (
            m.dims[1] as u32,
            m.dims[2].max(1) as u32,
            m.dims[3].max(1) as u32,
        ),
        None => (1u32, 1u32, 1u32),
    };
    let mask_enable: u32 = if mask.is_some() { 1 } else { 0 };
    let nb01 = q.element_stride[1] as u32;
    let nb02 = q.element_stride[2] as u32;
    let nb03 = q.element_stride[3] as u32;
    let nb11 = k.element_stride[1] as u32;
    let nb12 = k.element_stride[2] as u32;
    let nb13 = k.element_stride[3] as u32;
    let nb21 = v.element_stride[1] as u32;
    let nb22 = v.element_stride[2] as u32;
    let nb23 = v.element_stride[3] as u32;

    // ---- split-K decode heuristic (replicates llama.cpp, clamped to FA_MAX_K_NUM) ----
    let base_wgs = n * ne2 * ne3;
    // Vision full-attention over long KV (no mask + many query rows): a single
    // workgroup walking all `kv` keys trips the per-dispatch watchdog past ~14k.
    // Force a KV-split so each split walks ≤ VISION_FA_KV_WALK keys. The split-K
    // branch below then uses a DIRECT dispatch (fresh forward each encode, no
    // decode-replay) with partials sized at the actual (small) k_num rather than
    // FA_MAX_K_NUM. Decode/prefill (mask present, or n == 1) is unaffected.
    let vision_split = mask.is_none() && n > 1 && kv > vision_fa_kv_walk();
    let (k_num, blocks_per_split) = if vision_split {
        let num_blocks = kv.div_ceil(FA_BC).max(1);
        let kf = kv.div_ceil(vision_fa_kv_walk()).clamp(2, num_blocks);
        (kf, num_blocks.div_ceil(kf))
    } else {
        pick_k_num_clamped(ctx.device.shader_core_count, base_wgs, kv)
    };
    // Mirror kv/k_num/blocks_per_split into the per-forward DecodeDyn
    // slot. The shader reads them from there so the recorded cmdbuf
    // is replay-stable (host overwrites these fields between submits
    // when the persistent-decode-cmdbuf path activates).
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 0, kv)?;
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 4, k_num)?;
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 8, blocks_per_split)?;

    // Pack push.
    let mut push = [0u8; FA_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, n);
    put_u(&mut push, &mut w, kv);
    put_u(&mut push, &mut w, ne1);
    put_u(&mut push, &mut w, ne2);
    put_u(&mut push, &mut w, ne3);
    put_u(&mut push, &mut w, neq2);
    put_u(&mut push, &mut w, neq3);
    put_u(&mut push, &mut w, nek2);
    put_u(&mut push, &mut w, nek3);
    put_u(&mut push, &mut w, nev2);
    put_u(&mut push, &mut w, nev3);
    put_u(&mut push, &mut w, nem1);
    put_u(&mut push, &mut w, nem2);
    put_u(&mut push, &mut w, nem3);
    put_u(&mut push, &mut w, nb01);
    put_u(&mut push, &mut w, nb02);
    put_u(&mut push, &mut w, nb03);
    put_u(&mut push, &mut w, nb11);
    put_u(&mut push, &mut w, nb12);
    put_u(&mut push, &mut w, nb13);
    put_u(&mut push, &mut w, nb21);
    put_u(&mut push, &mut w, nb22);
    put_u(&mut push, &mut w, nb23);
    put_f(&mut push, &mut w, params.scale);
    put_f(&mut push, &mut w, 0.0); // max_bias
    put_f(&mut push, &mut w, 0.0); // logit_softcap
    put_u(&mut push, &mut w, prefix_len); // mask_kv_offset (was mask_n_head_log2)
    put_f(&mut push, &mut w, 0.0); // m0
    put_f(&mut push, &mut w, 0.0); // m1
    put_u(&mut push, &mut w, params.gqa_ratio);
    put_u(&mut push, &mut w, blocks_per_split); // split_kv (blocks per split)
    put_u(&mut push, &mut w, k_num);

    let spec_constants = vec![
        FA_BC,             // Bc
        params.head_dim_k, // HSK
        params.head_dim_v, // HSV
        mask_enable,       // MASK_ENABLE
        0,                 // Clamp
    ];

    // `flash_attn.slang` runs one workgroup per (query-row, head, batch)
    // with WORKGROUP_SIZE=32 — pin to wave32 so the workgroup maps to
    // exactly one subgroup (rather than half a wave64).
    let key = PipelineKey {
        name: variant_name.to_string(),
        binding_indices: vec![0, 1, 2, 3, 4, 5, 6],
        push_size: FA_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant_spv)?;

    // Binding 3 (data_m) needs a valid descriptor even when MASK_ENABLE=0;
    // the shader guards every read on the spec constant, so bind any live
    // storage buffer (q) as a harmless stand-in.
    let mask_range = mask.map(|m| m.range()).unwrap_or_else(|| q.range());
    let dyn_range = ctx.decode_dyn;

    if k_num <= 1 {
        // Single pass: writes the final normalized output to data_o
        // (binding 5). data_o_split (binding 4) is unused here but still
        // needs a valid descriptor, so bind `out` to it too.
        //
        // Direct dispatch (no replay benefit on the single-pass branch
        // — grid is just `[n, ne2, ne3]`, all model-static, so a
        // recorded cmdbuf already replays correctly).
        let workgroups = [n, ne2, ne3];
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5, 6],
            &[
                q.range(),
                k.range(),
                v.range(),
                mask_range,
                out.range(),
                out.range(),
                dyn_range,
            ],
            &push,
            workgroups,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, out.range());
    } else {
        // Split-K: variable-size grid in y (= ne2 * k_num) — switch to
        // `vkCmdDispatchIndirect` so the recorded cmdbuf can serve any
        // runtime k_num via a host-side write to the 12-byte
        // `indirect_wg` slot. Partials are sized once at `FA_MAX_K_NUM`
        // so the binding (offset, size) is constant across calls.
        // Decode replay sizes partials at FA_MAX_K_NUM for a stable binding; the
        // vision direct-dispatch path has no replay, so size at the actual k_num
        // (the combine indexes partials by runtime k_num — see
        // flash_attn_split_k_reduce.slang). At a large vision n this keeps the
        // partials ~200 MB (k_num·(hd+2)·n·heads) instead of FA_MAX_K_NUM×.
        let knum_partials = if vision_split { k_num } else { FA_MAX_K_NUM };
        let partials_floats = (params.head_dim_v as u64 + 2)
            * n as u64
            * ne2 as u64
            * ne3 as u64
            * knum_partials as u64;
        let partials = ctx.alloc_tensor([partials_floats, 1, 1, 1], GgmlType::F32)?;

        if vision_split {
            // Direct dispatch — one workgroup per (query row, head, split):
            // grid [n, ne2 * k_num, ne3]. No indirect args (those exist only for
            // decode-cmdbuf replay where k_num varies between submits).
            super::bind_and_dispatch(
                ctx,
                &pipeline,
                &[0, 1, 2, 3, 4, 5, 6],
                &[
                    q.range(),
                    k.range(),
                    v.range(),
                    mask_range,
                    partials.range(),
                    out.range(),
                    dyn_range,
                ],
                &push,
                [n, ne2 * k_num, ne3],
            )?;
            record_compute_barrier(ctx.device, ctx.cmd, partials.range());
            record_split_k_combine(
                ctx, partials, out, params.head_dim_v, n, ne2, ne3, k_num, dyn_range,
            )?;
            return Ok(());
        }

        let indirect_wg = ctx.alloc_scratch(12)?;
        // Write the (wg_x, wg_y, wg_z) tuple via cmd_update_buffer —
        // recorded into the cmdbuf as a transfer op, naturally ordered
        // before the indirect dispatch by a transfer→draw_indirect
        // barrier. A direct host-mapped write empirically triggers
        // DEVICE_LOST on RADV STRIX_HALO when the indirect dispatch
        // fires past kv=33 — likely because indirect reads from
        // HOST_COHERENT memory aren't covered by the implicit
        // submit-time HOST→DRAW_INDIRECT dependency on this driver.
        //
        // NOTE: cmd_update_buffer bakes the value into the cmdbuf, so
        // this approach is incompatible with Phase 4 replay if k_num
        // varies between submits. Phase 4 will switch to a host→staging
        // → cmd_copy_buffer chain with the right barriers.
        let wg_data: [u32; 3] = [n, ne2 * k_num, ne3];
        let wg_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(wg_data.as_ptr() as *const u8, 12)
        };
        unsafe {
            ctx.device.device.cmd_update_buffer(
                ctx.cmd,
                indirect_wg.buffer,
                indirect_wg.offset,
                wg_bytes,
            );
            let bar = ash::vk::BufferMemoryBarrier::default()
                .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(ash::vk::AccessFlags::INDIRECT_COMMAND_READ)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .buffer(indirect_wg.buffer)
                .offset(indirect_wg.offset)
                .size(12);
            ctx.device.device.cmd_pipeline_barrier(
                ctx.cmd,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::PipelineStageFlags::DRAW_INDIRECT,
                ash::vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&bar),
                &[],
            );
        }
        super::bind_and_dispatch_indirect(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5, 6],
            &[
                q.range(),
                k.range(),
                v.range(),
                mask_range,
                partials.range(),
                out.range(),
                dyn_range,
            ],
            &push,
            indirect_wg,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, partials.range());
        // `dyn_range` (= ctx.decode_dyn) carries this call's k_num for the combine.
        record_split_k_combine(ctx, partials, out, params.head_dim_v, n, ne2, ne3, k_num, dyn_range)?;
    }
    Ok(())
}

/// Batched **decode** flash attention: B independent sequences, one query
/// token each, each attending to its own KV slab over its own length.
///
/// `q` is `[head_dim, 1, n_head, B]`, `k`/`v` are `[head_dim, kv_pos, n_head_kv,
/// B]` where the batch (dim 3, stride `element_stride[3]`) selects a sequence's
/// slab. `kv_lens[b]` is sequence `b`'s cache length. `out` is `[hidden, B]`
/// (`[HSV, n_head, B]`). No mask (a decode query attends to its whole cache).
///
/// Split-K (M4): base parallelism is `n_head * B` workgroups (one per head per
/// sequence). When that under-fills the GPU — long context, few sequences — KV
/// is split across `k_num` workgroups per head and merged by
/// `flash_attn_split_k_reduce`, exactly as the single-sequence decode path. As
/// `B` grows the base already saturates, so the heuristic falls back to
/// `k_num = 1` (no split). `blocks_per_split` is sized from the longest
/// sequence so every sequence's full range is covered; a shorter sequence stops
/// early via the per-element `kv_base + c < p_KV` guard (`p_KV = kv_lens[seq]`),
/// and its over-long splits run zero iterations (the combine weights them to
/// ~0). The batched forward always re-records (no decode replay), so this uses
/// a plain direct dispatch — none of the single-seq indirect-dispatch machinery
/// that exists only to vary `k_num` across replayed submits. `SEEKER_FA_SPLIT=0`
/// disables it; `SEEKER_FA_SPLIT_KNUM=<n>` pins it.
pub fn record_batched(
    ctx: &mut DispatchContext,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    out: TensorView,
    params: FlashAttnParams,
    kv_lens: &[u32],
    dyn_range: crate::inference::buffer::BufferRange,
    // Per-sequence KV slab index (which `BatchKvCache` slab each sequence's K/V
    // lives in). `None` → identity (sequence `b` reads slab `b`), the
    // contiguous layout. `Some` lets a gathered batch read non-contiguous slabs
    // (continuous-batching with prefix-reuse parking). Must be `kv_lens.len()`.
    slots: Option<&[u32]>,
    // Per-sequence query-row counts (`L_s`). `None` → decode: each sequence
    // contributes one query row, split-K enabled, legacy Q/out layout
    // (`[head_dim, 1, n_head, B]`, batch stride nb03). `Some` → unified varlen:
    // sequence `s` contributes `query_lens[s]` rows packed flat in the token
    // dimension; the shader masks causally in-place (row r of seq s sees
    // `[0, kv_lens[s] - query_lens[s] + r]`). q/out are the flat
    // `[head_dim, N_total, n_head]` / `[hidden_v, N_total]` layouts (per-token
    // stride nb01, per-head nb02); split-K is disabled (the `L_s` rows already
    // fill the grid). Must be `kv_lens.len()`; each `query_lens[s] <= kv_lens[s]`.
    query_lens: Option<&[u32]>,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(q.dtype, GgmlType::F32);
    debug_assert_eq!(out.dtype, GgmlType::F32);
    debug_assert_eq!(k.dtype, v.dtype, "flash_attn requires K and V to share a dtype");
    let b = kv_lens.len() as u32;
    debug_assert!(b >= 1, "record_batched needs at least one sequence");
    let varlen = query_lens.is_some();
    if let Some(ql) = query_lens {
        debug_assert_eq!(ql.len(), kv_lens.len(), "query_lens must match kv_lens");
        debug_assert!(
            ql.iter().zip(kv_lens).all(|(&q, &kv)| q >= 1 && q <= kv),
            "each query_lens[s] must be in 1..=kv_lens[s]"
        );
    } else {
        debug_assert_eq!(q.dims[3].max(1) as u32, b, "q batch dim must equal kv_lens.len()");
    }
    // Flat per-sequence query offsets (prefix sum of query_lens) and the grid's
    // x-dim (max query rows). Decode: query_lens = all-1 → q_start[s]=s, 1 row.
    let query_lens_vec: Vec<u32> =
        query_lens.map(|q| q.to_vec()).unwrap_or_else(|| vec![1; b as usize]);
    let q_starts: Vec<u32> = query_lens_vec
        .iter()
        .scan(0u32, |acc, &l| {
            let s = *acc;
            *acc += l;
            Some(s)
        })
        .collect();
    let max_rows = query_lens_vec.iter().copied().max().unwrap_or(1);

    let (variant_name, variant_spv) = match k.dtype {
        GgmlType::F32 => ("flash_attn_f32_f32", shaders::FLASH_ATTN_F32_F32_SPV.as_bytes()),
        GgmlType::F16 => ("flash_attn_f32_f16", shaders::FLASH_ATTN_F32_F16_SPV.as_bytes()),
        GgmlType::BF16 => ("flash_attn_f32_bf16", shaders::FLASH_ATTN_F32_BF16_SPV.as_bytes()),
        GgmlType::Q4_0 => ("flash_attn_f32_q4_0", shaders::FLASH_ATTN_F32_Q4_0_SPV.as_bytes()),
        GgmlType::Q4_1 => ("flash_attn_f32_q4_1", shaders::FLASH_ATTN_F32_Q4_1_SPV.as_bytes()),
        GgmlType::Q5_0 => ("flash_attn_f32_q5_0", shaders::FLASH_ATTN_F32_Q5_0_SPV.as_bytes()),
        GgmlType::Q5_1 => ("flash_attn_f32_q5_1", shaders::FLASH_ATTN_F32_Q5_1_SPV.as_bytes()),
        GgmlType::Q8_0 => ("flash_attn_f32_q8_0", shaders::FLASH_ATTN_F32_Q8_0_SPV.as_bytes()),
        GgmlType::IQ4_NL => (
            "flash_attn_f32_iq4_nl",
            shaders::FLASH_ATTN_F32_IQ4_NL_SPV.as_bytes(),
        ),
        other => return Err(format!("flash_attn: no shader variant for K/V dtype {other:?}").into()),
    };

    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);

    // Split-K decode heuristic for the batch (see the doc comment). `base_wgs`
    // counts one workgroup per (head, sequence); `k_num` splits each head's KV
    // only when that under-fills the GPU. When `k_num == 1`, `pick_k_num`
    // returns `blocks_per_split == ceil(max_kv / Bc)` — the whole range in one
    // split, identical to the pre-split single pass.
    let ne2 = q.dims[2] as u32; // n_head
    let ne3 = b;
    // Grid x-dim = query rows. Decode: 1 row/seq, split KV via the heuristic.
    // Varlen: max_rows rows/seq (the L_s rows already fill the grid → no split).
    let (n, k_num, blocks_per_split) = if varlen {
        (max_rows, 1u32, max_kv.div_ceil(FA_BC).max(1))
    } else {
        let base_wgs = ne2 * ne3; // one wg per (head, seq)
        let (k, bps) = pick_k_num_clamped(ctx.device.shader_core_count, base_wgs, max_kv);
        (1u32, k, bps)
    };

    // Per-sequence DecodeDyn array: entry b's kv_len bounds sequence b's KV
    // loop; entry 0's k_num / blocks_per_split are read uniformly by every
    // workgroup. Zero the rest so stale scratch never leaks into a field.
    //
    // CRITICAL: `dyn_range` is caller-provided and MUST live in scratch that
    // is NOT reclaimed by `scratch_restore` before the submit. The host writes
    // these fields now (record time) but the shader reads them at execute time
    // (after submit); if a later layer's host-write reuses this offset, the
    // shader reads a garbage `kv_len` → unbounded KV loop → GPU hang. (The
    // qwen hybrid hit exactly this: per-layer scratch reuse across SSM/attn
    // layers corrupted the length. Allocate once per forward, before the
    // layer checkpoint.)
    debug_assert!(
        dyn_range.size >= crate::inference::decode_dyn::DecodeDyn::SIZE * b as u64,
        "record_batched: dyn_range too small for {b} sequences"
    );
    {
        let host_ptr = ctx
            .scratch
            .host_ptr
            .ok_or("scratch not host-visible — record_batched requires mapped memory")?;
        let mut entries = vec![crate::inference::decode_dyn::DecodeDyn::default(); b as usize];
        for (i, e) in entries.iter_mut().enumerate() {
            e.kv_len = kv_lens[i];
            e.slot = slots.map_or(i as u32, |s| s[i]);
            e.n_query = query_lens_vec[i];
            e.q_start = q_starts[i];
        }
        entries[0].k_num = k_num;
        entries[0].blocks_per_split = blocks_per_split;
        unsafe {
            let dst = host_ptr.add(dyn_range.offset as usize)
                as *mut crate::inference::decode_dyn::DecodeDyn;
            std::ptr::copy_nonoverlapping(entries.as_ptr(), dst, entries.len());
        }
    }

    let ne1 = n; // n / ne2 / ne3 computed with the split-K heuristic above
    let push_strides = [
        n,
        max_kv,
        ne1,
        ne2,
        ne3,
        q.dims[2] as u32,             // neq2
        b,                            // neq3
        k.dims[2] as u32,             // nek2
        b,                            // nek3
        v.dims[2] as u32,             // nev2
        b,                            // nev3
        1,                            // nem1 (mask disabled)
        1,                            // nem2
        1,                            // nem3
        q.element_stride[1] as u32,   // nb01
        q.element_stride[2] as u32,   // nb02
        q.element_stride[3] as u32,   // nb03
        k.element_stride[1] as u32,   // nb11
        k.element_stride[2] as u32,   // nb12
        k.element_stride[3] as u32,   // nb13
        v.element_stride[1] as u32,   // nb21
        v.element_stride[2] as u32,   // nb22
        v.element_stride[3] as u32,   // nb23
    ];

    let mut push = [0u8; FA_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    for s in push_strides {
        put_u(&mut push, &mut w, s);
    }
    put_f(&mut push, &mut w, params.scale);
    put_f(&mut push, &mut w, 0.0); // max_bias
    put_f(&mut push, &mut w, 0.0); // logit_softcap
    put_u(&mut push, &mut w, 0); // mask_kv_offset (no prefix mask on decode)
    put_f(&mut push, &mut w, 0.0); // m0
    put_f(&mut push, &mut w, 0.0); // m1
    put_u(&mut push, &mut w, params.gqa_ratio);
    put_u(&mut push, &mut w, blocks_per_split); // split_kv (blocks per split)
    put_u(&mut push, &mut w, k_num);

    let spec_constants = vec![
        FA_BC,
        params.head_dim_k,
        params.head_dim_v,
        0,                          // MASK_ENABLE (varlen masks causally in-shader)
        0,                          // Clamp
        if varlen { 1 } else { 0 }, // VARLEN
    ];
    let key = PipelineKey {
        name: variant_name.to_string(),
        binding_indices: vec![0, 1, 2, 3, 4, 5, 6],
        push_size: FA_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant_spv)?;

    if k_num <= 1 {
        // Single pass: one workgroup per (query-row, head, sequence) writes the
        // normalized output directly. data_o_split (binding 4) is unused but
        // still needs a live descriptor → bind `out`.
        let workgroups = [n, ne2, ne3];
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5, 6],
            &[
                q.range(),
                k.range(),
                v.range(),
                q.range(),   // data_m stand-in (MASK_ENABLE=0 → never read)
                out.range(), // data_o_split stand-in (k_num=1 → unused)
                out.range(),
                dyn_range,
            ],
            &push,
            workgroups,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, out.range());
    } else {
        // Split-K: grid.y packs (head, split) = ne2 * k_num; .z is the
        // sequence. Each workgroup writes its split's unnormalized O partial +
        // (L, M) into `partials`, then the combine kernel merges k_num partials
        // per (head, sequence). Direct dispatch — the batched forward always
        // re-records, so no indirect-dispatch / FA_MAX_K_NUM padding needed:
        // size partials at exactly k_num (the layout indexes by the runtime
        // k_num) and pin the grid to k_num here.
        let partials_floats = (params.head_dim_v as u64 + 2)
            * n as u64
            * ne2 as u64
            * ne3 as u64
            * k_num as u64;
        let partials = ctx.alloc_tensor([partials_floats, 1, 1, 1], GgmlType::F32)?;
        let workgroups = [n, ne2 * k_num, ne3];
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5, 6],
            &[
                q.range(),
                k.range(),
                v.range(),
                q.range(), // data_m stand-in (MASK_ENABLE=0 → never read)
                partials.range(),
                out.range(),
                dyn_range,
            ],
            &push,
            workgroups,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, partials.range());
        // The combine reads k_num from `data_dyn[0]` — pass the same batched
        // dyn_range whose entry 0 we set above (NOT ctx.decode_dyn).
        record_split_k_combine(ctx, partials, out, params.head_dim_v, n, ne2, ne3, k_num, dyn_range)?;
    }
    Ok(())
}

/// Pick the KV-split factor `(k_num, blocks_per_split)` for the main
/// flash-attn dispatch, replicating llama.cpp's heuristic
/// (`ggml_vk_flash_attn` in ggml-vulkan.cpp).
///
/// llama.cpp aims for ~`2 × shader_core_count` total workgroups, then snaps
/// each split to a `Bc`-aligned KV chunk and re-derives the split count.
/// `base_wgs` is our no-split workgroup count (`n * n_head * batch`). Because
/// our kernel launches one workgroup per *full* query head (no GQA grouping),
/// `base_wgs` already equals llama.cpp's grouped total occupancy, so dividing
/// the target by it reproduces the same total workgroup count. (We omit
/// llama.cpp's Intel-Alchemist `×2` multiplier — irrelevant on this path.)
///
/// `SEEKER_FA_SPLIT=0` disables splitting; `SEEKER_FA_SPLIT_KNUM=<n>` pins it.
pub fn pick_k_num(shader_core_count: u32, base_wgs: u32, kv: u32) -> (u32, u32) {
    let num_blocks = kv.div_ceil(FA_BC).max(1);
    if *crate::runtime_flags::FA_SPLIT_DISABLED {
        return (1, num_blocks);
    }
    if let Some(k) = *crate::runtime_flags::FA_SPLIT_KNUM {
        let k = k.clamp(1, num_blocks);
        return (k, num_blocks.div_ceil(k));
    }

    // Placeholder core count when the device doesn't report one; target is
    // 2× that, the same as llama.cpp.
    let core_count = if shader_core_count != 0 {
        shader_core_count
    } else {
        FA_SPLIT_CORE_COUNT_FALLBACK
    };
    let target = core_count * 2;

    let mut split_k = 1u32;
    if base_wgs < target {
        split_k = target / base_wgs.max(1);
    }
    if split_k <= 1 {
        return (1, num_blocks);
    }

    // Snap KV into `split_k` chunks rounded up to a multiple of Bc, then
    // re-derive split_k from the chunk size (matches llama.cpp's
    // `ROUNDUP_POW2(KV/split_k, alignment)` + `CEIL_DIV(KV, split_kv)`). This
    // self-corrects to never request more splits than there are blocks.
    let split_kv = roundup_mult((kv / split_k).max(1), FA_BC);
    let split_k = kv.div_ceil(split_kv);
    let blocks_per_split = split_kv / FA_BC; // exact: split_kv is a multiple of Bc
    (split_k, blocks_per_split)
}

/// Round `m` up to a multiple of `n` (n a power of two) — llama.cpp's
/// `ROUNDUP_POW2`.
fn roundup_mult(m: u32, n: u32) -> u32 {
    (m + n - 1) & !(n - 1)
}

/// Merge `k_num` per-split (O, L, M) partials into the final attention output
/// via `flash_attn_split_k_reduce`. One workgroup per (row, head, batch);
/// grid.y tiles HSV in BLOCK_SIZE(=32)-wide chunks.
fn record_split_k_combine(
    ctx: &mut DispatchContext,
    partials: TensorView,
    out: TensorView,
    head_dim_v: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
    k_num: u32,
    // DecodeDyn buffer the reduce kernel reads `k_num` from (`data_dyn[0]`).
    // Single-seq decode passes `ctx.decode_dyn` (k_num written via
    // `write_field_ctx`); batched decode passes its per-forward dyn_range whose
    // entry 0 carries the same k_num.
    dyn_range: crate::inference::buffer::BufferRange,
) -> Result<(), Box<dyn Error>> {
    let mut push = [0u8; FA_SPLIT_K_PUSH_BYTES as usize];
    let mut w = 0usize;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, head_dim_v); // D
    put_u(&mut push, &mut w, ne1);
    put_u(&mut push, &mut w, ne2);
    put_u(&mut push, &mut w, ne3);
    put_u(&mut push, &mut w, k_num);
    put_u(&mut push, &mut w, 0); // sinks (unused)

    let key = PipelineKey {
        name: "flash_attn_split_k_reduce_f32".to_string(),
        binding_indices: vec![0, 1, 2, 3],
        push_size: FA_SPLIT_K_PUSH_BYTES,
        spec_constants: vec![32], // BLOCK_SIZE
        required_subgroup_size: None,
    };
    let pipeline = *ctx.pipelines.get(
        ctx.device,
        key,
        shaders::FLASH_ATTN_SPLIT_K_REDUCE_F32_SPV.as_bytes(),
    )?;
    // data_a = partials, data_s = sinks (unused → dummy `out`), data_d = out, data_dyn = DecodeDyn.
    let workgroups = [ne1, head_dim_v.div_ceil(32), ne2 * ne3];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3],
        &[partials.range(), out.range(), out.range(), dyn_range],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, out.range());
    Ok(())
}

/// Cooperative-matrix flash-attention path (`flash_attn_cm1.slang`).
///
/// Same push-constant layout as the scalar path, but:
///   - Bc is fixed at 16 in the shader (not a spec constant) — only HSK,
///     HSV, MASK_ENABLE are spec-tunable. Three spec constants total.
///   - Workgroup processes `Br=16` query rows, so dispatch.x = ceil(N/16)
///     (vs the scalar shader's one workgroup per query row).
///   - The shader's `data_m` is `KV_TYPE` (F16). The model emits an F32
///     causal mask, so we cast a transient F16 copy into scratch before
///     dispatch. Mask is small (`L × L` per layer; F16 is 2× cheaper than
///     F32) so the cast cost is negligible.
///   - Pinned to wave32 (single subgroup, hardcoded in shader).
fn record_cm1(
    ctx: &mut DispatchContext,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    mask: TensorView,
    out: TensorView,
    params: FlashAttnParams,
    kv_actual: u32,
) -> Result<(), Box<dyn Error>> {
    // Cast F32 mask → F16 in scratch. Mask layout is preserved (only the
    // element dtype changes), so dims/strides flow through unchanged via
    // `record_cast`.
    let mask_f16 = ctx.alloc_tensor(mask.dims, GgmlType::F16)?;
    crate::inference::ops::cast::record_cast(ctx, mask, mask_f16)?;

    let n = q.dims[1] as u32;
    let kv = kv_actual;
    let ne1 = n;
    let ne2 = q.dims[2] as u32;
    let ne3 = q.dims[3].max(1) as u32;
    let neq2 = q.dims[2] as u32;
    let neq3 = q.dims[3].max(1) as u32;
    let nek2 = k.dims[2] as u32;
    let nek3 = k.dims[3].max(1) as u32;
    let nev2 = v.dims[2] as u32;
    let nev3 = v.dims[3].max(1) as u32;
    let nem1 = mask.dims[1] as u32;
    let nem2 = mask.dims[2].max(1) as u32;
    let nem3 = mask.dims[3].max(1) as u32;
    let nb01 = q.element_stride[1] as u32;
    let nb02 = q.element_stride[2] as u32;
    let nb03 = q.element_stride[3] as u32;
    let nb11 = k.element_stride[1] as u32;
    let nb12 = k.element_stride[2] as u32;
    let nb13 = k.element_stride[3] as u32;
    let nb21 = v.element_stride[1] as u32;
    let nb22 = v.element_stride[2] as u32;
    let nb23 = v.element_stride[3] as u32;

    let mut push = [0u8; FA_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, n);
    put_u(&mut push, &mut w, kv);
    put_u(&mut push, &mut w, ne1);
    put_u(&mut push, &mut w, ne2);
    put_u(&mut push, &mut w, ne3);
    put_u(&mut push, &mut w, neq2);
    put_u(&mut push, &mut w, neq3);
    put_u(&mut push, &mut w, nek2);
    put_u(&mut push, &mut w, nek3);
    put_u(&mut push, &mut w, nev2);
    put_u(&mut push, &mut w, nev3);
    put_u(&mut push, &mut w, nem1);
    put_u(&mut push, &mut w, nem2);
    put_u(&mut push, &mut w, nem3);
    put_u(&mut push, &mut w, nb01);
    put_u(&mut push, &mut w, nb02);
    put_u(&mut push, &mut w, nb03);
    put_u(&mut push, &mut w, nb11);
    put_u(&mut push, &mut w, nb12);
    put_u(&mut push, &mut w, nb13);
    put_u(&mut push, &mut w, nb21);
    put_u(&mut push, &mut w, nb22);
    put_u(&mut push, &mut w, nb23);
    put_f(&mut push, &mut w, params.scale);
    put_f(&mut push, &mut w, 0.0);
    put_f(&mut push, &mut w, 0.0);
    put_u(&mut push, &mut w, 0);
    put_f(&mut push, &mut w, 0.0);
    put_f(&mut push, &mut w, 0.0);
    put_u(&mut push, &mut w, params.gqa_ratio);
    put_u(&mut push, &mut w, 1);
    put_u(&mut push, &mut w, 1);

    let spec_constants = vec![
        params.head_dim_k, // HSK
        params.head_dim_v, // HSV
        1,                 // MASK_ENABLE
    ];

    let key = PipelineKey {
        name: "flash_attn_cm1_f32_f16".to_string(),
        binding_indices: vec![0, 1, 2, 3, 5],
        push_size: FA_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::FLASH_ATTN_CM1_F32_F16_SPV.as_bytes())?;
    // Workgroup.x covers Br=16 query rows; .y is heads; .z is batch.
    let workgroups = [n.div_ceil(16), ne2, ne3];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3, 5],
        &[q.range(), k.range(), v.range(), mask_f16.range(), out.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, out.range());
    Ok(())
}
