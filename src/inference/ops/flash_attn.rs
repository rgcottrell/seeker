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
    if ctx.device.coop_matrix
        && k.dtype == GgmlType::F16
        && v.dtype == GgmlType::F16
        && std::env::var("SEEKER_FA_CM").is_ok_and(|v| v == "1")
    {
        if let Some(m) = mask {
            return record_cm1(ctx, q, k, v, m, out, params);
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
    let kv = k.dims[1] as u32;
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
    let (k_num, blocks_per_split) =
        pick_k_num_clamped(ctx.device.shader_core_count, base_wgs, kv);
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
    put_u(&mut push, &mut w, 0);   // mask_n_head_log2
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
        let partials_floats = (params.head_dim_v as u64 + 2)
            * n as u64
            * ne2 as u64
            * ne3 as u64
            * FA_MAX_K_NUM as u64;
        let partials = ctx.alloc_tensor([partials_floats, 1, 1, 1], GgmlType::F32)?;

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
        record_split_k_combine(ctx, partials, out, params.head_dim_v, n, ne2, ne3, k_num)?;
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
    if std::env::var("SEEKER_FA_SPLIT").is_ok_and(|v| v == "0") {
        return (1, num_blocks);
    }
    if let Ok(v) = std::env::var("SEEKER_FA_SPLIT_KNUM") {
        if let Ok(k) = v.parse::<u32>() {
            let k = k.clamp(1, num_blocks);
            return (k, num_blocks.div_ceil(k));
        }
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
    let dyn_range = ctx.decode_dyn;
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
) -> Result<(), Box<dyn Error>> {
    // Cast F32 mask → F16 in scratch. Mask layout is preserved (only the
    // element dtype changes), so dims/strides flow through unchanged via
    // `record_cast`.
    let mask_f16 = ctx.alloc_tensor(mask.dims, GgmlType::F16)?;
    crate::inference::ops::cast::record_cast(ctx, mask, mask_f16)?;

    let n = q.dims[1] as u32;
    let kv = k.dims[1] as u32;
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
