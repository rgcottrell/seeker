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
/// `k` and `v` are `[head_dim, L, n_head_kv]`. `mask` is `[L, L]` F16 with
/// causal -inf in masked positions. `out` is `[hidden, L]` contiguous.
pub fn record(
    ctx: &mut DispatchContext,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    mask: TensorView,
    out: TensorView,
    params: FlashAttnParams,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(q.dtype, GgmlType::F32);
    debug_assert_eq!(out.dtype, GgmlType::F32);
    debug_assert_eq!(mask.dtype, GgmlType::F32, "mask is always F32 now");
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
    if ctx.device.coop_matrix
        && k.dtype == GgmlType::F16
        && v.dtype == GgmlType::F16
        && std::env::var("SEEKER_FA_CM").is_ok_and(|v| v == "1")
    {
        return record_cm1(ctx, q, k, v, mask, out, params);
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
    let nem1 = mask.dims[1] as u32; // L
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
    put_u(&mut push, &mut w, 1);   // split_kv
    put_u(&mut push, &mut w, 1);   // k_num

    let spec_constants = vec![
        32,                       // Bc default
        params.head_dim_k,        // HSK
        params.head_dim_v,        // HSV
        1,                        // MASK_ENABLE
        0,                        // Clamp
    ];

    // `flash_attn.slang` runs one workgroup per (query-row, head, batch)
    // with WORKGROUP_SIZE=32 — pin to wave32 so the workgroup maps to
    // exactly one subgroup (rather than half a wave64).
    let key = PipelineKey {
        name: variant_name.to_string(),
        binding_indices: vec![0, 1, 2, 3, 5],
        push_size: FA_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant_spv)?;
    let workgroups = [n, ne2, ne3];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3, 5],
        &[q.range(), k.range(), v.range(), mask.range(), out.range()],
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
