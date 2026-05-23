//! `mul_mm` dispatch — scalar blocked-GEMM (no cooperative matrix).
//! Variant `f16_f32`: F16 weights × F32 activations → F32 output.
//!
//! Push constants follow `MulMmParams` (shaders/compute/mul_mm.slang), which
//! mirrors llama.cpp's `vk_mat_mat_push_constants`. Workgroup count is
//! `(ceil(M/BM), ceil(N/BN), num_batches)` with BM=BN=32.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::command::{record_compute_barrier, record_dispatch};
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::{CachedPipeline, PipelineKey};
use crate::inference::weights::TensorView;
use crate::shaders;

const BM: u32 = 32;
const BN: u32 = 32;

const MUL_MM_PARAMS_BYTES: u32 = 14 * 4;

/// Record a single matmul: `dst[m, n] = sum_k a[k, m] * b[k, n]`. ggml's
/// natural layout — A has shape [K, M], B has shape [K, N], D has shape
/// [M, N]. Inner dim K is the contracting one. Both A and B store K as
/// `ne[0]` (innermost). Output stride_d = M.
///
/// A is expected to be F16 (weight); B and D are F32 (activations and output).
pub fn record(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(a.dtype, GgmlType::F16, "matmul A must be F16");
    debug_assert_eq!(b.dtype, GgmlType::F32, "matmul B must be F32");
    debug_assert_eq!(d.dtype, GgmlType::F32, "matmul D must be F32");

    // ggml: src0=A (weight) shape [K, M], src1=B (activations) shape [K, N].
    let k = a.dims[0] as u32;
    let m = a.dims[1] as u32;
    let n = b.dims[1] as u32;
    debug_assert_eq!(b.dims[0] as u32, k, "matmul K mismatch");
    debug_assert_eq!(d.dims[0] as u32, m, "matmul output M mismatch");
    debug_assert_eq!(d.dims[1] as u32, n, "matmul output N mismatch");

    let stride_a = a.element_stride[1] as u32; // = K
    let stride_b = b.element_stride[1] as u32; // = K
    let stride_d = d.element_stride[1] as u32; // = M
    let batch_stride_a = a.element_stride[2] as u32;
    let batch_stride_b = b.element_stride[2] as u32;
    let batch_stride_d = d.element_stride[2] as u32;
    let ne02 = a.dims[2].max(1) as u32;
    let ne12 = b.dims[2].max(1) as u32;
    let num_batches = (ne12 * b.dims[3].max(1) as u32).max(1);
    let broadcast2 = (ne12 / ne02).max(1);
    let broadcast3 = (b.dims[3].max(1) / a.dims[3].max(1)) as u32;

    let mut push = [0u8; MUL_MM_PARAMS_BYTES as usize];
    let fields = [
        m,
        n,
        k,
        stride_a,
        stride_b,
        stride_d,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d,
        num_batches,
        ne02,
        ne12,
        broadcast2,
        broadcast3,
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }

    let key = PipelineKey::dense("mul_mm_f16_f32", 3, MUL_MM_PARAMS_BYTES, Vec::new());
    let (pipeline, layout, set_layout) = {
        let p: &CachedPipeline = ctx
            .pipelines
            .get(ctx.device, key, shaders::MUL_MM_F16_F32_SPV.as_bytes())?;
        (p.pipeline, p.layout, p.set_layout)
    };

    let set = ctx.descriptors.allocate_and_write(
        ctx.device,
        set_layout,
        &[a.range(), b.range(), d.range()],
    )?;

    let cached = CachedPipeline {
        pipeline,
        layout,
        set_layout,
    };
    let workgroups = [m.div_ceil(BM), n.div_ceil(BN), num_batches];
    record_dispatch(ctx.device, ctx.cmd, &cached, set, &push, workgroups);
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}
