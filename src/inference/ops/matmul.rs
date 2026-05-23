//! Matrix multiplication, two paths:
//!
//! * `mul_mm` — general scalar blocked-GEMM (32×32 tiles, no cooperative
//!   matrix). Used for prefill (`N > 1`), where there's enough output reuse
//!   to amortize the shared-memory tile dance.
//! * `mul_mat_vec` — vector × matrix kernel optimized for `N = 1`
//!   (autoregressive decode). One workgroup per output row, 32-thread
//!   parallel dot-product reduction over K. Without this, decode wastes
//!   31/32 of each workgroup's compute and runs ~20× slower than expected.
//!
//! Push constants follow `MulMmParams` and `MulMatVecParams` respectively
//! (see `shaders/compute/mul_mm.slang` and `shaders/include/mul_mat_vec_head.slang`).

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
const MUL_MAT_VEC_PARAMS_BYTES: u32 = 13 * 4;
/// `numthreads(BLOCK_SIZE, 1, 1)` in `mul_mat_vec.slang`. Each WG produces
/// one output row via a 32-thread dot-product reduction.
const MUL_MAT_VEC_BLOCK_SIZE: u32 = 32;

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
    debug_assert_eq!(b.dims[0], a.dims[0], "matmul K mismatch");
    debug_assert_eq!(d.dims[0], a.dims[1], "matmul output M mismatch");
    debug_assert_eq!(d.dims[1], b.dims[1], "matmul output N mismatch");

    if b.dims[1] == 1 {
        record_mul_mat_vec(ctx, a, b, d)
    } else {
        record_mul_mm(ctx, a, b, d)
    }
}

/// General GEMM (`mul_mm.slang`, F16 × F32 → F32). Worth running when
/// `N > 1` (prefill or batched eval).
fn record_mul_mm(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
) -> Result<(), Box<dyn Error>> {
    let k = a.dims[0] as u32;
    let m = a.dims[1] as u32;
    let n = b.dims[1] as u32;

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

/// Vector-matrix kernel (`mul_mat_vec.slang`, F16 × F32 → F32, N=1).
/// Dispatches `[M, batch, 1]` workgroups; each WG computes one output row
/// via 32 threads' dot-product reduction across K.
///
/// Bindings:
///   0 → data_a              (weight, F16)
///   1 → data_b              (input vector, F32)
///   2 → data_d              (output, F32)
///   4 → data_b_v4 (= b)     (float4 alias of B; the F16 path's `K_PER_ITER=2`
///                            doesn't read it, but the head declares it
///                            unconditionally for the variant macros, so the
///                            descriptor must be valid).
fn record_mul_mat_vec(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(b.dims[1], 1, "mul_mat_vec requires N=1");

    let ncols = a.dims[0] as u32; // K
    let m = a.dims[1] as u32; // output rows
    let stride_a = a.element_stride[1] as u32; // = K
    let stride_b = b.element_stride[1] as u32; // = K
    let stride_d = m; // shader's row bound: `if first_row < stride_d`
    let batch_stride_a = a.element_stride[2] as u32;
    let batch_stride_b = b.element_stride[2] as u32;
    let batch_stride_d = d.element_stride[2] as u32;
    let ne02 = a.dims[2].max(1) as u32;
    let ne12 = b.dims[2].max(1) as u32;
    let num_batches = (ne12 * b.dims[3].max(1) as u32).max(1);
    let broadcast2 = (ne12 / ne02).max(1);
    let broadcast3 = (b.dims[3].max(1) / a.dims[3].max(1)) as u32;

    let mut push = [0u8; MUL_MAT_VEC_PARAMS_BYTES as usize];
    let fields = [
        ncols,
        stride_a,
        stride_b,
        stride_d,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d,
        0, // fusion_flags (reserved)
        0, // base_work_group_y — we issue a single dispatch per call
        ne02,
        ne12,
        broadcast2,
        broadcast3,
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }

    // Sparse bindings: 0 (data_a), 1 (data_b), 2 (data_d), 4 (data_b_v4).
    // Binding 3 (data_a_packed16) is unused for the F16 variant.
    let binding_indices: Vec<u32> = vec![0, 1, 2, 4];
    let bindings = [a.range(), b.range(), d.range(), b.range()];

    let key = PipelineKey {
        name: "mul_mat_vec_f16".to_string(),
        binding_indices: binding_indices.clone(),
        push_size: MUL_MAT_VEC_PARAMS_BYTES,
        spec_constants: Vec::new(),
    };
    let (pipeline, layout, set_layout) = {
        let p: &CachedPipeline = ctx
            .pipelines
            .get(ctx.device, key, shaders::MUL_MAT_VEC_F16_SPV.as_bytes())?;
        (p.pipeline, p.layout, p.set_layout)
    };
    let set = ctx.descriptors.allocate_and_write_indexed(
        ctx.device,
        set_layout,
        &binding_indices,
        &bindings,
    )?;

    // One WG per output row; y dim covers batch (unused here but kept for
    // shape consistency with the shader's `wg_id.y` batch lookup).
    let workgroups = [m, num_batches, 1];
    let cached = CachedPipeline { pipeline, layout, set_layout };
    record_dispatch(ctx.device, ctx.cmd, &cached, set, &push, workgroups);
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    let _ = MUL_MAT_VEC_BLOCK_SIZE; // documented above; not currently used host-side
    Ok(())
}
