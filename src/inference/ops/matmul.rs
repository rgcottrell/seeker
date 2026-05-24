//! Matrix multiplication, two paths:
//!
//! * `mul_mm` — general scalar blocked-GEMM (32×32 tiles, no cooperative
//!   matrix). Used for F16 prefill (`N > 1`), where there's enough output
//!   reuse to amortize the shared-memory tile dance.
//! * `mul_mat_vec` — vector × matrix kernel optimized for `N = 1`
//!   (autoregressive decode). One workgroup per output row, 32-thread
//!   parallel dot-product reduction over K. Without this, decode wastes
//!   31/32 of each workgroup's compute and runs ~20× slower than expected.
//!
//! Dtype dispatch (A's `dtype`): F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
//! Q8_0, IQ4_NL, MXFP4 are wired through `mul_mat_vec.<variant>.spv`. For
//! N>1 with non-F16 weights, we fall back to issuing one `mul_mat_vec`
//! dispatch per output column — correct but not bandwidth-optimal for
//! large prefills. K-quants (Q2_K / Q4_K / Q5_K / Q6_K) and the IQ-family
//! quants have their own SPV variants compiled but not yet wired here.
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

/// Per-dtype dispatch info for `mul_mat_vec`. `binding_indices` tracks
/// which slots in `mul_mat_vec_head.slang` the variant actually declares
/// (descriptor-set layout must match the SPIR-V's binding decorations
/// exactly, even when the path through the kernel doesn't read every
/// alias). Slot 0 → A, 1 → B, 2 → D, 3 → A as packed16 (active when the
/// quant defines `A_TYPE_PACKED16`), 4 → B as float4 (active when
/// `B_TYPEV4` is defined — true for every wired variant).
struct MmvVariant {
    name: &'static str,
    spv: &'static [u8],
    binding_indices: &'static [u32],
}

const MMV_BINDINGS_NO_PACKED16: &[u32] = &[0, 1, 2, 4];
const MMV_BINDINGS_PACKED16: &[u32] = &[0, 1, 2, 3, 4];
/// K-quant variants that need both packed16 and packed32 aliases of A
/// (slots 3 and 6 — see `mul_mat_vec_head.slang`). Q4_K and Q5_K want
/// 32-bit-wide reads of `qs[]`; Q6_K is packed16-only.
const MMV_BINDINGS_PACKED16_AND_32: &[u32] = &[0, 1, 2, 3, 4, 6];

fn mmv_variant(dtype: GgmlType) -> Option<MmvVariant> {
    let v = match dtype {
        GgmlType::F32 => MmvVariant {
            name: "mul_mat_vec_f32",
            spv: shaders::MUL_MAT_VEC_F32_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_NO_PACKED16,
        },
        GgmlType::F16 => MmvVariant {
            name: "mul_mat_vec_f16",
            spv: shaders::MUL_MAT_VEC_F16_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_NO_PACKED16,
        },
        GgmlType::BF16 => MmvVariant {
            name: "mul_mat_vec_bf16",
            spv: shaders::MUL_MAT_VEC_BF16_SPV.as_bytes(),
            // BF16 now defines A_TYPE_PACKED16 = uint (see types.slang) so
            // the vectorized K_PER_ITER=8 path lands and the host has to
            // bind A on slot 3 as well as 0.
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        GgmlType::Q4_0 => MmvVariant {
            name: "mul_mat_vec_q4_0",
            spv: shaders::MUL_MAT_VEC_Q4_0_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        GgmlType::Q4_1 => MmvVariant {
            name: "mul_mat_vec_q4_1",
            spv: shaders::MUL_MAT_VEC_Q4_1_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        GgmlType::Q5_0 => MmvVariant {
            name: "mul_mat_vec_q5_0",
            spv: shaders::MUL_MAT_VEC_Q5_0_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        GgmlType::Q5_1 => MmvVariant {
            name: "mul_mat_vec_q5_1",
            spv: shaders::MUL_MAT_VEC_Q5_1_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        GgmlType::Q8_0 => MmvVariant {
            name: "mul_mat_vec_q8_0",
            spv: shaders::MUL_MAT_VEC_Q8_0_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        GgmlType::IQ4_NL => MmvVariant {
            name: "mul_mat_vec_iq4_nl",
            spv: shaders::MUL_MAT_VEC_IQ4_NL_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        GgmlType::MXFP4 => MmvVariant {
            name: "mul_mat_vec_mxfp4",
            spv: shaders::MUL_MAT_VEC_MXFP4_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_NO_PACKED16,
        },
        // K-quants — these have a separate kernel per dtype (no `q4_k`
        // variant in `mul_mat_vec.slang`; see `mul_mat_vec_q4_k.slang`
        // and siblings). All use the same `MulMatVecParams` push-constant
        // layout via `mul_mat_vec_head.slang`.
        GgmlType::Q4_K => MmvVariant {
            name: "mul_mat_vec_q4_k",
            spv: shaders::MUL_MAT_VEC_Q4_K_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16_AND_32,
        },
        GgmlType::Q5_K => MmvVariant {
            name: "mul_mat_vec_q5_k",
            spv: shaders::MUL_MAT_VEC_Q5_K_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16_AND_32,
        },
        GgmlType::Q6_K => MmvVariant {
            name: "mul_mat_vec_q6_k",
            spv: shaders::MUL_MAT_VEC_Q6_K_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        _ => return None,
    };
    Some(v)
}

/// Record a single matmul: `dst[m, n] = sum_k a[k, m] * b[k, n]`. ggml's
/// natural layout — A has shape [K, M], B has shape [K, N], D has shape
/// [M, N]. Inner dim K is the contracting one. Both A and B store K as
/// `ne[0]` (innermost). Output stride_d = M.
///
/// A may be any wired dtype (see `mmv_variant`); B and D are F32. Emits a
/// trailing compute→compute barrier on the scratch buffer; callers that
/// dispatch independent matmuls back-to-back (same input, disjoint
/// outputs — e.g. Q/K/V or ffn_gate/ffn_up) should use [`record_nofence`]
/// for all but the last and let one barrier cover the group.
pub fn record(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
) -> Result<(), Box<dyn Error>> {
    record_inner(ctx, a, b, d, /*fence=*/ true)
}

/// Same as [`record`] but skips the trailing barrier. Use only when the
/// next dispatch you record either reads disjoint memory or is followed
/// by a real barrier itself; otherwise you'll race the output buffer.
pub fn record_nofence(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
) -> Result<(), Box<dyn Error>> {
    record_inner(ctx, a, b, d, /*fence=*/ false)
}

fn record_inner(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(b.dtype, GgmlType::F32, "matmul B must be F32");
    debug_assert_eq!(d.dtype, GgmlType::F32, "matmul D must be F32");
    debug_assert_eq!(b.dims[0], a.dims[0], "matmul K mismatch");
    debug_assert_eq!(d.dims[0], a.dims[1], "matmul output M mismatch");
    debug_assert_eq!(d.dims[1], b.dims[1], "matmul output N mismatch");

    let n = b.dims[1];
    if a.dtype == GgmlType::F16 && n > 1 {
        return record_mul_mm(ctx, a, b, d, fence);
    }

    let variant = mmv_variant(a.dtype).ok_or_else(|| {
        format!(
            "matmul: weight dtype {:?} not yet wired (shader may exist — see ops/matmul.rs)",
            a.dtype
        )
    })?;

    if n == 1 {
        record_mul_mat_vec(ctx, &variant, a, b, d, fence)?;
    } else {
        // Per-column fallback: dispatch one `mul_mat_vec` per output column.
        // Correct but bandwidth-suboptimal — replace with a dedicated
        // dtype-specific `mul_mm` once those kernels are wired (or use
        // cooperative-matrix `mul_mm_cm` when the device exposes it).
        // Intra-group barriers are unnecessary (each column writes a
        // disjoint slice of D), so only fence after the last column.
        let last = n - 1;
        for col in 0..n {
            let b_col = slice_col(b, col);
            let d_col = slice_col(d, col);
            let col_fence = fence && col == last;
            record_mul_mat_vec(ctx, &variant, a, b_col, d_col, col_fence)?;
        }
    }
    Ok(())
}

/// Take column `col` of a `[K, N, …]` tensor as a `[K, 1, …]` view.
/// Byte_stride is preserved so the underlying buffer offsets stay correct;
/// only `byte_offset`, `byte_size`, and `dims[1]` change.
fn slice_col(t: TensorView, col: u64) -> TensorView {
    let col_bytes = t.byte_stride[1];
    let mut dims = t.dims;
    dims[1] = 1;
    TensorView {
        byte_offset: t.byte_offset + col * col_bytes,
        byte_size: col_bytes,
        dims,
        ..t
    }
}

/// General GEMM (`mul_mm.slang`, F16 × F32 → F32). Worth running when
/// `N > 1` (prefill or batched eval).
fn record_mul_mm(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
    fence: bool,
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
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    }
    Ok(())
}

/// Vector-matrix kernel (`mul_mat_vec.slang`, A × F32 → F32, N=1).
/// Dispatches `[M, batch, 1]` workgroups; each WG computes one output row
/// via 32 threads' dot-product reduction across K.
///
/// Bindings (see `mul_mat_vec_head.slang`):
///   0 → data_a              (weight, dtype = `a.dtype`)
///   1 → data_b              (input vector, F32)
///   2 → data_d              (output, F32)
///   3 → data_a_packed16     (alias of A for quants with `A_TYPE_PACKED16`;
///                            same VkBuffer, just a different element type)
///   4 → data_b_v4           (float4 alias of B; declared by every wired
///                            variant via `B_TYPEV4=float4`)
fn record_mul_mat_vec(
    ctx: &mut DispatchContext,
    variant: &MmvVariant,
    a: TensorView,
    b: TensorView,
    d: TensorView,
    fence: bool,
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

    // Build the bindings array in the same order as `variant.binding_indices`.
    // Slot meanings (see `mul_mat_vec_head.slang`):
    //   0 = A,  3 = A aliased as packed16,  6 = A aliased as packed32
    //   1 = B,  4 = B aliased as float4,    5 = B aliased as float2 (Q5_K)
    //   2 = D
    let bindings: Vec<_> = variant
        .binding_indices
        .iter()
        .map(|&slot| match slot {
            0 | 3 | 6 => a.range(),
            1 | 4 | 5 => b.range(),
            2 => d.range(),
            other => panic!("unexpected mul_mat_vec binding slot {other}"),
        })
        .collect();

    let key = PipelineKey {
        name: variant.name.to_string(),
        binding_indices: variant.binding_indices.to_vec(),
        push_size: MUL_MAT_VEC_PARAMS_BYTES,
        spec_constants: Vec::new(),
    };
    let (pipeline, layout, set_layout) = {
        let p: &CachedPipeline = ctx.pipelines.get(ctx.device, key, variant.spv)?;
        (p.pipeline, p.layout, p.set_layout)
    };
    let set = ctx.descriptors.allocate_and_write_indexed(
        ctx.device,
        set_layout,
        variant.binding_indices,
        &bindings,
    )?;

    // Each workgroup produces `NUM_ROWS` output rows (see
    // `mul_mat_vec_head.slang`). Keep this in sync with the shader's
    // `static const uint NUM_ROWS = …`. The shader's per-thread bounds
    // check tolerates an over-dispatch on the last X tile when M is not
    // divisible by NUM_ROWS, so `div_ceil` is correct.
    const NUM_ROWS: u32 = 2;
    let workgroups = [m.div_ceil(NUM_ROWS), num_batches, 1];
    let cached = CachedPipeline { pipeline, layout, set_layout };
    record_dispatch(ctx.device, ctx.cmd, &cached, set, &push, workgroups);
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    }
    let _ = MUL_MAT_VEC_BLOCK_SIZE; // documented above; not currently used host-side
    Ok(())
}
