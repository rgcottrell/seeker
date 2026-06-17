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
//! Dtype dispatch (A's `dtype`): F32, F16, BF16, the legacy quants (Q4_0,
//! Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, MXFP4), the K-quants (Q2_K, Q3_K, Q4_K,
//! Q5_K, Q6_K), and the IQ-family (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
//! IQ3_XXS, IQ3_S, IQ4_XS) are all wired here, each through its own
//! `mul_mat_vec.<variant>.spv`. For N>1 with a weight whose dtype has no
//! coopmat (`mul_mm_cm`) variant, we fall back to issuing one `mul_mat_vec`
//! dispatch per output column — correct but not bandwidth-optimal for large
//! prefills (the K-quants and IQ4_XS do have coopmat; Q2_K/Q3_K and the other
//! IQ quants take the per-column path).
//!
//! Push constants follow `MulMmParams` and `MulMatVecParams` respectively
//! (see `shaders/compute/mul_mm.slang` and `shaders/include/mul_mat_vec_head.slang`).

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

const BM: u32 = 32;
const BN: u32 = 32;

/// Coopmat (`mul_mm_cm.slang`) output-tile dims — must match the shader's
/// `BM`/`BN`. Separate from the scalar `mul_mm` 32×32 tile: the coopmat kernel
/// register-blocks a 64×64 tile (2×2 subgroups × 2×2 fragments). The last
/// row/col tile straddles unless M/N are multiples of these, so a non-multiple
/// selects the kernel's `ALLOW_PARTIAL_N` staged store.
const MM_CM_BM: u32 = 64;
const MM_CM_BN: u32 = 64;
/// `mul_mm_cm.slang`'s `BK` (K-step). Split-K slices must stay BK-aligned.
const MM_CM_BK: u32 = 32;

const MUL_MM_PARAMS_BYTES: u32 = 14 * 4;
const MUL_MM_CM_PARAMS_BYTES: u32 = 14 * 4;
const MUL_MAT_VEC_PARAMS_BYTES: u32 = 13 * 4;
/// `numthreads(BLOCK_SIZE, 1, 1)` in `mul_mat_vec.slang`. Each WG produces
/// one output row via a 32-thread dot-product reduction.
const MUL_MAT_VEC_BLOCK_SIZE: u32 = 32;

/// Per-dtype dispatch info for `mul_mat_vec`. `binding_indices` lists the
/// descriptor slots the host binds for this variant; it must be a *superset*
/// of the bindings the compiled SPIR-V declares (Vulkan permits a descriptor
/// layout wider than the shader's static use — slangc `-O3` strips any alias
/// the kernel never reads, so the surviving set is often smaller than the
/// aliases `mul_mat_vec_head.slang` defines). Determine the minimal-correct
/// set per shader with `spirv-dis … | grep Binding`. Slot map:
///   0 → A,  3 → A as packed16,  6 → A as packed32
///   1 → B,  4 → B as float4,    5 → B as float2
///   2 → D
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
/// Q5_K: like Q4_K it reads A via packed16 (3) + packed32 (6), but the B side
/// uses the float2 alias `data_b_v2` (slot 5) rather than the float4 alias
/// (slot 4) Q4_K uses — so its binding set is `{0,1,2,3,5,6}`, not `…,4,…`.
const MMV_BINDINGS_Q5_K: &[u32] = &[0, 1, 2, 3, 5, 6];
/// Q2_K / Q3_K (`mul_mat_vec_q2_k.slang`, `…q3_k.slang`): A (0) + packed16
/// alias (3), D (2), B via the float2 alias `data_b_v2` (5). slangc -O3 leaves
/// the SPIR-V decorating exactly `{0,2,3,5}` (verified with spirv-dis) — the
/// same alias set Q5_K's kernel uses, minus the over-declared slots.
const MMV_BINDINGS_K_DMIN_V2: &[u32] = &[0, 2, 3, 5];
/// i-quant matvec reading the block scale straight from `data_a` (no packed16
/// alias), B as float4: IQ1_S / IQ1_M / IQ2_XS. SPIR-V decorates `{0,2,4}`.
const MMV_BINDINGS_IQ_V4: &[u32] = &[0, 2, 4];
/// i-quant matvec that also reads `data_a_packed16` (3) for its sign/scale
/// field, B as float4: IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S. SPIR-V `{0,2,3,4}`.
const MMV_BINDINGS_IQ_P16_V4: &[u32] = &[0, 2, 3, 4];
/// IQ4_XS (generic `mul_mat_vec.slang` iq4_xs variant): B as float4 (4) plus
/// the packed32 alias of A (6). SPIR-V decorates `{0,2,4,6}`.
const MMV_BINDINGS_IQ4_XS: &[u32] = &[0, 2, 4, 6];

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
            binding_indices: MMV_BINDINGS_Q5_K,
        },
        GgmlType::Q6_K => MmvVariant {
            name: "mul_mat_vec_q6_k",
            spv: shaders::MUL_MAT_VEC_Q6_K_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_PACKED16,
        },
        // Q2_K / Q3_K — dedicated per-dtype kernels (`mul_mat_vec_q2_k.slang`,
        // `…q3_k.slang`), same `MulMatVecParams` layout via the shared head.
        GgmlType::Q2_K => MmvVariant {
            name: "mul_mat_vec_q2_k",
            spv: shaders::MUL_MAT_VEC_Q2_K_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_K_DMIN_V2,
        },
        GgmlType::Q3_K => MmvVariant {
            name: "mul_mat_vec_q3_k",
            spv: shaders::MUL_MAT_VEC_Q3_K_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_K_DMIN_V2,
        },
        // IQ-family — per-dtype kernels ported from llama.cpp; codebook grids
        // are embedded constants copied to groupshared via `init_iq_shmem()`
        // (no extra buffer binding). Prefill (N>1) reuses these via the
        // per-column matvec fallback in `record_inner` (no coopmat variant).
        GgmlType::IQ1_S => MmvVariant {
            name: "mul_mat_vec_iq1_s",
            spv: shaders::MUL_MAT_VEC_IQ1_S_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_IQ_V4,
        },
        GgmlType::IQ1_M => MmvVariant {
            name: "mul_mat_vec_iq1_m",
            spv: shaders::MUL_MAT_VEC_IQ1_M_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_IQ_V4,
        },
        GgmlType::IQ2_XS => MmvVariant {
            name: "mul_mat_vec_iq2_xs",
            spv: shaders::MUL_MAT_VEC_IQ2_XS_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_IQ_V4,
        },
        GgmlType::IQ2_XXS => MmvVariant {
            name: "mul_mat_vec_iq2_xxs",
            spv: shaders::MUL_MAT_VEC_IQ2_XXS_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_IQ_P16_V4,
        },
        GgmlType::IQ2_S => MmvVariant {
            name: "mul_mat_vec_iq2_s",
            spv: shaders::MUL_MAT_VEC_IQ2_S_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_IQ_P16_V4,
        },
        GgmlType::IQ3_XXS => MmvVariant {
            name: "mul_mat_vec_iq3_xxs",
            spv: shaders::MUL_MAT_VEC_IQ3_XXS_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_IQ_P16_V4,
        },
        GgmlType::IQ3_S => MmvVariant {
            name: "mul_mat_vec_iq3_s",
            spv: shaders::MUL_MAT_VEC_IQ3_S_DEFAULT_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_IQ_P16_V4,
        },
        GgmlType::IQ4_XS => MmvVariant {
            name: "mul_mat_vec_iq4_xs",
            spv: shaders::MUL_MAT_VEC_IQ4_XS_SPV.as_bytes(),
            binding_indices: MMV_BINDINGS_IQ4_XS,
        },
        _ => return None,
    };
    Some(v)
}

/// Cooperative-matrix matmul variant for `mul_mm_cm.slang`. B is always
/// F32 in our call sites (the model casts the activation side to F32 at
/// the rms_norm boundary), so we pick the `*_f32` shader variant for
/// every A dtype.
struct MmCmVariant {
    name: &'static str,
    spv: &'static [u8],
}

fn mmcm_variant(dtype: GgmlType) -> Option<MmCmVariant> {
    let v = match dtype {
        GgmlType::F32 => MmCmVariant {
            name: "mul_mm_cm_f32",
            spv: shaders::MUL_MM_CM_F32_SPV.as_bytes(),
        },
        GgmlType::F16 => MmCmVariant {
            name: "mul_mm_cm_f16_f32",
            spv: shaders::MUL_MM_CM_F16_F32_SPV.as_bytes(),
        },
        GgmlType::BF16 => MmCmVariant {
            name: "mul_mm_cm_bf16",
            spv: shaders::MUL_MM_CM_BF16_SPV.as_bytes(),
        },
        GgmlType::Q4_0 => MmCmVariant {
            name: "mul_mm_cm_q4_0_f32",
            spv: shaders::MUL_MM_CM_Q4_0_F32_SPV.as_bytes(),
        },
        GgmlType::Q8_0 => MmCmVariant {
            name: "mul_mm_cm_q8_0_f32",
            spv: shaders::MUL_MM_CM_Q8_0_F32_SPV.as_bytes(),
        },
        GgmlType::Q4_K => MmCmVariant {
            name: "mul_mm_cm_q4_k_f32",
            spv: shaders::MUL_MM_CM_Q4_K_F32_SPV.as_bytes(),
        },
        GgmlType::Q5_K => MmCmVariant {
            name: "mul_mm_cm_q5_k_f32",
            spv: shaders::MUL_MM_CM_Q5_K_F32_SPV.as_bytes(),
        },
        GgmlType::Q6_K => MmCmVariant {
            name: "mul_mm_cm_q6_k_f32",
            spv: shaders::MUL_MM_CM_Q6_K_F32_SPV.as_bytes(),
        },
        GgmlType::MXFP4 => MmCmVariant {
            name: "mul_mm_cm_mxfp4_f32",
            spv: shaders::MUL_MM_CM_MXFP4_F32_SPV.as_bytes(),
        },
        // IQ4_XS has a coopmat variant (the other newly-wired quants don't —
        // they fall through to the per-column matvec prefill path, correct but
        // slower). Gives IQ4_XS the fast N≥32 prefill path like the K-quants.
        GgmlType::IQ4_XS => MmCmVariant {
            name: "mul_mm_cm_iq4_xs_f32",
            spv: shaders::MUL_MM_CM_IQ4_XS_F32_SPV.as_bytes(),
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

/// `d += a @ b`. Fuses the residual-add that follows out-projection matmuls in
/// the attention and SSM blocks into the matmul kernel via the `ACCUMULATE` spec
/// constant on `mul_mat_vec_head.slang` — `d` is the residual buffer
/// (read-modify-write), so no separate `proj` scratch + elementwise add is
/// needed. Handles any N:
///   - N == 1 (single-seq decode): the fused matvec-accumulate.
///   - N ≤ MAX_BATCH_COLS (B>1 decode): the fused NUM_COLS matvec-accumulate
///     (reads the weight once across columns AND fuses the add).
///   - otherwise (prefill large-N → coopmat, or unwired): matmul into a scratch
///     then a separate elementwise add — identical to the prior call-site code.
///
/// Byte-identical across all three (the add is the same F32 `residual + sum`).
pub fn record_accumulate(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(d.dtype, GgmlType::F32);
    let n = b.dims[1];
    let variant = mmv_variant(a.dtype);

    if n == 1 {
        let variant = variant.as_ref().ok_or_else(|| {
            format!(
                "matmul accumulate: weight dtype {:?} not yet wired",
                a.dtype
            )
        })?;
        return record_mul_mat_vec_with_flags(
            ctx, variant, a, b, d, /*fence=*/ true, /*accumulate=*/ true,
        );
    }

    // B>1 decode: fuse the residual add into the batched-column matvec. Same
    // eligibility as the non-accumulate batched path (no tensor batch, N ≤ cap).
    // Fire for exactly the dtypes `record_inner` sends to its batched-column
    // path (quant + F32), so the fused result matches the fallback. F16 n>1
    // routes to `record_mul_mm` in record_inner, and coopmat needs n≥32 (never
    // at decode) — both land in the fallback below, byte-identical to before.
    if let Some(variant) = variant.as_ref()
        && a.dtype != GgmlType::F16
        && !*crate::runtime_flags::MM_BATCH_DISABLED
        && n <= MAX_BATCH_COLS as u64
        && b.dims[2] <= 1
        && b.dims[3] <= 1
        && a.dims[2] <= 1
        && a.dims[3] <= 1
    {
        return record_mul_mat_vec_cols(
            ctx, variant, a, b, d, n as u32, /*accumulate=*/ true, /*fence=*/ true,
        );
    }

    // Fallback: matmul into a scratch then add into `d` (prefill large-N takes
    // the coopmat `mul_mm` path inside `record_inner`). Identical to the prior
    // `matmul::record(..) + elementwise::record_add(..)` at the call sites.
    let proj = ctx.alloc_tensor([a.dims[1], n, 1, 1], GgmlType::F32)?;
    record_inner(ctx, a, b, proj, /*fence=*/ true)?;
    super::elementwise::record_add(ctx, d, proj, d)
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

    // Count the weight bytes this matmul reads toward the prefill flush budget
    // (no-op unless this forward has flushing enabled). All dense matmul callers
    // (record / record_nofence / record_accumulate-prefill-fallback) funnel here.
    ctx.account_matmul(a.byte_size);

    let n = b.dims[1];

    // Cooperative-matrix prefill path (KHR_cooperative_matrix). Pulls in
    // the `mul_mm_cm.slang` 16×16×16 fp16-acc-fp32 fragment kernel. Gated
    // by:
    //   - device.coop_matrix (the KHR extension is enabled and the device
    //     reports CoopMat support)
    //   - mmcm_variant for this A dtype is wired
    //   - n >= 32 — N need NOT be a multiple of 16: a non-16-aligned last
    //     tile is served by the kernel's `ALLOW_PARTIAL_N` store (staged
    //     through groupshared, in-bounds columns only; see the dispatch
    //     below). M in the Llama path is always a multiple of 32 (hidden=2048,
    //     n_ff=8192, vocab=128256, n_kv*head_dim=512), so we don't gate on
    //     M alignment beyond the 16-multiple requirement just below.
    //
    // Verified ~3× prefill win on Llama-1B Q4_K_M and ~1.4× on
    // qwen35moe Q4_K_XL @ N=320; default-on. The partial-N path also lets the
    // qwen3-vl vision tower (n_pos e.g. 16104) use coopmat instead of the
    // per-column matvec fallback — 13–26% faster encode on small/medium images,
    // ~neutral at full-res (attention-bound). `SEEKER_MM_CM=0` opts out.
    // M (output rows) must also be a multiple of 16: the CoopMat store writes
    // full 16×16 fragments, so a non-16-aligned M would mishandle / overrun the
    // partial output fragment — observed as NONDETERMINISTIC output (and likely
    // OOB writes) for qwen35moe's shared-expert gate, an M=1 (hidden→1)
    // projection. (Llama's M values are all multiples of 32, which is why this
    // was latent.) Non-aligned M falls through to mul_mm / per-column matvec,
    // which handle arbitrary M deterministically.
    let mm_cm_enabled = !*crate::runtime_flags::MM_CM_DISABLED;
    if ctx.device.coop_matrix
        && n >= 32
        && a.dims[1].is_multiple_of(16)
        && mm_cm_enabled
        && let Some(variant) = mmcm_variant(a.dtype)
    {
        // N need not be a multiple of 16: when it isn't, the kernel's
        // `ALLOW_PARTIAL_N` path stages the straddling last tile through
        // groupshared and writes only in-bounds columns (the vision tower's
        // n_pos, e.g. 16104, is not 16-aligned). The aligned path (N % 16 == 0,
        // text decoder) is byte-identical and skips that staging.
        return record_mul_mm_cm(ctx, &variant, a, b, d, fence);
    }

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
        // Split-K opt-in for huge-M Q8_0 matvecs (lm_head — 248k × 2048).
        // The single-pass kernel already issues vocab/NUM_ROWS workgroups
        // which saturates RDNA's CU count; split-K trades a reduce pass
        // for fewer iterations per WG, occasionally helping when the
        // per-WG work is large enough to bottleneck on memory issue rate.
        // Gate on `SEEKER_MM_SPLIT_K=<n>` for now (no auto-pick); 0
        // disables. Only Q8_0 (the lm_head dtype) is wired.
        if a.dtype == GgmlType::Q8_0
            && let Some(split_k) = pick_mm_split_k(a.dims[0] as u32, a.dims[1] as u32)
        {
            return record_mul_mat_vec_split_k(ctx, &variant, a, b, d, split_k, fence);
        }
        record_mul_mat_vec(ctx, &variant, a, b, d, fence)?;
    } else if !*crate::runtime_flags::MM_BATCH_DISABLED
        && n <= MAX_BATCH_COLS as u64
        && b.dims[2] <= 1
        && b.dims[3] <= 1
        && a.dims[2] <= 1
        && a.dims[3] <= 1
    {
        // Batched-column dispatch: read the weight once for all N columns
        // (NUM_COLS=N) instead of re-reading it per column. Byte-identical.
        // Gated to the no-tensor-batch case (lm_head + dense projections at B>1)
        // and N ≤ MAX_BATCH_COLS so the `temp[NUM_COLS][NUM_ROWS]` register
        // tile stays bounded; wider N falls back to the per-column loop.
        record_mul_mat_vec_cols(
            ctx, &variant, a, b, d, n as u32, /*accumulate=*/ false, fence,
        )?;
    } else {
        // Per-column fallback: dispatch one `mul_mat_vec` per output column.
        // Correct but bandwidth-suboptimal (re-reads the full weight per
        // column). Reached when `SEEKER_MM_BATCH=0`, N > MAX_BATCH_COLS, or a
        // real tensor (dim-2/3) batch — otherwise the batched path above runs.
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

    // `mul_mm.slang` bakes in `SUBGROUP_SIZE = 32` (mul_mm.slang:29) — pin
    // the pipeline to wave32.
    let key = PipelineKey::dense("mul_mm_f16_f32", 3, MUL_MM_PARAMS_BYTES, Vec::new())
        .with_subgroup_size(32);
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::MUL_MM_F16_F32_SPV.as_bytes())?;
    let workgroups = [m.div_ceil(BM), n.div_ceil(BN), num_batches];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[a.range(), b.range(), d.range()],
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, d.range());
    }
    Ok(())
}

/// Cooperative-matrix prefill (`mul_mm_cm.slang`, A × F32 → F32, N ≥ 32).
/// Same push-constant layout and binding layout as `record_mul_mm`. Pinned
/// to wave32 (the shader hardcodes `SUBGROUP_SIZE = 32` and lays out four
/// 16×16 CoopMat fragments across a 32×32 output tile assuming 4 wave32s
/// per workgroup).
///
/// Strides are passed in *elements* regardless of A's dtype — for
/// quantized weights the shader divides by `QUANT_K` to recover the
/// block-major index (see `mul_mm_cm.slang:167` and similar). M and N are
/// taken from `a.dims[1]` and `b.dims[1]` directly so this works
/// uniformly for F16/F32/BF16 and quantized A.
fn record_mul_mm_cm(
    ctx: &mut DispatchContext,
    variant: &MmCmVariant,
    a: TensorView,
    b: TensorView,
    d: TensorView,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    let k = a.dims[0] as u32;
    let m = a.dims[1] as u32;
    let n = b.dims[1] as u32;

    // A last row/col tile straddles the M/N edge unless both are multiples of
    // the coopmat output tile (MM_CM_BM/BN); a straddling tile takes the kernel's
    // `ALLOW_PARTIAL_N` staged store (in-bounds elements only). Aligned tiles use
    // the direct full-fragment store.
    let partial = !m.is_multiple_of(MM_CM_BM) || !n.is_multiple_of(MM_CM_BN);

    // For the llama path there's no batched matmul: ne02 = ne12 = 1,
    // num_batches = 1, broadcasts = 1. Keep the wiring simple; bail if a
    // future model tries to dispatch a real batch through here.
    if a.dims[2].max(1) != 1
        || a.dims[3].max(1) != 1
        || b.dims[2].max(1) != 1
        || b.dims[3].max(1) != 1
    {
        return Err("mul_mm_cm: batched dims > 1 not yet supported in dispatcher".into());
    }

    // Strides in elements. For quantized A, the shader divides by QUANT_K
    // on the fly — we pass element counts, not block counts.
    let stride_a = k;
    let stride_b = k;
    let stride_d = m;
    let batch_stride_a = m * k;
    let batch_stride_b = n * k;
    let batch_stride_d = m * n;

    let mut push = [0u8; MUL_MM_CM_PARAMS_BYTES as usize];
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
        1, // num_batches
        1, // ne02
        1, // ne12
        1, // broadcast2
        1, // broadcast3
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }

    let key = PipelineKey::dense(
        variant.name,
        3,
        MUL_MM_CM_PARAMS_BYTES,
        vec![partial as u32],
    )
    .with_subgroup_size(32);
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant.spv)?;
    let workgroups = [m.div_ceil(MM_CM_BM), n.div_ceil(MM_CM_BN), 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[a.range(), b.range(), d.range()],
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, d.range());
    }
    Ok(())
}

/// Coopmat matmul with split-K: parallelise the K reduction across `split_k`
/// workgroups (on grid.z) so a huge-K / small-N matmul launches `split_k`× more
/// tiles. Each split writes its partial M×N into a `[split_k, M, N]` scratch
/// buffer; a reduce pass sums them. NOT bit-identical to the single-pass path
/// (the partials are summed in a different order), so only the
/// reproducibility-tolerant diffusion self-conditioning matmul uses it.
/// `record` (auto-pick) never split-Ks, so every other caller is unchanged.
pub fn record_split_k(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    d: TensorView,
    split_k: u32,
) -> Result<(), Box<dyn Error>> {
    let variant = mmcm_variant(a.dtype)
        .ok_or("matmul::record_split_k: no coopmat variant for this A dtype")?;
    let k = a.dims[0] as u32;
    let m = a.dims[1] as u32;
    let n = b.dims[1] as u32;
    if split_k <= 1 || k / split_k < MM_CM_BK {
        // Too few K per split to bother — fall back to the single-pass path.
        return record_mul_mm_cm(ctx, &variant, a, b, d, true);
    }

    let partial = !m.is_multiple_of(MM_CM_BM) || !n.is_multiple_of(MM_CM_BN);
    let stride_a = k;
    let stride_b = k;
    let stride_d = m;
    let batch_stride_a = m * k;
    let batch_stride_b = n * k;
    let batch_stride_d = m * n;
    let mut push = [0u8; MUL_MM_CM_PARAMS_BYTES as usize];
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
        1,
        1,
        1,
        1,
        1,
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }

    let mn = m as u64 * n as u64;
    let partials = ctx.alloc_scratch(split_k as u64 * mn * 4)?;

    let key = PipelineKey::dense(
        variant.name,
        3,
        MUL_MM_CM_PARAMS_BYTES,
        vec![partial as u32, split_k],
    )
    .with_subgroup_size(32);
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant.spv)?;
    let workgroups = [m.div_ceil(MM_CM_BM), n.div_ceil(MM_CM_BN), split_k];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[a.range(), b.range(), partials],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, partials);

    // Reduce the split_k partial tiles into the final output.
    let mut rpush = [0u8; 8];
    rpush[0..4].copy_from_slice(&(mn as u32).to_ne_bytes());
    rpush[4..8].copy_from_slice(&split_k.to_ne_bytes());
    let rkey = PipelineKey::dense("mul_mm_split_k_reduce", 2, 8, Vec::new());
    let rpipeline = *ctx.pipelines.get(
        ctx.device,
        rkey,
        crate::shaders::MUL_MM_SPLIT_K_REDUCE_SPV.as_bytes(),
    )?;
    super::bind_and_dispatch(
        ctx,
        &rpipeline,
        &[0, 1],
        &[partials, d.range()],
        &rpush,
        [(mn as u32).div_ceil(256), 1, 1],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, d.range());
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
    record_mul_mat_vec_with_flags(ctx, variant, a, b, d, fence, /*accumulate=*/ false)
}

fn record_mul_mat_vec_with_flags(
    ctx: &mut DispatchContext,
    variant: &MmvVariant,
    a: TensorView,
    b: TensorView,
    d: TensorView,
    fence: bool,
    accumulate: bool,
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

    // `mul_mat_vec.slang` reduces lane-wise partials with `WaveActiveSum`
    // assuming `subgroupSize == 32` (mul_mat_vec_head.slang:101-106). Pin
    // the pipeline to wave32 — required for correctness on any device that
    // supports both wave32 and wave64 (RDNA, Intel Xe).
    //
    // `NUM_ROWS` is a spec constant on the shared `mul_mat_vec_head.slang`
    // (default 2). For Q8_0 the optimal tile on RDNA is 1 row per
    // workgroup — matching llama.cpp's `rm_stdq` table for that dtype;
    // empirically saves ~6% on qwen35moe decode. Other dtypes keep the
    // 2-rows default. Spec-const order is `[BLOCK_SIZE, NUM_ROWS]` per
    // the head's declaration order.
    // NUM_ROWS picks the per-WG output-row tile:
    //   - vocab-sized outputs (lm_head, m ≥ 32k) → 4 (saves dispatch
    //     overhead vs the 124k-WG default)
    //   - Q8_0 elsewhere → 1 (matches llama.cpp's rm_stdq on RDNA)
    //   - everything else → 2 (default)
    let num_rows: u32 = if m >= 32_768 {
        4
    } else {
        match a.dtype {
            crate::gguf::GgmlType::Q8_0 => 1,
            _ => 2,
        }
    };
    // Spec-const order in `mul_mat_vec_head.slang`: BLOCK_SIZE, NUM_ROWS,
    // ACCUMULATE. The accumulate flag is purely a spec constant and
    // doesn't need to disambiguate the name — the spec_constants vec
    // already makes the pipeline key unique, so we can keep the name
    // as the static variant string and avoid a per-dispatch
    // `format!()` allocation.
    let key = PipelineKey {
        name: variant.name.to_string(),
        binding_indices: variant.binding_indices.to_vec(),
        push_size: MUL_MAT_VEC_PARAMS_BYTES,
        spec_constants: vec![MUL_MAT_VEC_BLOCK_SIZE, num_rows, accumulate as u32],
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant.spv)?;

    let workgroups = [m.div_ceil(num_rows), num_batches, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        variant.binding_indices,
        &bindings,
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, d.range());
    }
    let _ = MUL_MAT_VEC_BLOCK_SIZE; // documented above; not currently used host-side
    Ok(())
}

/// Max output columns the batched-column matvec handles in one dispatch. The
/// `temp[NUM_COLS][NUM_ROWS]` accumulator is register-resident, so wide N would
/// spill; the serve decode batch is ≤ 8, and wider N falls back to per-column.
const MAX_BATCH_COLS: u32 = 8;

/// Batched-column matvec: one dispatch computes all `num_cols` output columns,
/// dequantizing each A (weight) row ONCE and reusing it across the columns (the
/// `NUM_COLS` spec const), so a large (> L2) weight is read once instead of
/// re-read per column. **Byte-identical** to `num_cols` separate
/// [`record_mul_mat_vec`] calls — each column's dot product is computed exactly
/// as before; only the weight dequant is shared. Requires NO tensor (dim-2/3)
/// batch: `b` is `[K, num_cols, 1, 1]`, `d` is `[M, num_cols, 1, 1]`, so the
/// `NUM_COLS` j-loop steps `b`/`d` by their column stride and the `wg.y` batch
/// is a single slice at offset 0.
#[allow(clippy::too_many_arguments)]
fn record_mul_mat_vec_cols(
    ctx: &mut DispatchContext,
    variant: &MmvVariant,
    a: TensorView,
    b: TensorView,
    d: TensorView,
    num_cols: u32,
    accumulate: bool,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    debug_assert!(num_cols >= 1);
    debug_assert_eq!(
        b.dims[1] as u32, num_cols,
        "mul_mat_vec_cols: b column count must equal num_cols"
    );
    debug_assert!(
        b.dims[2] <= 1 && b.dims[3] <= 1 && a.dims[2] <= 1 && a.dims[3] <= 1,
        "mul_mat_vec_cols: tensor (dim-2/3) batch unsupported"
    );

    let ncols = a.dims[0] as u32; // K
    let m = a.dims[1] as u32; // output rows
    let stride_a = a.element_stride[1] as u32; // = K
    let stride_b = b.element_stride[1] as u32; // = K (within a column)
    let stride_d = m;
    // The NUM_COLS j-loop steps b/d by their COLUMN stride; with no tensor
    // batch (ne02=ne12=1) the wg.y batch resolves to a single slice at 0.
    let batch_stride_a = a.element_stride[2] as u32; // unused (a_offset = 0)
    let batch_stride_b = b.element_stride[1] as u32; // column step of b
    let batch_stride_d = d.element_stride[1] as u32; // column step of d

    let mut push = [0u8; MUL_MAT_VEC_PARAMS_BYTES as usize];
    let fields = [
        ncols,
        stride_a,
        stride_b,
        stride_d,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d,
        0, // fusion_flags
        0, // base_work_group_y
        1, // ne02
        1, // ne12
        1, // broadcast2
        1, // broadcast3
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }

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

    // Base per-WG row tile (same heuristic as the single-column path), capped so
    // the `temp[NUM_COLS][NUM_ROWS]` accumulator stays ≤ 8 — at B>1 the
    // weight-read-once (NUM_COLS) win dominates the row-tiling (NUM_ROWS) win.
    let base_rows: u32 = if m >= 32_768 {
        4
    } else {
        match a.dtype {
            crate::gguf::GgmlType::Q8_0 => 1,
            _ => 2,
        }
    };
    let num_rows = base_rows.min((MAX_BATCH_COLS / num_cols).max(1));
    debug_assert!(
        num_cols <= MAX_BATCH_COLS && num_rows * num_cols <= MAX_BATCH_COLS,
        "mul_mat_vec_cols: register tile num_rows*num_cols={} exceeds budget {MAX_BATCH_COLS}",
        num_rows * num_cols
    );

    let key = PipelineKey {
        name: variant.name.to_string(),
        binding_indices: variant.binding_indices.to_vec(),
        push_size: MUL_MAT_VEC_PARAMS_BYTES,
        // Spec-const order: [BLOCK_SIZE, NUM_ROWS, ACCUMULATE, NUM_COLS].
        spec_constants: vec![
            MUL_MAT_VEC_BLOCK_SIZE,
            num_rows,
            accumulate as u32,
            num_cols,
        ],
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant.spv)?;

    let workgroups = [m.div_ceil(num_rows), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        variant.binding_indices,
        &bindings,
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, d.range());
    }
    Ok(())
}

/// Heuristic: opt into split-K for the lm_head matvec specifically.
/// `ncols` = K (hidden dim), `nrows` = M (vocab). Returns the split factor
/// to use, or None for the single-pass path.
///
/// `SEEKER_MM_SPLIT_K=<n>` forces a value (0 = disable). Otherwise enable
/// for huge-M matvecs where the per-WG K work is large enough to bottleneck
/// on memory issue rate; threshold tuned for STRIX_HALO's 40 CUs.
fn pick_mm_split_k(ncols: u32, nrows: u32) -> Option<u32> {
    if let Some(v) = *crate::runtime_flags::MM_SPLIT_K {
        return match v {
            0 | 1 => None,
            n => Some(n),
        };
    }
    // Auto-enable only for the lm_head-shaped case: vocab-scale rows and
    // K divisible by `4 * K_PER_ITER * BLOCK_SIZE = 1024` so 4-way split-K
    // partitions cleanly. Skip when rows are small enough that the
    // single-pass kernel already issues plenty of workgroups.
    if nrows >= 32_768 && ncols.is_multiple_of(1024) {
        Some(4)
    } else {
        None
    }
}

/// Split-K mul_mat_vec — multiple workgroups cooperate on each output row.
/// Each WG writes one partial value per (row, split_idx) to a `[M, SPLIT_K]`
/// scratch buffer; a follow-up reduce shader sums the SPLIT_K partials
/// into the final `[M]` output. Used by the lm_head matvec on STRIX_HALO.
fn record_mul_mat_vec_split_k(
    ctx: &mut DispatchContext,
    variant: &MmvVariant,
    a: TensorView,
    b: TensorView,
    d: TensorView,
    split_k: u32,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(b.dims[1], 1, "split-K matvec requires N=1");
    debug_assert_eq!(
        a.dtype,
        GgmlType::Q8_0,
        "split-K matvec wired for Q8_0 only"
    );

    let ncols = a.dims[0] as u32;
    let m = a.dims[1] as u32;
    let num_rows: u32 = match a.dtype {
        crate::gguf::GgmlType::Q8_0 => 1,
        _ => 2,
    };

    // Partials buffer: [M, SPLIT_K] F32, laid out split-major per row so
    // the reduce kernel sweeps with unit stride.
    let partials_bytes = (m as u64) * (split_k as u64) * 4;
    let partials = ctx.alloc_scratch(partials_bytes)?;

    // Build the split-K push constants — share the existing matvec layout
    // (the shader reuses `mul_mat_vec_head.slang`). `stride_d` carries M
    // (rows in this split-K dispatch's output slice); other fields are
    // single-batch defaults.
    let mut push = [0u8; MUL_MAT_VEC_PARAMS_BYTES as usize];
    let fields = [
        ncols,
        ncols,
        ncols,
        m, // ncols, stride_a, stride_b, stride_d
        ncols * m,
        ncols,
        m, // batch_stride_a, _b, _d (single batch)
        0,
        0,
        1,
        1,
        1,
        1, // fusion_flags, base_work_group_y, ne02..broadcast3
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }

    // Pipeline: `mul_mat_vec_split_k.slang` Q8_0 variant. Spec-const order is
    // the head's [BLOCK_SIZE, NUM_ROWS, ACCUMULATE, NUM_COLS] (ids 0-3) plus
    // this shader's own SPLIT_K (id 4). NUM_COLS pins to 1 (split-K is the
    // single-column lm_head path); accumulate stays 0 (partials are fresh).
    // NOTE: NUM_COLS must appear here — it sits at id 3 in the head now, so
    // omitting it would push SPLIT_K's value onto NUM_COLS and break split-K.
    let key = PipelineKey {
        name: "mul_mat_vec_split_k_q8_0".to_string(),
        binding_indices: variant.binding_indices.to_vec(),
        push_size: MUL_MAT_VEC_PARAMS_BYTES,
        spec_constants: vec![MUL_MAT_VEC_BLOCK_SIZE, num_rows, 0, 1, split_k],
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(
        ctx.device,
        key,
        shaders::MUL_MAT_VEC_SPLIT_K_Q8_0_SPV.as_bytes(),
    )?;

    // Dispatch: (M / NUM_ROWS, SPLIT_K, 1).
    let workgroups = [m.div_ceil(num_rows), split_k, 1];

    // The split-K shader writes into `partials` via the data_d binding.
    // We slice the regular `MmvVariant` bindings list and swap slot 2's
    // buffer for the partials buffer (slot 0 = A, 1 = B, 2 = D=partials,
    // 3 = A packed16, 4 = B vec4).
    let bindings: Vec<_> = variant
        .binding_indices
        .iter()
        .map(|&slot| match slot {
            0 | 3 | 6 => a.range(),
            1 | 4 | 5 => b.range(),
            2 => partials,
            other => panic!("unexpected split-K binding slot {other}"),
        })
        .collect();
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        variant.binding_indices,
        &bindings,
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, partials);

    // Reduce: sum SPLIT_K partials per row into `d`. The shader includes
    // `generic_head.slang`, whose `GenericParams` push block is 24 bytes
    // (KX, KY + 4 floats) even though this kernel only reads KX — the
    // pipeline-layout push range must cover the whole declared block or
    // the validation layer flags it (VUID-…-layout-10069). Only KX is set;
    // the rest stay zero (unread).
    const GENERIC_PARAMS_BYTES: u32 = 6 * 4;
    let reduce_key = PipelineKey::dense(
        "mul_mat_vec_split_k_reduce_f32",
        2,
        GENERIC_PARAMS_BYTES,
        vec![split_k],
    );
    let reduce_pipeline = *ctx.pipelines.get(
        ctx.device,
        reduce_key,
        shaders::MUL_MAT_VEC_SPLIT_K_REDUCE_F32_SPV.as_bytes(),
    )?;
    let mut reduce_push = [0u8; GENERIC_PARAMS_BYTES as usize];
    reduce_push[0..4].copy_from_slice(&m.to_ne_bytes()); // KX = M
    let reduce_workgroups = [m.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &reduce_pipeline,
        &[0, 1],
        &[partials, d.range()],
        &reduce_push,
        reduce_workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, d.range());
    }
    Ok(())
}
