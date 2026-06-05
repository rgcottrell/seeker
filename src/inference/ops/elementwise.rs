//! Element-wise ops: add, mul, silu, plus `get_rows` for embedding lookup.
//! All share the same simple dispatch pattern; only push-constant shape and
//! workgroup math differ.
//!
//! - `add` / `mul` use `BinaryParams` and `wg_denoms = {512, 1, 1}` so
//!   workgroups = `ceil(nelements / 512)` in the X direction.
//! - `silu` uses `GenericParams` (KX = nelements).
//! - `get_rows` uses `BinaryParams` and `wg_denoms = {512, 1, 1}` with
//!   workgroups = `(ceil(ne00/512), ne10, ne11*ne12)`.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::buffer::BufferRange;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

use super::{UNARY_PARAMS_BYTES, binary_params_bytes, unary_params_bytes};

const GENERIC_PARAMS_BYTES: u32 = 6 * 4;

pub fn record_add(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    record_binary_f32(ctx, "add_f32", shaders::ADD_F32_SPV.as_bytes(), a, b, dst)
}

pub fn record_mul(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    record_binary_f32(ctx, "mul_f32", shaders::MUL_F32_SPV.as_bytes(), a, b, dst)
}

pub fn record_sub(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    record_binary_f32(ctx, "sub_f32", shaders::SUB_F32_SPV.as_bytes(), a, b, dst)
}

fn record_binary_f32(
    ctx: &mut DispatchContext,
    name: &str,
    spirv: &[u8],
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    let key = PipelineKey::dense(name, 3, super::BINARY_PARAMS_BYTES, vec![0]);
    let pipeline = *ctx.pipelines.get(ctx.device, key, spirv)?;
    let push = binary_params_bytes(&a, &b, &dst, 0.0, 0.0, 0);

    // Shader: 256 threads × num_iter=2, each workgroup owning a UNIQUE 512-wide
    // block (base = workgroup * 512). Non-overlapping, so every element is
    // written exactly once — required for in-place ops where dst aliases a
    // source (e.g. `residual += proj`); overlapping writes would double-apply
    // the op nondeterministically. Dispatch ceil(N/512) workgroups.
    let nelements: u64 = dst.dims.iter().product();
    let workgroups = [(nelements as u32).div_ceil(512), 1, 1];

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[a.range(), b.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// Fused MoE residual tail for qwen35moe.
///
/// Computes `residual = residual + routed + shared * sigmoid(gate)`
/// where `gate` is a per-token scalar broadcast across the hidden
/// dimension. Replaces four sequential dispatches (sigmoid →
/// broadcast mul → two adds) and the `shared_gate_sig` /
/// `shared_scaled` scratch buffers. `hidden` is the broadcast period
/// (= L's stride in `residual`).
pub fn record_moe_residual_fuse(
    ctx: &mut DispatchContext,
    residual_in: TensorView,
    routed: TensorView,
    shared: TensorView,
    gate: TensorView,
    residual_out: TensorView,
    hidden: u32,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(residual_in.dtype, GgmlType::F32);
    debug_assert_eq!(routed.dtype, GgmlType::F32);
    debug_assert_eq!(shared.dtype, GgmlType::F32);
    debug_assert_eq!(gate.dtype, GgmlType::F32);
    debug_assert_eq!(residual_out.dtype, GgmlType::F32);
    debug_assert_eq!(residual_in.dims, residual_out.dims);

    let nelements: u32 = residual_in.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());
    push[4..8].copy_from_slice(&hidden.to_ne_bytes()); // KY = broadcast period

    let key = PipelineKey::dense("moe_residual_fuse_f32", 5, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx.pipelines.get(
        ctx.device,
        key,
        shaders::MOE_RESIDUAL_FUSE_F32_SPV.as_bytes(),
    )?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3, 4],
        &[
            residual_in.range(),
            routed.range(),
            shared.range(),
            gate.range(),
            residual_out.range(),
        ],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, residual_out.range());
    Ok(())
}

/// Fused 3-way element-wise add: `dst = a + b + c`. Used by qwen35moe's
/// MoE FFN to collapse the two sequential residual updates
/// (`residual += routed`; `residual += shared_scaled`) into one dispatch.
/// Same workgroup decomposition as the binary add/mul ops.
pub fn record_add3(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    c: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(a.dtype, GgmlType::F32);
    debug_assert_eq!(b.dtype, GgmlType::F32);
    debug_assert_eq!(c.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    debug_assert_eq!(a.dims, dst.dims);
    debug_assert_eq!(b.dims, dst.dims);
    debug_assert_eq!(c.dims, dst.dims);

    let nelements: u32 = a.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());

    let key = PipelineKey::dense("add3_f32", 4, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::ADD3_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3],
        &[a.range(), b.range(), c.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// Fused `silu(a) * b → dst` ("split-mode" swiglu). a, b, dst must share
/// the same F32 contiguous layout. Saves two dispatches (silu + mul) and
/// one barrier vs the unfused path — the FFN gate path of every MoE expert
/// and the SSM `output * silu(z)` step both fit this shape.
///
/// Push-constants are `GluParams` from `shaders/include/glu_head.slang`
/// (16 × 4 bytes); `nb*` are in **element** units (not bytes) — see
/// `glu_main.slang`. For contiguous tensors that means `nb01 = ne00`,
/// `nb02 = ne00 * ne01`, etc., and the same for the dst side.
pub fn record_swiglu_split(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(a.dtype, GgmlType::F32);
    debug_assert_eq!(b.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    debug_assert_eq!(a.dims, b.dims);
    debug_assert_eq!(a.dims, dst.dims);

    let ne00 = a.dims[0] as u32;
    let ne01 = a.dims[1] as u32;
    let ne02 = a.dims[2] as u32;
    let n_elements: u64 = a.dims.iter().product();

    let nb01 = ne00;
    let nb02 = ne00 * ne01;
    let nb03 = ne00 * ne01 * ne02;

    const SWIGLU_PARAMS_BYTES: u32 = 16 * 4;
    let mut push = [0u8; SWIGLU_PARAMS_BYTES as usize];
    let fields: [u32; 16] = [
        n_elements as u32, // N
        ne00,              // ne00 (src col count)
        ne00,              // ne20 (dst col count — same shape)
        2,                 // mode = 2 (split)
        0,                 // alpha (unused)
        0,                 // limit (unused)
        nb01,
        nb02,
        nb03,
        ne01,
        ne02,
        nb01, // nb11 = same as nb01 (dst contiguous, same shape)
        nb02,
        nb03,
        ne01,
        ne02,
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }
    // `alpha`/`limit` are floats — the shader reads them through the same
    // 16-byte aligned struct; the zero u32 bit-pattern is 0.0f32, so the
    // u32 write above already encodes `alpha = 0.0`, `limit = 0.0`.

    let key = PipelineKey::dense("swiglu_f32", 3, SWIGLU_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SWIGLU_F32_SPV.as_bytes())?;
    record_glu_dispatch(
        ctx,
        pipeline,
        &push,
        n_elements,
        &[a.range(), b.range(), dst.range()],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// Fused `sigmoid(a) * b → dst` ("split-mode" sigmoid_mul) — same dispatch
/// shape as `record_swiglu_split` but using the `sigmoid_mul.slang`
/// kernel. The attention block's q-gate `sigmoid(q_gate) * attn_out`
/// chain fits this exactly.
pub fn record_sigmoid_mul_split(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(a.dtype, GgmlType::F32);
    debug_assert_eq!(b.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    debug_assert_eq!(a.dims, b.dims);
    debug_assert_eq!(a.dims, dst.dims);

    let ne00 = a.dims[0] as u32;
    let ne01 = a.dims[1] as u32;
    let ne02 = a.dims[2] as u32;
    let n_elements: u64 = a.dims.iter().product();

    let nb01 = ne00;
    let nb02 = ne00 * ne01;
    let nb03 = ne00 * ne01 * ne02;

    const SIGMOID_MUL_PARAMS_BYTES: u32 = 16 * 4;
    let mut push = [0u8; SIGMOID_MUL_PARAMS_BYTES as usize];
    let fields: [u32; 16] = [
        n_elements as u32,
        ne00,
        ne00,
        2,
        0,
        0,
        nb01,
        nb02,
        nb03,
        ne01,
        ne02,
        nb01,
        nb02,
        nb03,
        ne01,
        ne02,
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }

    let key = PipelineKey::dense("sigmoid_mul_f32", 3, SIGMOID_MUL_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SIGMOID_MUL_F32_SPV.as_bytes())?;
    record_glu_dispatch(
        ctx,
        pipeline,
        &push,
        n_elements,
        &[a.range(), b.range(), dst.range()],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// Dispatch helper for the `glu_main.slang` family. The shader uses 512
/// threads/WG and packs the flat index as `z * 262144 + y * 512 + x`, so
/// dispatch ceil(N/512) on X up to the 65535 hw limit and spill into Y.
fn record_glu_dispatch(
    ctx: &mut DispatchContext,
    pipeline: crate::inference::pipeline::CachedPipeline,
    push: &[u8],
    n_elements: u64,
    bindings: &[crate::inference::buffer::BufferRange],
) -> Result<(), Box<dyn Error>> {
    let total_wgs = (n_elements as u32).div_ceil(512);
    let max_x: u32 = 65535;
    let wg_x = total_wgs.min(max_x);
    let wg_y = total_wgs.div_ceil(max_x);
    let workgroups = [wg_x, wg_y, 1];
    super::bind_and_dispatch(ctx, &pipeline, &[0, 1, 2], bindings, push, workgroups)
}

/// Fused SSM post-GDN normalize + silu(z) gate. Combines
/// `rms_norm(gdn_attn, ssm_norm) * silu(z) → gated_attn` into a single
/// dispatch — saves the intermediate `attn_normed` allocation, the
/// rms_norm dispatch, and one barrier per SSM layer. Dispatches one
/// workgroup per `(head, token)` pair.
#[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
pub fn record_ssm_norm_gate(
    ctx: &mut DispatchContext,
    gdn_attn: TensorView,
    ssm_norm: TensorView,
    z: TensorView,
    dst: TensorView,
    s_v: u32,
    num_v: u32,
    l: u32,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(gdn_attn.dtype, GgmlType::F32);
    debug_assert_eq!(ssm_norm.dtype, GgmlType::F32);
    debug_assert_eq!(z.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    const SSM_NORM_GATE_PUSH_BYTES: u32 = 5 * 4;
    let mut push = [0u8; SSM_NORM_GATE_PUSH_BYTES as usize];
    let value_dim = s_v * num_v;
    let fields: [u32; 5] = [
        s_v,
        num_v,
        value_dim,
        l,
        eps.to_bits(), // float reinterpret
    ];
    for (i, v) in fields.iter().enumerate() {
        push[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
    }

    let key = PipelineKey::dense("ssm_norm_gate_f32", 4, SSM_NORM_GATE_PUSH_BYTES, Vec::new());
    let pipeline =
        *ctx.pipelines
            .get(ctx.device, key, shaders::SSM_NORM_GATE_F32_SPV.as_bytes())?;
    let workgroups = [num_v, l, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3],
        &[gdn_attn.range(), ssm_norm.range(), z.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// Fused SSM alpha pipeline: `alpha = softplus(alpha_pre + bias) * ssm_a`,
/// with `bias` and `ssm_a` shape `[num_v]` broadcasting along the L
/// dimension of `alpha_pre`. Replaces three sequential dispatches in
/// `ssm_block`; everything F32. KY is the broadcast period (`num_v`).
pub fn record_ssm_alpha_fuse(
    ctx: &mut DispatchContext,
    alpha_pre: TensorView,
    bias: TensorView,
    ssm_a: TensorView,
    dst: TensorView,
    num_v: u32,
) -> Result<(), Box<dyn Error>> {
    record_ssm_alpha_fuse_inner(
        ctx, alpha_pre, bias, ssm_a, dst, num_v, /*fence=*/ true,
    )
}

/// As [`record_ssm_alpha_fuse`] but skips the trailing barrier — caller is
/// responsible for fencing `dst` before any downstream read.
pub fn record_ssm_alpha_fuse_nofence(
    ctx: &mut DispatchContext,
    alpha_pre: TensorView,
    bias: TensorView,
    ssm_a: TensorView,
    dst: TensorView,
    num_v: u32,
) -> Result<(), Box<dyn Error>> {
    record_ssm_alpha_fuse_inner(
        ctx, alpha_pre, bias, ssm_a, dst, num_v, /*fence=*/ false,
    )
}

fn record_ssm_alpha_fuse_inner(
    ctx: &mut DispatchContext,
    alpha_pre: TensorView,
    bias: TensorView,
    ssm_a: TensorView,
    dst: TensorView,
    num_v: u32,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(alpha_pre.dtype, GgmlType::F32);
    debug_assert_eq!(bias.dtype, GgmlType::F32);
    debug_assert_eq!(ssm_a.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let n_elements: u64 = alpha_pre.dims.iter().product();
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&(n_elements as u32).to_ne_bytes());
    push[4..8].copy_from_slice(&num_v.to_ne_bytes()); // KY

    let key = PipelineKey::dense("ssm_alpha_fuse_f32", 4, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline =
        *ctx.pipelines
            .get(ctx.device, key, shaders::SSM_ALPHA_FUSE_F32_SPV.as_bytes())?;
    let total_wgs = (n_elements as u32).div_ceil(512);
    let max_x: u32 = 65535;
    let wg_x = total_wgs.min(max_x);
    let wg_y = total_wgs.div_ceil(max_x);
    let workgroups = [wg_x, wg_y, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3],
        &[alpha_pre.range(), bias.range(), ssm_a.range(), dst.range()],
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    }
    Ok(())
}

pub fn record_silu(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());
    // KY, param1..4 all zero — leave as default.

    let key = PipelineKey::dense("silu_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SILU_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// GELU (tanh approximation, = ggml `gelu`). Same generic-unary dispatch as
/// `silu`. Gemma's GeGLU FFN runs this on the gate branch (then `× up`).
pub fn record_gelu(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());

    let key = PipelineKey::dense("gelu_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::GELU_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// `tanh(x)` elementwise. Same generic-unary dispatch as `silu`. Used (with
/// [`record_scale`]) for Gemma's final-logit softcap `cap·tanh(x/cap)`.
pub fn record_tanh(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());

    let key = PipelineKey::dense("tanh_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::TANH_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// `α·x + β` over a (possibly multi-dim, contiguous) tensor via `scale.slang`.
/// In-place safe (`src == dst`). Gemma uses it for the `× sqrt(n_embd)`
/// embedding scale, the per-layer `× layer_output_scale`, and the final-logit
/// softcap halves.
pub fn record_scale(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    alpha: f32,
    beta: f32,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let push = unary_params_bytes(&src, &dst, alpha, beta);
    let key = PipelineKey::dense("scale_f32", 2, UNARY_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SCALE_F32_SPV.as_bytes())?;
    let nelements: u64 = src.dims.iter().product();
    // scale.slang: 128 threads × num_iter=4 (stepping +128) cover one 512-block,
    // and `get_idx = y*512 + x` keys the block off the Y workgroup index. The
    // 512-block count MUST go in Y — putting it in X spaces consecutive
    // workgroups by 128 (= numthreads), so their +128 stepping OVERLAPS and an
    // in-place scale compounds (×alpha^k). (Non-in-place callers tolerate the
    // overlap since they re-write the same value, but in-place ones do not.)
    let blocks = (nelements as u32).div_ceil(512);
    let workgroups = [1, blocks, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// `softplus(x) = log(1 + exp(x))`. Same generic-unary dispatch shape
/// as `silu` / `sigmoid`. Used by the SSM block to map raw `α` projection
/// logits to a positive value before the `ssm_a` scaling.
pub fn record_softplus(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());
    let key = PipelineKey::dense("softplus_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SOFTPLUS_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// As [`record_l2_norm`] but skips the trailing barrier — caller fences.
pub fn record_l2_norm_nofence(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    record_l2_norm_inner(ctx, src, dst, eps, /*fence=*/ false)
}

/// Per-row L2 normalize: `dst[r, c] = src[r, c] / max(sqrt(sum_c src^2), eps)`.
/// Dispatched with `src.dims[1..]` workgroups, each reducing over
/// `src.dims[0]` — so passing `src` shape `[head_dim, n_head, L, 1]`
/// gives per-(head, token) L2 normalization (which is what the SSM
/// block needs for Q and K before gated-delta-net).
pub fn record_l2_norm(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    record_l2_norm_inner(ctx, src, dst, eps, /*fence=*/ true)
}

fn record_l2_norm_inner(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    eps: f32,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    let push = super::unary_params_bytes(&src, &dst, eps, 0.0);
    let key = PipelineKey::dense("l2_norm_f32", 2, super::UNARY_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::L2_NORM_F32_SPV.as_bytes())?;
    // `l2_norm.slang` decodes the row index as a packed
    // `z*262144 + y*512 + x` value — DIFFERENT convention from `rms_norm`,
    // which uses (x, y, z) → (row, channel, batch) directly. Pack total
    // rows (= dim1 × dim2 × dim3) along x, capped at 512 per stripe, with
    // overflow into y and (z × 262144).
    let total_rows = (src.dims[1].max(1) * src.dims[2].max(1) * src.dims[3].max(1)) as u32;
    let wg_z = total_rows / 262144;
    let wg_after_z = total_rows - wg_z * 262144;
    let wg_y = wg_after_z.div_ceil(512);
    let wg_x = (wg_after_z + wg_y.max(1) - 1).min(512);
    // Simpler when total_rows <= 512: just (total_rows, 1, 1).
    let workgroups = if total_rows <= 512 {
        [total_rows, 1, 1]
    } else if total_rows <= 512 * 512 {
        [512, total_rows.div_ceil(512), 1]
    } else {
        [512, 512, total_rows.div_ceil(262144)]
    };
    let _ = (wg_x, wg_y, wg_z);
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    }
    Ok(())
}

/// Element-wise sigmoid (`σ(x) = 1 / (1 + exp(-x))`). Same dispatch shape
/// and push-constant layout as `record_silu` — `sigmoid.slang` is the
/// same generic-unary template, just a different SPV.
/// As [`record_sigmoid`] but skips the trailing barrier — caller fences `dst`.
pub fn record_sigmoid_nofence(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());
    let key = PipelineKey::dense("sigmoid_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SIGMOID_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    Ok(())
}

pub fn record_sigmoid(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());

    let key = PipelineKey::dense("sigmoid_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SIGMOID_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// `get_rows`: dst[col, row] = src[col, indices[row]]. src has shape
/// `[hidden, vocab]` (ggml: ne[0]=hidden, ne[1]=vocab), indices is `[L]` of
/// u32, dst is `[hidden, L]`.
pub fn record_get_rows(
    ctx: &mut DispatchContext,
    src: TensorView,
    indices: BufferRange,
    indices_len: u32,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    // Synthetic TensorView for the indices buffer with shape [L, 1, 1, 1].
    let indices_view = TensorView {
        buffer: indices.buffer,
        byte_offset: indices.offset,
        byte_size: indices.size,
        dims: [indices_len as u64, 1, 1, 1],
        byte_stride: [
            4,
            4 * indices_len as u64,
            4 * indices_len as u64,
            4 * indices_len as u64,
        ],
        element_stride: [
            1,
            indices_len as u64,
            indices_len as u64,
            indices_len as u64,
        ],
        dtype: GgmlType::I32,
    };
    let mut push = binary_params_bytes(&src, &indices_view, &dst, 0.0, 0.0, 0);

    // `elems_per_x` = elements covered per workgroup along the X (column)
    // axis, which sets the dispatch divisor:
    //   - plain get_rows (`get_rows.slang`): 512 threads × 1 elem  = 512
    //   - get_rows_quant (`get_rows_quant.slang`): 512 × 2 elems   = 1024
    //   - get_rows_q6_k (`get_rows_q6_k.slang`): 1 block / WG       = 256
    // All variants here declare only bindings [0,1,2] (slangc -O3 strips
    // the unused packed16 alias from the quant kernels' scalar path).
    let (name, spirv, elems_per_x) = match (src.dtype, dst.dtype) {
        (GgmlType::F32, GgmlType::F32) => {
            ("get_rows_f32", shaders::GET_ROWS_F32_SPV.as_bytes(), 512)
        }
        (GgmlType::F16, GgmlType::F16) => {
            ("get_rows_f16", shaders::GET_ROWS_F16_SPV.as_bytes(), 512)
        }
        (GgmlType::F16, GgmlType::F32) => (
            "get_rows_f16_f32",
            shaders::GET_ROWS_F16_F32_SPV.as_bytes(),
            512,
        ),
        (GgmlType::BF16, GgmlType::F32) => {
            ("get_rows_bf16", shaders::GET_ROWS_BF16_SPV.as_bytes(), 512)
        }
        (GgmlType::I32, GgmlType::I32) => {
            ("get_rows_i32", shaders::GET_ROWS_I32_SPV.as_bytes(), 512)
        }
        (GgmlType::Q5_K, GgmlType::F32) => (
            "get_rows_q5_k",
            shaders::GET_ROWS_Q5_K_DEFAULT_SPV.as_bytes(),
            256,
        ),
        (GgmlType::Q6_K, GgmlType::F32) => (
            "get_rows_q6_k",
            shaders::GET_ROWS_Q6_K_DEFAULT_SPV.as_bytes(),
            256,
        ),
        (GgmlType::Q4_0, GgmlType::F32) => (
            "get_rows_quant_q4_0",
            shaders::GET_ROWS_QUANT_Q4_0_SPV.as_bytes(),
            1024,
        ),
        (GgmlType::Q4_1, GgmlType::F32) => (
            "get_rows_quant_q4_1",
            shaders::GET_ROWS_QUANT_Q4_1_SPV.as_bytes(),
            1024,
        ),
        (GgmlType::Q5_0, GgmlType::F32) => (
            "get_rows_quant_q5_0",
            shaders::GET_ROWS_QUANT_Q5_0_SPV.as_bytes(),
            1024,
        ),
        (GgmlType::Q5_1, GgmlType::F32) => (
            "get_rows_quant_q5_1",
            shaders::GET_ROWS_QUANT_Q5_1_SPV.as_bytes(),
            1024,
        ),
        (GgmlType::Q8_0, GgmlType::F32) => (
            "get_rows_quant_q8_0",
            shaders::GET_ROWS_QUANT_Q8_0_SPV.as_bytes(),
            1024,
        ),
        (GgmlType::IQ4_NL, GgmlType::F32) => (
            "get_rows_quant_iq4_nl",
            shaders::GET_ROWS_QUANT_IQ4_NL_SPV.as_bytes(),
            1024,
        ),
        (s, d) => return Err(format!("get_rows: unsupported src/dst combo {s:?}/{d:?}").into()),
    };

    let ne00 = src.dims[0] as u32;
    let ne10 = indices_len;

    // `get_rows_quant.slang` indexes `data_a[a_off + i00/QUANT_K]` in *block*
    // units, so it needs `nb01` = blocks-per-row. `binary_params_bytes`
    // fills `nb01` from `element_stride[1]`, which for a quant tensor is
    // `byte_stride[1] / rounded_elem_size` (e.g. Q8_0: (64·34)/2 = 1088) —
    // not the block count. Patch `nb01` (field index 6, byte offset 24) to
    // the true blocks-per-row. (`get_rows_q6_k` doesn't need this — it
    // derives the block index from `ne00/QUANT_K` directly.)
    if elems_per_x == 1024 {
        let (block_size, _) = src.dtype.block_layout();
        let blocks_per_row = ne00 / block_size as u32;
        push[24..28].copy_from_slice(&blocks_per_row.to_ne_bytes());
    }

    let key = PipelineKey::dense(name, 3, super::BINARY_PARAMS_BYTES, vec![0]);
    let pipeline = *ctx.pipelines.get(ctx.device, key, spirv)?;

    let workgroups = [ne00.div_ceil(elems_per_x), ne10, 1];

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[src.range(), indices, dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}
