//! `rms_norm` with multiply-by-weight (the LLaMA path). Dispatches
//! `shaders::RMS_NORM_F32_SPV` with the `do_multiply` spec constant set to
//! 1. Push constants follow llama.cpp's `vk_op_binary_push_constants`
//!    exactly (ggml-vulkan.cpp:11322).
//!
//! Shader contract: see `shaders/compute/rms_norm.slang` +
//! `shaders/include/generic_binary_head.slang`. Workgroup count is
//! `(ne01, ne02, ne03)` — one workgroup per row.

use std::error::Error;

use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

use super::binary_params_bytes;

pub fn record(
    ctx: &mut DispatchContext,
    src: TensorView,
    weight: TensorView,
    dst: TensorView,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    record_inner(ctx, src, weight, dst, eps, /*fence=*/ true)
}

/// As [`record`] but skips the trailing barrier — caller fences `dst`.
pub fn record_nofence(
    ctx: &mut DispatchContext,
    src: TensorView,
    weight: TensorView,
    dst: TensorView,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    record_inner(ctx, src, weight, dst, eps, /*fence=*/ false)
}

/// RMSNorm with NO learned weight (`scale·x` only). Gemma4 applies this to V
/// before caching (a weightless per-head RMSNorm). Skips the trailing barrier.
pub fn record_noweight_nofence(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    record_noweight_inner(ctx, src, dst, eps, /*fence=*/ false)
}

fn record_noweight_inner(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    eps: f32,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    // do_multiply=false ⇒ the weight binding (slot 1) is declared but never
    // read; bind `src` as a harmless placeholder. Output is written contiguous.
    let key = PipelineKey::dense(
        "rms_norm_f32",
        3,
        super::BINARY_PARAMS_BYTES,
        vec![0, 0], // norepeat=false, do_multiply=false
    );
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::RMS_NORM_F32_SPV.as_bytes())?;
    let push = binary_params_bytes(&src, &src, &dst, eps, 0.0, 0);
    let workgroups = [
        src.dims[1].max(1) as u32,
        src.dims[2].max(1) as u32,
        src.dims[3].max(1) as u32,
    ];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[src.range(), src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    }
    Ok(())
}

fn record_inner(
    ctx: &mut DispatchContext,
    src: TensorView,
    weight: TensorView,
    dst: TensorView,
    eps: f32,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    let key = PipelineKey::dense(
        "rms_norm_f32",
        3,
        super::BINARY_PARAMS_BYTES,
        vec![0, 1], // norepeat=false, do_multiply=true
    );
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::RMS_NORM_F32_SPV.as_bytes())?;

    let push = binary_params_bytes(&src, &weight, &dst, eps, 0.0, 0);
    let workgroups = [
        src.dims[1].max(1) as u32,
        src.dims[2].max(1) as u32,
        src.dims[3].max(1) as u32,
    ];

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[src.range(), weight.range(), dst.range()],
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    }
    Ok(())
}
