//! `rms_norm` with multiply-by-weight (the LLaMA path). Dispatches
//! `shaders::RMS_NORM_F32_SPV` with the `do_multiply` spec constant set to
//! 1. Push constants follow llama.cpp's `vk_op_binary_push_constants`
//! exactly (ggml-vulkan.cpp:11322).
//!
//! Shader contract: see `shaders/compute/rms_norm.slang` +
//! `shaders/include/generic_binary_head.slang`. Workgroup count is
//! `(ne01, ne02, ne03)` — one workgroup per row.

use std::error::Error;

use crate::inference::command::{record_compute_barrier, record_dispatch};
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::{CachedPipeline, PipelineKey};
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
    let key = PipelineKey::dense(
        "rms_norm_f32",
        3,
        super::BINARY_PARAMS_BYTES,
        vec![0, 1], // norepeat=false, do_multiply=true
    );
    let (pipeline, set_layout) = {
        let p: &CachedPipeline = ctx
            .pipelines
            .get(ctx.device, key, shaders::RMS_NORM_F32_SPV.as_bytes())?;
        ((p.pipeline, p.layout), p.set_layout)
    };

    let set = ctx.descriptors.allocate_and_write(
        ctx.device,
        set_layout,
        &[src.range(), weight.range(), dst.range()],
    )?;

    let push = binary_params_bytes(&src, &weight, &dst, eps, 0.0, 0);
    let workgroups = [
        src.dims[1].max(1) as u32,
        src.dims[2].max(1) as u32,
        src.dims[3].max(1) as u32,
    ];

    let cached = CachedPipeline {
        pipeline: pipeline.0,
        layout: pipeline.1,
        set_layout,
    };
    record_dispatch(ctx.device, ctx.cmd, &cached, set, &push, workgroups);
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}
