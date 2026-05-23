//! `rope_norm` dispatch — LLaMA-style RoPE (NORM, not NEOX/IMROPE/MROPE).
//! Push constants follow `rope_params` from
//! shaders/include/rope_params.slang, populated per llama.cpp's
//! `ggml_vk_make_rope_constants` (ggml-vulkan.cpp:11252).
//!
//! Workgroup count: `(nrows, ceil(n_dims / 512), 1)`. For LLaMA with
//! head_dim ≤ 512, that's `(nrows, 1, 1)`.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::buffer::BufferRange;
use crate::inference::command::{record_compute_barrier, record_dispatch};
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::{CachedPipeline, PipelineKey};
use crate::inference::weights::TensorView;
use crate::shaders;

const ROPE_PARAMS_BYTES: u32 = 116;

#[derive(Clone, Copy)]
pub struct RopeParams {
    pub n_dims: u32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub corr_dims: [f32; 2],
}

impl RopeParams {
    pub fn llama_default(n_dims: u32, freq_base: f32) -> Self {
        Self {
            n_dims,
            freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
        }
    }
}

/// Record an in-place RoPE on a tensor reshaped to `[head_dim, n_head, L]`.
/// `positions` is a scratch slot holding `L` `i32`s = [0, 1, …, L-1] (or
/// whatever absolute position indices apply).
pub fn record(
    ctx: &mut DispatchContext,
    src: TensorView,
    positions: BufferRange,
    dst: TensorView,
    params: RopeParams,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let ne00 = src.dims[0] as u32; // head_dim
    let ne01 = src.dims[1] as u32; // n_head
    let ne02 = src.dims[2] as u32; // L (tokens)
    let nrows: u32 = ne01 * ne02 * src.dims[3].max(1) as u32;

    let theta_scale = (params.freq_base).powf(-2.0 / params.n_dims as f32);

    // Bind a dummy buffer for freq_factors / set_rows indices slots. Any
    // valid storage buffer works since the shader only reads them when
    // has_ff != 0 / set_rows_stride != 0, both of which are false here.
    let dummy = positions;

    // Pack push constants.
    let mut push = [0u8; ROPE_PARAMS_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_i(out: &mut [u8], w: &mut usize, v: i32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }

    put_u(&mut push, &mut w, 0);             // rope_mode = NORM
    put_u(&mut push, &mut w, nrows);
    put_u(&mut push, &mut w, params.n_dims);
    put_f(&mut push, &mut w, params.freq_scale);
    put_f(&mut push, &mut w, params.freq_base);
    put_f(&mut push, &mut w, params.ext_factor);
    put_f(&mut push, &mut w, params.attn_factor);
    put_f(&mut push, &mut w, params.corr_dims[0]);
    put_f(&mut push, &mut w, params.corr_dims[1]);
    put_f(&mut push, &mut w, theta_scale);
    put_u(&mut push, &mut w, 0);             // has_ff
    for _ in 0..4 {
        put_i(&mut push, &mut w, 0);         // sections[4]
    }
    put_u(&mut push, &mut w, 0);             // is_imrope
    put_u(&mut push, &mut w, 0);             // is_back
    put_u(&mut push, &mut w, 0);             // set_rows_stride
    put_u(&mut push, &mut w, ne00);
    put_u(&mut push, &mut w, ne01);
    put_u(&mut push, &mut w, ne02);
    put_u(&mut push, &mut w, src.element_stride[1] as u32); // nb01
    put_u(&mut push, &mut w, src.element_stride[2] as u32); // nb02
    put_u(&mut push, &mut w, src.element_stride[3] as u32); // nb03
    put_u(&mut push, &mut w, dst.element_stride[1] as u32); // nb11
    put_u(&mut push, &mut w, dst.element_stride[2] as u32); // nb12
    put_u(&mut push, &mut w, dst.element_stride[3] as u32); // nb13
    put_u(&mut push, &mut w, 0);             // a_offset
    put_u(&mut push, &mut w, 0);             // d_offset

    let key = PipelineKey::dense("rope_norm_f32", 5, ROPE_PARAMS_BYTES, Vec::new());
    let (pipeline, layout, set_layout) = {
        let p: &CachedPipeline = ctx
            .pipelines
            .get(ctx.device, key, shaders::ROPE_NORM_F32_SPV.as_bytes())?;
        (p.pipeline, p.layout, p.set_layout)
    };
    let set = ctx.descriptors.allocate_and_write(
        ctx.device,
        set_layout,
        &[src.range(), positions, dummy, dst.range(), dummy],
    )?;

    let workgroups = [nrows, ne00.div_ceil(512), 1];
    let cached = CachedPipeline {
        pipeline,
        layout,
        set_layout,
    };
    record_dispatch(ctx.device, ctx.cmd, &cached, set, &push, workgroups);
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}
