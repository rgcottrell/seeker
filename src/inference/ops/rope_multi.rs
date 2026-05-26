//! Multi-section (M-RoPE) dispatch. Mirrors `ops/rope.rs` but binds
//! `rope_multi.slang`'s F32 variant and populates `rope_params.sections[]`
//! + `is_imrope`. Used for Qwen 3.5 / Qwen-VL models where the rotated
//! dimensions are partitioned into 3 axis groups (typically text +
//! image height + image width). For text-only inference we still set the
//! sections array (per the GGUF metadata) but route every axis to the
//! same `positions` buffer — the shader handles that case.
//!
//! Push-constant struct: shared `rope_params` from
//! shaders/include/rope_params.slang (116 bytes).

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::buffer::BufferRange;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

const ROPE_PARAMS_BYTES: u32 = 116;

#[derive(Clone, Copy)]
pub struct RopeMultiParams {
    pub n_dims: u32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub corr_dims: [f32; 2],
    /// Section widths in *rotated-pair units*. Sum should be `n_dims / 2`.
    /// For Qwen3.5 with `dimension_count=64` and `sections=[11,11,10,0]`,
    /// the first three sum to 32 = 64/2.
    pub sections: [u32; 4],
    /// 0 = standard M-RoPE; non-zero selects Qwen-VL's "IM-RoPE" sequence.
    /// Always 0 for plain qwen35moe text inference.
    pub is_imrope: u32,
}

impl RopeMultiParams {
    pub fn qwen_default(n_dims: u32, freq_base: f32, sections: [u32; 4]) -> Self {
        Self {
            n_dims,
            freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
            sections,
            is_imrope: 0,
        }
    }
}

pub fn record(
    ctx: &mut DispatchContext,
    src: TensorView,
    positions: BufferRange,
    dst: TensorView,
    params: RopeMultiParams,
) -> Result<(), Box<dyn Error>> {
    record_inner(ctx, src, positions, dst, params, true)
}

pub fn record_nofence(
    ctx: &mut DispatchContext,
    src: TensorView,
    positions: BufferRange,
    dst: TensorView,
    params: RopeMultiParams,
) -> Result<(), Box<dyn Error>> {
    record_inner(ctx, src, positions, dst, params, false)
}

fn record_inner(
    ctx: &mut DispatchContext,
    src: TensorView,
    positions: BufferRange,
    dst: TensorView,
    params: RopeMultiParams,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let ne00 = src.dims[0] as u32;
    let ne01 = src.dims[1] as u32;
    let ne02 = src.dims[2] as u32;
    let nrows: u32 = ne01 * ne02 * src.dims[3].max(1) as u32;
    let theta_scale = (params.freq_base).powf(-2.0 / params.n_dims as f32);
    let dummy = positions;

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

    put_u(&mut push, &mut w, 0); // rope_mode (unused by rope_multi)
    put_u(&mut push, &mut w, nrows);
    put_u(&mut push, &mut w, params.n_dims);
    put_f(&mut push, &mut w, params.freq_scale);
    put_f(&mut push, &mut w, params.freq_base);
    put_f(&mut push, &mut w, params.ext_factor);
    put_f(&mut push, &mut w, params.attn_factor);
    put_f(&mut push, &mut w, params.corr_dims[0]);
    put_f(&mut push, &mut w, params.corr_dims[1]);
    put_f(&mut push, &mut w, theta_scale);
    put_u(&mut push, &mut w, 0); // has_ff
    for s in params.sections {
        put_i(&mut push, &mut w, s as i32);
    }
    put_u(&mut push, &mut w, params.is_imrope);
    put_u(&mut push, &mut w, 0); // is_back
    put_u(&mut push, &mut w, 0); // set_rows_stride
    put_u(&mut push, &mut w, ne00);
    put_u(&mut push, &mut w, ne01);
    put_u(&mut push, &mut w, ne02);
    put_u(&mut push, &mut w, src.element_stride[1] as u32);
    put_u(&mut push, &mut w, src.element_stride[2] as u32);
    put_u(&mut push, &mut w, src.element_stride[3] as u32);
    put_u(&mut push, &mut w, dst.element_stride[1] as u32);
    put_u(&mut push, &mut w, dst.element_stride[2] as u32);
    put_u(&mut push, &mut w, dst.element_stride[3] as u32);
    put_u(&mut push, &mut w, 0); // a_offset
    put_u(&mut push, &mut w, 0); // d_offset

    let key = PipelineKey::dense("rope_multi_f32", 5, ROPE_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::ROPE_MULTI_F32_SPV.as_bytes())?;
    let workgroups = [nrows, ne00.div_ceil(512), 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3, 4],
        &[src.range(), positions, dummy, dst.range(), dummy],
        &push,
        workgroups,
    )?;
    if fence {
        record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    }
    Ok(())
}
