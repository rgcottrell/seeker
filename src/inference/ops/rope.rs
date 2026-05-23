//! `rope_norm` dispatch — see shaders/compute/rope_norm.slang.

use std::error::Error;

use crate::inference::buffer::BufferRange;
use crate::inference::context::DispatchContext;
use crate::inference::weights::TensorView;

#[allow(dead_code)]
pub struct RopeParams {
    pub freq_base: f32,
    pub freq_scale: f32,
    pub n_dims: u32,
    pub n_ctx_orig: u32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

#[allow(dead_code)]
pub fn record(
    _ctx: &mut DispatchContext,
    _src: TensorView,
    _positions: BufferRange,
    _freq_factors: Option<TensorView>,
    _dst: TensorView,
    _params: RopeParams,
) -> Result<(), Box<dyn Error>> {
    Err("rope::record not yet implemented".into())
}
