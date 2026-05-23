//! `flash_attn` (scalar variant f32_f16) — see shaders/compute/flash_attn.slang.
//! Push constants per the `FaParams` struct in ggml-vulkan.cpp.

use std::error::Error;

use crate::inference::context::DispatchContext;
use crate::inference::weights::TensorView;

#[allow(dead_code)]
pub struct FlashAttnParams {
    pub scale: f32,
    pub max_bias: f32,
    pub logit_softcap: f32,
    pub gqa_ratio: u32,
}

#[allow(dead_code)]
pub fn record(
    _ctx: &mut DispatchContext,
    _q: TensorView,
    _k: TensorView,
    _v: TensorView,
    _mask: Option<TensorView>,
    _out: TensorView,
    _params: FlashAttnParams,
) -> Result<(), Box<dyn Error>> {
    Err("flash_attn::record not yet implemented".into())
}
