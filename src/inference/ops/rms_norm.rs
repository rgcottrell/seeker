//! `rms_norm` dispatch — see shaders/compute/rms_norm.slang and llama.cpp's
//! `vk_op_binary_push_constants` (ggml-vulkan.cpp:1283).
//!
//! Currently a stub: real recording lands in the next implementation pass.
//! Signature reserved here so the model layer can compile against it.

use std::error::Error;

use crate::inference::context::DispatchContext;
use crate::inference::weights::TensorView;

#[allow(dead_code)]
pub fn record(
    _ctx: &mut DispatchContext,
    _src: TensorView,
    _weight: TensorView,
    _dst: TensorView,
    _eps: f32,
) -> Result<(), Box<dyn Error>> {
    Err("rms_norm::record not yet implemented".into())
}
