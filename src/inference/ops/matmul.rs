//! `mul_mm` / `mul_mat_vec` dispatch — see shaders/compute/mul_mm.slang and
//! mul_mat_vec.slang. Push constants per llama.cpp ggml-vulkan.cpp:1010 /
//! :995.
//!
//! MVP uses the scalar `mul_mm.slang` (no cooperative_matrix). For B==1
//! cases (single-token decode), `mul_mat_vec.slang` is preferable but
//! deferred.

use std::error::Error;

use crate::inference::context::DispatchContext;
use crate::inference::weights::TensorView;

#[allow(dead_code)]
pub fn record(
    _ctx: &mut DispatchContext,
    _a: TensorView, // weight matrix [K, N]
    _b: TensorView, // activations  [M, K]
    _d: TensorView, // out          [M, N]
) -> Result<(), Box<dyn Error>> {
    Err("matmul::record not yet implemented".into())
}
