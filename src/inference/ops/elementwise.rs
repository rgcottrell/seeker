//! Element-wise ops shared by every transformer block: add, mul, silu, plus
//! `get_rows` for the initial embedding lookup. Each is a tiny push-constant
//! + dispatch.

use std::error::Error;

use crate::inference::buffer::BufferRange;
use crate::inference::context::DispatchContext;
use crate::inference::weights::TensorView;

#[allow(dead_code)]
pub fn record_add(
    _ctx: &mut DispatchContext,
    _a: TensorView,
    _b: TensorView,
    _dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    Err("elementwise::record_add not yet implemented".into())
}

#[allow(dead_code)]
pub fn record_mul(
    _ctx: &mut DispatchContext,
    _a: TensorView,
    _b: TensorView,
    _dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    Err("elementwise::record_mul not yet implemented".into())
}

#[allow(dead_code)]
pub fn record_silu(
    _ctx: &mut DispatchContext,
    _src: TensorView,
    _dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    Err("elementwise::record_silu not yet implemented".into())
}

#[allow(dead_code)]
pub fn record_get_rows(
    _ctx: &mut DispatchContext,
    _src: TensorView,        // embedding table [hidden, vocab]
    _indices: BufferRange,   // u32 token ids
    _dst: TensorView,        // out [hidden, n_tokens]
) -> Result<(), Box<dyn Error>> {
    Err("elementwise::record_get_rows not yet implemented".into())
}
