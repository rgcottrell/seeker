//! Per-op dispatch recorders. Each op fills its shader's push-constant
//! struct exactly per llama.cpp's `vk_op_*_push_constants` patterns, binds
//! the right slots, and records a single compute dispatch + barrier into
//! the active command buffer.

pub mod elementwise;
pub mod flash_attn;
pub mod matmul;
pub mod rms_norm;
pub mod rope;
