//! Vulkan inference runtime — architecture-agnostic.
//!
//! This module owns the Vulkan device, GPU memory regions, pipeline cache,
//! descriptor pool, and the per-op dispatch recorders in [`ops`]. It knows
//! nothing about LLaMA, Qwen, or any specific transformer arch — those live
//! in `crate::models::*` and use [`DispatchContext`] + the [`ops`] helpers
//! to record their forward pass.

pub mod buffer;
pub mod command;
pub mod context;
pub mod descriptor;
pub mod device;
pub mod memory;
pub mod ops;
pub mod pipeline;
pub mod sample;
pub mod weights;

