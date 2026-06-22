//! The Vulkan backend for seeker: the inference `Engine`, every model
//! architecture, the GPU op kernels, and the compiled Slang shaders.
//!
//! This crate is the concrete compute backend that `seeker-cli` drives. It
//! builds on the backend-neutral [`seeker_core`] crate, which it re-exports
//! below so the engine/model code can keep referring to `crate::gguf`,
//! `crate::tokenizer`, etc. unchanged.

// Re-export the backend-neutral leaves at the crate root so existing
// `crate::{gguf,tokenizer,chat_template,runtime_flags}` paths inside
// `inference/`, `models/`, and the encoders resolve without rewriting. The
// vision/audio host-side modules are re-exported under their own modules (see
// `vision`/`audio` below) because this crate has its own `vision`/`audio`
// modules for the GPU encoders.
pub use seeker_core::{chat_template, gguf, runtime_flags, tokenizer};

/// SPIR-V shader binaries, generated from `shaders/compute/*.slang` by
/// `build.rs`. The `include!` must live in this crate because `env!("OUT_DIR")`
/// resolves to the OUT_DIR of the crate being compiled.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

// These modules carry GPU helper functions kept for completeness / llama.cpp
// parity that aren't all exercised by every build (the original single binary
// allowed dead code here too — see the old `main.rs` module declarations).
#[allow(dead_code)]
pub mod audio;
#[allow(dead_code)]
pub mod inference;
#[allow(dead_code)]
pub mod models;
#[allow(dead_code)]
pub mod vision;
