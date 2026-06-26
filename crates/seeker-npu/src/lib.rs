//! seeker-npu — the AMD Strix Halo XDNA2 NPU backend.
//!
//! Runs Qwen3-Embedding on the NPU via XRT + AIE kernels. The low-level XRT
//! plumbing ([`npu::Context`]/[`npu::Buffer`]) is vendored from the
//! `~/workspace/gpu-npu-demo` bring-up; [`Qwen3EmbeddingNpu`] implements the
//! backend-neutral [`seeker_core::embed::TextEmbedder`] trait.
//!
//! [`Qwen3EmbeddingNpu`] runs the full forward on the NPU (see
//! [`qwen3::Qwen3Forward`]) and returns the pre-`output_norm` residual; the shared
//! host-side `output_norm` + pool + L2 in `seeker_core::embed` finishes the
//! embedding. The AIE xclbins are fixed-shape (built for a token block of
//! [`qwen3::L_PAD`]); see `crates/seeker-npu/kernels/` for the build scripts.

mod sys;

pub mod npu;
pub mod qwen3;

use std::error::Error;

use seeker_core::embed::TextEmbedder;
use seeker_core::gguf::GgufFile;
use seeker_core::tokenizer::{TokenizerBundle, build_tokenizer};

use crate::qwen3::Qwen3Forward;

/// Qwen3-Embedding on the Strix Halo NPU.
pub struct Qwen3EmbeddingNpu {
    tokenizer: TokenizerBundle,
    model: Qwen3Forward,
}

impl Qwen3EmbeddingNpu {
    /// Read the tokenizer + config and dequantize all weights to bf16 from `gguf`.
    pub fn new(gguf: &GgufFile) -> Result<Self, Box<dyn Error>> {
        let tokenizer = build_tokenizer(gguf)?;
        let model = Qwen3Forward::load(gguf)?;
        Ok(Self { tokenizer, model })
    }
}

impl TextEmbedder for Qwen3EmbeddingNpu {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error>> {
        self.tokenizer
            .tokenizer
            .encode(text, /*add_special=*/ true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| -> Box<dyn Error> { format!("tokenize failed: {e}").into() })
    }

    fn embed_residual(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
        self.model.forward(tokens)
    }

    fn n_embd(&self) -> usize {
        self.model.n_embd
    }

    fn device_name(&self) -> String {
        "Strix Halo NPU (XDNA2)".to_string()
    }
}
