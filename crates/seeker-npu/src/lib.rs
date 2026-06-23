//! seeker-npu — the AMD Strix Halo XDNA2 NPU backend.
//!
//! Runs Qwen3-Embedding on the NPU via XRT + AIE kernels. The low-level XRT
//! plumbing ([`npu::Context`]/[`npu::Buffer`]) is vendored from the
//! `~/workspace/gpu-npu-demo` bring-up; [`Qwen3EmbeddingNpu`] implements the
//! backend-neutral [`seeker_core::embed::TextEmbedder`] trait.
//!
//! M1 (this commit) brings up the crate + the XRT plumbing and a `vadd` example;
//! the AIE kernel library and the real forward pass land in later milestones, so
//! [`Qwen3EmbeddingNpu::embed_residual`] currently returns a not-implemented error.

mod sys;

pub mod npu;

use std::error::Error;

use seeker_core::embed::TextEmbedder;
use seeker_core::gguf::GgufFile;
use seeker_core::tokenizer::{TokenizerBundle, build_tokenizer};

/// Qwen3-Embedding on the Strix Halo NPU.
///
/// Holds the GGUF-derived tokenizer + dims now; the on-NPU weight upload + forward
/// land with the AIE kernel library (M2+).
pub struct Qwen3EmbeddingNpu {
    tokenizer: TokenizerBundle,
    n_embd: usize,
}

impl Qwen3EmbeddingNpu {
    /// Read the tokenizer + `n_embd` from `gguf`. Does not yet upload weights to
    /// the NPU (that arrives with the kernel library).
    pub fn new(gguf: &GgufFile) -> Result<Self, Box<dyn Error>> {
        let tokenizer = build_tokenizer(gguf)?;
        let arch = gguf.architecture().unwrap_or("");
        let n_embd = gguf
            .meta_u32(&format!("{arch}.embedding_length"))
            .ok_or("missing <arch>.embedding_length")? as usize;
        Ok(Self { tokenizer, n_embd })
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

    fn embed_residual(&mut self, _tokens: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
        Err(
            "NPU embedding backend not yet implemented (AIE kernels land in a later milestone)"
                .into(),
        )
    }

    fn n_embd(&self) -> usize {
        self.n_embd
    }

    fn device_name(&self) -> String {
        "Strix Halo NPU (XDNA2)".to_string()
    }
}
