//! Per-architecture model definitions. Each architecture (LLaMA, Qwen, …)
//! provides a [`Model`] implementation that knows its parameter layout,
//! tensor naming, and forward-pass graph. Architectures depend on
//! [`crate::inference`] for the dispatch primitives.

use std::error::Error;

use crate::gguf::GgufFile;
use crate::inference::context::DispatchContext;
use crate::inference::weights::WeightsHandle;
use crate::tokenizer::TokenizerBundle;

pub mod llama;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("unsupported architecture: {0}")]
    Unsupported(String),
    #[error("missing required GGUF metadata: {0}")]
    MissingMetadata(&'static str),
    #[error("missing required tensor: {0}")]
    MissingTensor(String),
    #[error("invalid metadata value for {key}: {detail}")]
    BadMetadata { key: &'static str, detail: String },
}

pub trait Model: Send + Sync {
    fn arch(&self) -> &'static str;
    fn vocab_size(&self) -> u32;
    /// Record the forward pass into `ctx`'s command buffer. The shape of the
    /// final logits slot is `[vocab_size]` F32; the engine is responsible
    /// for copying it out.
    fn record_forward(
        &self,
        ctx: &mut DispatchContext,
        tokens: &[u32],
    ) -> Result<(), Box<dyn Error>>;
}

/// Construct the right `Model` for the given GGUF based on its
/// `general.architecture` metadata key.
pub fn open(
    gguf: &GgufFile,
    weights: WeightsHandle,
    tokenizer: TokenizerBundle,
) -> Result<Box<dyn Model>, Box<dyn Error>> {
    let arch = gguf
        .architecture()
        .ok_or(ModelError::MissingMetadata("general.architecture"))?;
    match arch {
        "llama" => Ok(Box::new(llama::LlamaModel::new(gguf, weights, tokenizer)?)),
        other => Err(ModelError::Unsupported(other.to_string()).into()),
    }
}
