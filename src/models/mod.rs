//! Per-architecture model definitions. Each architecture (LLaMA, Qwen, …)
//! provides a [`Model`] implementation that knows its parameter layout,
//! tensor naming, and forward-pass graph. Architectures depend on
//! [`crate::inference`] for the dispatch primitives.

use std::error::Error;

use crate::gguf::GgufFile;
use crate::inference::buffer::BufferRange;
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
    /// Borrow the model's uploaded weight buffer. Needed by the engine so
    /// it can pass `&WeightsHandle` into the dispatch context.
    fn weights(&self) -> &WeightsHandle;
    /// Borrow the model's tokenizer (for prompt encoding / sampled-token
    /// decoding by callers).
    fn tokenizer(&self) -> &TokenizerBundle;
    /// Record the forward pass into `ctx`'s command buffer and return the
    /// scratch slot that, after submission, will hold the next-token logits
    /// (`vocab_size` F32s).
    fn record_forward(
        &self,
        ctx: &mut DispatchContext,
        tokens: &[u32],
    ) -> Result<BufferRange, Box<dyn Error>>;
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
