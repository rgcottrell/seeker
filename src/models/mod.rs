//! Per-architecture model definitions. Each architecture (LLaMA, Qwen, …)
//! provides a [`Model`] implementation that knows its parameter layout,
//! tensor naming, and forward-pass graph. Architectures depend on
//! [`crate::inference`] for the dispatch primitives.

use std::error::Error;

use crate::gguf::GgufFile;
use crate::inference::context::DispatchContext;
use crate::inference::kv_cache::KvCache;
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::tokenizer::TokenizerBundle;

pub mod llama;
pub mod qwen35moe;

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
    /// Architecture params needed to allocate a KV cache.
    fn cache_dims(&self) -> CacheDims;

    /// Optional per-layer SSM state. Pure-attention models return None.
    /// Hybrid models (qwen35moe) return Some so the engine allocates a
    /// persistent state region on the KvCache.
    fn ssm_state_dims(&self) -> Option<SsmStateDims> {
        None
    }
    /// Borrow the model's uploaded weight buffer. Needed by the engine so
    /// it can pass `&WeightsHandle` into the dispatch context.
    fn weights(&self) -> &WeightsHandle;
    /// Borrow the model's tokenizer (for prompt encoding / sampled-token
    /// decoding by callers).
    fn tokenizer(&self) -> &TokenizerBundle;

    /// Per-model constants the engine needs to drive the
    /// persistent-decode-cmdbuf replay path. Returns `None` when the
    /// model doesn't support replay yet (default).
    fn replay_constants(&self) -> Option<crate::inference::decode_dyn::ModelReplayConstants> {
        None
    }

    /// The (k_num, blocks_per_split) the model would feed into
    /// flash_attn for the upcoming decode call. Engine uses this both
    /// to decide whether a cached decode cmdbuf can be replayed (it
    /// compares against the captured pair) and to re-stamp DecodeDyn
    /// on the replay path. None means the model doesn't support replay
    /// (default).
    fn decode_grid(&self, _kv: u32, _shader_core_count: u32) -> Option<(u32, u32)> {
        None
    }

    /// Refresh the host-side scratch slots a cached decode cmdbuf reads
    /// from before each replay submit. Mirrors the work that
    /// `record_forward` would have done on a fresh-record path —
    /// writing the new input token, the M-RoPE position values, etc. —
    /// without re-recording any GPU dispatches. Default returns an
    /// error so non-replay-capable models can't be silently mis-driven.
    fn refresh_replay_inputs(
        &self,
        _host_ptr: *mut u8,
        _plan: &crate::inference::decode_dyn::ReplayPlan,
        _tokens: &[u32],
        _position_offset: u32,
    ) -> Result<(), Box<dyn Error>> {
        Err("model does not support decode replay".into())
    }

    /// Record a forward pass into `ctx`'s command buffer.
    ///
    /// `tokens` are the new tokens being added at absolute positions
    /// `[position_offset, position_offset + tokens.len())`. The model
    /// writes the K/V for those positions into `cache`, reads back the
    /// full prefix `[0, position_offset + tokens.len())` for attention,
    /// and on success advances `cache.position` by `tokens.len()`.
    ///
    /// Returns the scratch tensor that, after submission, will hold the
    /// next-token logits (`vocab_size` F32s) for the last token — shape
    /// `[vocab_size, 1, 1, 1]`, dtype F32.
    fn record_forward(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
    ) -> Result<TensorView, Box<dyn Error>>;
}

/// Per-layer SSM state dimensions for hybrid (attention + Mamba/GDN)
/// models. None for pure-attention models. Hooked into KvCache
/// allocation in `seeker run` via `Model::ssm_state_dims`.
#[derive(Debug, Clone, Copy)]
pub struct SsmStateDims {
    pub n_ssm_layers: u32,
    pub conv_state_floats: u32,
    pub gdn_state_floats: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct CacheDims {
    pub n_layer: u32,
    pub head_dim: u32,
    pub n_head_kv: u32,
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
        "qwen35moe" => Ok(Box::new(qwen35moe::Qwen35MoeModel::new(
            gguf, weights, tokenizer,
        )?)),
        other => Err(ModelError::Unsupported(other.to_string()).into()),
    }
}
