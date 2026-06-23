//! [`VulkanEmbedder`] — the Vulkan implementation of
//! [`seeker_core::embed::TextEmbedder`].
//!
//! A thin wrapper that owns an [`Engine`] + model + KV cache and runs the
//! existing `forward_full_readback` path. It exists so the CLI `embedding`
//! command (and future callers) can target the backend-agnostic trait while the
//! Vulkan numerics stay byte-identical to the previous inline command code.

use std::error::Error;

use seeker_core::embed::TextEmbedder;

use crate::gguf::{GgmlType, GgufFile};
use crate::inference::Engine;
use crate::inference::kv_cache::{KvCache, KvCacheConfig};
use crate::models::{self, Model};
use crate::tokenizer::build_tokenizer;

/// Vulkan-backed text embedder: opens a model on the GPU and produces the
/// pre-`output_norm` residual for a token sequence via the engine's
/// `forward_full_readback`.
pub struct VulkanEmbedder {
    engine: Engine,
    model: Box<dyn Model>,
    /// Allocated lazily by [`Self::ensure_capacity`]; grows to fit the longest
    /// sequence seen (or the [`reserve`](TextEmbedder::reserve) hint).
    cache: Option<KvCache>,
    ubatch: u32,
    cache_k: GgmlType,
    cache_v: GgmlType,
    /// `max_seq_len` the current scratch + KV cache are sized for (0 = unsized).
    capacity: u32,
    n_embd: usize,
}

impl VulkanEmbedder {
    /// Open `gguf` on the GPU: upload weights, build the model + tokenizer. The
    /// KV cache + scratch are sized lazily on the first embed (or via
    /// [`reserve`](TextEmbedder::reserve)).
    pub fn new(
        gguf: &GgufFile,
        ubatch: u32,
        cache_k: GgmlType,
        cache_v: GgmlType,
    ) -> Result<Self, Box<dyn Error>> {
        let bundle = build_tokenizer(gguf)?;
        let engine = Engine::new(ubatch, 2048)?;
        let weights = engine.upload_weights(gguf)?;
        let model = models::open(gguf, weights, bundle, /*spec_enabled=*/ false)?;
        let arch = gguf.architecture().unwrap_or("");
        let n_embd = gguf
            .meta_u32(&format!("{arch}.embedding_length"))
            .ok_or("missing <arch>.embedding_length")? as usize;
        Ok(Self {
            engine,
            model,
            cache: None,
            ubatch,
            cache_k,
            cache_v,
            capacity: 0,
            n_embd,
        })
    }

    /// (Re)allocate scratch + KV cache for sequences up to `max_len`. Grow-only:
    /// a no-op when already sized large enough. Output is independent of the
    /// allocation size, so growing mid-batch does not change results.
    fn ensure_capacity(&mut self, max_len: u32) -> Result<(), Box<dyn Error>> {
        let max_len = max_len.max(1);
        if self.cache.is_some() && max_len <= self.capacity {
            return Ok(());
        }
        let scratch = self.model.scratch_bytes_estimate(
            self.ubatch,
            max_len,
            self.cache_k,
            self.cache_v,
            /*max_batch=*/ 1,
        );
        self.engine.allocate_scratch(scratch)?;
        let dims = self.model.cache_dims();
        let config = KvCacheConfig {
            k_dtype: self.cache_k,
            v_dtype: self.cache_v,
            max_seq_len: max_len,
            n_head: dims.n_head,
        };
        self.cache = Some(self.engine.allocate_kv_cache(
            dims.n_layer,
            dims.head_dim,
            dims.n_head_kv,
            config,
        )?);
        self.capacity = max_len;
        Ok(())
    }
}

impl TextEmbedder for VulkanEmbedder {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, Box<dyn Error>> {
        self.model
            .tokenizer()
            .tokenizer
            .encode(text, /*add_special=*/ true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| -> Box<dyn Error> { format!("tokenize failed: {e}").into() })
    }

    fn reserve(&mut self, max_seq_len: u32) -> Result<(), Box<dyn Error>> {
        self.ensure_capacity(max_seq_len)
    }

    fn embed_residual(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
        self.ensure_capacity(tokens.len() as u32)?;
        let cache = self
            .cache
            .as_mut()
            .expect("cache allocated by ensure_capacity");
        cache.reset();
        let (_logits, residual) = self.engine.forward_full_readback(
            &*self.model,
            cache,
            tokens,
            0,
            /*full_logits=*/ false,
        )?;
        Ok(residual)
    }

    fn n_embd(&self) -> usize {
        self.n_embd
    }

    fn device_name(&self) -> String {
        self.engine.device.name()
    }
}
