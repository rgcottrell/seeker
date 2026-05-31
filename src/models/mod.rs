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
    /// When `compute_logits` is true, returns `Some(view)` where `view`
    /// is the scratch tensor that, after submission, holds the next-token
    /// logits (`vocab_size` F32s) for the last token — shape
    /// `[vocab_size, 1, 1, 1]`, dtype F32.
    ///
    /// When `compute_logits` is false (intermediate prefill ubatches that
    /// only need to populate the KV / recurrent state), the model skips
    /// the final norm + lm_head epilogue entirely and returns `None`. The
    /// K/V writes and `cache.position` advance still happen.
    fn record_forward(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        compute_logits: bool,
    ) -> Result<Option<TensorView>, Box<dyn Error>>;

    /// Batched **decode** forward: advance `B = tokens.len()` independent
    /// sequences by one token each in a single pass. `tokens[s]` is sequence
    /// `s`'s input token and `positions[s]` is its current cache position
    /// (its new K/V lands there; it attends over `[0, positions[s] + 1)`).
    /// `slots[s]` is the `BatchKvCache` slab that sequence `s`'s K/V and
    /// recurrent state live in — the batch can gather arbitrary, non-contiguous
    /// slabs (so prefix-reuse can park a conversation in any slab and still join
    /// the batch). Returns the next-token logits for all sequences — shape
    /// `[vocab_size, B]`, dtype F32, column `s` being sequence `s`'s logits.
    ///
    /// Default: unimplemented. Implemented per-architecture (llama, qwen35moe).
    /// The dense ops already process the `B`-wide token dimension; the
    /// per-sequence work is attention (own KV slab + length) and, for hybrids,
    /// the recurrent state.
    fn record_forward_batch(
        &self,
        _ctx: &mut DispatchContext,
        _batch: &mut crate::inference::kv_cache::BatchKvCache,
        _tokens: &[u32],
        _positions: &[u32],
        _slots: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        Err("record_forward_batch (batched decode) not implemented for this model".into())
    }

    /// Unified varlen forward (M5): `B` sequences, sequence `s` contributing
    /// `seq_lens[s]` tokens packed flat (in order) into `tokens` / `positions`
    /// (`N_total = sum seq_lens`). Mixes prefill chunks (`L_s > 1`) and decode
    /// (`L_s = 1`) in one forward; attention masks causally per sequence over
    /// its own slab (`slots[s]`), continuing from the cached prefix at
    /// `positions[first token of s]`. `positions[t]` is token `t`'s absolute
    /// cache position; the new K/V land there. Returns the last-token logits of
    /// each sequence — shape `[vocab, B]`, column `s` = logits at sequence `s`'s
    /// final supplied token (sample it iff `s` just finished prefilling / is
    /// decoding). Generalizes [`Self::record_forward_batch`] (the `L_s = 1`
    /// case). Default: unimplemented.
    fn record_forward_unified(
        &self,
        _ctx: &mut DispatchContext,
        _batch: &mut crate::inference::kv_cache::BatchKvCache,
        _tokens: &[u32],
        _positions: &[u32],
        _seq_lens: &[u32],
        _slots: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        Err("record_forward_unified (varlen prefill+decode) not implemented for this model".into())
    }

    /// Conservative upper bound (in bytes) on the transient scratch one
    /// forward pass of `≤ n_ubatch` tokens needs, used to size the engine's
    /// scratch region (llama.cpp-style worst-case compute-buffer reservation).
    ///
    /// `max_seq_len` only affects the estimate for heterogeneous
    /// (`k_dtype != v_dtype`) caches, which materialize the KV prefix to F32
    /// scratch; with a homogeneous cache the estimate is context-independent.
    /// The bump allocator's region-OOM error remains the precise backstop if
    /// this under-estimates.
    fn scratch_bytes_estimate(
        &self,
        n_ubatch: u32,
        max_seq_len: u32,
        k_dtype: crate::gguf::GgmlType,
        v_dtype: crate::gguf::GgmlType,
    ) -> u64;

    // ─── MTP speculative-decode hooks (optional; default unsupported) ───

    /// True when this model loaded its MTP/NextN draft head (only when
    /// the GGUF carries the tensors *and* spec decoding was requested at
    /// load time). The engine's `decode_speculative` path requires this.
    fn supports_mtp_spec(&self) -> bool {
        false
    }

    /// Record a forward pass that also exposes the per-position hidden
    /// state (the pre-final-norm residual), used by MTP speculative
    /// decode. When `full_logits` is true the returned `logits` covers
    /// **all** `L` positions (`[vocab, L]`) for batched verification;
    /// otherwise just the last position (`[vocab, 1]`). `residual` is the
    /// pre-`output_norm` hidden state `[n_embd, L]` for every position.
    /// Advances `cache.position` by `tokens.len()` like `record_forward`.
    /// `checkpoint` (spec-decode verify only): the SSM layers emit
    /// per-position recurrent-state snapshots into the cache's snapshot
    /// buffers instead of writing the live state, so a partial-acceptance
    /// step can roll back to the accepted position via
    /// [`record_ssm_finalize`] without re-running the model.
    fn record_forward_full(
        &self,
        _ctx: &mut DispatchContext,
        _cache: &mut KvCache,
        _tokens: &[u32],
        _position_offset: u32,
        _full_logits: bool,
        _checkpoint: bool,
    ) -> Result<ForwardFullOut, Box<dyn Error>> {
        Err("model does not support record_forward_full (MTP spec decode)".into())
    }

    /// Commit the per-position SSM snapshots from a checkpoint verify into
    /// the live recurrent state, selecting the state as of the accepted
    /// position (`accept_len`). Replaces the partial-acceptance re-run.
    fn record_ssm_finalize(
        &self,
        _ctx: &mut DispatchContext,
        _cache: &mut KvCache,
        _accept_len: u32,
    ) -> Result<(), Box<dyn Error>> {
        Err("model does not support record_ssm_finalize (MTP spec decode)".into())
    }

    /// Populate the MTP draft head's KV cache for positions
    /// `[position_offset, position_offset + tokens.len())` from the main
    /// model's hidden states (`hiddens`, `[n_embd, L]` row-major by
    /// position) and the corresponding next-token ids (`tokens[i]` =
    /// `t_{position_offset+i+1}`). Runs the NextN block's KV projections
    /// only (no MoE / output head). Used to seed the draft head from the
    /// prompt after prefill so drafting attends to real prior context.
    fn record_mtp_seed(
        &self,
        _ctx: &mut DispatchContext,
        _cache: &mut KvCache,
        _hiddens: &[f32],
        _tokens: &[u32],
        _position_offset: u32,
    ) -> Result<(), Box<dyn Error>> {
        Err("model does not support record_mtp_seed (MTP spec decode)".into())
    }

    /// Record one autoregressive MTP draft step (`L=1`). Given the last
    /// hidden state `h_last` (`[n_embd]`, host-uploaded) and the previously
    /// accepted/drafted token `prev_token`, runs the NextN head and returns
    /// the draft `logits` (`[vocab, 1]`) plus the MTP block output
    /// `block_out` (`[n_embd, 1]`) that seeds the next draft step's hidden.
    /// `rel_pos` is the position within the ephemeral per-step MTP KV slot.
    fn record_mtp_draft(
        &self,
        _ctx: &mut DispatchContext,
        _cache: &mut KvCache,
        _h_last: &[f32],
        _prev_token: u32,
        _rel_pos: u32,
    ) -> Result<MtpDraftOut, Box<dyn Error>> {
        Err("model does not support record_mtp_draft (MTP spec decode)".into())
    }
}

/// Output of [`Model::record_forward_full`]: logits (last-position or all
/// `L` positions depending on `full_logits`) plus the per-position
/// pre-final-norm hidden state. `logits` is `None` only when the shared
/// forward body was invoked with `compute_logits=false` (the chunked-prefill
/// intermediate-ubatch path via `record_forward`); `record_forward_full`
/// always populates it.
pub struct ForwardFullOut {
    pub logits: Option<TensorView>,
    pub residual: TensorView,
}

/// Output of [`Model::record_mtp_draft`]: the greedily-selected draft
/// token (a 4-byte `u32` on the GPU — drafting is always argmax, so we do
/// the reduction on-device and avoid reading back full vocab logits) and
/// the MTP block output that becomes the next step's hidden input.
pub struct MtpDraftOut {
    pub draft_token: crate::inference::buffer::BufferRange,
    pub block_out: TensorView,
}

/// Per-layer SSM state dimensions for hybrid (attention + Mamba/GDN)
/// models. None for pure-attention models. Hooked into KvCache
/// allocation in `seeker run` via `Model::ssm_state_dims`.
#[derive(Debug, Clone, Copy)]
pub struct SsmStateDims {
    pub n_ssm_layers: u32,
    pub conv_state_floats: u32,
    pub gdn_state_floats: u32,
    /// Conv1d channel count (`conv_state_floats = (conv_kernel-1) * conv_channels`).
    /// Needed to size the per-position conv checkpoint backup for MTP spec decode.
    pub conv_channels: u32,
    /// Conv1d kernel size.
    pub conv_kernel: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct CacheDims {
    pub n_layer: u32,
    pub head_dim: u32,
    pub n_head_kv: u32,
}

/// Construct the right `Model` for the given GGUF based on its
/// `general.architecture` metadata key.
///
/// `spec_enabled` requests loading the MTP/NextN draft head (for
/// `--spec-draft-n-max > 0`). Architectures without MTP support ignore it.
pub fn open(
    gguf: &GgufFile,
    weights: WeightsHandle,
    tokenizer: TokenizerBundle,
    spec_enabled: bool,
) -> Result<Box<dyn Model>, Box<dyn Error>> {
    let arch = gguf
        .architecture()
        .ok_or(ModelError::MissingMetadata("general.architecture"))?;
    match arch {
        "llama" => Ok(Box::new(llama::LlamaModel::new(gguf, weights, tokenizer)?)),
        "qwen35moe" => Ok(Box::new(qwen35moe::Qwen35MoeModel::new(
            gguf,
            weights,
            tokenizer,
            spec_enabled,
        )?)),
        other => Err(ModelError::Unsupported(other.to_string()).into()),
    }
}
