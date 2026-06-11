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

pub mod gemma4;
pub mod gemma4_assistant;
pub mod llama;
pub mod qwen3;
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

    /// Per-layer `(head_dims, n_head_kvs)` for architectures whose KV
    /// dimensions vary by layer (gemma4's interleaved sliding-window / global
    /// attention). `None` ⇒ uniform dims from [`cache_dims`]. When `Some`, both
    /// vectors have length `cache_dims().n_layer` and callers must allocate via
    /// [`crate::inference::Engine::allocate_kv_cache_per_layer`].
    fn cache_per_layer_dims(&self) -> Option<(Vec<u32>, Vec<u32>)> {
        None
    }

    /// Per-layer KV slab token capacity for a serve `BatchKvCache`. `None` ⇒
    /// every layer holds the full `max_seq_len` (the default). A model with
    /// sliding-window-attention layers (gemma4) returns `Some(depths)` (length
    /// `cache_dims().n_layer`) capping its SWA layers at the ring-buffer depth
    /// `sliding_window + (n_ubatch − 1)` — so those slabs wrap instead of
    /// growing with context, cutting KV memory at long context. A depth `>=
    /// max_seq_len` means that layer is a normal full slab. The model's forward
    /// must drive the matching ring write/read for any capped layer (detected
    /// from the slab view's depth). Serve-only for now (single-seq run/chat
    /// keep full slabs).
    fn cache_slab_depths(&self, _max_seq_len: u32, _n_ubatch: u32) -> Option<Vec<u32>> {
        None
    }

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

    /// Record ONE chunk of an image-containing prefill (chunked so a large image
    /// + prompt doesn't run as one oversized forward — which TDR-hangs the GPU).
    ///   `chunk_tokens` are this chunk's tokens; `chunk_global_start` is the global
    ///   prompt index of `chunk_tokens[0]`. `image_embeddings` is the FULL encoder
    ///   output `[n_embd, n_tok]` (column = merged token, n_embd contiguous); the
    ///   model splices only the columns whose `<|image_pad|>` slots fall in this
    ///   chunk. `image_global_start` is the first image-pad token's global index,
    ///   `image_nx`/`image_ny` the merged grid (`n_tok = nx*ny`), `prompt_pos0` the
    ///   absolute position of global token 0 (so the M-RoPE 2D cursor is continuous
    ///   across chunks). `compute_logits` is true only for the final chunk (which
    ///   samples). Advances `cache.position` by the chunk length; does NOT touch
    ///   `rope_position_lag` (the caller sets it once after the whole prefill).
    ///   Returns the last-token logits when `compute_logits`. Default unsupported.
    #[allow(clippy::too_many_arguments)]
    fn record_forward_image_chunk(
        &self,
        _ctx: &mut DispatchContext,
        _cache: &mut KvCache,
        _chunk_tokens: &[u32],
        _chunk_global_start: usize,
        _image_embeddings: &[f32],
        _image_global_start: usize,
        _image_nx: usize,
        _image_ny: usize,
        _prompt_pos0: u32,
        _compute_logits: bool,
    ) -> Result<Option<TensorView>, Box<dyn Error>> {
        Err("model does not support image input".into())
    }

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

    /// Spec batched-verify forward: like [`Self::record_forward_unified`] but
    /// returns logits for ALL `N_total` positions (not just each sequence's last)
    /// plus the per-position residual, and does NOT commit per-slot `positions`
    /// (the caller truncates each slot to its accepted length). See
    /// [`UnifiedVerifyOut`]. Default: unimplemented.
    fn record_forward_unified_verify(
        &self,
        _ctx: &mut DispatchContext,
        _batch: &mut crate::inference::kv_cache::BatchKvCache,
        _tokens: &[u32],
        _positions: &[u32],
        _seq_lens: &[u32],
        _slots: &[u32],
    ) -> Result<UnifiedVerifyOut, Box<dyn Error>> {
        Err(
            "record_forward_unified_verify (batched spec verify) not implemented for this model"
                .into(),
        )
    }

    /// Whether image tokens use a 2D M-RoPE cursor (qwen-VL) vs plain sequential
    /// 1D positions (gemma4). When `true` (default), the engine advances the
    /// M-RoPE cursor by `max(nx,ny)` after an image and tracks the
    /// `rope_position_lag` (KV slots − cursor). gemma4 returns `false`: its image
    /// tokens advance the cursor 1:1 like text, so there is no lag.
    fn image_uses_mrope(&self) -> bool {
        true
    }

    /// Whether [`Self::record_forward_unified`] is implemented — lets the server
    /// scheduler use the token-budget / chunked-prefill loop (mixing prefill and
    /// decode of different requests in one forward). Models without it fall back
    /// to the serial-prefill + batched-decode path. Default `false`.
    fn supports_unified(&self) -> bool {
        false
    }

    /// Whether [`Self::record_forward_batch`] is implemented. Keep this in sync
    /// with the `record_forward_batch` impl: when `false`, the server clamps
    /// `--parallel` to 1 and decodes through the single-sequence path
    /// (`forward_sampled` on the borrowed slot cache) instead of the batched
    /// step — without it, serve generation fails after prefill on the first
    /// decode token. Default `false`.
    fn supports_batch_decode(&self) -> bool {
        false
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
    /// `max_batch` is the largest batched-decode width the scratch must
    /// serve (serve: the resolved/explicit `--parallel`, capped by
    /// `--parallel-max` in auto mode; single-sequence callers pass 1) —
    /// the epilogue allocates `[vocab, B]` logits in batched mode.
    fn scratch_bytes_estimate(
        &self,
        n_ubatch: u32,
        max_seq_len: u32,
        k_dtype: crate::gguf::GgmlType,
        v_dtype: crate::gguf::GgmlType,
        max_batch: u32,
    ) -> u64;

    // ─── MTP speculative-decode hooks (optional; default unsupported) ───

    /// True when this model loaded its MTP/NextN draft head (only when
    /// the GGUF carries the tensors *and* spec decoding was requested at
    /// load time). The engine's `decode_speculative` path requires this.
    fn supports_mtp_spec(&self) -> bool {
        false
    }

    /// Attach a *separate* MTP/EAGLE draft model from its own GGUF (gemma4's
    /// `gemma4-assistant`), paired with this base model for speculative
    /// decoding. `handle` is the draft GGUF's uploaded weights. After a
    /// successful attach, [`supports_mtp_spec`](Self::supports_mtp_spec)
    /// returns true. Models with an in-GGUF NextN head (qwen35moe) or no MTP
    /// support return an error. Default: unsupported.
    fn attach_mtp_draft(
        &mut self,
        _gguf: &GgufFile,
        _handle: WeightsHandle,
    ) -> Result<(), Box<dyn Error>> {
        Err("model does not support a separate MTP draft head".into())
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

    /// Batched embedding prefill: process `seq_lens.len()` independent texts
    /// packed flat into `tokens` (text `s` is `seq_lens[s]` tokens, in order)
    /// in ONE forward, and return the `[n_embd, N_total]` pre-output-norm
    /// residual (column `t` = packed token `t`'s hidden). Each text attends
    /// only within itself (block-diagonal causal mask) and restarts RoPE
    /// positions at 0, so the result is identical to prefilling each text
    /// separately — but the weights are read once for the whole batch instead
    /// of once per text. The caller slices the residual by `seq_lens` and
    /// pools/normalizes each text. `cache` is a scratch slab (position 0,
    /// holds `N_total` tokens); no state persists. Default unsupported;
    /// implemented by embedding models (qwen3).
    fn record_forward_embed_batch(
        &self,
        _ctx: &mut DispatchContext,
        _cache: &mut KvCache,
        _tokens: &[u32],
        _seq_lens: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        Err("model does not support batched embedding".into())
    }

    /// Whether [`Self::record_forward_embed_batch`] is implemented — lets the
    /// embedding server pack multiple texts into one forward. Default `false`
    /// (callers fall back to one forward per text).
    fn supports_embed_batch(&self) -> bool {
        false
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

    /// Batched form of [`Self::record_ssm_finalize`] for the concurrent spec
    /// verify: for each verified sequence `s`, commit lane `s`'s per-position
    /// snapshots at `accept_lens[s]` into slot `slots[s]`'s live recurrent state
    /// (one `cmd_copy_buffer` per SSM layer for the GDN state + a strided conv
    /// extract). `slots`/`accept_lens` are in the same batch order as the verify;
    /// lane `s` is `batch.snapshot_lane(s)`. Default: unimplemented.
    fn record_ssm_finalize_batched(
        &self,
        _ctx: &mut DispatchContext,
        _batch: &mut crate::inference::kv_cache::BatchKvCache,
        _slots: &[u32],
        _accept_lens: &[u32],
    ) -> Result<(), Box<dyn Error>> {
        Err("model does not support record_ssm_finalize_batched (concurrent spec)".into())
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

/// Output of [`Model::record_forward_unified_verify`]: per-position logits +
/// residual over the flat varlen batch. Sequence `s` owns columns
/// `[q_starts[s], q_starts[s] + seq_lens[s])` (prefix-sum of `seq_lens`), which
/// the caller slices to sample/compare its `n+1` draft positions.
pub struct UnifiedVerifyOut {
    /// `[vocab, N_total]`.
    pub logits: TensorView,
    /// `[n_embd, N_total]`.
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
    /// Number of attention (query) heads — for the GQA ratio used by
    /// TurboQuant auto-asymmetric K-protection (see `KvCacheConfig::n_head`).
    pub n_head: u32,
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
        "gemma4" => Ok(Box::new(gemma4::Gemma4Model::new(
            gguf, weights, tokenizer,
        )?)),
        "llama" => Ok(Box::new(llama::LlamaModel::new(gguf, weights, tokenizer)?)),
        "qwen3" => Ok(Box::new(qwen3::Qwen3Model::new(gguf, weights, tokenizer)?)),
        "qwen35moe" => Ok(Box::new(qwen35moe::Qwen35MoeModel::new(
            gguf,
            weights,
            tokenizer,
            spec_enabled,
        )?)),
        other => Err(ModelError::Unsupported(other.to_string()).into()),
    }
}
