//! Qwen3.6-A3B MoE model (GGUF architecture name `qwen35moe`).
//!
//! Hybrid architecture, fundamentally different from LLaMA:
//!   - `n_layer` blocks (41 in the reference checkpoint); the last
//!     `nextn_predict_layers` block(s) are MTP (NextN) — *skipped here*
//!     per user direction. The "main trunk" is the first
//!     `n_main = n_layer - nextn_predict_layers` blocks.
//!   - Of the main trunk, every `full_attention_interval`-th block is
//!     full attention; the rest are SSM / gated-delta-net blocks (Mamba-2
//!     family). Llama.cpp's marker is
//!     `recurrent_layer_arr[i] = (i < n_main) && ((i+1) % full_attn_interval != 0)`,
//!     so an attention layer is `(i+1) % full_attn_interval == 0`.
//!   - Every block has an MoE FFN: top-`expert_used_count` of `expert_count`
//!     routed experts, plus a sigmoid-gated shared expert.
//!   - Attention blocks apply per-head RMS norm on Q and K before M-RoPE,
//!     and the Q projection is *double width* — second half is sliced
//!     off as a sigmoid gate over the attention output.
//!
//! Phase-1 status: this file builds the param + weight schema and stubs
//! `record_forward` so the model loads cleanly. Phases 2 and 3 wire the
//! attention and SSM forward passes respectively.

use std::error::Error;

use crate::gguf::{GgmlType, GgufFile, MetadataValue};
use crate::inference::context::DispatchContext;
use crate::inference::kv_cache::KvCache;
use crate::inference::ops::{
    cache_io, cast, elementwise, flash_attn, matmul, moe, rms_norm, rope_multi, ssm,
};
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::tokenizer::TokenizerBundle;

use super::{CacheDims, Model, ModelError};

const ARCH: &str = "qwen35moe";

#[derive(Debug, Clone)]
pub struct Qwen35MoeParams {
    pub n_layer: u32,
    /// `n_layer - nextn_predict_layers` — the count of "real" forward blocks.
    /// Layers `[n_main, n_layer)` are MTP/NextN blocks we skip.
    pub n_main: u32,
    pub nextn_predict_layers: u32,
    pub full_attn_interval: u32,

    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,

    // RoPE — multi-section (M-RoPE).
    pub rope_dim: u32,
    pub rope_sections: [u32; 4],
    pub rope_freq_base: f32,

    // MoE.
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub expert_ff: u32,
    pub shared_expert_ff: u32,

    // SSM (Mamba-2-style state-space block).
    pub ssm_state: u32,    // d_state
    pub ssm_conv: u32,     // conv kernel size
    pub ssm_groups: u32,   // num_k_heads
    pub ssm_dt_rank: u32,  // num_v_heads
    pub ssm_inner: u32,    // value_dim = head_v_dim * num_v_heads

    pub rms_eps: f32,
}

impl Qwen35MoeParams {
    /// True when layer `i` (0-indexed) is a full-attention block in the
    /// main trunk. Layers in the MTP range (`i >= n_main`) return false and
    /// the caller is expected to skip them.
    pub fn is_attention_layer(&self, i: u32) -> bool {
        i < self.n_main && ((i + 1) % self.full_attn_interval == 0)
    }

    /// Total dimensionality of all attention heads on the K side
    /// (`head_dim_k * n_head_kv`).
    pub fn n_embd_k_gqa(&self) -> u32 {
        self.head_dim_k * self.n_head_kv
    }

    /// Total dimensionality of all attention heads on the V side.
    pub fn n_embd_v_gqa(&self) -> u32 {
        self.head_dim_v * self.n_head_kv
    }

    /// Output width of `wq` in attention blocks. `wq` projects to Q + a
    /// sigmoid gate of the same width; total is `2 * head_dim_k * n_head`.
    pub fn wq_out(&self) -> u32 {
        2 * self.head_dim_k * self.n_head
    }
}

/// MoE FFN weights — shared between attention and SSM blocks. The routed
/// experts share a flat `[n_embd, expert_ff, n_expert]` tensor layout
/// (per-expert slabs along the last axis); the shared expert is a normal
/// pair of `[n_embd, shared_expert_ff]` matmuls.
pub struct MoeFfnWeights {
    pub ffn_gate_inp: TensorView,       // [n_embd, n_expert] — router
    pub ffn_gate_inp_shexp: TensorView, // [n_embd] — scalar shared-expert gate
    pub ffn_gate_exps: TensorView,      // [n_embd, expert_ff, n_expert]
    pub ffn_up_exps: TensorView,        // [n_embd, expert_ff, n_expert]
    pub ffn_down_exps: TensorView,      // [expert_ff, n_embd, n_expert]
    pub ffn_gate_shexp: TensorView,     // [n_embd, shared_expert_ff]
    pub ffn_up_shexp: TensorView,       // [n_embd, shared_expert_ff]
    pub ffn_down_shexp: TensorView,     // [shared_expert_ff, n_embd]
}

pub struct AttentionBlockWeights {
    pub attn_norm: TensorView,
    pub post_attn_norm: TensorView,
    pub wq: TensorView,          // [n_embd, 2 * head_dim_k * n_head]
    pub wk: TensorView,          // [n_embd, head_dim_k * n_head_kv]
    pub wv: TensorView,          // [n_embd, head_dim_v * n_head_kv]
    pub wo: TensorView,          // [head_dim_v * n_head, n_embd]
    pub attn_q_norm: TensorView, // [head_dim_k]
    pub attn_k_norm: TensorView, // [head_dim_k]
    pub moe: MoeFfnWeights,
}

pub struct SsmBlockWeights {
    pub attn_norm: TensorView,
    pub post_attn_norm: TensorView,
    pub attn_qkv: TensorView,     // [n_embd, key_dim*2 + value_dim]
    pub attn_gate: TensorView,    // [n_embd, value_dim] — z gate
    pub ssm_alpha: TensorView,    // [n_embd, num_v_heads]
    pub ssm_beta: TensorView,     // [n_embd, num_v_heads]
    pub ssm_a: TensorView,        // [num_v_heads] — log-scale "A"
    pub ssm_dt_bias: TensorView,  // [num_v_heads]
    pub ssm_conv1d: TensorView,   // [conv_kernel, conv_channels]
    pub ssm_norm: TensorView,     // [head_v_dim]
    pub ssm_out: TensorView,      // [value_dim, n_embd]
    pub moe: MoeFfnWeights,
}

pub enum BlockWeights {
    Attention(AttentionBlockWeights),
    Ssm(SsmBlockWeights),
}

impl BlockWeights {
    /// Both block flavors carry the same MoE FFN weights — common access
    /// lets `record_forward` invoke the same MoE pass after either an
    /// attention or SSM step.
    pub fn moe(&self) -> &MoeFfnWeights {
        match self {
            BlockWeights::Attention(a) => &a.moe,
            BlockWeights::Ssm(s) => &s.moe,
        }
    }

    pub fn attn_norm(&self) -> TensorView {
        match self {
            BlockWeights::Attention(a) => a.attn_norm,
            BlockWeights::Ssm(s) => s.attn_norm,
        }
    }

    pub fn post_attn_norm(&self) -> TensorView {
        match self {
            BlockWeights::Attention(a) => a.post_attn_norm,
            BlockWeights::Ssm(s) => s.post_attn_norm,
        }
    }
}

/// MTP / NextN draft-head weights (the block at GGUF index `n_main`).
/// Loaded only when speculative decoding is requested and the checkpoint
/// actually ships the `blk.{n_main}.nextn.*` tensors. The output head is
/// shared with the main model (`Qwen35MoeWeights::output` / `token_embd`),
/// so only the NextN-specific norms + projection and the transformer
/// `body` are held here.
pub struct MtpWeights {
    /// RMSNorm applied to the embedding of the accepted token.
    pub enorm: TensorView,
    /// RMSNorm applied to the previous-layer hidden state.
    pub hnorm: TensorView,
    /// `[2*n_embd, n_embd]` projection of concat(hnorm(h), enorm(emb)).
    pub eh_proj: TensorView,
    /// Final RMSNorm before the shared output head (replaces `output_norm`).
    pub shared_head_norm: TensorView,
    /// The MTP transformer block — full attention + MoE FFN, structurally
    /// identical to a main attention block.
    pub body: AttentionBlockWeights,
}

pub struct Qwen35MoeWeights {
    pub token_embd: TensorView,
    pub output_norm: TensorView,
    /// `None` ⇒ tied weights: lm_head uses `token_embd`.
    pub output: Option<TensorView>,
    /// One entry per main-trunk block. MTP blocks are not stored here.
    pub blocks: Vec<BlockWeights>,
    /// MTP / NextN draft head — `Some` only when spec decoding requested
    /// and the tensors exist. `None` ⇒ model behaves exactly as before.
    pub mtp: Option<MtpWeights>,
}

pub struct Qwen35MoeModel {
    pub params: Qwen35MoeParams,
    pub weights: Qwen35MoeWeights,
    pub handle: WeightsHandle,
    #[allow(dead_code)]
    pub tokenizer: TokenizerBundle,
}

impl Qwen35MoeModel {
    pub fn new(
        gguf: &GgufFile,
        handle: WeightsHandle,
        tokenizer: TokenizerBundle,
        spec_enabled: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let params = parse_params(gguf, &handle)?;
        let weights = collect_weights(&handle, &params, spec_enabled)?;
        tracing::info!(
            arch = ARCH,
            n_layer = params.n_layer,
            n_main = params.n_main,
            attention_layers = (0..params.n_main).filter(|&i| params.is_attention_layer(i)).count(),
            ssm_layers = (0..params.n_main).filter(|&i| !params.is_attention_layer(i)).count(),
            n_expert = params.n_expert,
            n_expert_used = params.n_expert_used,
            mtp = weights.mtp.is_some(),
            "qwen35moe model loaded",
        );
        Ok(Self {
            params,
            weights,
            handle,
            tokenizer,
        })
    }
}

impl Model for Qwen35MoeModel {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn vocab_size(&self) -> u32 {
        self.params.n_vocab
    }

    fn supports_mtp_spec(&self) -> bool {
        self.weights.mtp.is_some()
    }

    fn cache_dims(&self) -> CacheDims {
        // Only attention blocks need a KV cache, but the engine indexes the
        // cache by layer (one slot per `n_layer`). For Phase 1 we expose the
        // attention block dims uniformly and accept the unused-SSM-layer
        // waste; a later optimization can compact to attention-only slots.
        //
        // When the MTP draft head is loaded it needs its own (ephemeral,
        // per-step) KV slot at index `n_main`, so allocate one extra layer.
        let mtp_slots = if self.weights.mtp.is_some() { 1 } else { 0 };
        CacheDims {
            n_layer: self.params.n_main + mtp_slots,
            head_dim: self.params.head_dim_k,
            n_head_kv: self.params.n_head_kv,
        }
    }

    fn ssm_state_dims(&self) -> Option<crate::models::SsmStateDims> {
        // One state per SSM layer (= 30 for the reference checkpoint).
        // - conv state: (conv_kernel - 1) * conv_channels per layer
        //   = 3 * 8192 = 24576 floats
        // - GDN recurrent state: S_v * S_v * num_v_heads per layer
        //   = 128 * 128 * 32 = 524288 floats
        let p = &self.params;
        let n_ssm_layers = (0..p.n_main).filter(|&i| !p.is_attention_layer(i)).count() as u32;
        let conv_channels = 2 * p.ssm_groups * p.ssm_state + p.ssm_dt_rank * p.ssm_state;
        let conv_state_floats = (p.ssm_conv - 1) * conv_channels;
        let gdn_state_floats = p.ssm_state * p.ssm_state * p.ssm_dt_rank;
        Some(crate::models::SsmStateDims {
            n_ssm_layers,
            conv_state_floats,
            gdn_state_floats,
            conv_channels,
            conv_kernel: p.ssm_conv,
        })
    }

    fn weights(&self) -> &WeightsHandle {
        &self.handle
    }

    fn tokenizer(&self) -> &TokenizerBundle {
        &self.tokenizer
    }

    fn replay_constants(&self) -> Option<crate::inference::decode_dyn::ModelReplayConstants> {
        Some(crate::inference::decode_dyn::ModelReplayConstants {
            rope_d_offset_per_position: self.params.head_dim_k * self.params.n_head_kv,
            v_cache_d_offset_per_position: self.params.head_dim_v * self.params.n_head_kv,
            mrope_axes: 4,
        })
    }

    fn decode_grid(&self, kv: u32, shader_core_count: u32) -> Option<(u32, u32)> {
        // L=1 for decode → base_wgs = n_head (= ne2 of the FA dispatch).
        // The cached cmdbuf bakes the flash_attn split-K wg count via
        // cmd_update_buffer, so the (k_num, blocks_per_split) pair is
        // the canonical "graph shape" the Engine compares against.
        let base_wgs = self.params.n_head;
        Some(crate::inference::ops::flash_attn::pick_k_num_clamped(
            shader_core_count,
            base_wgs,
            kv,
        ))
    }

    fn refresh_replay_inputs(
        &self,
        host_ptr: *mut u8,
        plan: &crate::inference::decode_dyn::ReplayPlan,
        tokens: &[u32],
        position_offset: u32,
    ) -> Result<(), Box<dyn Error>> {
        let token_off = plan
            .token_buf_offset
            .ok_or("replay plan missing token_buf_offset")?;
        let pos_off = plan
            .positions_buf_offset
            .ok_or("replay plan missing positions_buf_offset")?;
        // SAFETY: host_ptr is the mapped pointer of the host-coherent
        // scratch region; both offsets were captured during the
        // recording pass into that same region.
        unsafe {
            let tok_dst = host_ptr.add(token_off as usize) as *mut u32;
            for (i, &t) in tokens.iter().enumerate() {
                std::ptr::write(tok_dst.add(i), t);
            }
            // M-RoPE positions: 4 axes × L tokens. Each axis gets the
            // same linear sequence (text decode — see `record_forward` for
            // the same layout at record time). The rope base lags the KV-slot
            // count by `rope_position_lag` once an image was prefilled
            // (captured into the plan at record time); zero for text-only, so
            // this matches the original `position_offset + i` layout exactly.
            let l = tokens.len();
            let rope_base = position_offset.saturating_sub(plan.rope_position_lag);
            let pos_dst = host_ptr.add(pos_off as usize) as *mut u32;
            for axis in 0..4usize {
                for i in 0..l {
                    std::ptr::write(pos_dst.add(axis * l + i), rope_base + i as u32);
                }
            }
        }
        Ok(())
    }

    fn scratch_bytes_estimate(
        &self,
        n_ubatch: u32,
        max_seq_len: u32,
        k_dtype: GgmlType,
        v_dtype: GgmlType,
    ) -> u64 {
        let p = &self.params;
        // Per-pass token count: bounded by n_ubatch, or the whole context when
        // chunking is disabled (n_ubatch == 0 ⇒ a single full-prompt pass).
        let l = if n_ubatch == 0 { max_seq_len.max(1) } else { n_ubatch } as u64;
        let hidden = p.n_embd as u64;
        let vocab = p.n_vocab as u64;
        let value_dim = (p.ssm_dt_rank * p.ssm_state) as u64;
        let key_dim = (p.ssm_groups * p.ssm_state) as u64;
        let conv_channels = 2 * key_dim + value_dim;
        let expert_intermediate = 3 * (p.expert_ff as u64) * (p.n_expert_used as u64);
        // Sum the widths of all [_, l] F32 buffers that can be live within a
        // single layer. We add attention + SSM + MoE widths together — a safe
        // over-count, since a given layer runs only one mixer type — so the
        // estimate bounds every layer kind.
        let summed_width = 7 * hidden                          // norms / proj / residual fan-out
            + 2 * p.wq_out() as u64                            // attention Q + gate
            + 3 * p.n_embd_v_gqa() as u64                      // attn_out / gated
            + p.n_embd_k_gqa() as u64
            + conv_channels + 3 * value_dim + key_dim          // SSM conv + GDN working set
            + expert_intermediate + 2 * p.n_expert as u64      // MoE experts + routing
            + p.shared_expert_ff as u64;
        let per_layer = summed_width * l * 4;
        let residual = hidden * l * 4;
        let mask = l * l * 4;
        let logits = vocab * 4;
        // Heterogeneous K/V caches materialize the [0, ctx) prefix to F32 per
        // layer; homogeneous caches bind directly.
        let staging = if k_dtype != v_dtype {
            2 * p.head_dim_k.max(p.head_dim_v) as u64 * p.n_head_kv as u64 * max_seq_len as u64 * 4
        } else {
            0
        };
        let raw = per_layer + residual + mask + logits + staging;
        raw + raw / 3 + (32 << 20) // +33% headroom + 32 MiB slack
    }

    fn record_forward(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        compute_logits: bool,
    ) -> Result<Option<TensorView>, Box<dyn Error>> {
        Ok(self
            .forward_impl(
                ctx,
                cache,
                tokens,
                position_offset,
                compute_logits,
                /*full_logits=*/ false,
                /*checkpoint=*/ false,
                /*image=*/ None,
            )?
            .logits)
    }

    fn record_forward_image_chunk(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        chunk_tokens: &[u32],
        chunk_global_start: usize,
        image_embeddings: &[f32],
        image_global_start: usize,
        image_nx: usize,
        image_ny: usize,
        prompt_pos0: u32,
        compute_logits: bool,
    ) -> Result<Option<TensorView>, Box<dyn Error>> {
        let n_embd = self.params.n_embd as usize;
        let n_tok = image_nx * image_ny;
        if image_embeddings.len() != n_embd * n_tok {
            return Err(format!(
                "image embeddings len {} != n_embd {} * n_tok {} — the vision \
                 projection_dim must equal the text model's embedding_length",
                image_embeddings.len(),
                n_embd,
                n_tok
            )
            .into());
        }
        let l = chunk_tokens.len();
        let span = ImageSpan { start: image_global_start, n_tok, nx: image_nx, ny: image_ny };

        // M-RoPE base lags the KV-slot cursor by the lag accrued *before* this
        // image (text-only ⇒ 0). Held constant across the prefill's chunks —
        // the caller advances `rope_position_lag` only after the whole prefill —
        // so every chunk builds positions off the same global base.
        let rope_base0 = prompt_pos0.saturating_sub(cache.rope_position_lag);
        let positions =
            build_decoder_mrope_positions_window(rope_base0, Some(span), chunk_global_start, l);

        // Which image columns (global [g, g+n_tok)) fall in this chunk's global
        // range [chunk_global_start, +l)? Splice just those, at their chunk-local
        // residual column.
        let g = image_global_start;
        let ov_start = chunk_global_start.max(g);
        let ov_end = (chunk_global_start + l).min(g + n_tok);
        let splice = if ov_start < ov_end {
            let col0 = ov_start - g;
            let count = ov_end - ov_start;
            let sub = &image_embeddings[col0 * n_embd..(col0 + count) * n_embd];
            Some((sub, ov_start - chunk_global_start))
        } else {
            None
        };

        Ok(self
            .forward_impl(
                ctx,
                cache,
                chunk_tokens,
                /*position_offset=*/ cache.position,
                compute_logits,
                /*full_logits=*/ false,
                /*checkpoint=*/ false,
                Some(ForwardImage { positions: &positions, splice }),
            )?
            .logits)
    }

    fn record_forward_full(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        full_logits: bool,
        checkpoint: bool,
    ) -> Result<crate::models::ForwardFullOut, Box<dyn Error>> {
        self.forward_impl(
            ctx,
            cache,
            tokens,
            position_offset,
            /*compute_logits=*/ true,
            full_logits,
            checkpoint,
            /*image=*/ None,
        )
    }

    fn record_ssm_finalize(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        accept_len: u32,
    ) -> Result<(), Box<dyn Error>> {
        self.ssm_finalize_impl(ctx, cache, accept_len)
    }

    fn record_mtp_seed(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        hiddens: &[f32],
        tokens: &[u32],
        position_offset: u32,
    ) -> Result<(), Box<dyn Error>> {
        self.mtp_seed_impl(ctx, cache, hiddens, tokens, position_offset)
    }

    fn record_mtp_draft(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        h_last: &[f32],
        prev_token: u32,
        position: u32,
    ) -> Result<crate::models::MtpDraftOut, Box<dyn Error>> {
        self.mtp_draft_impl(ctx, cache, h_last, prev_token, position)
    }

    fn record_forward_batch(
        &self,
        ctx: &mut DispatchContext,
        batch: &mut crate::inference::kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        self.forward_batch_impl(ctx, batch, tokens, positions, slots)
    }

    fn supports_unified(&self) -> bool {
        true
    }

    fn record_forward_unified(
        &self,
        ctx: &mut DispatchContext,
        batch: &mut crate::inference::kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        seq_lens: &[u32],
        slots: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        self.forward_unified_impl(ctx, batch, tokens, positions, seq_lens, slots)
    }
}

impl Qwen35MoeModel {
    /// Shared forward body for both [`Model::record_forward`] (last-token
    /// logits) and [`Model::record_forward_full`] (all-position logits +
    /// hidden). Returns the logits tensor and the per-position
    /// pre-`output_norm` residual `[n_embd, L]`.
    ///
    /// `compute_logits=false` (chunked-prefill intermediate ubatch) runs the
    /// layers to populate the KV / recurrent state but skips the epilogue;
    /// the returned `logits` is then `None`.
    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        compute_logits: bool,
        full_logits: bool,
        checkpoint: bool,
        image: Option<ForwardImage<'_>>,
    ) -> Result<crate::models::ForwardFullOut, Box<dyn Error>> {
        let p = &self.params;
        let l = tokens.len() as u32;
        if l == 0 {
            return Err("empty prompt".into());
        }
        if let Some(fi) = image.as_ref() {
            debug_assert_eq!(fi.positions.len(), 4 * l as usize, "image positions must be 4*l");
            if let Some((emb, start)) = fi.splice {
                let cols = emb.len() / p.n_embd as usize;
                if start + cols > l as usize {
                    return Err("image splice overruns the chunk".into());
                }
            }
        }

        let hidden = p.n_embd as u64;
        let head_dim_k = p.head_dim_k as u64;
        let head_dim_v = p.head_dim_v as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv as u64;
        let n_embd_kv = p.n_embd_k_gqa() as u64; // K-side projection width
        let n_embd_vv = p.n_embd_v_gqa() as u64; // V-side
        let wq_out = p.wq_out() as u64; // 2 * head_dim_k * n_head
        let hidden_v = head_dim_v * n_head;

        if cache.position != position_offset {
            return Err(format!(
                "cache.position {} != position_offset {position_offset}",
                cache.position
            )
            .into());
        }
        let total_len = position_offset + l;
        let kv_len_u = total_len as u64;

        // ─── Prologue: token ids, mask, embedding lookup, positions ───
        let token_buf = ctx.alloc_scratch((l as u64) * 4)?;
        write_u32(ctx, token_buf, tokens)?;

        // M-RoPE positions: shader reads pos[i2 + ne02 * axis] for axes
        // 0..3. With `sections=[11,11,10,0]` only axes 0/1/2 are accessed
        // (axis 3 has width 0). For text-only inference we replicate the
        // same position sequence to all 4 sub-arrays so any axis read
        // resolves to the linear text position.
        let positions_buf = ctx.alloc_scratch(4 * (l as u64) * 4)?;
        // M-RoPE positions, axis-major `pos[axis*l + tok]`. For text-only this
        // is the linear position replicated to all 4 axes (so imrope 0≡1). For an
        // image chunk the caller precomputes them globally (the image's 2D cursor
        // is continuous across chunks), so we use those verbatim. The text base
        // lags the KV-slot count (`position_offset`) by `rope_position_lag` once
        // an image was prefilled; text-only ⇒ lag 0 ⇒ `rope_base == position_offset`
        // (the validated text path, unchanged).
        let positions: Vec<u32> = match image.as_ref() {
            Some(fi) => fi.positions.to_vec(),
            None => {
                let rope_base = position_offset.saturating_sub(cache.rope_position_lag);
                build_decoder_mrope_positions(l as usize, rope_base, &[])
            }
        };
        write_u32(ctx, positions_buf, &positions)?;

        // Snapshot the replay-input offsets so the persistent-decode-cmdbuf
        // path can re-write these slots between submits via
        // `refresh_replay_inputs`. `rope_position_lag` is captured here so the
        // replay path can re-derive the lagged rope base each submit.
        if let Some(plan) = ctx.replay_plan.as_mut() {
            plan.token_buf_offset = Some(token_buf.offset);
            plan.positions_buf_offset = Some(positions_buf.offset);
            plan.rope_position_lag = cache.rope_position_lag;
        }

        // Single-token decode (l == 1) needs no mask: every KV slot is
        // causally visible, so flash_attn runs with MASK_ENABLE=0 and we skip
        // the O(total_len) host-side mask build per step. See llama.rs.
        let mask = if l > 1 {
            // Within-chunk mask only: [l × l] (not [kv_len × l]). The cached
            // prefix [0, position_offset) is always visible and is handled
            // shader-side via mask_kv_offset. See llama.rs.
            let m = ctx.alloc_tensor([l as u64, l as u64, 1, 1], GgmlType::F32)?;
            write_causal_mask(ctx, m, l)?;
            Some(m)
        } else {
            None
        };

        let residual = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, l, residual)?;
        // Splice the vision embeddings over the `<|image_pad|>` rows that fall in
        // THIS chunk: `count` consecutive columns from chunk-local `start`. Both
        // the vision output and the residual are n_embd-contiguous per column, so
        // this is a contiguous F32 block copy over `residual`'s sub-columns
        // (get_rows already barriered `residual`, so the overwrite is ordered).
        if let Some((emb, start)) = image.as_ref().and_then(|fi| fi.splice) {
            let count = (emb.len() / p.n_embd as usize) as u64;
            let emb_buf = ctx.alloc_scratch((emb.len() as u64) * 4)?;
            write_f32(ctx, emb_buf, emb)?;
            let img_src = TensorView {
                buffer: emb_buf.buffer,
                byte_offset: emb_buf.offset,
                byte_size: emb_buf.size,
                dims: [hidden, count, 1, 1],
                byte_stride: [4, hidden * 4, hidden * count * 4, hidden * count * 4],
                element_stride: [1, hidden, hidden * count, hidden * count],
                dtype: GgmlType::F32,
            };
            let img_dst = TensorView {
                buffer: residual.buffer,
                byte_offset: residual.byte_offset + (start as u64) * hidden * 4,
                byte_size: count * hidden * 4,
                dims: [hidden, count, 1, 1],
                byte_stride: [4, hidden * 4, hidden * count * 4, hidden * count * 4],
                element_stride: [1, hidden, hidden * count, hidden * count],
                dtype: GgmlType::F32,
            };
            cast::record_cast(ctx, img_src, img_dst)?;
        }
        ctx.tap("input_embed", residual)?;
        // Boundary marker: everything emitted from here until the
        // next `mark(…)` is attributed to BlockClass::Embed. No-op
        // unless the `profile_gpu` Cargo feature is set.
        ctx.mark(crate::inference::profile::BlockClass::Embed);

        let layer_checkpoint = ctx.scratch_checkpoint();

        let mut rope_params =
            rope_multi::RopeMultiParams::qwen_default(p.rope_dim, p.rope_freq_base, p.rope_sections);
        // Qwen3-VL text decoder uses interleaved M-RoPE. For text-only every
        // axis is equal so is_imrope 0≡1 (kept 0 to preserve the validated text
        // path); when an image is present the axes differ, so enable it.
        if image.is_some() {
            rope_params.is_imrope = 1;
        }
        let scale = 1.0 / (head_dim_k as f32).sqrt();
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim_k as u32,
            head_dim_v: head_dim_v as u32,
            gqa_ratio: (p.n_head / p.n_head_kv).max(1),
            scale,
        };
        let cache_direct = cache.config.k_dtype == cache.config.v_dtype;

        // ─── Per-layer loop ───
        // All diagnostic toggles are behind the `gpu_debug` feature (see
        // `runtime_flags`): each accessor reads its `SEEKER_*` env var once
        // (cached) when the feature is on, and constant-folds to
        // `false`/`None` when it's off so every branch below is eliminated
        // from production builds — no per-layer instrumentation at all.
        let max_layers = crate::runtime_flags::qwen_max_layers()
            .map(|n| n as usize)
            .unwrap_or(self.weights.blocks.len());
        // When diff-dumping intermediates, each layer's taps must remain in
        // their own scratch slots until the GPU has executed all dispatches
        // and the host reads them back. Restoring scratch between layers
        // makes subsequent layers overwrite the tap data at the same byte
        // offsets — making every per-layer tap report the same value.
        let dump_mode = crate::runtime_flags::qwen_diff_dump();
        let skip_attn = crate::runtime_flags::qwen_no_attn();
        let skip_ssm = crate::runtime_flags::qwen_no_ssm();
        let skip_moe = crate::runtime_flags::qwen_no_moe();
        for (layer_idx, block) in self.weights.blocks.iter().take(max_layers).enumerate() {
            if !dump_mode {
                ctx.scratch_restore(layer_checkpoint);
            }

            // Attention-or-SSM "communication" step.
            //   SEEKER_QWEN_NO_ATTN=1 → skip only the 10 attention layers
            //   SEEKER_QWEN_NO_SSM=1  → skip only the 30 SSM layers
            // Used to bisect which block type contributes a bug.
            match block {
                BlockWeights::Attention(att) if !skip_attn => {
                    ctx.mark(crate::inference::profile::BlockClass::Attn);
                    attention_block(
                        ctx,
                        att,
                        cache,
                        residual,
                        mask,
                        positions_buf,
                        rope_params,
                        fa_params,
                        layer_idx as u32,
                        position_offset,
                        total_len,
                        kv_len_u,
                        l,
                        p,
                        head_dim_k,
                        head_dim_v,
                        n_head,
                        n_head_kv,
                        n_embd_kv,
                        n_embd_vv,
                        wq_out,
                        hidden,
                        hidden_v,
                        cache_direct,
                    )?;
                }
                BlockWeights::Ssm(ssm_w) if !skip_ssm => {
                    ctx.mark(crate::inference::profile::BlockClass::Ssm);
                    // Map block index to SSM-layer index (counting only SSM
                    // blocks). cache.ssm_gdn_states is indexed in SSM-layer
                    // order, not block order.
                    let ssm_layer_idx = (0..layer_idx)
                        .filter(|&i| !p.is_attention_layer(i as u32))
                        .count();
                    let gdn_state = cache.ssm_gdn_states.get(ssm_layer_idx).copied();
                    let conv_state = cache.ssm_conv_states.get(ssm_layer_idx).copied();
                    let ssm_host_ptr = cache.ssm_region.as_ref().and_then(|r| r.host_ptr);
                    // Checkpoint mode (spec-decode verify): write per-position
                    // GDN snapshots + a conv-input backup instead of the live
                    // recurrent state, so a partial-accept step can roll back.
                    let checkpoint_bufs = if checkpoint {
                        match (
                            cache.ssm_gdn_snapshots.get(ssm_layer_idx).copied(),
                            cache.ssm_conv_backups.get(ssm_layer_idx).copied(),
                        ) {
                            (Some(g), Some(c)) => Some((g, c)),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    ssm_block(ctx, ssm_w, residual, p, hidden, l, layer_idx as u32, gdn_state, conv_state, ssm_host_ptr, checkpoint_bufs)?;
                }
                _ => {
                    // block type currently passthrough'd via the matching
                    // SEEKER_QWEN_NO_* env flag.
                }
            }

            // Common "FFN" step for both block types: MoE with shared expert.
            // SEEKER_QWEN_NO_MOE=1 skips the FFN entirely — leaves residual
            // unchanged. Used to isolate whether the MoE-FFN accumulation
            // chain (and not the prologue / epilogue) is the source of bugs.
            if !skip_moe {
                ctx.mark(crate::inference::profile::BlockClass::MoE);
                moe_ffn(ctx, block.moe(), block.post_attn_norm(), residual, p, hidden, l, layer_idx as u32)?;
            }
        }
        if !dump_mode {
            ctx.scratch_restore(layer_checkpoint);
        }

        // Intermediate prefill ubatches only populate the KV / recurrent
        // state — skip the epilogue (no logits needed until the last
        // ubatch). cache.position is still advanced.
        if !compute_logits {
            cache_io::advance(cache, l);
            return Ok(crate::models::ForwardFullOut {
                logits: None,
                residual,
            });
        }

        // ─── Epilogue: final norm + lm_head ───
        let elem_size = 4u64;
        let vocab = p.n_vocab as u64;
        ctx.mark(crate::inference::profile::BlockClass::Epilogue);
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let logits = Some(if full_logits {
            // Batched verify path: final-norm + project ALL L positions.
            // n_vocab × L scratch is fine here because L = n_draft+1 is small.
            let final_norm = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, self.weights.output_norm, final_norm, p.rms_eps)?;
            ctx.mark(crate::inference::profile::BlockClass::LmHead);
            let all_logits = ctx.alloc_tensor([vocab, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, lm_head, final_norm, all_logits)?;
            all_logits
        } else {
            // Default decode/prefill: only the LAST token's residual is
            // normalized + projected — full-batch logits would burn
            // n_vocab × L scratch (~318MB at L=320 with vocab=248k).
            let residual_last = TensorView {
                buffer: residual.buffer,
                byte_offset: residual.byte_offset + (l as u64 - 1) * hidden * elem_size,
                byte_size: hidden * elem_size,
                dims: [hidden, 1, 1, 1],
                byte_stride: [elem_size, hidden * elem_size, hidden * elem_size, hidden * elem_size],
                element_stride: [1, hidden, hidden, hidden],
                dtype: residual.dtype,
            };
            let final_norm = ctx.alloc_tensor([hidden, 1, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual_last, self.weights.output_norm, final_norm, p.rms_eps)?;
            ctx.tap("final_norm", final_norm)?;
            ctx.mark(crate::inference::profile::BlockClass::LmHead);
            let last_logits = ctx.alloc_tensor([vocab, 1, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, lm_head, final_norm, last_logits)?;
            last_logits
        });

        cache_io::advance(cache, l);
        Ok(crate::models::ForwardFullOut { logits, residual })
    }

    /// Record one autoregressive MTP / NextN draft step (`L = 1`).
    ///
    /// Mirrors the DeepSeek-V3 / Qwen3-Next MTP module:
    /// `proj = eh_proj · concat(hnorm(h_last), enorm(emb(prev_token)))`,
    /// then the MTP transformer block (attention + MoE), then
    /// `logits = output · shared_head_norm(block_out)`. The block output
    /// (`block_out`) is the recurrence hidden for the next draft step.
    ///
    /// The MTP attention uses its own persistent KV slot (index `n_main`)
    /// at absolute `position`, so it attends to the full prior context
    /// [0, position) — seeded from the prompt's main hidden states
    /// ([`mtp_seed_impl`]) and extended by prior steps' drafts — plus the
    /// current draft token. Draft-window quality only affects acceptance;
    /// verification keeps the emitted tokens lossless regardless.
    #[allow(clippy::too_many_arguments)]
    fn mtp_draft_impl(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        h_last: &[f32],
        prev_token: u32,
        position: u32,
    ) -> Result<crate::models::MtpDraftOut, Box<dyn Error>> {
        let p = &self.params;
        let mtp = self
            .weights
            .mtp
            .as_ref()
            .ok_or("mtp_draft called but MTP weights are not loaded")?;

        let hidden = p.n_embd as u64;
        let head_dim_k = p.head_dim_k as u64;
        let head_dim_v = p.head_dim_v as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv as u64;
        let n_embd_kv = p.n_embd_k_gqa() as u64;
        let n_embd_vv = p.n_embd_v_gqa() as u64;
        let wq_out = p.wq_out() as u64;
        let hidden_v = head_dim_v * n_head;
        let vocab = p.n_vocab as u64;
        let mtp_layer = p.n_main; // dedicated KV slot index

        if h_last.len() as u64 != hidden {
            return Err(format!(
                "mtp_draft: h_last len {} != n_embd {hidden}",
                h_last.len()
            )
            .into());
        }

        // Upload the seed hidden state and embed the previous token.
        let h_tensor = ctx.alloc_tensor([hidden, 1, 1, 1], GgmlType::F32)?;
        write_f32(ctx, h_tensor.range(), h_last)?;
        let tok_buf = ctx.alloc_scratch(4)?;
        write_u32(ctx, tok_buf, &[prev_token])?;
        let emb = ctx.alloc_tensor([hidden, 1, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, tok_buf, 1, emb)?;

        // combined = concat(enorm(emb), hnorm(h)) — EMBEDDING FIRST, then
        // hidden, matching llama.cpp's `ggml_concat(e_norm, h_norm, dim=0)`
        // (qwen35moe.cpp). Normalize directly into the two halves of a
        // [2*n_embd] buffer (no separate copy).
        let combined = ctx.alloc_tensor([2 * hidden, 1, 1, 1], GgmlType::F32)?;
        let half = |off_elems: u64| -> TensorView {
            TensorView {
                buffer: combined.buffer,
                byte_offset: combined.byte_offset + off_elems * 4,
                byte_size: hidden * 4,
                dims: [hidden, 1, 1, 1],
                byte_stride: [4, hidden * 4, hidden * 4, hidden * 4],
                element_stride: [1, hidden, hidden, hidden],
                dtype: GgmlType::F32,
            }
        };
        rms_norm::record(ctx, emb, mtp.enorm, half(0), p.rms_eps)?;
        rms_norm::record(ctx, h_tensor, mtp.hnorm, half(hidden), p.rms_eps)?;
        crate::inference::command::record_compute_barrier(ctx.device, ctx.cmd, combined.range());

        // projected = eh_proj @ combined  → [n_embd, 1]. Becomes the block
        // input residual that attention + MoE accumulate into.
        let residual = ctx.alloc_tensor([hidden, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, mtp.eh_proj, combined, residual)?;

        // M-RoPE positions for this single token (4 axes, all = absolute position).
        let positions_buf = ctx.alloc_scratch(4 * 4)?;
        write_u32(ctx, positions_buf, &[position; 4])?;

        let rope_params =
            rope_multi::RopeMultiParams::qwen_default(p.rope_dim, p.rope_freq_base, p.rope_sections);
        let scale = 1.0 / (head_dim_k as f32).sqrt();
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim_k as u32,
            head_dim_v: head_dim_v as u32,
            gqa_ratio: (p.n_head / p.n_head_kv).max(1),
            scale,
        };
        let cache_direct = cache.config.k_dtype == cache.config.v_dtype;
        // Persistent MTP KV: the draft at absolute `position` attends to the
        // full prior context [0, position) (seeded from the prompt's main
        // hidden states + prior steps' drafts) plus itself.
        let total_len = position + 1;

        attention_block(
            ctx,
            &mtp.body,
            cache,
            residual,
            /*mask=*/ None,
            positions_buf,
            rope_params,
            fa_params,
            mtp_layer,
            /*position_offset=*/ position,
            total_len,
            total_len as u64,
            /*l=*/ 1,
            p,
            head_dim_k,
            head_dim_v,
            n_head,
            n_head_kv,
            n_embd_kv,
            n_embd_vv,
            wq_out,
            hidden,
            hidden_v,
            cache_direct,
        )?;
        moe_ffn(ctx, &mtp.body.moe, mtp.body.post_attn_norm, residual, p, hidden, 1, mtp_layer)?;
        // `residual` now holds the MTP block output (the recurrence hidden).

        // logits = output · shared_head_norm(block_out), then argmax on the
        // GPU (drafting is greedy) so only a 4-byte token id is read back.
        let normed = ctx.alloc_tensor([hidden, 1, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, residual, mtp.shared_head_norm, normed, p.rms_eps)?;
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let logits = ctx.alloc_tensor([vocab, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, normed, logits)?;
        let draft_token = crate::inference::ops::sampler::record_greedy(ctx, logits)?;

        Ok(crate::models::MtpDraftOut {
            draft_token,
            block_out: residual,
        })
    }

    /// Populate the MTP draft head's KV (slot `n_main`) for `L` positions
    /// `[position_offset, position_offset+L)` from the main model's hidden
    /// states + next-token ids — the batched (`L>1`) form of the NextN
    /// prologue + attention, run for its K/V side effect only (no MoE / no
    /// output head). Used to seed/extend the draft head's context so
    /// drafting attends to the real prior sequence.
    #[allow(clippy::too_many_arguments)]
    fn mtp_seed_impl(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        hiddens: &[f32],
        tokens: &[u32],
        position_offset: u32,
    ) -> Result<(), Box<dyn Error>> {
        let p = &self.params;
        let mtp = self
            .weights
            .mtp
            .as_ref()
            .ok_or("mtp_seed called but MTP weights are not loaded")?;
        let l = tokens.len() as u32;
        if l == 0 {
            return Ok(());
        }
        let hidden = p.n_embd as u64;
        let head_dim_k = p.head_dim_k as u64;
        let head_dim_v = p.head_dim_v as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv as u64;
        let n_embd_kv = p.n_embd_k_gqa() as u64;
        let n_embd_vv = p.n_embd_v_gqa() as u64;
        let wq_out = p.wq_out() as u64;
        let hidden_v = head_dim_v * n_head;
        let mtp_layer = p.n_main;
        let l_u = l as u64;
        if hiddens.len() as u64 != hidden * l_u {
            return Err(format!(
                "mtp_seed: hiddens len {} != n_embd*L {}",
                hiddens.len(),
                hidden * l_u
            )
            .into());
        }

        // Upload hiddens [n_embd, L] and embed the next tokens.
        let h_tensor = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
        write_f32(ctx, h_tensor.range(), hiddens)?;
        let tok_buf = ctx.alloc_scratch(l_u * 4)?;
        write_u32(ctx, tok_buf, tokens)?;
        let emb = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, tok_buf, l, emb)?;

        // combined[:, i] = concat(enorm(emb_i), hnorm(h_i)) — per-position,
        // embedding first. combined is [2*n_embd, L]; the two halves are
        // strided views (gap of 2*n_embd between columns).
        let combined = ctx.alloc_tensor([2 * hidden, l_u, 1, 1], GgmlType::F32)?;
        let half = |off_elems: u64| -> TensorView {
            TensorView {
                buffer: combined.buffer,
                byte_offset: combined.byte_offset + off_elems * 4,
                byte_size: combined.byte_size - off_elems * 4,
                dims: [hidden, l_u, 1, 1],
                byte_stride: [4, 2 * hidden * 4, 2 * hidden * l_u * 4, 2 * hidden * l_u * 4],
                element_stride: [1, 2 * hidden, 2 * hidden * l_u, 2 * hidden * l_u],
                dtype: GgmlType::F32,
            }
        };
        rms_norm::record(ctx, emb, mtp.enorm, half(0), p.rms_eps)?;
        rms_norm::record(ctx, h_tensor, mtp.hnorm, half(hidden), p.rms_eps)?;
        crate::inference::command::record_compute_barrier(ctx.device, ctx.cmd, combined.range());

        let residual = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, mtp.eh_proj, combined, residual)?;

        // M-RoPE positions (4 axes × L) + causal mask for the batched attn.
        let positions_buf = ctx.alloc_scratch(4 * l_u * 4)?;
        let mut positions: Vec<u32> = Vec::with_capacity(4 * l as usize);
        for _axis in 0..4 {
            for pos in position_offset..position_offset + l {
                positions.push(pos);
            }
        }
        write_u32(ctx, positions_buf, &positions)?;

        let total_len = position_offset + l;
        let kv_len_u = total_len as u64;
        let mask = if l > 1 {
            // Within-chunk mask [l × l]; the cached prefix [0, position_offset)
            // is always visible and handled shader-side via mask_kv_offset
            // (matches forward_impl + attention_block after the chunked-prefill
            // mask change).
            let m = ctx.alloc_tensor([l_u, l_u, 1, 1], GgmlType::F32)?;
            write_causal_mask(ctx, m, l)?;
            Some(m)
        } else {
            None
        };

        let rope_params =
            rope_multi::RopeMultiParams::qwen_default(p.rope_dim, p.rope_freq_base, p.rope_sections);
        let scale = 1.0 / (head_dim_k as f32).sqrt();
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim_k as u32,
            head_dim_v: head_dim_v as u32,
            gqa_ratio: (p.n_head / p.n_head_kv).max(1),
            scale,
        };
        let cache_direct = cache.config.k_dtype == cache.config.v_dtype;

        // Attention block writes K/V into the MTP slot for these positions.
        // Its attention output is discarded — we only want the KV side effect,
        // so MoE + output head are skipped.
        attention_block(
            ctx,
            &mtp.body,
            cache,
            residual,
            mask,
            positions_buf,
            rope_params,
            fa_params,
            mtp_layer,
            position_offset,
            total_len,
            kv_len_u,
            l,
            p,
            head_dim_k,
            head_dim_v,
            n_head,
            n_head_kv,
            n_embd_kv,
            n_embd_vv,
            wq_out,
            hidden,
            hidden_v,
            cache_direct,
        )?;
        Ok(())
    }

    /// Commit per-position SSM snapshots from a checkpoint verify into the
    /// live recurrent state, selecting position `accept_len`. For each SSM
    /// layer: copy GDN snapshot slot `accept_len` → live GDN state, and
    /// extract the conv state at `accept_len` (rows `[accept_len+1 ..
    /// accept_len+conv_kernel-1]` of the backed-up conv window) → live conv
    /// state. Replaces the partial-acceptance re-run.
    fn ssm_finalize_impl(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        accept_len: u32,
    ) -> Result<(), Box<dyn Error>> {
        let elem = 4u64;
        let conv_kernel = cache.ssm_conv_kernel as u64;
        let conv_channels = cache.ssm_conv_channels as u64;
        let n_padded = conv_kernel - 1 + cache.ssm_max_snapshots as u64;
        let state_dim_inner = conv_kernel - 1;
        let n_ssm = cache.ssm_gdn_states.len();

        for i in 0..n_ssm {
            // GDN: contiguous copy of snapshot slot[accept_len] → live state.
            let live_gdn = cache.ssm_gdn_states[i];
            let snap = cache.ssm_gdn_snapshots[i];
            let state_floats = live_gdn.size / elem;
            unsafe {
                use ash::vk;
                let copy = vk::BufferCopy::default()
                    .src_offset(snap.offset + accept_len as u64 * state_floats * elem)
                    .dst_offset(live_gdn.offset)
                    .size(state_floats * elem);
                ctx.device.device.cmd_copy_buffer(
                    ctx.cmd,
                    snap.buffer,
                    live_gdn.buffer,
                    std::slice::from_ref(&copy),
                );
                let bar = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(live_gdn.buffer)
                    .offset(live_gdn.offset)
                    .size(state_floats * elem);
                ctx.device.device.cmd_pipeline_barrier(
                    ctx.cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&bar),
                    &[],
                );
            }

            // Conv: strided cast of backup rows [accept_len+1 .. +kernel-1]
            // → live conv state (same layout as the normal writeback).
            let backup = cache.ssm_conv_backups[i];
            let live_conv = cache.ssm_conv_states[i];
            let src = TensorView {
                buffer: backup.buffer,
                byte_offset: backup.offset + (accept_len as u64 + 1) * elem,
                byte_size: backup.size - (accept_len as u64 + 1) * elem,
                dims: [state_dim_inner, conv_channels, 1, 1],
                byte_stride: [
                    elem,
                    n_padded * elem,
                    n_padded * conv_channels * elem,
                    n_padded * conv_channels * elem,
                ],
                element_stride: [1, n_padded, n_padded * conv_channels, n_padded * conv_channels],
                dtype: GgmlType::F32,
            };
            let dst = TensorView {
                buffer: live_conv.buffer,
                byte_offset: live_conv.offset,
                byte_size: live_conv.size,
                dims: [state_dim_inner, conv_channels, 1, 1],
                byte_stride: [
                    elem,
                    state_dim_inner * elem,
                    state_dim_inner * conv_channels * elem,
                    state_dim_inner * conv_channels * elem,
                ],
                element_stride: [
                    1,
                    state_dim_inner,
                    state_dim_inner * conv_channels,
                    state_dim_inner * conv_channels,
                ],
                dtype: GgmlType::F32,
            };
            cast::record_cast(ctx, src, dst)?;
        }
        Ok(())
    }

    /// Batched **decode** forward (M2): `B = tokens.len()` sequences, one token
    /// each, in one pass. Mirrors [`Self::forward_impl`] but per-sequence:
    /// attention layers use per-slab KV + batched flash-attn; SSM layers use
    /// per-sequence conv/GDN state at `n_seqs = B`. The MoE FFN footer is
    /// token-parallel and sequence-agnostic, so the existing `moe_ffn` is
    /// reused with `l = B`. Returns `[vocab, B]` logits (column s = sequence s).
    fn forward_batch_impl(
        &self,
        ctx: &mut DispatchContext,
        batch: &mut crate::inference::kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        let p = &self.params;
        let b = tokens.len() as u64;
        if b == 0 {
            return Err("forward_batch_impl: empty batch".into());
        }
        if positions.len() != tokens.len() || slots.len() != tokens.len() {
            return Err("forward_batch_impl: tokens/positions/slots length mismatch".into());
        }
        let hidden = p.n_embd as u64;
        let head_dim_k = p.head_dim_k as u64;
        let head_dim_v = p.head_dim_v as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv as u64;
        let n_embd_kv = p.n_embd_k_gqa() as u64;
        let n_embd_vv = p.n_embd_v_gqa() as u64;
        let wq_out = p.wq_out() as u64;
        let hidden_v = head_dim_v * n_head;
        let vocab = p.n_vocab as u64;

        // Prologue: B token ids; M-RoPE positions [4 axes × B] (each axis gets
        // the same per-sequence text position); embedding → residual [hidden, B].
        let token_buf = ctx.alloc_scratch(b * 4)?;
        write_u32(ctx, token_buf, tokens)?;
        let positions_buf = ctx.alloc_scratch(4 * b * 4)?;
        let mut pos: Vec<u32> = Vec::with_capacity(4 * b as usize);
        for _axis in 0..4 {
            pos.extend_from_slice(positions);
        }
        write_u32(ctx, positions_buf, &pos)?;

        let residual = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, b as u32, residual)?;
        // Persistent per-forward DecodeDyn array for the batched flash-attn.
        // MUST be allocated before `layer_checkpoint` so per-layer
        // `scratch_restore` never reclaims it — the shader reads its `kv_len`
        // at execute time, long after the host write here, and a reclaimed
        // offset would be overwritten by a later layer (e.g. the SSM
        // conv_input memset) → garbage length → unbounded KV loop → GPU hang.
        let fa_dyn_range = crate::inference::decode_dyn::alloc_array(ctx, b as u32)?;
        let layer_checkpoint = ctx.scratch_checkpoint();

        let rope_params = rope_multi::RopeMultiParams::qwen_default(
            p.rope_dim,
            p.rope_freq_base,
            p.rope_sections,
        );
        let scale = 1.0 / (head_dim_k as f32).sqrt();
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim_k as u32,
            head_dim_v: head_dim_v as u32,
            gqa_ratio: (p.n_head / p.n_head_kv).max(1),
            scale,
        };
        let kv_lens: Vec<u32> = positions.iter().map(|&pp| pp + 1).collect();

        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_checkpoint);
            match block {
                BlockWeights::Attention(att) => {
                    attention_block_batch(
                        ctx, att, batch, residual, positions_buf, rope_params, fa_params,
                        layer_idx as u32, positions, slots, &kv_lens, p, head_dim_k, head_dim_v,
                        n_head, n_head_kv, n_embd_kv, n_embd_vv, wq_out, hidden, hidden_v, b,
                        fa_dyn_range,
                    )?;
                }
                BlockWeights::Ssm(ssm_w) => {
                    let ssm_layer_idx = (0..layer_idx)
                        .filter(|&i| !p.is_attention_layer(i as u32))
                        .count() as u32;
                    ssm_block_batch(
                        ctx, ssm_w, batch, residual, p, hidden, b, ssm_layer_idx, positions, slots,
                    )?;
                }
            }
            moe_ffn(
                ctx,
                block.moe(),
                block.post_attn_norm(),
                residual,
                p,
                hidden,
                b as u32,
                layer_idx as u32,
            )?;
        }
        ctx.scratch_restore(layer_checkpoint);

        // Epilogue: final norm + lm_head over ALL B columns (each is its
        // sequence's last — and only — token this step).
        let final_norm = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, residual, self.weights.output_norm, final_norm, p.rms_eps)?;
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let logits = ctx.alloc_tensor([vocab, b, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, final_norm, logits)?;

        for (s, &pp) in positions.iter().enumerate() {
            batch.positions[slots[s] as usize] = pp + 1;
        }
        Ok(logits)
    }

    /// Unified varlen forward (M5 Phase 4): `B` sequences, sequence `s`
    /// contributing `seq_lens[s]` tokens packed flat into `tokens`/`positions`
    /// (`N_total = sum`). Dense ops + M-RoPE run on the flat `[hidden, N_total]`
    /// stream; attention layers use the per-slab varlen causal flash (Phase 1);
    /// SSM layers loop per-sequence over [`ssm_block`] — the GDN/conv recurrence
    /// is sequential, so each sequence runs its own `L_s`-token scan over its
    /// slab's conv/GDN state (a global barrier + scratch restore between
    /// sequences). MoE FFN is token-parallel → reused on `N_total`. Returns each
    /// sequence's last-token logits, `[vocab, B]`.
    #[allow(clippy::too_many_arguments)]
    fn forward_unified_impl(
        &self,
        ctx: &mut DispatchContext,
        batch: &mut crate::inference::kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        seq_lens: &[u32],
        slots: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        use crate::inference::command::record_global_barrier;
        let p = &self.params;
        let b = seq_lens.len();
        let n_total = tokens.len() as u64;
        if b == 0 || n_total == 0 {
            return Err("forward_unified_impl: empty batch".into());
        }
        if positions.len() != tokens.len() || slots.len() != b {
            return Err("forward_unified_impl: tokens/positions/seq_lens/slots length mismatch".into());
        }
        if seq_lens.iter().map(|&l| l as u64).sum::<u64>() != n_total {
            return Err("forward_unified_impl: sum(seq_lens) != tokens.len()".into());
        }
        let hidden = p.n_embd as u64;
        let head_dim_k = p.head_dim_k as u64;
        let head_dim_v = p.head_dim_v as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv as u64;
        let n_embd_kv = p.n_embd_k_gqa() as u64;
        let n_embd_vv = p.n_embd_v_gqa() as u64;
        let wq_out = p.wq_out() as u64;
        let hidden_v = head_dim_v * n_head;
        let vocab = p.n_vocab as u64;
        let elem = 4u64;

        let q_starts: Vec<u64> = seq_lens
            .iter()
            .scan(0u64, |a, &l| { let s = *a; *a += l as u64; Some(s) })
            .collect();
        // KV write base + attention length come from the slab's cache position
        // (the KV-slot count), NOT the `positions` arg. They are equal for text
        // (the scheduler sets positions[q_start[s]] = batch.positions[slot]), so
        // this is byte-identical there; decoupling them lets `positions` carry
        // the M-RoPE rope base (= cache_pos − rope_lag) for image sequences,
        // whose rope cursor trails their KV-slot count. See the rope buffer below.
        let kv_lens: Vec<u32> = (0..b)
            .map(|s| batch.positions[slots[s] as usize] + seq_lens[s])
            .collect();

        // Prologue: flat token ids, M-RoPE positions ([4 axes × N_total], each
        // axis the flat per-token position — text-only), embedding → residual.
        let token_buf = ctx.alloc_scratch(n_total * 4)?;
        write_u32(ctx, token_buf, tokens)?;
        let positions_buf = ctx.alloc_scratch(4 * n_total * 4)?;
        let mut pos: Vec<u32> = Vec::with_capacity(4 * n_total as usize);
        for _axis in 0..4 {
            pos.extend_from_slice(positions);
        }
        write_u32(ctx, positions_buf, &pos)?;

        let residual = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, n_total as u32, residual)?;
        let fa_dyn_range = crate::inference::decode_dyn::alloc_array(ctx, b as u32)?;
        let layer_checkpoint = ctx.scratch_checkpoint();

        let rope_params =
            rope_multi::RopeMultiParams::qwen_default(p.rope_dim, p.rope_freq_base, p.rope_sections);
        let scale = 1.0 / (head_dim_k as f32).sqrt();
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim_k as u32,
            head_dim_v: head_dim_v as u32,
            gqa_ratio: (p.n_head / p.n_head_kv).max(1),
            scale,
        };

        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_checkpoint);
            match block {
                BlockWeights::Attention(att) => {
                    attention_block_unified(
                        ctx, att, batch, residual, positions_buf, rope_params, fa_params,
                        layer_idx as u32, &kv_lens, seq_lens, &q_starts, slots, p, head_dim_k,
                        head_dim_v, n_head, n_head_kv, n_embd_kv, n_embd_vv, wq_out, hidden,
                        hidden_v, n_total, fa_dyn_range,
                    )?;
                }
                BlockWeights::Ssm(ssm_w) => {
                    let ssm_layer_idx = (0..layer_idx)
                        .filter(|&i| !p.is_attention_layer(i as u32))
                        .count() as u32;
                    // Per-sequence GDN/conv recurrence (sequential): each runs
                    // its L_s-token scan over its slab's state, writing its own
                    // disjoint residual columns. Each ssm_block uses FRESH
                    // scratch (no mid-layer restore — matching single-seq
                    // forward_impl) so its GPU work is never raced by a later
                    // op reusing the same scratch offsets. The layer-top
                    // scratch_restore reclaims it all next layer. Total scratch
                    // ≈ a single-seq forward over N_total tokens (within the
                    // n_ubatch reservation). Disjoint residual slices + fresh
                    // scratch ⇒ no inter-sequence barrier needed.
                    for s in 0..b {
                        let l = seq_lens[s];
                        let off = q_starts[s] * hidden * elem;
                        let residual_slice = TensorView {
                            byte_offset: residual.byte_offset + off,
                            byte_size: l as u64 * hidden * elem,
                            dims: [hidden, l as u64, 1, 1],
                            byte_stride: [elem, hidden * elem, hidden * elem * l as u64, hidden * elem * l as u64],
                            element_stride: [1, hidden, hidden * l as u64, hidden * l as u64],
                            ..residual
                        };
                        let gdn_state = Some(batch.gdn_state_slot(ssm_layer_idx, slots[s]));
                        let conv_state = Some(batch.conv_state_slot(ssm_layer_idx, slots[s]));
                        ssm_block(
                            ctx, ssm_w, residual_slice, p, hidden, l, layer_idx as u32,
                            gdn_state, conv_state, None, None,
                        )?;
                    }
                }
            }
            // Release-capable dump: snapshot the residual after the block (attn
            // /ssm) and after MoE. No-op unless ctx.dump is set (debug harness).
            ctx.dump(&format!("L{layer_idx:02}-block"), residual);
            moe_ffn(ctx, block.moe(), block.post_attn_norm(), residual, p, hidden, n_total as u32, layer_idx as u32)?;
            ctx.dump(&format!("L{layer_idx:02}-moe"), residual);
        }
        ctx.scratch_restore(layer_checkpoint);

        // Epilogue: gather each sequence's last-token column → norm + lm_head.
        let last_hidden = ctx.alloc_tensor([hidden, b as u64, 1, 1], GgmlType::F32)?;
        record_global_barrier(ctx.device, ctx.cmd);
        unsafe {
            use ash::vk;
            for s in 0..b {
                let src_col = q_starts[s] + seq_lens[s] as u64 - 1;
                let copy = vk::BufferCopy::default()
                    .src_offset(residual.byte_offset + src_col * hidden * elem)
                    .dst_offset(last_hidden.byte_offset + s as u64 * hidden * elem)
                    .size(hidden * elem);
                ctx.device.device.cmd_copy_buffer(
                    ctx.cmd, residual.buffer, last_hidden.buffer, std::slice::from_ref(&copy),
                );
            }
        }
        record_global_barrier(ctx.device, ctx.cmd);
        let final_norm = ctx.alloc_tensor([hidden, b as u64, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, last_hidden, self.weights.output_norm, final_norm, p.rms_eps)?;
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let logits = ctx.alloc_tensor([vocab, b as u64, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, final_norm, logits)?;

        for s in 0..b {
            batch.positions[slots[s] as usize] = kv_lens[s];
        }
        Ok(logits)
    }
}

// ───────────────────────────────────────────────────────────────────────
// Forward helpers
// ───────────────────────────────────────────────────────────────────────

/// Attention block forward, mirroring `qwen35moe.cpp:283-362`:
///   1. x_norm  = rms_norm(residual, attn_norm)
///   2. q_full  = wq @ x_norm  (width = 2 * head_dim_k * n_head — doubled)
///   3. q_attn  = first-half-per-head view of q_full
///      q_gate  = second-half-per-head view of q_full
///   4. k       = wk @ x_norm  → [head_dim_k, n_head_kv, L]
///      v       = wv @ x_norm  → [head_dim_v, n_head_kv, L]
///   5. q_attn ← rms_norm(q_attn, attn_q_norm)   (per-head)
///      k       ← rms_norm(k,      attn_k_norm)   (per-head)
///   6. q_roped, k_roped ← rope_multi(...)
///   7. cache write/read, flash_attn  →  attn_out [head_dim_v, n_head, L]
///   8. attn_out *= sigmoid(q_gate)               (gated attention output)
///   9. residual += wo @ attn_out
fn attention_block(
    ctx: &mut DispatchContext,
    att: &AttentionBlockWeights,
    cache: &mut KvCache,
    residual: TensorView,
    mask: Option<TensorView>,
    positions_buf: crate::inference::buffer::BufferRange,
    rope_params: rope_multi::RopeMultiParams,
    fa_params: flash_attn::FlashAttnParams,
    layer_idx: u32,
    position_offset: u32,
    total_len: u32,
    kv_len_u: u64,
    l: u32,
    p: &Qwen35MoeParams,
    head_dim_k: u64,
    head_dim_v: u64,
    n_head: u64,
    n_head_kv: u64,
    n_embd_kv: u64,
    n_embd_vv: u64,
    wq_out: u64,
    hidden: u64,
    hidden_v: u64,
    cache_direct: bool,
) -> Result<(), Box<dyn Error>> {
    // 1. attn_norm
    let x_norm = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
    rms_norm::record(ctx, residual, att.attn_norm, x_norm, p.rms_eps)?;
    ctx.tap(&format!("attn_norm-{layer_idx}"), x_norm)?;

    // 2. wq @ x_norm  →  q_full [wq_out, L]  (wq_out = 2 * head_dim_k * n_head)
    let q_full = ctx.alloc_tensor([wq_out, l as u64, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wq, x_norm, q_full)?;
    // 4. wk, wv
    let k = ctx.alloc_tensor([n_embd_kv, l as u64, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wk, x_norm, k)?;
    let v = ctx.alloc_tensor([n_embd_vv, l as u64, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wv, x_norm, v)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_full.range(), k.range(), v.range()],
    );
    ctx.tap(&format!("Qcur_full-{layer_idx}"), q_full)?;

    // 3. q_attn and q_gate are non-contiguous views into q_full:
    //    within each of the n_head heads, the first head_dim_k elements
    //    are the actual Q, the next head_dim_k are the sigmoid gate.
    let q_attn_view = slice_q_half(q_full, head_dim_k, n_head, l as u64, /*gate=*/ false);
    let q_gate_view = slice_q_half(q_full, head_dim_k, n_head, l as u64, /*gate=*/ true);

    let k_view = reshape_for_rope(k, head_dim_k, n_head_kv, l as u64);
    let v_view = reshape_for_rope(v, head_dim_v, n_head_kv, l as u64);

    // 5+6+7a. Per-head Q/K RMS norm + M-RoPE fused. For Q the result
    //         lands in an F32 scratch slot (flash_attn reads it
    //         directly). For K, when the cache is F16, the fused
    //         kernel writes the rotated K directly into the cache
    //         buffer at the right position offset — eliminating the
    //         `k_roped` scratch, the cast F32→F16 dispatch, the
    //         transfer copy, and one global barrier per attention
    //         layer.
    let q_roped = ctx.alloc_tensor([head_dim_k, n_head, l as u64, 1], GgmlType::F32)?;
    rope_multi::record_rms_norm_rope_nofence(
        ctx, q_attn_view, positions_buf, att.attn_q_norm, q_roped, rope_params, p.rms_eps,
    )?;
    let k_cache_layer = cache.k_layers[layer_idx as usize];
    let cache_max_seq_len = cache.config.max_seq_len;
    let k_cache_fused = k_cache_layer.dtype == GgmlType::F16;
    if k_cache_fused {
        rope_multi::record_rms_norm_rope_to_cache_f16_nofence(
            ctx,
            k_view,
            positions_buf,
            att.attn_k_norm,
            k_cache_layer,
            rope_params,
            p.rms_eps,
            position_offset,
            cache_max_seq_len,
        )?;
    } else {
        // Fallback for non-F16 caches: separate rope into a scratch
        // buffer, then go through the regular cache_io::record_write
        // (cast + copy) below.
        let k_roped_fb = ctx.alloc_tensor([head_dim_k, n_head_kv, l as u64, 1], GgmlType::F32)?;
        rope_multi::record_rms_norm_rope_nofence(
            ctx, k_view, positions_buf, att.attn_k_norm, k_roped_fb, rope_params, p.rms_eps,
        )?;
        crate::inference::command::record_compute_barrier(ctx.device, ctx.cmd, k_roped_fb.range());
        cache_io::record_write_nofence(ctx, k_roped_fb, k_cache_layer, position_offset)?;
    }
    // Diff-dump diagnostics — the `Kcur*` taps used to point at the
    // F32 `k_roped` scratch. With cache-fused K that intermediate
    // doesn't exist; the rotated values live as F16 inside the cache
    // buffer. Tap only the Q side; downstream `attn_pregate` / etc.
    // still expose end-to-end values.
    ctx.tap(&format!("Qcur_normed-{layer_idx}"), q_roped)?;
    ctx.tap(&format!("Qcur-{layer_idx}"), q_roped)?;

    // 7a (V only — K already in cache via the fused kernel above).
    // F16 cache fast path: dyn-offset write into the cache via
    // `v_cache_write_f16` so the descriptor binding doesn't bake the
    // position offset (required for the persistent-decode-cmdbuf
    // replay path).
    let v_cache_layer = cache.v_layers[layer_idx as usize];
    if v_cache_layer.dtype == GgmlType::F16 {
        cache_io::record_v_cache_write_f16_nofence(ctx, v_view, v_cache_layer, position_offset)?;
        // ONE barrier covering (q_roped, K cache slot, V cache slot)
        // before flash_attn reads.
        crate::inference::command::record_compute_barrier(
            ctx.device,
            ctx.cmd,
            q_roped.range(),
        );
    } else {
        // Non-F16 cache: the old cast+copy chain still emits its own
        // trailing global barrier which covers Q rope and K rope.
        // (Replay path is not supported on this branch — V write
        // bakes position into cmd_copy_buffer.)
        cache_io::record_write(ctx, v_view, v_cache_layer, position_offset)?;
    }

    // 7b. cache read — bind the *full* cache layers (no
    // `slice_cache_prefix`). Strides are `total_len`-independent at
    // `iq3 = 0` (`nb11 = head_dim * n_head_kv`, `nb12 = head_dim`); the
    // shader bounds reads by `DecodeDyn::kv_len` so OOB into the
    // unwritten max-seq-len tail never happens. For non-F16 caches
    // (materialize path) we still slice to `total_len` — those callers
    // don't use replay.
    let (k_src, v_src) = if cache_direct {
        (
            cache.k_layers[layer_idx as usize],
            cache.v_layers[layer_idx as usize],
        )
    } else {
        (
            cache_io::record_read(ctx, cache.k_layers[layer_idx as usize], total_len)?,
            cache_io::record_read(ctx, cache.v_layers[layer_idx as usize], total_len)?,
        )
    };

    // 7c. flash_attn — permute Q to [hd_k, L, n_head], K/V to
    //     [hd_kv, kv_len_for_strides, n_head_kv]. For the direct-cache
    //     fast path the permute uses max_seq_len so the strides match
    //     the actual cache layout; the shader bounds its iteration by
    //     `DecodeDyn::kv_len` via the `kv_actual` arg here, which feeds
    //     `pick_k_num` and the dyn write.
    let kv_for_perm = if cache_direct {
        cache.k_layers[layer_idx as usize].dims[2]
    } else {
        kv_len_u
    };
    let q_perm = permute_to_attn(q_roped, head_dim_k, l as u64, n_head);
    let k_perm = permute_to_attn(k_src, head_dim_k, kv_for_perm, n_head_kv);
    let v_perm = permute_to_attn(v_src, head_dim_v, kv_for_perm, n_head_kv);
    let attn_out = ctx.alloc_tensor([hidden_v, l as u64, 1, 1], GgmlType::F32)?;
    flash_attn::record(ctx, q_perm, k_perm, v_perm, mask, attn_out, fa_params, total_len)?;
    ctx.tap(&format!("attn_pregate-{layer_idx}"), attn_out)?;

    // 8. Sigmoid-gate the attention output by q_gate, fused as one
    //    `sigmoid(q_gate) * attn_out` dispatch via `sigmoid_mul.slang`.
    //    The kernel reads a flat buffer so we still materialize the
    //    strided q_gate view into a contiguous slot via F32→F32 cast.
    let attn_gated = {
        let q_gate_contig = ctx.alloc_tensor([head_dim_k, n_head, l as u64, 1], GgmlType::F32)?;
        cast::record_cast(ctx, q_gate_view, q_gate_contig)?;
        let q_gate_flat = TensorView {
            dims: [hidden_v, l as u64, 1, 1],
            byte_size: q_gate_contig.byte_size,
            byte_stride: [4, 4 * hidden_v, 4 * hidden_v * (l as u64), 4 * hidden_v * (l as u64)],
            element_stride: [1, hidden_v, hidden_v * (l as u64), hidden_v * (l as u64)],
            ..q_gate_contig
        };
        let attn_gated = ctx.alloc_tensor([hidden_v, l as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_sigmoid_mul_split(ctx, q_gate_flat, attn_out, attn_gated)?;
        attn_gated
    };
    ctx.tap(&format!("attn_gated-{layer_idx}"), attn_gated)?;

    // 9. residual += wo @ attn_gated  — fused accumulate. The matvec
    //    kernel adds its output into the existing residual buffer in
    //    place via the ACCUMULATE spec constant on
    //    `mul_mat_vec_head.slang`. Eliminates the `proj` scratch slot,
    //    one dispatch (record_add), and one barrier per attention layer.
    //    Decode-only (matvec path); prefill (L>1) falls back to the
    //    unfused chain because mul_mm doesn't yet implement accumulate.
    if l == 1 {
        matmul::record_accumulate(ctx, att.wo, attn_gated, residual)?;
    } else {
        let proj = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, att.wo, attn_gated, proj)?;
        elementwise::record_add(ctx, residual, proj, residual)?;
    }
    ctx.tap(&format!("attn_output-{layer_idx}"), residual)?;
    ctx.tap(&format!("attn_residual-{layer_idx}"), residual)?;

    Ok(())
}

/// SSM / gated-delta-net block forward.
///
/// Mirrors the sequence from llama.cpp's `qwen35moe.cpp::build_layer_attn_linear`:
///
///   1. x_norm = rms_norm(residual, attn_norm)
///   2. qkv   = wqkv      @ x_norm        // [key_dim*2 + value_dim, L]
///      z    = attn_gate  @ x_norm        // [value_dim, L]
///   3. beta_pre  = ssm_beta  @ x_norm    // [num_v_heads, L]
///      alpha_pre = ssm_alpha @ x_norm    // [num_v_heads, L]
///   4. beta = sigmoid(beta_pre)
///      alpha = softplus(alpha_pre + ssm_dt_bias) * ssm_a
///   5. qkv_conv = ssm_conv1d(qkv_padded_with_prefix)
///   6. slice q, k, v out of qkv_conv; L2-norm q and k per head
///   7. gated_delta_net(q, k, v, gate=alpha, beta, state_in=0) → attn_out [value_dim, L]
///   8. attn_normed = rms_norm(attn_out_per_head, ssm_norm) * silu(z)
///   9. proj = ssm_out @ attn_normed
///  10. residual += proj
///
/// PHASE 3 SIMPLIFICATIONS:
///   - Conv1d prefix is fresh zeros every forward (no cross-call state).
///     Correct for the first forward, drops conv context on subsequent
///     decode steps. Persistent state is Phase 4.
///   - GDN state buffer is fresh zeros every forward (same caveat).
fn ssm_block(
    ctx: &mut DispatchContext,
    ssm_w: &SsmBlockWeights,
    residual: TensorView,
    p: &Qwen35MoeParams,
    hidden: u64,
    l: u32,
    layer_idx: u32,
    gdn_state_persistent: Option<crate::inference::buffer::BufferRange>,
    conv_state_persistent: Option<crate::inference::buffer::BufferRange>,
    ssm_region_host_ptr: Option<*mut u8>,
    // Spec-decode checkpoint mode: `(gdn_snapshot, conv_backup)` per-layer
    // buffers. When Some, the GDN scan emits L per-token state snapshots
    // into `gdn_snapshot` and the full conv input window is copied into
    // `conv_backup`, and the live recurrent-state writebacks are SKIPPED
    // (finalize commits the accepted position later).
    checkpoint: Option<(
        crate::inference::buffer::BufferRange,
        crate::inference::buffer::BufferRange,
    )>,
) -> Result<(), Box<dyn Error>> {
    let l_u = l as u64;
    let num_k = p.ssm_groups as u64;          // 16
    let num_v = p.ssm_dt_rank as u64;         // 32
    let s_v = p.ssm_state as u64;             // 128 = head_v_dim = head_k_dim
    let conv_kernel = p.ssm_conv as u64;      // 4
    let key_dim = num_k * s_v;                // 2048
    let value_dim = num_v * s_v;              // 4096 = ssm_inner
    let conv_channels = 2 * key_dim + value_dim; // 8192

    // 1. attn_norm
    let x_norm = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    rms_norm::record(ctx, residual, ssm_w.attn_norm, x_norm, p.rms_eps)?;
    ctx.tap(&format!("attn_norm-{layer_idx}"), x_norm)?;
    if crate::runtime_flags::qwen_only_rms() {
        return Ok(());
    }

    // 2. wqkv and attn_gate (z)
    let qkv = ctx.alloc_tensor([conv_channels, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, ssm_w.attn_qkv, x_norm, qkv)?;
    let z = ctx.alloc_tensor([value_dim, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, ssm_w.attn_gate, x_norm, z)?;
    // 3. beta_pre and alpha_pre
    let beta_pre = ctx.alloc_tensor([num_v, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, ssm_w.ssm_beta, x_norm, beta_pre)?;
    let alpha_pre = ctx.alloc_tensor([num_v, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, ssm_w.ssm_alpha, x_norm, alpha_pre)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[qkv.range(), z.range(), beta_pre.range(), alpha_pre.range()],
    );
    ctx.tap(&format!("qkv_mixed-{layer_idx}"), qkv)?;
    ctx.tap(&format!("z-{layer_idx}"), z)?;
    ctx.tap(&format!("beta_pre-{layer_idx}"), beta_pre)?;
    ctx.tap(&format!("alpha_pre-{layer_idx}"), alpha_pre)?;

    // 4a. beta = sigmoid(beta_pre)
    // 4a/4b: sigmoid(beta_pre) and the fused alpha pipeline are
    // independent — same inputs (matmul outputs), disjoint outputs
    // (beta vs alpha). Dispatch both nofence so the GPU is free to
    // overlap them with each other (and with the conv1d below, which
    // is also independent until gated_delta_net reads its output);
    // emit one coalesced barrier covering (beta, alpha, conv_out) just
    // before the L2 norms further down.
    let beta = ctx.alloc_tensor([num_v, l_u, 1, 1], GgmlType::F32)?;
    elementwise::record_sigmoid_nofence(ctx, beta_pre, beta)?;
    ctx.tap(&format!("beta_sigmoid-{layer_idx}"), beta)?;

    let alpha = ctx.alloc_tensor([num_v, l_u, 1, 1], GgmlType::F32)?;
    elementwise::record_ssm_alpha_fuse_nofence(
        ctx,
        alpha_pre,
        ssm_w.ssm_dt_bias,
        ssm_w.ssm_a,
        alpha,
        num_v as u32,
    )?;
    ctx.tap(&format!("a_softplus-{layer_idx}"), alpha)?;
    ctx.tap(&format!("gate-{layer_idx}"), alpha)?;

    // 5. ssm_conv1d on qkv.
    // The shader expects `src0[i3*nb02 + i1*nb01 + i2]` with i2 (token)
    // innermost and i1 (channel) outer. The matmul output `qkv` is
    // ne0=channels, ne1=tokens — channels innermost in memory. We need a
    // transposed view of qkv that pretends "channels are outer, tokens
    // are inner", AND a fresh contiguous scratch buffer with a zero
    // prefix to feed as conv input.
    let n_padded = (conv_kernel - 1) + l_u;
    let conv_input = ctx.alloc_tensor([n_padded, conv_channels, 1, 1], GgmlType::F32)?;
    // Initialize the (conv_kernel-1)-token prefix of conv_input. For the
    // first forward (persistent state still zero) this is a zero-fill via
    // host; for subsequent forwards we cast the persistent state into the
    // prefix with stride conversion (persistent is contiguous [(kernel-1) ×
    // conv_channels] but conv_input has stride n_padded between channels).
    // The L "new" tokens (tail) get overwritten by the qkv cast below.
    {
        let host_ptr = ctx
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible")?;
        unsafe {
            std::ptr::write_bytes(
                host_ptr.add(conv_input.byte_offset as usize) as *mut u8,
                0,
                conv_input.byte_size as usize,
            );
        }
    }
    // Copy persistent conv state into the prefix via a strided cast.
    // (Note: a host-side memcpy version was attempted but produced
    // incorrect output — the GPU's preceding cast-write to persistent
    // wasn't reliably host-visible by the time the next forward's host
    // read fired, despite HOST_COHERENT. GPU-side cast is the safe path.)
    let _ = ssm_region_host_ptr;
    if let Some(persistent) = conv_state_persistent {
        let elem_local = 4u64;
        let state_dim_inner = conv_kernel - 1; // 3
        let persistent_src = TensorView {
            buffer: persistent.buffer,
            byte_offset: persistent.offset,
            byte_size: persistent.size,
            dims: [state_dim_inner, conv_channels, 1, 1],
            byte_stride: [
                elem_local,
                state_dim_inner * elem_local,
                state_dim_inner * conv_channels * elem_local,
                state_dim_inner * conv_channels * elem_local,
            ],
            element_stride: [
                1,
                state_dim_inner,
                state_dim_inner * conv_channels,
                state_dim_inner * conv_channels,
            ],
            dtype: GgmlType::F32,
        };
        let conv_input_prefix = TensorView {
            buffer: conv_input.buffer,
            byte_offset: conv_input.byte_offset,
            byte_size: conv_input.byte_size,
            dims: [state_dim_inner, conv_channels, 1, 1],
            byte_stride: [
                elem_local,
                n_padded * elem_local,
                n_padded * conv_channels * elem_local,
                n_padded * conv_channels * elem_local,
            ],
            element_stride: [
                1,
                n_padded,
                n_padded * conv_channels,
                n_padded * conv_channels,
            ],
            dtype: GgmlType::F32,
        };
        cast::record_cast(ctx, persistent_src, conv_input_prefix)?;
    }
    // Transpose-cast `qkv` (channels-inner, tokens-outer) into the *tail*
    // of `conv_input` (tokens-inner, channels-outer). The src view has
    // matching dims `[L, conv_channels]` but element strides that index
    // the original chan-inner memory; dst points to conv_input with
    // offset (conv_kernel - 1) elements into the token dim.
    let elem = 4u64;
    let qkv_as_token_inner = TensorView {
        buffer: qkv.buffer,
        byte_offset: qkv.byte_offset,
        byte_size: qkv.byte_size,
        dims: [l_u, conv_channels, 1, 1],
        byte_stride: [
            conv_channels * elem,
            elem,
            conv_channels * l_u * elem,
            conv_channels * l_u * elem,
        ],
        element_stride: [conv_channels, 1, conv_channels * l_u, conv_channels * l_u],
        dtype: qkv.dtype,
    };
    let conv_input_tail = TensorView {
        buffer: conv_input.buffer,
        byte_offset: conv_input.byte_offset + (conv_kernel - 1) * elem,
        byte_size: conv_input.byte_size - (conv_kernel - 1) * elem,
        dims: [l_u, conv_channels, 1, 1],
        byte_stride: [
            elem,
            n_padded * elem,
            l_u * conv_channels * elem,
            l_u * conv_channels * elem,
        ],
        element_stride: [1, n_padded, l_u * conv_channels, l_u * conv_channels],
        dtype: conv_input.dtype,
    };
    cast::record_cast(ctx, qkv_as_token_inner, conv_input_tail)?;

    // Checkpoint mode: back up the entire conv input window (contiguous)
    // so finalize can extract the conv state at the accepted position.
    // Skips the live conv-state writeback below.
    if let Some((_, conv_backup)) = checkpoint {
        unsafe {
            use ash::vk;
            let bytes = conv_input.byte_size;
            let copy = vk::BufferCopy::default()
                .src_offset(conv_input.byte_offset)
                .dst_offset(conv_backup.offset)
                .size(bytes);
            ctx.device.device.cmd_copy_buffer(
                ctx.cmd,
                conv_input.buffer,
                conv_backup.buffer,
                std::slice::from_ref(&copy),
            );
            let bar = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(conv_backup.buffer)
                .offset(conv_backup.offset)
                .size(bytes);
            ctx.device.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&bar),
                &[],
            );
        }
    } else if let Some(persistent) = conv_state_persistent {
        // Save the last (conv_kernel-1) tokens of conv_input as the new
        // persistent conv state, for the next forward to read. Mirrors
        // llama.cpp's `conv_state_last` view at offset `s_idx = L`.
        let state_dim_inner = conv_kernel - 1;
        let s_idx = l_u; // = n_padded - (kernel - 1)
        let conv_input_last = TensorView {
            buffer: conv_input.buffer,
            byte_offset: conv_input.byte_offset + s_idx * elem,
            byte_size: conv_input.byte_size - s_idx * elem,
            dims: [state_dim_inner, conv_channels, 1, 1],
            byte_stride: [
                elem,
                n_padded * elem,
                n_padded * conv_channels * elem,
                n_padded * conv_channels * elem,
            ],
            element_stride: [
                1,
                n_padded,
                n_padded * conv_channels,
                n_padded * conv_channels,
            ],
            dtype: GgmlType::F32,
        };
        let persistent_dst = TensorView {
            buffer: persistent.buffer,
            byte_offset: persistent.offset,
            byte_size: persistent.size,
            dims: [state_dim_inner, conv_channels, 1, 1],
            byte_stride: [
                elem,
                state_dim_inner * elem,
                state_dim_inner * conv_channels * elem,
                state_dim_inner * conv_channels * elem,
            ],
            element_stride: [
                1,
                state_dim_inner,
                state_dim_inner * conv_channels,
                state_dim_inner * conv_channels,
            ],
            dtype: GgmlType::F32,
        };
        cast::record_cast(ctx, conv_input_last, persistent_dst)?;
    }

    // Conv output: `ssm_conv.slang` writes `dst[batch, token, channel]`
    // with channel-innermost (`dst_nb0 = elem`, `dst_nb1 = channels*elem`),
    // so allocate as `[conv_channels, L]` in ggml convention — ne0 (=
    // channels) is innermost, ne1 (= tokens) is the outer axis with
    // stride channels.
    let conv_out = ctx.alloc_tensor([conv_channels, l_u, 1, 1], GgmlType::F32)?;
    // SEEKER_QWEN_NO_CONV=1 bypasses ssm_conv1d (uses raw qkv directly as
    // conv_out, treating the conv as identity). Helps isolate stride
    // bugs in the conv path from bugs in the downstream GDN math.
    if crate::runtime_flags::qwen_no_conv() {
        // qkv has the same memory layout as conv_out should (channel-inner,
        // token-outer) — `matmul::record` produces D[m, n] at offset
        // n*M + m, i.e. m (channel) innermost. Just memcpy.
        cast::record_cast(ctx, qkv, conv_out)?;
    } else {
        // Nofence — coalesced barrier below covers conv_out alongside
        // beta and alpha.
        ssm::record_ssm_conv_nofence(
            ctx,
            conv_input,
            ssm_w.ssm_conv1d,
            conv_out,
            conv_channels as u32,
            n_padded as u32,
            l as u32,
            1,
            conv_kernel as u32,
            /* fuse_silu = */ true,
        )?;
    }
    // Coalesced barrier on (beta, alpha, conv_out): the three branches
    // are independent on the GPU and the driver is now free to overlap
    // them. None of the subsequent dispatches read any of these
    // buffers before this barrier point.
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[beta.range(), alpha.range(), conv_out.range()],
    );

    // 5b. SiLU is fused into the conv1d kernel above (FUSE_SILU=1), so
    // `conv_out` already holds `silu(raw_conv_output)`. Diagnostic taps
    // expect the post-silu values too, which is fine — both names point
    // at the same buffer when fused.
    ctx.tap(&format!("conv_output_raw-{layer_idx}"), conv_out)?;
    ctx.tap(&format!("conv_output_silu-{layer_idx}"), conv_out)?;

    // 6. Slice Q, K, V out of conv_out. Layout is `[L, conv_channels]`
    //    (token innermost). The channel axis spans `[key_dim, key_dim,
    //    value_dim]` contiguous. For each slice we need a view with the
    //    sub-channel range; the byte_offset shifts to the start of that
    //    channel range and the channel dim shrinks.
    //
    //    Q (key_dim wide), K (key_dim wide), V (value_dim wide). We want
    //    them shaped as `[s_v=128, num_*_heads, L, 1]` for downstream
    //    L2 norm + gated_delta_net.
    //
    //    conv_out memory layout is `[token][channel]` with channel
    //    innermost (one row = one token's full channel vector). To slice
    //    out Q (channels 0..key_dim), K (key_dim..2*key_dim), V
    //    (2*key_dim..conv_channels), we want a view that picks
    //    `chan_count` consecutive channels at byte offset
    //    `chan_offset * elem` within each token, with the token stride
    //    unchanged at `conv_channels * elem`. The slice is
    //    non-contiguous along the token axis (gaps for the un-picked
    //    channels of K and V) but contiguous within a single token.
    let slice_qkv = |chan_offset: u64, chan_count: u64| -> TensorView {
        TensorView {
            buffer: conv_out.buffer,
            byte_offset: conv_out.byte_offset + chan_offset * elem,
            // byte_size needs to span the LAST channel of the LAST token —
            // = (l_u - 1) * token_stride + chan_count * elem. Use a
            // conservative upper bound that covers the natural slice.
            byte_size: conv_out.byte_size - chan_offset * elem,
            dims: [chan_count, l_u, 1, 1],
            byte_stride: [
                elem,
                conv_channels * elem, // skip past the OTHER channels of this token
                conv_channels * l_u * elem,
                conv_channels * l_u * elem,
            ],
            element_stride: [
                1,
                conv_channels,
                conv_channels * l_u,
                conv_channels * l_u,
            ],
            dtype: conv_out.dtype,
        }
    };
    let q_slice = slice_qkv(0, key_dim);
    let k_slice = slice_qkv(key_dim, key_dim);
    let v_slice = slice_qkv(2 * key_dim, value_dim);

    // Reshape Q/K/V into `[s_v, num_heads, L, 1]` views — same memory as
    // the slice, just splitting the channel axis into (head, s_v_idx)
    // with s_v_idx innermost. Q and K have `num_k` heads, V has `num_v`.
    let head_view = |slice: TensorView, num_heads: u64| -> TensorView {
        TensorView {
            buffer: slice.buffer,
            byte_offset: slice.byte_offset,
            byte_size: slice.byte_size,
            dims: [s_v, num_heads, l_u, 1],
            byte_stride: [
                elem,
                s_v * elem,
                conv_channels * elem, // token stride (across all channels)
                conv_channels * l_u * elem,
            ],
            element_stride: [1, s_v, conv_channels, conv_channels * l_u],
            dtype: slice.dtype,
        }
    };
    let q_view = head_view(q_slice, num_k);
    let k_view = head_view(k_slice, num_k);
    let v_view = head_view(v_slice, num_v);
    ctx.tap(&format!("v_conv_predelta-{layer_idx}"), v_view)?;

    // L2-normalize Q and K per (head, token). The two are independent —
    // nofence both, then emit one coalesced barrier so they're free to
    // overlap on the GPU.
    let ssm_norm_eps = 1e-6;
    let q_normed = ctx.alloc_tensor([s_v, num_k, l_u, 1], GgmlType::F32)?;
    elementwise::record_l2_norm_nofence(ctx, q_view, q_normed, ssm_norm_eps)?;
    let k_normed = ctx.alloc_tensor([s_v, num_k, l_u, 1], GgmlType::F32)?;
    elementwise::record_l2_norm_nofence(ctx, k_view, k_normed, ssm_norm_eps)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_normed.range(), k_normed.range()],
    );
    ctx.tap(&format!("q_conv_predelta-{layer_idx}"), q_normed)?;
    ctx.tap(&format!("k_conv_predelta-{layer_idx}"), k_normed)?;

    // gated_delta_net dispatch. Output layout (per shader):
    //   data_dst[seq * n_tokens * H * S_V + t * H * S_V + head * S_V + col]
    // i.e. `[S_V, H, n_tokens, n_seqs]` with S_V innermost. Plus the
    // updated state at byte offset `s_off * 4`. We allocate one scratch
    // buffer big enough for both regions: attn_out is L*num_v*s_v
    // floats, state is num_v * s_v * s_v floats.
    let attn_floats = l_u * num_v * s_v; // per-token outputs
    let state_floats = num_v * s_v * s_v; // single-snapshot state
    // Checkpoint mode emits L per-token state snapshots (K = L) instead of 1.
    let k_snapshots: u64 = if checkpoint.is_some() { l_u } else { 1 };
    let gdn_total_floats = attn_floats + k_snapshots * state_floats;
    let gdn_dst = ctx.alloc_scratch(gdn_total_floats * elem)?;
    // state_in: either persistent (carried over from previous forward via
    // KvCache.ssm_gdn_states) or zero-initialized fallback if the engine
    // didn't allocate persistent state.
    let gdn_state_in = if let Some(persistent) = gdn_state_persistent {
        persistent
    } else {
        let r = ctx.alloc_scratch(state_floats * elem)?;
        let host_ptr = ctx
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible")?;
        unsafe {
            std::ptr::write_bytes(
                host_ptr.add(r.offset as usize) as *mut u8,
                0,
                r.size as usize,
            );
        }
        r
    };
    // Beta and alpha for the shader are `[num_v_heads, n_tokens, n_seqs]`
    // with num_v_heads innermost (we allocated them as `[num_v, L]`).
    // Strides match: head=1, token=num_v, seq=num_v*L.
    // Gated delta-net output scale. llama.cpp uses 1/sqrt(S_v) — see
    // ggml-vulkan.cpp:10837 (`const float scale = 1.0f / sqrtf((float)S_v)`).
    // Without this scale, GDN output is sqrt(S_v) too large, and the
    // downstream ssm_norm + silu(z) gating amplifies the discrepancy enough
    // that residual diverges from llama by orders of magnitude after the
    // first few SSM layers. `SEEKER_QWEN_GDN_SCALE=one` (gpu_debug only)
    // bypasses for testing; `qwen_gdn_scale_one()` constant-folds to
    // `false` in production builds so this resolves to the real scale.
    let gdn_scale = if crate::runtime_flags::qwen_gdn_scale_one() {
        1.0
    } else {
        1.0 / (s_v as f32).sqrt()
    };
    let q_strides = ssm::GdnStrides {
        s1: s_v as u32,          // head stride within Q (= s_v)
        s2: conv_channels as u32, // token stride
        s3: (conv_channels * l_u) as u32,
    };
    let v_strides = ssm::GdnStrides {
        s1: s_v as u32,
        s2: conv_channels as u32,
        s3: (conv_channels * l_u) as u32,
    };
    let b_strides = ssm::GdnStrides {
        s1: 1,
        s2: num_v as u32,
        s3: (num_v * l_u) as u32,
    };

    // For Q and K we want to feed in the L2-normalized versions
    // (`q_normed`, `k_normed`), which are contiguous in `[s_v, num_k,
    // L]`. The shader strides into them via sq1/sq2/sq3 (assumed same
    // layout for Q and K).
    let q_normed_strides = ssm::GdnStrides {
        s1: s_v as u32,           // head stride
        s2: (s_v * num_k) as u32, // token stride
        s3: (s_v * num_k * l_u) as u32,
    };

    ssm::record_gated_delta_net(
        ctx,
        q_normed,
        k_normed,
        v_view,
        alpha,  // gate (g)
        beta,
        gdn_state_in,
        gdn_dst,
        num_v as u32,
        num_k as u32,
        l as u32,
        1, // n_seqs
        attn_floats as u32, // s_off in F32 elements
        gdn_scale,
        q_normed_strides,
        v_strides,
        b_strides,
        s_v as u32,
        k_snapshots as u32,
    )?;
    let _ = (q_strides,); // q_normed_strides supersedes q_strides above

    // Checkpoint mode: copy ALL K=L per-token state snapshots (contiguous in
    // gdn_dst's state region) into the per-layer snapshot buffer for later
    // finalize. Skips the live-state writeback below.
    if let Some((gdn_snapshot, _)) = checkpoint {
        unsafe {
            use ash::vk;
            let copy = vk::BufferCopy::default()
                .src_offset(gdn_dst.offset + attn_floats * elem)
                .dst_offset(gdn_snapshot.offset)
                .size(k_snapshots * state_floats * elem);
            ctx.device.device.cmd_copy_buffer(
                ctx.cmd,
                gdn_dst.buffer,
                gdn_snapshot.buffer,
                std::slice::from_ref(&copy),
            );
            let bar = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(gdn_snapshot.buffer)
                .offset(gdn_snapshot.offset)
                .size(k_snapshots * state_floats * elem);
            ctx.device.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&bar),
                &[],
            );
        }
    } else if let Some(persistent) = gdn_state_persistent {
        // Normal decode: copy the single final GDN state back to the
        // persistent state buffer. gdn_dst: [attn outputs][state].
        unsafe {
            use ash::vk;
            let copy = vk::BufferCopy::default()
                .src_offset(gdn_dst.offset + attn_floats * elem)
                .dst_offset(persistent.offset)
                .size(state_floats * elem);
            ctx.device.device.cmd_copy_buffer(
                ctx.cmd,
                gdn_dst.buffer,
                persistent.buffer,
                std::slice::from_ref(&copy),
            );
            // Barrier so the next forward's read of persistent sees this write.
            let bar = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(persistent.buffer)
                .offset(persistent.offset)
                .size(state_floats * elem);
            ctx.device.device.cmd_pipeline_barrier(
                ctx.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&bar),
                &[],
            );
        }
    }

    // GDN output (attention portion) lives at the start of gdn_dst.
    // Wrap it back as a TensorView shaped `[s_v, num_v, L, 1]`.
    let gdn_attn = TensorView {
        buffer: gdn_dst.buffer,
        byte_offset: gdn_dst.offset,
        byte_size: attn_floats * elem,
        dims: [s_v, num_v, l_u, 1],
        byte_stride: [
            elem,
            s_v * elem,
            s_v * num_v * elem,
            s_v * num_v * l_u * elem,
        ],
        element_stride: [1, s_v, s_v * num_v, s_v * num_v * l_u],
        dtype: GgmlType::F32,
    };

    ctx.tap(&format!("gdn_attn_raw-{layer_idx}"), gdn_attn)?;

    // Fused: gated_attn = (rms_norm(gdn_attn) * ssm_norm) * silu(z).
    // Was rms_norm → swiglu_split (two dispatches); now one kernel writes
    // gated_attn directly. The `ssm_norm_out` tap is kept for parity with
    // diff-dump runs but now points at `gated_attn` rather than the
    // intermediate normed buffer.
    let gated_attn = ctx.alloc_tensor([value_dim, l_u, 1, 1], GgmlType::F32)?;
    elementwise::record_ssm_norm_gate(
        ctx,
        gdn_attn,
        ssm_w.ssm_norm,
        z,
        gated_attn,
        s_v as u32,
        num_v as u32,
        l_u as u32,
        p.rms_eps,
    )?;
    ctx.tap(&format!("ssm_norm_out-{layer_idx}"), gated_attn)?;

    ctx.tap(&format!("attn_output_pre_proj-{layer_idx}"), gated_attn)?;

    // ssm_out @ gated_attn — when no debug dump is active, fuse the
    // residual add into the matvec via the ACCUMULATE spec constant
    // and skip the `proj` scratch slot entirely. Decode-only (L=1
    // matvec path).
    let dump = crate::runtime_flags::qwen_ssm_dump();
    if dump.is_none() && l == 1 {
        matmul::record_accumulate(ctx, ssm_w.ssm_out, gated_attn, residual)?;
        ctx.tap(&format!("attn_output-{layer_idx}"), residual)?;
        ctx.tap(&format!("attn_residual-{layer_idx}"), residual)?;
        return Ok(());
    }

    // Slow path (prefill or dump mode): allocate proj, matmul, optional
    // dump, then a separate residual add.
    let proj = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, ssm_w.ssm_out, gated_attn, proj)?;
    ctx.tap(&format!("attn_output-{layer_idx}"), proj)?;

    if let Some(stage) = dump {
        let src = match stage {
            "alpha" => Some(alpha),
            "beta" => Some(beta),
            "qkv" => Some(qkv),
            "conv" => Some(conv_out),
            "gdn_attn" => {
                // gdn_attn has shape [s_v=128, num_v=32, L, 1] = 4096 elems
                // per token. We need a [hidden=2048, L] view for the cast
                // to work with residual's shape. Flatten the first two
                // dims and TAKE only the first hidden elements (= first
                // 16 heads).
                let trimmed = TensorView {
                    buffer: gdn_attn.buffer,
                    byte_offset: gdn_attn.byte_offset,
                    byte_size: hidden * l_u * elem,
                    dims: [hidden, l_u, 1, 1],
                    byte_stride: [elem, value_dim * elem, value_dim * l_u * elem, value_dim * l_u * elem],
                    element_stride: [1, value_dim, value_dim * l_u, value_dim * l_u],
                    dtype: gdn_attn.dtype,
                };
                Some(trimmed)
            }
            "gated" => {
                // gated_attn has shape [value_dim=4096, L]; take first hidden.
                let trimmed = TensorView {
                    buffer: gated_attn.buffer,
                    byte_offset: gated_attn.byte_offset,
                    byte_size: hidden * l_u * elem,
                    dims: [hidden, l_u, 1, 1],
                    byte_stride: [elem, value_dim * elem, value_dim * l_u * elem, value_dim * l_u * elem],
                    element_stride: [1, value_dim, value_dim * l_u, value_dim * l_u],
                    dtype: gated_attn.dtype,
                };
                Some(trimmed)
            }
            "proj" => Some(proj),
            _ => None,
        };
        if let Some(s) = src {
            // Cast src into residual (truncating or padding via the cast
            // shader's stride math).
            cast::record_cast(ctx, s, residual)?;
            return Ok(());
        }
    }

    elementwise::record_add(ctx, residual, proj, residual)?;
    ctx.tap(&format!("attn_residual-{layer_idx}"), residual)?;

    Ok(())
}

fn n_padded_l_stride(l_u: u64, elem: u64) -> u64 {
    // For the conv_out slice the per-channel byte stride is l_u * elem
    // (each channel occupies L F32s contiguously, since the conv output
    // is laid out token-innermost).
    l_u * elem
}

/// MoE FFN forward — top-k routing on gpu, per-expert matvec via
/// `mul_mat_vec_q4_k_id`, fused weighted sum down via `moe_down_q5_k`,
/// and a sigmoid-gated shared expert added afterward. The whole step
/// stays on the GPU.
fn moe_ffn(
    ctx: &mut DispatchContext,
    w: &MoeFfnWeights,
    post_attn_norm: TensorView,
    residual: TensorView,
    p: &Qwen35MoeParams,
    hidden: u64,
    l: u32,
    layer_idx: u32,
) -> Result<(), Box<dyn Error>> {
    let ff = p.expert_ff as u64;
    let shexp_ff = p.shared_expert_ff as u64;
    let n_experts = p.n_expert;
    let n_used = p.n_expert_used;
    let l_u = l as u64;

    // post_attention_norm of the current residual.
    let x_norm = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    rms_norm::record(ctx, residual, post_attn_norm, x_norm, p.rms_eps)?;
    ctx.tap(&format!("attn_post_norm-{layer_idx}"), x_norm)?;

    // Dispatch all four "first-touch" matmuls in parallel — each reads
    // x_norm and writes a disjoint output. The driver can overlap them
    // on RDNA. Coalesced barrier covers the entire set.
    //   - router logits   (gate_inp)        : drives topk_moe
    //   - shared gate     (ffn_gate_shexp)  : drives shared FFN chain
    //   - shared up       (ffn_up_shexp)    : drives shared FFN chain
    //   - shared scalar   (ffn_gate_inp_shexp): drives sigmoid gate
    let gate_logits = ctx.alloc_tensor([n_experts as u64, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, w.ffn_gate_inp, x_norm, gate_logits)?;
    let sgate = ctx.alloc_tensor([shexp_ff, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, w.ffn_gate_shexp, x_norm, sgate)?;
    let sup = ctx.alloc_tensor([shexp_ff, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, w.ffn_up_shexp, x_norm, sup)?;
    let shared_gate_pre = ctx.alloc_tensor([1, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, w.ffn_gate_inp_shexp, x_norm, shared_gate_pre)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[
            gate_logits.range(),
            sgate.range(),
            sup.range(),
            shared_gate_pre.range(),
        ],
    );
    ctx.tap(&format!("ffn_moe_logits-{layer_idx}"), gate_logits)?;

    // topk_moe → ids + weights (GPU-only).
    let ids = ctx.alloc_scratch((n_experts as u64) * l_u * 4)?;
    let weights_buf = ctx.alloc_scratch((n_used as u64) * l_u * 4)?;
    moe::record_topk_moe(
        ctx,
        gate_logits,
        weights_buf,
        ids,
        moe::TopkMoeParams {
            n_experts,
            n_expert_used: n_used,
            gating_func: moe::GATING_SOFTMAX,
            // llama.cpp normalizes the top-k routing weights to sum to 1
            // (see ffn_moe_weights_norm in qwen35moe.cpp). Without this,
            // routed expert output is scaled down by sum_of_topk weights
            // (typically ~0.3), which biases the residual significantly.
            with_norm: true,
        },
    )?;

    // Per-expert gate + up matvecs. The Q4_K_XL checkpoint mixes Q4_K
    // and Q5_K for gate/up — branch on each tensor's dtype. (The two
    // sides always share a dtype, but branching independently keeps
    // the code robust to future checkpoints that don't.)
    //
    // Shape `[ff, n_expert_used, L, 1]` — one slot per (token, expert).
    // `dispatch_matvec_id` loops `expert_i1` over L internally (mirrors
    // llama.cpp's MUL_MAT_ID dispatch at ggml-vulkan.cpp:9020). For L=1
    // (decode) it's a single dispatch; for L>1 (prefill) it's L
    // sequential dispatches with per-token expert_i1 push constants.
    // gate_exps and up_exps matvec_ids are independent (read same x_norm /
    // same ids, write disjoint outputs). Dispatch both nofence so the
    // driver can overlap them, then emit one coalesced barrier before the
    // swiglu_split below reads them.
    let gate = ctx.alloc_tensor([ff, n_used as u64, l_u, 1], GgmlType::F32)?;
    dispatch_matvec_id_nofence(ctx, w.ffn_gate_exps, x_norm, ids, gate, n_used)?;
    let up = ctx.alloc_tensor([ff, n_used as u64, l_u, 1], GgmlType::F32)?;
    dispatch_matvec_id_nofence(ctx, w.ffn_up_exps, x_norm, ids, up, n_used)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[gate.range(), up.range()],
    );

    ctx.tap(&format!("ffn_moe_gate-{layer_idx}"), gate)?;
    ctx.tap(&format!("ffn_moe_up-{layer_idx}"), up)?;
    // SwiGLU: silu(gate) * up — single fused dispatch.
    let ffn_h = ctx.alloc_tensor([ff, n_used as u64, l_u, 1], GgmlType::F32)?;
    elementwise::record_swiglu_split(ctx, gate, up, ffn_h)?;
    ctx.tap(&format!("ffn_moe_swiglu-{layer_idx}"), ffn_h)?;

    // Fused routing-weighted down. Checkpoints mix dtypes for
    // `ffn_down_exps`: Q4_K_XL uses Q5_K/Q6_K; Q5_K_XL uses Q6_K/Q8_0.
    // Dispatch on dtype.
    let routed = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    match w.ffn_down_exps.dtype {
        GgmlType::Q5_K => {
            moe::record_moe_down_q5k(ctx, w.ffn_down_exps, ffn_h, ids, weights_buf, routed, n_used)?;
        }
        GgmlType::Q6_K => {
            moe::record_moe_down_q6k(ctx, w.ffn_down_exps, ffn_h, ids, weights_buf, routed, n_used)?;
        }
        GgmlType::Q8_0 => {
            moe::record_moe_down_q8_0(ctx, w.ffn_down_exps, ffn_h, ids, weights_buf, routed, n_used)?;
        }
        other => {
            return Err(format!(
                "qwen35moe: ffn_down_exps dtype {other:?} not supported (need Q5_K, Q6_K or Q8_0)"
            )
            .into());
        }
    }
    ctx.tap(&format!("ffn_moe_out-{layer_idx}"), routed)?;

    // Shared expert FFN — `sgate` and `sup` were already dispatched in
    // parallel with the router (see hoist above). Continue the chain
    // here: swiglu_split fuses silu(sgate)*sup, then down_shexp matmul.
    let sh = ctx.alloc_tensor([shexp_ff, l_u, 1, 1], GgmlType::F32)?;
    elementwise::record_swiglu_split(ctx, sgate, sup, sh)?;
    let shared = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, w.ffn_down_shexp, sh, shared)?;

    // Final residual update: residual += routed + shared * sigmoid(gate)
    // — single fused kernel collapsing what used to be four dispatches
    // (sigmoid, broadcast-mul, two adds). Diagnostic bisection flags
    // fall back to the unfused form so each branch can be dropped
    // independently.
    let no_routed = crate::runtime_flags::qwen_no_routed();
    let no_shared = crate::runtime_flags::qwen_no_shared();
    if !no_routed && !no_shared {
        elementwise::record_moe_residual_fuse(
            ctx,
            residual,
            routed,
            shared,
            shared_gate_pre,
            residual,
            hidden as u32,
        )?;
    } else {
        // Fallback path retained only for diagnostic flag combinations.
        let shared_gate_sig = ctx.alloc_tensor([1, l_u, 1, 1], GgmlType::F32)?;
        elementwise::record_sigmoid(ctx, shared_gate_pre, shared_gate_sig)?;
        let shared_gate_broadcast = TensorView {
            dims: [hidden, l_u, 1, 1],
            byte_stride: [0, 4, 4 * l_u, 4 * l_u],
            element_stride: [0, 1, l_u, l_u],
            byte_size: shared_gate_sig.byte_size,
            ..shared_gate_sig
        };
        let shared_scaled = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
        elementwise::record_mul(ctx, shared, shared_gate_broadcast, shared_scaled)?;
        if !no_routed {
            elementwise::record_add(ctx, residual, routed, residual)?;
        }
        if !no_shared {
            elementwise::record_add(ctx, residual, shared_scaled, residual)?;
        }
    }
    let _ = shared;
    let _ = routed;
    ctx.tap(&format!("l_out-{layer_idx}"), residual)?;

    Ok(())
}

// ───────────────────────────────────────────────────────────────────────
// Batched-decode (M2) block helpers: B sequences, one token each.
// ───────────────────────────────────────────────────────────────────────

/// A single-column `[d0, d1, 1, 1]` view of a contiguous `[d0, d1, B]` tensor
/// (sequence `s`), for the per-sequence KV cache write.
fn col3(t: TensorView, s: u64, col_stride: u64, d0: u64, d1: u64) -> TensorView {
    let elem = t.byte_stride[0];
    TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset + s * col_stride,
        byte_size: d0 * d1 * elem,
        dims: [d0, d1, 1, 1],
        byte_stride: [elem, elem * d0, elem * d0 * d1, elem * d0 * d1],
        element_stride: [1, d0, d0 * d1, d0 * d1],
        dtype: t.dtype,
    }
}

/// Reinterpret a contiguous `[head_dim, n_head, B]` (post-RoPE) Q as the
/// `[head_dim, 1, n_head, B]` flash-attn batched-decode layout (batch stride =
/// `batch_stride`, = head_dim*n_head).
fn batched_q_attn_view(t: TensorView, head_dim: u64, n_head: u64, b: u64, batch_stride: u64) -> TensorView {
    let elem = t.byte_stride[0];
    TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset,
        byte_size: t.byte_size,
        dims: [head_dim, 1, n_head, b],
        byte_stride: [elem, elem * head_dim, elem * head_dim, elem * batch_stride],
        element_stride: [1, head_dim, head_dim, batch_stride],
        dtype: t.dtype,
    }
}

/// Batched qwen attention block: per-slab KV + batched flash-attn, M-RoPE +
/// per-head Q/K norm + Q-gate, B sequences at their own positions.
#[allow(clippy::too_many_arguments)]
fn attention_block_batch(
    ctx: &mut DispatchContext,
    att: &AttentionBlockWeights,
    batch: &crate::inference::kv_cache::BatchKvCache,
    residual: TensorView,
    positions_buf: crate::inference::buffer::BufferRange,
    rope_params: rope_multi::RopeMultiParams,
    fa_params: flash_attn::FlashAttnParams,
    layer_idx: u32,
    positions: &[u32],
    slots: &[u32],
    kv_lens: &[u32],
    p: &Qwen35MoeParams,
    head_dim_k: u64,
    head_dim_v: u64,
    n_head: u64,
    n_head_kv: u64,
    n_embd_kv: u64,
    n_embd_vv: u64,
    wq_out: u64,
    hidden: u64,
    hidden_v: u64,
    b: u64,
    fa_dyn_range: crate::inference::buffer::BufferRange,
) -> Result<(), Box<dyn Error>> {
    let elem = 4u64;
    let x_norm = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
    rms_norm::record(ctx, residual, att.attn_norm, x_norm, p.rms_eps)?;

    let q_full = ctx.alloc_tensor([wq_out, b, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wq, x_norm, q_full)?;
    let k = ctx.alloc_tensor([n_embd_kv, b, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wk, x_norm, k)?;
    let v = ctx.alloc_tensor([n_embd_vv, b, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wv, x_norm, v)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_full.range(), k.range(), v.range()],
    );

    let q_attn_view = slice_q_half(q_full, head_dim_k, n_head, b, /*gate=*/ false);
    let q_gate_view = slice_q_half(q_full, head_dim_k, n_head, b, /*gate=*/ true);
    let k_view = reshape_for_rope(k, head_dim_k, n_head_kv, b);
    let v_view = reshape_for_rope(v, head_dim_v, n_head_kv, b);

    // Per-head RMS norm + M-RoPE into F32 scratch (per-sequence cache writes
    // below — the fused K-to-cache write bakes a single position offset, so it
    // can't serve B sequences at different positions/slabs).
    let q_roped = ctx.alloc_tensor([head_dim_k, n_head, b, 1], GgmlType::F32)?;
    rope_multi::record_rms_norm_rope_nofence(
        ctx, q_attn_view, positions_buf, att.attn_q_norm, q_roped, rope_params, p.rms_eps,
    )?;
    let k_roped = ctx.alloc_tensor([head_dim_k, n_head_kv, b, 1], GgmlType::F32)?;
    rope_multi::record_rms_norm_rope_nofence(
        ctx, k_view, positions_buf, att.attn_k_norm, k_roped, rope_params, p.rms_eps,
    )?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_roped.range(), k_roped.range()],
    );

    // Per-sequence K/V cache writes: column s → slab `slots[s]` at positions[s].
    let k_col_stride = head_dim_k * n_head_kv * elem;
    let v_col_stride = head_dim_v * n_head_kv * elem;
    for s in 0..b as usize {
        let k_col = col3(k_roped, s as u64, k_col_stride, head_dim_k, n_head_kv);
        let v_col = col3(v_view, s as u64, v_col_stride, head_dim_v, n_head_kv);
        cache_io::record_write(ctx, k_col, batch.slot_k_view(slots[s], layer_idx), positions[s])?;
        cache_io::record_write(ctx, v_col, batch.slot_v_view(slots[s], layer_idx), positions[s])?;
    }

    // Batched attention. The K/V views bind every slab; the flash reads each
    // sequence's slab via `DecodeDyn::slot` (= slots[s]).
    let q_attn = batched_q_attn_view(q_roped, head_dim_k, n_head, b, hidden_v);
    let k_attn = batch.batched_k_attn_view(layer_idx);
    let v_attn = batch.batched_v_attn_view(layer_idx);
    let attn_out = ctx.alloc_tensor([hidden_v, b, 1, 1], GgmlType::F32)?;
    flash_attn::record_batched(
        ctx, q_attn, k_attn, v_attn, attn_out, fa_params, kv_lens, fa_dyn_range, Some(slots),
        /*query_lens=*/ None,
    )?;

    // Sigmoid-gate the attention output by q_gate.
    let q_gate_contig = ctx.alloc_tensor([head_dim_k, n_head, b, 1], GgmlType::F32)?;
    cast::record_cast(ctx, q_gate_view, q_gate_contig)?;
    let q_gate_flat = TensorView {
        dims: [hidden_v, b, 1, 1],
        byte_size: q_gate_contig.byte_size,
        byte_stride: [4, 4 * hidden_v, 4 * hidden_v * b, 4 * hidden_v * b],
        element_stride: [1, hidden_v, hidden_v * b, hidden_v * b],
        ..q_gate_contig
    };
    let attn_gated = ctx.alloc_tensor([hidden_v, b, 1, 1], GgmlType::F32)?;
    elementwise::record_sigmoid_mul_split(ctx, q_gate_flat, attn_out, attn_gated)?;

    // residual += wo @ attn_gated. B > 1 so the fused matvec-accumulate path
    // (N=1 only) doesn't apply — use the general matmul + add.
    let proj = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, att.wo, attn_gated, proj)?;
    elementwise::record_add(ctx, residual, proj, residual)?;
    Ok(())
}

/// Unified varlen qwen attention block (M5 Phase 4): like
/// [`attention_block_batch`] but each sequence contributes `seq_lens[s]` query
/// rows (a prefill chunk or 1 for decode), packed flat in the `N_total` token
/// dimension. Dense ops + M-RoPE + Q-gate run on the flat stream; each sequence
/// writes its `L_s`-token K/V chunk to its slab at `base_s = kv_lens[s] -
/// seq_lens[s]`; the flash is the varlen causal path (`query_lens = seq_lens`).
#[allow(clippy::too_many_arguments)]
fn attention_block_unified(
    ctx: &mut DispatchContext,
    att: &AttentionBlockWeights,
    batch: &crate::inference::kv_cache::BatchKvCache,
    residual: TensorView,
    positions_buf: crate::inference::buffer::BufferRange,
    rope_params: rope_multi::RopeMultiParams,
    fa_params: flash_attn::FlashAttnParams,
    layer_idx: u32,
    kv_lens: &[u32],
    seq_lens: &[u32],
    q_starts: &[u64],
    slots: &[u32],
    p: &Qwen35MoeParams,
    head_dim_k: u64,
    head_dim_v: u64,
    n_head: u64,
    n_head_kv: u64,
    n_embd_kv: u64,
    n_embd_vv: u64,
    wq_out: u64,
    hidden: u64,
    hidden_v: u64,
    n_total: u64,
    fa_dyn_range: crate::inference::buffer::BufferRange,
) -> Result<(), Box<dyn Error>> {
    let elem = 4u64;
    let b = seq_lens.len();
    let x_norm = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
    rms_norm::record(ctx, residual, att.attn_norm, x_norm, p.rms_eps)?;

    let q_full = ctx.alloc_tensor([wq_out, n_total, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wq, x_norm, q_full)?;
    let k = ctx.alloc_tensor([n_embd_kv, n_total, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wk, x_norm, k)?;
    let v = ctx.alloc_tensor([n_embd_vv, n_total, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, att.wv, x_norm, v)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_full.range(), k.range(), v.range()],
    );

    let q_attn_view = slice_q_half(q_full, head_dim_k, n_head, n_total, /*gate=*/ false);
    let q_gate_view = slice_q_half(q_full, head_dim_k, n_head, n_total, /*gate=*/ true);
    let k_view = reshape_for_rope(k, head_dim_k, n_head_kv, n_total);
    let v_view = reshape_for_rope(v, head_dim_v, n_head_kv, n_total);

    let q_roped = ctx.alloc_tensor([head_dim_k, n_head, n_total, 1], GgmlType::F32)?;
    rope_multi::record_rms_norm_rope_nofence(
        ctx, q_attn_view, positions_buf, att.attn_q_norm, q_roped, rope_params, p.rms_eps,
    )?;
    let k_roped = ctx.alloc_tensor([head_dim_k, n_head_kv, n_total, 1], GgmlType::F32)?;
    rope_multi::record_rms_norm_rope_nofence(
        ctx, k_view, positions_buf, att.attn_k_norm, k_roped, rope_params, p.rms_eps,
    )?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_roped.range(), k_roped.range()],
    );

    // Per-sequence K/V chunk write: seq s's L_s columns → slab slots[s] at base_s.
    let k_tok_stride = head_dim_k * n_head_kv * elem;
    let v_tok_stride = head_dim_v * n_head_kv * elem;
    let chunk = |t: TensorView, qs: u64, l: u64, d0: u64, d1: u64, tok_stride: u64| -> TensorView {
        TensorView {
            byte_offset: t.byte_offset + qs * tok_stride,
            byte_size: l * tok_stride,
            dims: [d0, d1, l, 1],
            byte_stride: [elem, elem * d0, tok_stride, tok_stride * l],
            element_stride: [1, d0, d0 * d1, d0 * d1 * l],
            ..t
        }
    };
    for s in 0..b {
        let l = seq_lens[s] as u64;
        let base = kv_lens[s] - seq_lens[s];
        let k_chunk = chunk(k_roped, q_starts[s], l, head_dim_k, n_head_kv, k_tok_stride);
        let v_chunk = chunk(v_view, q_starts[s], l, head_dim_v, n_head_kv, v_tok_stride);
        cache_io::record_write(ctx, k_chunk, batch.slot_k_view(slots[s], layer_idx), base)?;
        cache_io::record_write(ctx, v_chunk, batch.slot_v_view(slots[s], layer_idx), base)?;
    }

    // Varlen attention: flat token-major Q view; the flash masks causally per
    // sequence over its own slab.
    let q_attn = TensorView {
        dims: [head_dim_k, n_total, n_head, 1],
        byte_stride: [elem, elem * head_dim_k * n_head, elem * head_dim_k, elem * head_dim_k * n_head * n_total],
        element_stride: [1, head_dim_k * n_head, head_dim_k, head_dim_k * n_head * n_total],
        ..q_roped
    };
    let k_attn = batch.batched_k_attn_view(layer_idx);
    let v_attn = batch.batched_v_attn_view(layer_idx);
    let attn_out = ctx.alloc_tensor([hidden_v, n_total, 1, 1], GgmlType::F32)?;
    flash_attn::record_batched(
        ctx, q_attn, k_attn, v_attn, attn_out, fa_params, kv_lens, fa_dyn_range, Some(slots),
        Some(seq_lens),
    )?;

    // Sigmoid-gate by q_gate (per-token).
    let q_gate_contig = ctx.alloc_tensor([head_dim_k, n_head, n_total, 1], GgmlType::F32)?;
    cast::record_cast(ctx, q_gate_view, q_gate_contig)?;
    let q_gate_flat = TensorView {
        dims: [hidden_v, n_total, 1, 1],
        byte_size: q_gate_contig.byte_size,
        byte_stride: [elem, elem * hidden_v, elem * hidden_v * n_total, elem * hidden_v * n_total],
        element_stride: [1, hidden_v, hidden_v * n_total, hidden_v * n_total],
        ..q_gate_contig
    };
    let attn_gated = ctx.alloc_tensor([hidden_v, n_total, 1, 1], GgmlType::F32)?;
    elementwise::record_sigmoid_mul_split(ctx, q_gate_flat, attn_out, attn_gated)?;

    let proj = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, att.wo, attn_gated, proj)?;
    elementwise::record_add(ctx, residual, proj, residual)?;
    Ok(())
}

/// Batched qwen SSM/GDN block: per-sequence conv-input packing + conv at
/// `n_seqs = B`, then gated-delta-net at `n_seqs = B` over per-sequence state.
#[allow(clippy::too_many_arguments)]
fn ssm_block_batch(
    ctx: &mut DispatchContext,
    ssm_w: &SsmBlockWeights,
    batch: &crate::inference::kv_cache::BatchKvCache,
    residual: TensorView,
    p: &Qwen35MoeParams,
    hidden: u64,
    b: u64,
    ssm_layer_idx: u32,
    _positions: &[u32],
    slots: &[u32],
) -> Result<(), Box<dyn Error>> {
    let elem = 4u64;
    let num_k = p.ssm_groups as u64;
    let num_v = p.ssm_dt_rank as u64;
    let s_v = p.ssm_state as u64;
    let conv_kernel = p.ssm_conv as u64;
    let key_dim = num_k * s_v;
    let value_dim = num_v * s_v;
    let conv_channels = 2 * key_dim + value_dim;
    let n_padded = (conv_kernel - 1) + 1; // L=1 per sequence
    let state_dim_inner = conv_kernel - 1;

    let x_norm = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
    rms_norm::record(ctx, residual, ssm_w.attn_norm, x_norm, p.rms_eps)?;

    let qkv = ctx.alloc_tensor([conv_channels, b, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, ssm_w.attn_qkv, x_norm, qkv)?;
    let z = ctx.alloc_tensor([value_dim, b, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, ssm_w.attn_gate, x_norm, z)?;
    let beta_pre = ctx.alloc_tensor([num_v, b, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, ssm_w.ssm_beta, x_norm, beta_pre)?;
    let alpha_pre = ctx.alloc_tensor([num_v, b, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, ssm_w.ssm_alpha, x_norm, alpha_pre)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[qkv.range(), z.range(), beta_pre.range(), alpha_pre.range()],
    );

    let beta = ctx.alloc_tensor([num_v, b, 1, 1], GgmlType::F32)?;
    elementwise::record_sigmoid_nofence(ctx, beta_pre, beta)?;
    let alpha = ctx.alloc_tensor([num_v, b, 1, 1], GgmlType::F32)?;
    elementwise::record_ssm_alpha_fuse_nofence(
        ctx, alpha_pre, ssm_w.ssm_dt_bias, ssm_w.ssm_a, alpha, num_v as u32,
    )?;

    // ── conv input [n_padded, conv_channels, B]: per-seq prefix + tail ──
    let conv_input = ctx.alloc_tensor([n_padded, conv_channels, b, 1], GgmlType::F32)?;
    {
        let host = ctx.scratch.host_ptr.ok_or("scratch not host-visible")?;
        unsafe {
            std::ptr::write_bytes(
                host.add(conv_input.byte_offset as usize) as *mut u8,
                0,
                conv_input.byte_size as usize,
            );
        }
    }
    // ── Per-sequence conv + GDN recurrent state. When the batch already
    // occupies contiguous slabs [0,B) (the dense / single-stream case), the
    // ops read/write the persistent seq-outermost blocks directly — zero copy.
    // Otherwise (a gathered, non-contiguous batch — prefix-reuse parked
    // conversations in arbitrary slabs) gather each sequence's state into
    // contiguous [0,B) working buffers, run, and scatter back after the GDN.
    let contiguous = slots.iter().enumerate().all(|(i, &s)| s as u64 == i as u64);
    let conv_floats = batch.conv_slot_floats();
    let gdn_floats = batch.gdn_slot_floats();
    let (conv_state, gdn_state_in, gathered) = if contiguous {
        (
            batch.conv_state_layer(ssm_layer_idx),
            batch.gdn_state_layer(ssm_layer_idx),
            None,
        )
    } else {
        let conv_work = ctx.alloc_scratch(b * conv_floats * elem)?;
        let gdn_work = ctx.alloc_scratch(b * gdn_floats * elem)?;
        unsafe {
            use ash::vk;
            for bi in 0..b {
                let cs = batch.conv_state_slot(ssm_layer_idx, slots[bi as usize]);
                let copy = vk::BufferCopy::default()
                    .src_offset(cs.offset)
                    .dst_offset(conv_work.offset + bi * conv_floats * elem)
                    .size(conv_floats * elem);
                ctx.device.device.cmd_copy_buffer(
                    ctx.cmd, cs.buffer, conv_work.buffer, std::slice::from_ref(&copy),
                );
                let gs = batch.gdn_state_slot(ssm_layer_idx, slots[bi as usize]);
                let copy = vk::BufferCopy::default()
                    .src_offset(gs.offset)
                    .dst_offset(gdn_work.offset + bi * gdn_floats * elem)
                    .size(gdn_floats * elem);
                ctx.device.device.cmd_copy_buffer(
                    ctx.cmd, gs.buffer, gdn_work.buffer, std::slice::from_ref(&copy),
                );
            }
        }
        crate::inference::command::record_global_barrier(ctx.device, ctx.cmd);
        (conv_work, gdn_work, Some((conv_work, gdn_work)))
    };
    // (a) conv_state (all B) → conv_input prefixes (all B).
    let prefix_src = TensorView {
        buffer: conv_state.buffer,
        byte_offset: conv_state.offset,
        byte_size: conv_state.size,
        dims: [state_dim_inner, conv_channels, b, 1],
        byte_stride: [elem, state_dim_inner * elem, state_dim_inner * conv_channels * elem, state_dim_inner * conv_channels * elem],
        element_stride: [1, state_dim_inner, state_dim_inner * conv_channels, state_dim_inner * conv_channels],
        dtype: GgmlType::F32,
    };
    let prefix_dst = TensorView {
        buffer: conv_input.buffer,
        byte_offset: conv_input.byte_offset,
        byte_size: conv_input.byte_size,
        dims: [state_dim_inner, conv_channels, b, 1],
        byte_stride: [elem, n_padded * elem, n_padded * conv_channels * elem, n_padded * conv_channels * elem],
        element_stride: [1, n_padded, n_padded * conv_channels, n_padded * conv_channels],
        dtype: GgmlType::F32,
    };
    cast::record_cast(ctx, prefix_src, prefix_dst)?;
    // (b) qkv [conv_channels, B] → conv_input tail (position kernel-1, all B).
    let tail_src = TensorView {
        buffer: qkv.buffer,
        byte_offset: qkv.byte_offset,
        byte_size: qkv.byte_size,
        dims: [1, conv_channels, b, 1],
        byte_stride: [conv_channels * elem, elem, conv_channels * elem, conv_channels * elem],
        element_stride: [conv_channels, 1, conv_channels, conv_channels],
        dtype: qkv.dtype,
    };
    let tail_dst = TensorView {
        buffer: conv_input.buffer,
        byte_offset: conv_input.byte_offset + (conv_kernel - 1) * elem,
        byte_size: conv_input.byte_size - (conv_kernel - 1) * elem,
        dims: [1, conv_channels, b, 1],
        byte_stride: [elem, n_padded * elem, n_padded * conv_channels * elem, n_padded * conv_channels * elem],
        element_stride: [1, n_padded, n_padded * conv_channels, n_padded * conv_channels],
        dtype: conv_input.dtype,
    };
    cast::record_cast(ctx, tail_src, tail_dst)?;
    // (c) conv state writeback: conv_input[1..kernel] (all B) → conv_state.
    let wb_src = TensorView {
        buffer: conv_input.buffer,
        byte_offset: conv_input.byte_offset + elem, // s_idx = L = 1
        byte_size: conv_input.byte_size - elem,
        dims: [state_dim_inner, conv_channels, b, 1],
        byte_stride: [elem, n_padded * elem, n_padded * conv_channels * elem, n_padded * conv_channels * elem],
        element_stride: [1, n_padded, n_padded * conv_channels, n_padded * conv_channels],
        dtype: GgmlType::F32,
    };
    cast::record_cast(ctx, wb_src, prefix_src)?;

    // conv1d (fused silu), n_seqs = B → conv_out [conv_channels, 1, B].
    let conv_out = ctx.alloc_tensor([conv_channels, 1, b, 1], GgmlType::F32)?;
    ssm::record_ssm_conv_nofence(
        ctx, conv_input, ssm_w.ssm_conv1d, conv_out, conv_channels as u32, n_padded as u32, 1,
        b as u32, conv_kernel as u32, /*fuse_silu=*/ true,
    )?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[beta.range(), alpha.range(), conv_out.range()],
    );

    // Slice Q/K/V from conv_out [seq][token=1][channel] → [s_v, num_heads, 1, B].
    let slice_qkv = |chan_offset: u64, chan_count: u64| -> TensorView {
        TensorView {
            buffer: conv_out.buffer,
            byte_offset: conv_out.byte_offset + chan_offset * elem,
            byte_size: conv_out.byte_size - chan_offset * elem,
            dims: [chan_count, 1, b, 1],
            byte_stride: [elem, conv_channels * elem, conv_channels * elem, conv_channels * elem],
            element_stride: [1, conv_channels, conv_channels, conv_channels],
            dtype: conv_out.dtype,
        }
    };
    let head_view = |slice: TensorView, num_heads: u64| -> TensorView {
        TensorView {
            buffer: slice.buffer,
            byte_offset: slice.byte_offset,
            byte_size: slice.byte_size,
            dims: [s_v, num_heads, 1, b],
            byte_stride: [elem, s_v * elem, conv_channels * elem, conv_channels * elem],
            element_stride: [1, s_v, conv_channels, conv_channels],
            dtype: slice.dtype,
        }
    };
    let q_view = head_view(slice_qkv(0, key_dim), num_k);
    let k_view = head_view(slice_qkv(key_dim, key_dim), num_k);
    let v_view = head_view(slice_qkv(2 * key_dim, value_dim), num_v);

    let ssm_norm_eps = 1e-6;
    let q_normed = ctx.alloc_tensor([s_v, num_k, 1, b], GgmlType::F32)?;
    elementwise::record_l2_norm_nofence(ctx, q_view, q_normed, ssm_norm_eps)?;
    let k_normed = ctx.alloc_tensor([s_v, num_k, 1, b], GgmlType::F32)?;
    elementwise::record_l2_norm_nofence(ctx, k_view, k_normed, ssm_norm_eps)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_normed.range(), k_normed.range()],
    );

    // gated-delta-net at n_seqs = B.
    let attn_floats = b * num_v * s_v; // B * (1 token * num_v * s_v)
    let state_floats = b * num_v * s_v * s_v;
    let gdn_dst = ctx.alloc_scratch((attn_floats + state_floats) * elem)?;
    let gdn_scale = 1.0 / (s_v as f32).sqrt();
    let q_normed_strides = ssm::GdnStrides {
        s1: s_v as u32,
        s2: (s_v * num_k) as u32,
        s3: (s_v * num_k) as u32,
    };
    let v_strides = ssm::GdnStrides {
        s1: s_v as u32,
        s2: conv_channels as u32,
        s3: conv_channels as u32,
    };
    let b_strides = ssm::GdnStrides {
        s1: 1,
        s2: num_v as u32,
        s3: num_v as u32,
    };
    ssm::record_gated_delta_net(
        ctx, q_normed, k_normed, v_view, alpha, beta, gdn_state_in, gdn_dst, num_v as u32,
        num_k as u32, 1, b as u32, attn_floats as u32, gdn_scale, q_normed_strides, v_strides,
        b_strides, s_v as u32, 1,
    )?;
    // Write the updated GDN state back. The conv writeback (cast) already
    // landed the new conv state in `conv_state` above. A compute→transfer
    // barrier (the conv cast / GDN dispatch fenced their writes) precedes; a
    // global barrier after makes the writes visible before the next read.
    crate::inference::command::record_global_barrier(ctx.device, ctx.cmd);
    unsafe {
        use ash::vk;
        match gathered {
            // Non-contiguous: scatter conv_work + gdn_dst[state] to each slab.
            Some((conv_work, _)) => {
                for bi in 0..b {
                    let cs = batch.conv_state_slot(ssm_layer_idx, slots[bi as usize]);
                    let copy = vk::BufferCopy::default()
                        .src_offset(conv_work.offset + bi * conv_floats * elem)
                        .dst_offset(cs.offset)
                        .size(conv_floats * elem);
                    ctx.device.device.cmd_copy_buffer(
                        ctx.cmd, conv_work.buffer, cs.buffer, std::slice::from_ref(&copy),
                    );
                    let gs = batch.gdn_state_slot(ssm_layer_idx, slots[bi as usize]);
                    let copy = vk::BufferCopy::default()
                        .src_offset(gdn_dst.offset + attn_floats * elem + bi * gdn_floats * elem)
                        .dst_offset(gs.offset)
                        .size(gdn_floats * elem);
                    ctx.device.device.cmd_copy_buffer(
                        ctx.cmd, gdn_dst.buffer, gs.buffer, std::slice::from_ref(&copy),
                    );
                }
            }
            // Contiguous: one B-wide copy straight to the persistent block.
            None => {
                let copy = vk::BufferCopy::default()
                    .src_offset(gdn_dst.offset + attn_floats * elem)
                    .dst_offset(gdn_state_in.offset)
                    .size(state_floats * elem);
                ctx.device.device.cmd_copy_buffer(
                    ctx.cmd, gdn_dst.buffer, gdn_state_in.buffer, std::slice::from_ref(&copy),
                );
            }
        }
    }
    crate::inference::command::record_global_barrier(ctx.device, ctx.cmd);

    // gated_attn = (rms_norm(gdn_attn) * ssm_norm) * silu(z); then ssm_out proj.
    let gdn_attn = TensorView {
        buffer: gdn_dst.buffer,
        byte_offset: gdn_dst.offset,
        byte_size: attn_floats * elem,
        dims: [s_v, num_v, b, 1],
        byte_stride: [elem, s_v * elem, s_v * num_v * elem, s_v * num_v * b * elem],
        element_stride: [1, s_v, s_v * num_v, s_v * num_v * b],
        dtype: GgmlType::F32,
    };
    let gated_attn = ctx.alloc_tensor([value_dim, b, 1, 1], GgmlType::F32)?;
    elementwise::record_ssm_norm_gate(
        ctx, gdn_attn, ssm_w.ssm_norm, z, gated_attn, s_v as u32, num_v as u32, b as u32, p.rms_eps,
    )?;
    // B > 1 → general matmul + add (the fused matvec-accumulate is N=1 only).
    let proj = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, ssm_w.ssm_out, gated_attn, proj)?;
    elementwise::record_add(ctx, residual, proj, residual)?;
    Ok(())
}

fn parse_params(gguf: &GgufFile, handle: &WeightsHandle) -> Result<Qwen35MoeParams, Box<dyn Error>> {
    let u32_key = |k: &'static str| -> Result<u32, Box<dyn Error>> {
        let v = gguf.get(k).ok_or(ModelError::MissingMetadata(k))?;
        coerce_u32(v).ok_or_else(|| {
            ModelError::BadMetadata {
                key: k,
                detail: format!("expected unsigned int, got {v:?}"),
            }
            .into()
        })
    };
    let u32_or = |k: &'static str, default: u32| -> u32 {
        gguf.get(k).and_then(coerce_u32).unwrap_or(default)
    };
    let f32_or = |k: &'static str, default: f32| -> f32 {
        gguf.get(k).and_then(coerce_f32).unwrap_or(default)
    };

    let n_layer = u32_key("qwen35moe.block_count")?;
    let nextn_predict_layers = u32_or("qwen35moe.nextn_predict_layers", 0);
    let n_main = n_layer.saturating_sub(nextn_predict_layers);
    let full_attn_interval = u32_or("qwen35moe.full_attention_interval", 4);

    let n_embd = u32_key("qwen35moe.embedding_length")?;
    let n_head = u32_key("qwen35moe.attention.head_count")?;
    let n_head_kv = u32_or("qwen35moe.attention.head_count_kv", n_head);
    let head_dim_k = u32_or("qwen35moe.attention.key_length", n_embd / n_head);
    let head_dim_v = u32_or("qwen35moe.attention.value_length", n_embd / n_head);
    let n_ctx_train = u32_or("qwen35moe.context_length", 4096);

    // RoPE — sections is a 4-element int32 array per the GGUF spec.
    let rope_dim = u32_or("qwen35moe.rope.dimension_count", head_dim_k);
    let rope_freq_base = f32_or("qwen35moe.rope.freq_base", 10000.0);
    let rope_sections = read_sections(gguf, "qwen35moe.rope.dimension_sections")?;

    let n_expert = u32_or("qwen35moe.expert_count", 0);
    let n_expert_used = u32_or("qwen35moe.expert_used_count", 0);
    let expert_ff = u32_or("qwen35moe.expert_feed_forward_length", 0);
    let shared_expert_ff = u32_or("qwen35moe.expert_shared_feed_forward_length", 0);

    let ssm_state = u32_or("qwen35moe.ssm.state_size", 0);
    let ssm_conv = u32_or("qwen35moe.ssm.conv_kernel", 0);
    let ssm_groups = u32_or("qwen35moe.ssm.group_count", 0);
    let ssm_dt_rank = u32_or("qwen35moe.ssm.time_step_rank", 0);
    let ssm_inner = u32_or("qwen35moe.ssm.inner_size", 0);

    let rms_eps = f32_or("qwen35moe.attention.layer_norm_rms_epsilon", 1e-5);

    // vocab_size is occasionally absent from the metadata block; derive
    // from the token embedding tensor (`token_embd.weight`) shape when so.
    let n_vocab = match gguf.get("qwen35moe.vocab_size").and_then(coerce_u32) {
        Some(v) => v,
        None => {
            let view = handle
                .view("token_embd.weight")
                .map_err(|_| ModelError::MissingTensor("token_embd.weight".to_string()))?;
            // `token_embd.weight` is `[n_embd, n_vocab]` in ggml layout —
            // ne[0] = n_embd, ne[1] = n_vocab.
            view.dims[1] as u32
        }
    };

    Ok(Qwen35MoeParams {
        n_layer,
        n_main,
        nextn_predict_layers,
        full_attn_interval,
        n_embd,
        n_head,
        n_head_kv,
        head_dim_k,
        head_dim_v,
        n_vocab,
        n_ctx_train,
        rope_dim,
        rope_sections,
        rope_freq_base,
        n_expert,
        n_expert_used,
        expert_ff,
        shared_expert_ff,
        ssm_state,
        ssm_conv,
        ssm_groups,
        ssm_dt_rank,
        ssm_inner,
        rms_eps,
    })
}

fn read_sections(gguf: &GgufFile, key: &'static str) -> Result<[u32; 4], Box<dyn Error>> {
    let v = gguf.get(key).ok_or(ModelError::MissingMetadata(key))?;
    let arr = match v {
        MetadataValue::Array(items) => items,
        other => {
            return Err(ModelError::BadMetadata {
                key,
                detail: format!("expected array, got {other:?}"),
            }
            .into());
        }
    };
    if arr.len() > 4 {
        return Err(ModelError::BadMetadata {
            key,
            detail: format!("too many sections: {} (max 4)", arr.len()),
        }
        .into());
    }
    let mut out = [0u32; 4];
    for (i, item) in arr.iter().enumerate() {
        out[i] = coerce_u32(item).ok_or_else(|| ModelError::BadMetadata {
            key,
            detail: format!("section[{i}] not a uint: {item:?}"),
        })?;
    }
    Ok(out)
}

fn collect_weights(
    handle: &WeightsHandle,
    params: &Qwen35MoeParams,
    spec_enabled: bool,
) -> Result<Qwen35MoeWeights, Box<dyn Error>> {
    let view = |name: &str| -> Result<TensorView, Box<dyn Error>> {
        handle
            .view(name)
            .map_err(|_| ModelError::MissingTensor(name.to_string()).into())
    };
    // Load the 8 MoE FFN tensors for block `i`. Shared by the main-trunk
    // loop and the MTP block loader below.
    let load_moe = |i: u32| -> Result<MoeFfnWeights, Box<dyn Error>> {
        Ok(MoeFfnWeights {
            ffn_gate_inp: view(&format!("blk.{i}.ffn_gate_inp.weight"))?,
            ffn_gate_inp_shexp: view(&format!("blk.{i}.ffn_gate_inp_shexp.weight"))?,
            ffn_gate_exps: view(&format!("blk.{i}.ffn_gate_exps.weight"))?,
            ffn_up_exps: view(&format!("blk.{i}.ffn_up_exps.weight"))?,
            ffn_down_exps: view(&format!("blk.{i}.ffn_down_exps.weight"))?,
            ffn_gate_shexp: view(&format!("blk.{i}.ffn_gate_shexp.weight"))?,
            ffn_up_shexp: view(&format!("blk.{i}.ffn_up_shexp.weight"))?,
            ffn_down_shexp: view(&format!("blk.{i}.ffn_down_shexp.weight"))?,
        })
    };
    // Load an attention block's transformer tensors for block `i`.
    let load_attn = |i: u32, moe: MoeFfnWeights| -> Result<AttentionBlockWeights, Box<dyn Error>> {
        Ok(AttentionBlockWeights {
            attn_norm: view(&format!("blk.{i}.attn_norm.weight"))?,
            post_attn_norm: view(&format!("blk.{i}.post_attention_norm.weight"))?,
            wq: view(&format!("blk.{i}.attn_q.weight"))?,
            wk: view(&format!("blk.{i}.attn_k.weight"))?,
            wv: view(&format!("blk.{i}.attn_v.weight"))?,
            wo: view(&format!("blk.{i}.attn_output.weight"))?,
            attn_q_norm: view(&format!("blk.{i}.attn_q_norm.weight"))?,
            attn_k_norm: view(&format!("blk.{i}.attn_k_norm.weight"))?,
            moe,
        })
    };

    let token_embd = view("token_embd.weight")?;
    let output_norm = view("output_norm.weight")?;
    let output = handle.view("output.weight").ok();

    let mut blocks = Vec::with_capacity(params.n_main as usize);
    for i in 0..params.n_main {
        let moe = load_moe(i)?;
        let block = if params.is_attention_layer(i) {
            BlockWeights::Attention(load_attn(i, moe)?)
        } else {
            BlockWeights::Ssm(SsmBlockWeights {
                attn_norm: view(&format!("blk.{i}.attn_norm.weight"))?,
                post_attn_norm: view(&format!("blk.{i}.post_attention_norm.weight"))?,
                attn_qkv: view(&format!("blk.{i}.attn_qkv.weight"))?,
                attn_gate: view(&format!("blk.{i}.attn_gate.weight"))?,
                ssm_alpha: view(&format!("blk.{i}.ssm_alpha.weight"))?,
                ssm_beta: view(&format!("blk.{i}.ssm_beta.weight"))?,
                // ssm_a / ssm_dt.bias are stored without the `.weight` suffix.
                ssm_a: view(&format!("blk.{i}.ssm_a"))?,
                ssm_dt_bias: view(&format!("blk.{i}.ssm_dt.bias"))?,
                ssm_conv1d: view(&format!("blk.{i}.ssm_conv1d.weight"))?,
                ssm_norm: view(&format!("blk.{i}.ssm_norm.weight"))?,
                ssm_out: view(&format!("blk.{i}.ssm_out.weight"))?,
                moe,
            })
        };
        blocks.push(block);
    }

    // MTP / NextN draft head — block index `n_main`. Loaded only when
    // speculative decoding was requested AND the checkpoint ships the
    // tensors. A missing tensor degrades gracefully to non-speculative
    // decode (mtp = None) rather than failing the whole load.
    let mtp = if spec_enabled && params.nextn_predict_layers >= 1 {
        let i = params.n_main;
        let load = || -> Result<MtpWeights, Box<dyn Error>> {
            let moe = load_moe(i)?;
            Ok(MtpWeights {
                enorm: view(&format!("blk.{i}.nextn.enorm.weight"))?,
                hnorm: view(&format!("blk.{i}.nextn.hnorm.weight"))?,
                eh_proj: view(&format!("blk.{i}.nextn.eh_proj.weight"))?,
                shared_head_norm: view(&format!("blk.{i}.nextn.shared_head_norm.weight"))?,
                body: load_attn(i, moe)?,
            })
        };
        match load() {
            Ok(w) => {
                if params.nextn_predict_layers > 1 {
                    tracing::warn!(
                        nextn_predict_layers = params.nextn_predict_layers,
                        "qwen35moe: only the first MTP layer is used; multi-MTP chains deferred",
                    );
                }
                Some(w)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "qwen35moe: MTP weights not loaded — falling back to non-speculative decode",
                );
                None
            }
        }
    } else {
        None
    };

    Ok(Qwen35MoeWeights {
        token_embd,
        output_norm,
        output,
        blocks,
        mtp,
    })
}

fn coerce_u32(v: &MetadataValue) -> Option<u32> {
    Some(match v {
        MetadataValue::U8(n) => *n as u32,
        MetadataValue::U16(n) => *n as u32,
        MetadataValue::U32(n) => *n,
        MetadataValue::U64(n) => *n as u32,
        MetadataValue::I8(n) if *n >= 0 => *n as u32,
        MetadataValue::I16(n) if *n >= 0 => *n as u32,
        MetadataValue::I32(n) if *n >= 0 => *n as u32,
        MetadataValue::I64(n) if *n >= 0 => *n as u32,
        _ => return None,
    })
}

/// Route to the right `matvec_*_id` dispatcher based on the expert
/// weight dtype. Qwen35MoE's gate / up tensors are mostly Q4_K but the
/// Q4_K_XL checkpoint upgrades blk.39 to Q5_K.
fn dispatch_matvec_id(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    ids: crate::inference::buffer::BufferRange,
    dst: TensorView,
    n_expert_used: u32,
) -> Result<(), Box<dyn Error>> {
    match a.dtype {
        GgmlType::Q4_K => moe::record_matvec_q4k_id(ctx, a, b, ids, dst, n_expert_used),
        GgmlType::Q5_K => moe::record_matvec_q5k_id(ctx, a, b, ids, dst, n_expert_used),
        GgmlType::Q6_K => moe::record_matvec_q6k_id(ctx, a, b, ids, dst, n_expert_used),
        other => Err(format!("matvec_id: expert weight dtype {other:?} not yet wired").into()),
    }
}

/// As [`dispatch_matvec_id`] but skips the trailing barrier — caller fences.
fn dispatch_matvec_id_nofence(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    ids: crate::inference::buffer::BufferRange,
    dst: TensorView,
    n_expert_used: u32,
) -> Result<(), Box<dyn Error>> {
    match a.dtype {
        GgmlType::Q4_K => moe::record_matvec_q4k_id_nofence(ctx, a, b, ids, dst, n_expert_used),
        GgmlType::Q5_K => moe::record_matvec_q5k_id_nofence(ctx, a, b, ids, dst, n_expert_used),
        GgmlType::Q6_K => moe::record_matvec_q6k_id_nofence(ctx, a, b, ids, dst, n_expert_used),
        other => Err(format!("matvec_id: expert weight dtype {other:?} not yet wired").into()),
    }
}

// ───────────────────────────────────────────────────────────────────────
// View / scratch helpers
// ───────────────────────────────────────────────────────────────────────

/// Host-write a `[L]` array of `u32` into a scratch slot via the mapped
/// pointer. Same pattern as `llama.rs::write_u32`.
fn write_u32(
    ctx: &mut DispatchContext,
    range: crate::inference::buffer::BufferRange,
    data: &[u32],
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch region not host-visible")?;
    unsafe {
        let dst = host_ptr.add(range.offset as usize) as *mut u32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
    Ok(())
}

/// Host-write a `[N]` array of `f32` into a scratch slot via the mapped
/// pointer. Used by the MTP draft path to upload the seed hidden state, and by
/// the Phase-3 image splice to upload the vision embeddings.
fn write_f32(
    ctx: &mut DispatchContext,
    range: crate::inference::buffer::BufferRange,
    data: &[f32],
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch region not host-visible")?;
    unsafe {
        let dst = host_ptr.add(range.offset as usize) as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
    Ok(())
}

/// Causal mask, F32, layout `[kv_len, L]`. Same logic as `llama.rs`.
fn write_causal_mask(
    ctx: &mut DispatchContext,
    mask: TensorView,
    l: u32,
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch region not host-visible")?;
    let l = l as usize;
    // Within-chunk causal mask only: [l × l] lower triangle (row-major
    // [l rows][l cols]). Query row i attends to within-chunk kv column jc iff
    // jc <= i. The cached prefix is always visible (handled shader-side via
    // mask_kv_offset), so it carries no entries — O(l²) instead of O(kv·l).
    let mut buf: Vec<f32> = vec![0.0; l * l];
    for i in 0..l {
        for jc in 0..l {
            buf[i * l + jc] = if jc <= i { 0.0 } else { f32::NEG_INFINITY };
        }
    }
    unsafe {
        let dst = host_ptr.add(mask.byte_offset as usize) as *mut f32;
        std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
    }
    Ok(())
}

/// View `q_full[2 * head_dim * n_head, L]` as a per-head slice — either
/// the first (Q) half or the second (sigmoid gate) half of each head's
/// 2*head_dim-wide region. Non-contiguous: stride between heads is
/// `2 * head_dim` so the view interleaves around the *other* half.
fn slice_q_half(q_full: TensorView, head_dim: u64, n_head: u64, l: u64, is_gate: bool) -> TensorView {
    let elem = q_full.byte_stride[0]; // F32 → 4
    let gate_offset = if is_gate { head_dim * elem } else { 0 };
    TensorView {
        buffer: q_full.buffer,
        byte_offset: q_full.byte_offset + gate_offset,
        byte_size: q_full.byte_size - gate_offset,
        dims: [head_dim, n_head, l, 1],
        byte_stride: [
            elem,
            2 * head_dim * elem,
            2 * head_dim * n_head * elem,
            2 * head_dim * n_head * l * elem,
        ],
        element_stride: [
            1,
            2 * head_dim,
            2 * head_dim * n_head,
            2 * head_dim * n_head * l,
        ],
        dtype: q_full.dtype,
    }
}

/// Reshape `[n_embd_kv, L]` into `[head_dim, n_head_kv, L]` (contiguous).
fn reshape_for_rope(t: TensorView, head_dim: u64, n_heads: u64, l: u64) -> TensorView {
    let elem = t.byte_stride[0];
    TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset,
        byte_size: t.byte_size,
        dims: [head_dim, n_heads, l, 1],
        byte_stride: [
            elem,
            elem * head_dim,
            elem * head_dim * n_heads,
            elem * head_dim * n_heads * l,
        ],
        element_stride: [
            1,
            head_dim,
            head_dim * n_heads,
            head_dim * n_heads * l,
        ],
        dtype: t.dtype,
    }
}

/// `[head_dim, n_heads, L]` → flash_attn input layout `[head_dim, L, n_heads]`
/// (non-contiguous view; same memory, different strides).
fn permute_to_attn(t: TensorView, head_dim: u64, l: u64, n_heads: u64) -> TensorView {
    let elem = t.byte_stride[0];
    TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset,
        byte_size: t.byte_size,
        dims: [head_dim, l, n_heads, 1],
        byte_stride: [
            elem,
            elem * head_dim * n_heads,
            elem * head_dim,
            elem * head_dim * n_heads * l,
        ],
        element_stride: [
            1,
            head_dim * n_heads,
            head_dim,
            head_dim * n_heads * l,
        ],
        dtype: t.dtype,
    }
}

fn coerce_f32(v: &MetadataValue) -> Option<f32> {
    Some(match v {
        MetadataValue::F32(x) => *x,
        MetadataValue::F64(x) => *x as f32,
        _ => return None,
    })
}

/// Placement of one image's merged tokens within a decoder token sequence
/// (Phase 3). The `<|image_pad|>` placeholders occupy `n_tok = nx*ny`
/// consecutive sequence slots starting at `start` (local index, i.e. relative
/// to the start of the tokens passed to `forward_impl`). `nx`/`ny` are the
/// **merged** grid dims (= patch grid / spatial_merge), so the vision encoder's
/// raster-ordered output token `k` lands at merged-grid `(col=k%nx, row=k/nx)`.
#[derive(Clone, Copy, Debug)]
pub struct ImageSpan {
    pub start: usize,
    pub n_tok: usize,
    pub nx: usize,
    pub ny: usize,
}

/// One chunk's image contribution to [`Qwen35MoeModel::forward_impl`]. Carries
/// the precomputed 4-axis M-RoPE positions for the chunk's tokens (built
/// globally so the image's 2D cursor is continuous across chunks) and, when any
/// image-pad tokens fall in this chunk, the vision-tower embedding columns to
/// splice + the chunk-local residual column where they start. `None` (text
/// path) builds positions internally and skips the splice — byte-identical to
/// the validated text-only forward. The caller advances `rope_position_lag`
/// once after the whole prefill (not per chunk), so `forward_impl` never touches
/// it: the global positions already encode the lag.
struct ForwardImage<'a> {
    /// `4 * tokens.len()` axis-major M-RoPE positions for this chunk's tokens.
    positions: &'a [u32],
    /// `(embeddings [n_embd, count], chunk-local start column)` for the image
    /// columns in this chunk, or `None` if none fall here.
    splice: Option<(&'a [f32], usize)>,
}

/// Build the 4-axis M-RoPE decoder positions for a token sequence that may
/// contain image spans, in the axis-major layout `forward_impl` uploads
/// (`out[axis*l + tok]`, read by the rope shader as `pos[tok + l*axis]`).
///
/// Exact port of llama's qwen-vl MROPE scheme (`mtmd.cpp
/// mtmd_image_tokens_get_decoder_pos` + `mtmd-helper.cpp set_position_mrope_2d`
/// / `set_position_normal`; advance from `mtmd_image_tokens_get_n_pos`):
///
/// * **Text** token at cursor `c`: all 4 axes = `c`; cursor advances by 1.
/// * **Image** at base `B` (= cursor when the span begins), token `k` in
///   `0..nx*ny`: `t=B`, `y(row)=B + k/nx`, `x(col)=B + k%nx`, `z=0`. The image
///   advances the cursor by `max(nx, ny)` (NOT `n_tok`): text after the image
///   resumes at `B + max(nx,ny)`.
///
/// `pos0` is the absolute position of the sequence's first token
/// (`position_offset`). `is_imrope=1` should accompany these (set by the
/// caller); for all-text sequences every axis is equal so imrope 0≡1.
pub fn build_decoder_mrope_positions(l: usize, pos0: u32, images: &[ImageSpan]) -> Vec<u32> {
    let mut out = vec![0u32; 4 * l];
    let mut cursor = pos0;
    let mut i = 0usize;
    while i < l {
        if let Some(img) = images.iter().find(|im| im.start == i) {
            debug_assert_eq!(img.n_tok, img.nx * img.ny, "image span n_tok != nx*ny");
            debug_assert!(i + img.n_tok <= l, "image span overruns sequence");
            let base = cursor;
            for k in 0..img.n_tok {
                let tok = i + k;
                out[tok] = base; // axis0 t
                out[l + tok] = base + (k / img.nx) as u32; // axis1 y (row)
                out[2 * l + tok] = base + (k % img.nx) as u32; // axis2 x (col)
                out[3 * l + tok] = 0; // axis3 z (unused; llama writes 0)
            }
            cursor = base + img.nx.max(img.ny) as u32;
            i += img.n_tok;
        } else {
            for axis in 0..4 {
                out[axis * l + i] = cursor;
            }
            cursor += 1;
            i += 1;
        }
    }
    out
}

/// Windowed variant of [`build_decoder_mrope_positions`] for **chunked** image
/// prefill: build the 4-axis M-RoPE positions for the global token window
/// `[window_start, window_start + window_len)` only, in chunk-local axis-major
/// layout (`out[axis*window_len + t]`, `t` the chunk-local index). `image`'s
/// `start` is a GLOBAL token index, so the image's 2D cursor stays continuous
/// across chunks and an image straddling a chunk boundary is handled correctly
/// (the formula is closed-form per token, not a left-to-right replay).
///
/// Single-image only (the current one-image-per-conversation scope) and assumes
/// every non-image token advances the cursor by 1 — identical to
/// `build_decoder_mrope_positions` for a `start..start+n` window, which the unit
/// tests assert. `pos0` is the rope base of global token 0.
pub fn build_decoder_mrope_positions_window(
    pos0: u32,
    image: Option<ImageSpan>,
    window_start: usize,
    window_len: usize,
) -> Vec<u32> {
    let mut out = vec![0u32; 4 * window_len];
    for t in 0..window_len {
        let j = window_start + t;
        let (a0, a1, a2, a3) = match image {
            // Inside the image span: token k shares t=base, with y/x the merged
            // grid row/col (raster order), z=0.
            Some(im) if j >= im.start && j < im.start + im.n_tok => {
                let k = j - im.start;
                let base = pos0 + im.start as u32;
                (base, base + (k / im.nx) as u32, base + (k % im.nx) as u32, 0)
            }
            // After the image: the cursor advanced by max(nx,ny) over the whole
            // image (not n_tok), then 1 per trailing text token.
            Some(im) if j >= im.start + im.n_tok => {
                let after = (j - (im.start + im.n_tok)) as u32;
                let c = pos0 + im.start as u32 + im.nx.max(im.ny) as u32 + after;
                (c, c, c, c)
            }
            // Before the image (or text-only): linear position on all axes.
            _ => {
                let c = pos0 + j as u32;
                (c, c, c, c)
            }
        };
        out[t] = a0;
        out[window_len + t] = a1;
        out[2 * window_len + t] = a2;
        out[3 * window_len + t] = a3;
    }
    out
}

#[cfg(test)]
mod vision_pos_tests {
    use super::{build_decoder_mrope_positions, build_decoder_mrope_positions_window, ImageSpan};

    /// The windowed builder must agree with the full builder for every window of
    /// a single-image sequence — including windows that start/end inside the
    /// image (chunk boundaries straddling the image). This is the invariant
    /// chunked prefill relies on.
    #[test]
    fn windowed_matches_full_for_all_splits() {
        let img = ImageSpan { start: 2, n_tok: 6, nx: 3, ny: 2 };
        let l = 2 + 6 + 3; // leading text + image + trailing text
        let pos0 = 7;
        let full = build_decoder_mrope_positions(l, pos0, &[img]);
        for ws in 0..=l {
            for we in ws..=l {
                let wl = we - ws;
                let win = build_decoder_mrope_positions_window(pos0, Some(img), ws, wl);
                for axis in 0..4 {
                    for t in 0..wl {
                        assert_eq!(
                            win[axis * wl + t],
                            full[axis * l + (ws + t)],
                            "mismatch axis {axis} window [{ws},{we}) tok {t}"
                        );
                    }
                }
            }
        }
        // Text-only window agrees too.
        let full_t = build_decoder_mrope_positions(l, pos0, &[]);
        let win_t = build_decoder_mrope_positions_window(pos0, None, 3, 4);
        for axis in 0..4 {
            for t in 0..4 {
                assert_eq!(win_t[axis * 4 + t], full_t[axis * l + (3 + t)]);
            }
        }
    }

    /// Extract the four per-token axis values at a local token index.
    fn at(p: &[u32], l: usize, tok: usize) -> (u32, u32, u32, u32) {
        (p[tok], p[l + tok], p[2 * l + tok], p[3 * l + tok])
    }

    /// Text-only: every axis is the sequential position (so imrope 0≡1).
    #[test]
    fn text_only_is_sequential_on_all_axes() {
        let l = 5;
        let p = build_decoder_mrope_positions(l, 0, &[]);
        for tok in 0..l {
            assert_eq!(at(&p, l, tok), (tok as u32, tok as u32, tok as u32, tok as u32));
        }
        // With a non-zero base offset, everything shifts.
        let p = build_decoder_mrope_positions(3, 100, &[]);
        assert_eq!(at(&p, 3, 0), (100, 100, 100, 100));
        assert_eq!(at(&p, 3, 2), (102, 102, 102, 102));
    }

    /// 2 text + (nx=3,ny=2) image + 2 text, base 0. Hand-derived from llama:
    /// image tokens share t=2; y = 2 + k/3; x = 2 + k%3; z = 0; the image
    /// advances the cursor by max(3,2)=3, so trailing text is at 5,6.
    #[test]
    fn single_image_3x2_hand_checked() {
        let l = 2 + 6 + 2;
        let img = ImageSpan { start: 2, n_tok: 6, nx: 3, ny: 2 };
        let p = build_decoder_mrope_positions(l, 0, &[img]);
        // leading text
        assert_eq!(at(&p, l, 0), (0, 0, 0, 0));
        assert_eq!(at(&p, l, 1), (1, 1, 1, 1));
        // image (t,y,x,z)
        assert_eq!(at(&p, l, 2), (2, 2, 2, 0)); // k=0
        assert_eq!(at(&p, l, 3), (2, 2, 3, 0)); // k=1
        assert_eq!(at(&p, l, 4), (2, 2, 4, 0)); // k=2
        assert_eq!(at(&p, l, 5), (2, 3, 2, 0)); // k=3
        assert_eq!(at(&p, l, 6), (2, 3, 3, 0)); // k=4
        assert_eq!(at(&p, l, 7), (2, 3, 4, 0)); // k=5
        // trailing text resumes at base(2) + max(nx,ny)(3) = 5
        assert_eq!(at(&p, l, 8), (5, 5, 5, 5));
        assert_eq!(at(&p, l, 9), (6, 6, 6, 6));
    }

    /// Tall image (ny>nx) advances by ny; non-zero base offset. The leading
    /// text token advances the cursor, so the image base is pos0+1.
    #[test]
    fn tall_image_advances_by_ny() {
        // 1 text + (nx=2,ny=3) image + 1 text, pos0=10. Image tokens are local
        // indices 1..=6 (k=0..5); index 7 is the trailing text.
        let l = 1 + 6 + 1;
        let img = ImageSpan { start: 1, n_tok: 6, nx: 2, ny: 3 };
        let p = build_decoder_mrope_positions(l, 10, &[img]);
        assert_eq!(at(&p, l, 0), (10, 10, 10, 10)); // text -> cursor advances to 11
        // image base = 11; k -> y=11+k/2, x=11+k%2, t=11, z=0
        assert_eq!(at(&p, l, 1), (11, 11, 11, 0)); // k0
        assert_eq!(at(&p, l, 2), (11, 11, 12, 0)); // k1
        assert_eq!(at(&p, l, 3), (11, 12, 11, 0)); // k2
        assert_eq!(at(&p, l, 4), (11, 12, 12, 0)); // k3
        assert_eq!(at(&p, l, 5), (11, 13, 11, 0)); // k4
        assert_eq!(at(&p, l, 6), (11, 13, 12, 0)); // k5
        // advance by max(2,3)=3 -> trailing text at 11+3 = 14
        assert_eq!(at(&p, l, 7), (14, 14, 14, 14));
    }

    /// The viz_test geometry: merged grid 6x5 (=30 tokens), preceded by one
    /// text token (so image base = 1). Verifies the realistic span and the
    /// +max(6,5)=6 advance.
    #[test]
    fn viz_test_geometry_6x5() {
        let l = 1 + 30 + 1;
        let img = ImageSpan { start: 1, n_tok: 30, nx: 6, ny: 5 };
        let p = build_decoder_mrope_positions(l, 0, &[img]);
        assert_eq!(at(&p, l, 0), (0, 0, 0, 0)); // leading text -> cursor 1
        assert_eq!(at(&p, l, 1), (1, 1, 1, 0)); // first image tok, base 1
        assert_eq!(at(&p, l, 7), (1, 2, 1, 0)); // k=6 -> row 1 col 0
        assert_eq!(at(&p, l, 30), (1, 5, 6, 0)); // k=29 -> row 4 col 5 (1+4, 1+5)
        assert_eq!(at(&p, l, 31), (7, 7, 7, 7)); // trailing text: 1 + max(6,5)
    }
}
