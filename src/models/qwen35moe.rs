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

pub struct Qwen35MoeWeights {
    pub token_embd: TensorView,
    pub output_norm: TensorView,
    /// `None` ⇒ tied weights: lm_head uses `token_embd`.
    pub output: Option<TensorView>,
    /// One entry per main-trunk block. MTP blocks are not stored here.
    pub blocks: Vec<BlockWeights>,
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
    ) -> Result<Self, Box<dyn Error>> {
        let params = parse_params(gguf, &handle)?;
        let weights = collect_weights(&handle, &params)?;
        tracing::info!(
            arch = ARCH,
            n_layer = params.n_layer,
            n_main = params.n_main,
            attention_layers = (0..params.n_main).filter(|&i| params.is_attention_layer(i)).count(),
            ssm_layers = (0..params.n_main).filter(|&i| !params.is_attention_layer(i)).count(),
            n_expert = params.n_expert,
            n_expert_used = params.n_expert_used,
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

    fn cache_dims(&self) -> CacheDims {
        // Only attention blocks need a KV cache, but the engine indexes the
        // cache by layer (one slot per `n_layer`). For Phase 1 we expose the
        // attention block dims uniformly and accept the unused-SSM-layer
        // waste; a later optimization can compact to attention-only slots.
        CacheDims {
            n_layer: self.params.n_main,
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
        })
    }

    fn weights(&self) -> &WeightsHandle {
        &self.handle
    }

    fn tokenizer(&self) -> &TokenizerBundle {
        &self.tokenizer
    }

    fn record_forward(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
    ) -> Result<TensorView, Box<dyn Error>> {
        let p = &self.params;
        let l = tokens.len() as u32;
        if l == 0 {
            return Err("empty prompt".into());
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
        let mut positions: Vec<u32> = Vec::with_capacity(4 * l as usize);
        for _axis in 0..4 {
            for pos in position_offset..position_offset + l {
                positions.push(pos);
            }
        }
        write_u32(ctx, positions_buf, &positions)?;

        // Single-token decode (l == 1) needs no mask: every KV slot is
        // causally visible, so flash_attn runs with MASK_ENABLE=0 and we skip
        // the O(total_len) host-side mask build per step. See llama.rs.
        let mask = if l > 1 {
            let m = ctx.alloc_tensor([kv_len_u, l as u64, 1, 1], GgmlType::F32)?;
            write_causal_mask(ctx, m, l, position_offset)?;
            Some(m)
        } else {
            None
        };

        let residual = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, l, residual)?;
        ctx.tap("input_embed", residual)?;

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
        let cache_direct = cache.config.k_dtype == cache.config.v_dtype;

        // ─── Per-layer loop ───
        // SEEKER_QWEN_MAX_LAYERS=N caps the loop at the first N layers —
        // helpful for bisecting the layer index at which a numeric issue
        // first appears.
        let max_layers = std::env::var("SEEKER_QWEN_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(self.weights.blocks.len());
        // When diff-dumping intermediates, each layer's taps must remain in
        // their own scratch slots until the GPU has executed all dispatches
        // and the host reads them back. Restoring scratch between layers
        // makes subsequent layers overwrite the tap data at the same byte
        // offsets — making every per-layer tap report the same value.
        let dump_mode = std::env::var("SEEKER_QWEN_DIFF_DUMP").is_ok();
        for (layer_idx, block) in self.weights.blocks.iter().take(max_layers).enumerate() {
            if !dump_mode {
                ctx.scratch_restore(layer_checkpoint);
            }

            // Attention-or-SSM "communication" step.
            //   SEEKER_QWEN_NO_ATTN=1 → skip only the 10 attention layers
            //   SEEKER_QWEN_NO_SSM=1  → skip only the 30 SSM layers
            // Used to bisect which block type contributes a bug.
            let skip_attn = std::env::var("SEEKER_QWEN_NO_ATTN").is_ok();
            let skip_ssm = std::env::var("SEEKER_QWEN_NO_SSM").is_ok();
            match block {
                BlockWeights::Attention(att) if !skip_attn => {
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
                    // Map block index to SSM-layer index (counting only SSM
                    // blocks). cache.ssm_gdn_states is indexed in SSM-layer
                    // order, not block order.
                    let ssm_layer_idx = (0..layer_idx)
                        .filter(|&i| !p.is_attention_layer(i as u32))
                        .count();
                    let gdn_state = cache.ssm_gdn_states.get(ssm_layer_idx).copied();
                    let conv_state = cache.ssm_conv_states.get(ssm_layer_idx).copied();
                    let ssm_host_ptr = cache.ssm_region.as_ref().and_then(|r| r.host_ptr);
                    ssm_block(ctx, ssm_w, residual, p, hidden, l, layer_idx as u32, gdn_state, conv_state, ssm_host_ptr)?;
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
            if std::env::var("SEEKER_QWEN_NO_MOE").is_err() {
                moe_ffn(ctx, block.moe(), block.post_attn_norm(), residual, p, hidden, l, layer_idx as u32)?;
            }
        }
        if !dump_mode {
            ctx.scratch_restore(layer_checkpoint);
        }

        // ─── Epilogue: final norm + lm_head ───
        // Only normalize + project the LAST token's residual: we sample
        // from the final position only, and full-batch logits would burn
        // n_vocab × L bytes of scratch (~318MB at L=320 with vocab=248k).
        // Slicing residual to the last token via a strided TensorView lets
        // both rms_norm and the lm_head matmul run with L=1.
        let elem_size = 4u64;
        let vocab = p.n_vocab as u64;
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

        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let last_logits = ctx.alloc_tensor([vocab, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, final_norm, last_logits)?;

        cache_io::advance(cache, l);
        Ok(last_logits)
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

    // 5. Per-head Q/K RMS norm. Independent pair — dispatch both nofence
    //    and emit one coalesced barrier so they can run concurrently.
    let q_normed = ctx.alloc_tensor([head_dim_k, n_head, l as u64, 1], GgmlType::F32)?;
    let k_normed = ctx.alloc_tensor([head_dim_k, n_head_kv, l as u64, 1], GgmlType::F32)?;
    rms_norm::record_nofence(ctx, q_attn_view, att.attn_q_norm, q_normed, p.rms_eps)?;
    rms_norm::record_nofence(ctx, k_view, att.attn_k_norm, k_normed, p.rms_eps)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_normed.range(), k_normed.range()],
    );
    ctx.tap(&format!("Qcur_normed-{layer_idx}"), q_normed)?;
    ctx.tap(&format!("Kcur_normed-{layer_idx}"), k_normed)?;

    // 6. M-RoPE on Q and K.
    let q_roped = ctx.alloc_tensor([head_dim_k, n_head, l as u64, 1], GgmlType::F32)?;
    rope_multi::record_nofence(ctx, q_normed, positions_buf, q_roped, rope_params)?;
    let k_roped = ctx.alloc_tensor([head_dim_k, n_head_kv, l as u64, 1], GgmlType::F32)?;
    rope_multi::record_nofence(ctx, k_normed, positions_buf, k_roped, rope_params)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[q_roped.range(), k_roped.range()],
    );
    ctx.tap(&format!("Qcur-{layer_idx}"), q_roped)?;
    ctx.tap(&format!("Kcur-{layer_idx}"), k_roped)?;

    // 7a. cache write K, V. The cache expects `[head_dim, n_head_kv, L]`
    //     natural layout; that's exactly what `k_roped` / `v_view` have.
    cache_io::record_write_nofence(ctx, k_roped, cache.k_layers[layer_idx as usize], position_offset)?;
    cache_io::record_write(ctx, v_view, cache.v_layers[layer_idx as usize], position_offset)?;

    // 7b. cache read (direct bind for matching K/V dtypes; materialize
    //     otherwise — same logic as the LLaMA path).
    let (k_src, v_src) = if cache_direct {
        (
            slice_cache_prefix(cache.k_layers[layer_idx as usize], kv_len_u),
            slice_cache_prefix(cache.v_layers[layer_idx as usize], kv_len_u),
        )
    } else {
        (
            cache_io::record_read(ctx, cache.k_layers[layer_idx as usize], total_len)?,
            cache_io::record_read(ctx, cache.v_layers[layer_idx as usize], total_len)?,
        )
    };

    // 7c. flash_attn — permute Q to [hd_k, L, n_head], K/V to
    //     [hd_kv, kv_len, n_head_kv]. Output is `[hidden_v, L]`.
    let q_perm = permute_to_attn(q_roped, head_dim_k, l as u64, n_head);
    let k_perm = permute_to_attn(k_src, head_dim_k, kv_len_u, n_head_kv);
    let v_perm = permute_to_attn(v_src, head_dim_v, kv_len_u, n_head_kv);
    let attn_out = ctx.alloc_tensor([hidden_v, l as u64, 1, 1], GgmlType::F32)?;
    flash_attn::record(ctx, q_perm, k_perm, v_perm, mask, attn_out, fa_params)?;
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

    // 9. wo @ attn_gated  →  proj [hidden, L];  residual += proj
    let proj = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, att.wo, attn_gated, proj)?;
    ctx.tap(&format!("attn_output-{layer_idx}"), proj)?;
    elementwise::record_add(ctx, residual, proj, residual)?;
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
    if std::env::var("SEEKER_QWEN_ONLY_RMS").is_ok() {
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

    // Save the last (conv_kernel-1) tokens of conv_input as the new
    // persistent conv state, for the next forward to read. Mirrors
    // llama.cpp's `conv_state_last` view at offset `s_idx = L`.
    if let Some(persistent) = conv_state_persistent {
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
    if std::env::var("SEEKER_QWEN_NO_CONV").is_ok() {
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
    let gdn_total_floats = attn_floats + state_floats;
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
    // first few SSM layers. SEEKER_QWEN_GDN_SCALE=one bypasses for testing.
    let gdn_scale = match std::env::var("SEEKER_QWEN_GDN_SCALE").as_deref() {
        Ok("one") => 1.0,
        _ => 1.0 / (s_v as f32).sqrt(),
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
    )?;
    let _ = (q_strides,); // q_normed_strides supersedes q_strides above

    // Copy new GDN state from gdn_dst's state region back to the persistent
    // state buffer. gdn_dst layout: [attn_floats outputs][state_floats state].
    // After this copy, subsequent forwards' GDN read picks up where this one
    // left off — needed for autoregressive decode quality.
    if let Some(persistent) = gdn_state_persistent {
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

    // ssm_out @ gated_attn → projection contribution [n_embd, L]
    let proj = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, ssm_w.ssm_out, gated_attn, proj)?;
    ctx.tap(&format!("attn_output-{layer_idx}"), proj)?;

    // Diagnostic: SEEKER_QWEN_SSM_DUMP=<stage> redirects the residual
    // contribution to a specific intermediate value (filling residual
    // with that tensor's contents via cast, instead of adding `proj`).
    // The lm_head then operates on the chosen intermediate, so the
    // dumped logits effectively contain the intermediate's data.
    //
    // Valid stages: alpha, beta, qkv, conv, gdn_attn, attn_normed,
    // gated, proj. Falls through to normal `residual += proj` otherwise.
    let dump = std::env::var("SEEKER_QWEN_SSM_DUMP").ok();
    if let Some(stage) = dump.as_deref() {
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

    // Router logits.
    let gate_logits = ctx.alloc_tensor([n_experts as u64, l_u, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, w.ffn_gate_inp, x_norm, gate_logits)?;
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

    // Fused routing-weighted down. The Q4_K_XL checkpoint mixes Q5_K
    // and Q6_K for `ffn_down_exps` — dispatch on dtype.
    let routed = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    match w.ffn_down_exps.dtype {
        GgmlType::Q5_K => {
            moe::record_moe_down_q5k(ctx, w.ffn_down_exps, ffn_h, ids, weights_buf, routed, n_used)?;
        }
        GgmlType::Q6_K => {
            moe::record_moe_down_q6k(ctx, w.ffn_down_exps, ffn_h, ids, weights_buf, routed, n_used)?;
        }
        other => {
            return Err(
                format!("qwen35moe: ffn_down_exps dtype {other:?} not supported (need Q5_K or Q6_K)")
                    .into(),
            );
        }
    }
    ctx.tap(&format!("ffn_moe_out-{layer_idx}"), routed)?;

    // Shared expert: standard FFN with `ffn_{gate,up,down}_shexp`.
    let sgate = ctx.alloc_tensor([shexp_ff, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, w.ffn_gate_shexp, x_norm, sgate)?;
    let sup = ctx.alloc_tensor([shexp_ff, l_u, 1, 1], GgmlType::F32)?;
    matmul::record_nofence(ctx, w.ffn_up_shexp, x_norm, sup)?;
    crate::inference::command::record_compute_barriers(
        ctx.device,
        ctx.cmd,
        &[sgate.range(), sup.range()],
    );
    let sh = ctx.alloc_tensor([shexp_ff, l_u, 1, 1], GgmlType::F32)?;
    elementwise::record_swiglu_split(ctx, sgate, sup, sh)?;
    let shared = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, w.ffn_down_shexp, sh, shared)?;

    // Shared-expert sigmoid scalar gate per token. Matmul layout is
    // a=[K=n_embd, M=1] @ b=[K=n_embd, N=L] → d=[M=1, N=L]. Allocating
    // d as [L, 1] would make `record_inner`'s per-column fallback for
    // N>1 write past the buffer end (one float per col into a slot
    // sized for L floats), corrupting downstream tensors.
    let shared_gate_pre = ctx.alloc_tensor([1, l_u, 1, 1], GgmlType::F32)?;
    matmul::record(ctx, w.ffn_gate_inp_shexp, x_norm, shared_gate_pre)?;
    let shared_gate_sig = ctx.alloc_tensor([1, l_u, 1, 1], GgmlType::F32)?;
    elementwise::record_sigmoid(ctx, shared_gate_pre, shared_gate_sig)?;
    // Broadcast scalar gate across hidden dim — view it as [1, L] and
    // rely on the element-wise mul shader's stride handling. The view
    // dims/strides need to match `shared` for the mul to broadcast
    // correctly: shape `[hidden, L]` with stride[0]=0 (broadcast) on the
    // gate side.
    let shared_gate_broadcast = TensorView {
        dims: [hidden, l_u, 1, 1],
        byte_stride: [0, 4, 4 * l_u, 4 * l_u],
        element_stride: [0, 1, l_u, l_u],
        byte_size: shared_gate_sig.byte_size,
        ..shared_gate_sig
    };
    let shared_scaled = ctx.alloc_tensor([hidden, l_u, 1, 1], GgmlType::F32)?;
    elementwise::record_mul(ctx, shared, shared_gate_broadcast, shared_scaled)?;

    // residual += routed (+ shared_scaled)
    // SEEKER_QWEN_NO_SHARED=1 disables the shared-expert add — bisects
    // between the broadcast/sigmoid gate path and the routed-experts path.
    // SEEKER_QWEN_NO_ROUTED=1 disables the routed-expert add — bisects
    // the other direction; combined with NO_SHARED it makes moe_ffn a
    // no-op so the forward becomes "embed → final_norm → lm_head".
    if std::env::var("SEEKER_QWEN_NO_ROUTED").is_err() {
        elementwise::record_add(ctx, residual, routed, residual)?;
    }
    if std::env::var("SEEKER_QWEN_NO_SHARED").is_err() {
        elementwise::record_add(ctx, residual, shared_scaled, residual)?;
    }
    let _ = shared_scaled;
    let _ = routed;
    ctx.tap(&format!("l_out-{layer_idx}"), residual)?;

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
) -> Result<Qwen35MoeWeights, Box<dyn Error>> {
    let view = |name: &str| -> Result<TensorView, Box<dyn Error>> {
        handle
            .view(name)
            .map_err(|_| ModelError::MissingTensor(name.to_string()).into())
    };

    let token_embd = view("token_embd.weight")?;
    let output_norm = view("output_norm.weight")?;
    let output = handle.view("output.weight").ok();

    let mut blocks = Vec::with_capacity(params.n_main as usize);
    for i in 0..params.n_main {
        let moe = MoeFfnWeights {
            ffn_gate_inp: view(&format!("blk.{i}.ffn_gate_inp.weight"))?,
            ffn_gate_inp_shexp: view(&format!("blk.{i}.ffn_gate_inp_shexp.weight"))?,
            ffn_gate_exps: view(&format!("blk.{i}.ffn_gate_exps.weight"))?,
            ffn_up_exps: view(&format!("blk.{i}.ffn_up_exps.weight"))?,
            ffn_down_exps: view(&format!("blk.{i}.ffn_down_exps.weight"))?,
            ffn_gate_shexp: view(&format!("blk.{i}.ffn_gate_shexp.weight"))?,
            ffn_up_shexp: view(&format!("blk.{i}.ffn_up_shexp.weight"))?,
            ffn_down_shexp: view(&format!("blk.{i}.ffn_down_shexp.weight"))?,
        };
        let block = if params.is_attention_layer(i) {
            BlockWeights::Attention(AttentionBlockWeights {
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

    Ok(Qwen35MoeWeights {
        token_embd,
        output_norm,
        output,
        blocks,
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

/// Causal mask, F32, layout `[kv_len, L]`. Same logic as `llama.rs`.
fn write_causal_mask(
    ctx: &mut DispatchContext,
    mask: TensorView,
    l: u32,
    position_offset: u32,
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch region not host-visible")?;
    let l = l as usize;
    let pos = position_offset as usize;
    let kv_len = pos + l;
    let mut buf: Vec<f32> = vec![0.0; l * kv_len];
    for i in 0..l {
        for j in 0..kv_len {
            buf[i * kv_len + j] = if j <= pos + i { 0.0 } else { f32::NEG_INFINITY };
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

/// Restrict a cache-layer view to the first `total_len` token slots.
fn slice_cache_prefix(layer: TensorView, total_len: u64) -> TensorView {
    let mut dims = layer.dims;
    dims[2] = total_len;
    let byte_size = layer.byte_stride[2] * total_len;
    TensorView {
        dims,
        byte_size,
        ..layer
    }
}

fn coerce_f32(v: &MetadataValue) -> Option<f32> {
    Some(match v {
        MetadataValue::F32(x) => *x,
        MetadataValue::F64(x) => *x as f32,
        _ => return None,
    })
}
