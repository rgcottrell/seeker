//! Gemma 4 (`general.architecture == "gemma4"`) — text decoder. A dense GQA
//! transformer with several Gemma-specific twists vs LLaMA, all matching
//! llama.cpp's `build_gemma4` (`src/models/gemma4.cpp`) and vllm's `gemma4.py`:
//!
//!   * **Hybrid per-layer attention**: a repeating `5×sliding-window + 1 global`
//!     pattern (from `sliding_window_pattern`). Sliding layers use head_dim 256,
//!     n_head_kv 8, rope θ=1e4; global layers use head_dim 512, n_head_kv 1,
//!     rope θ=1e6, and reuse the K projection for V (no `attn_v`). KV dims
//!     therefore vary per layer (→ [`Model::cache_per_layer_dims`]).
//!   * **Attention scale = 1.0** (NOT 1/√head_dim) — the learned per-head Q/K
//!     RMSNorm absorbs the magnitude.
//!   * **Per-head Q-norm + K-norm** (RMSNorm over head_dim, learned weight)
//!     before RoPE; **V-norm** is a *weightless* RMSNorm (no weight, no RoPE).
//!   * **NEOX RoPE** with full rotation (n_rot == head_dim); global layers apply
//!     `rope_freqs` freq-factors (high pairs → no rotation).
//!   * **Sandwich norms**: input→attn→post_attention_norm→+res; then
//!     ffn_norm→GeGLU(gelu-tanh)→post_ffw_norm→+res.
//!   * Embedding × √n_embd; per-layer `× layer_output_scale`; tied lm_head;
//!     final-logit softcap `cap·tanh(x/cap)`.
//!   * RMSNorm `(1+w)` is baked into the GGUF weights → a plain `rms_norm(x)·w`
//!     is correct (no runtime +1).
//!
//! First cut: single-sequence [`Model::record_forward`] (run / chat / bench /
//! probe). Sliding-window layers apply an analytical in-shader window (the
//! flash-attn `swa_window` slot); global layers attend the full context. Vision
//! input is spliced into the residual after the embedding scale via
//! [`Model::record_forward_image_chunk`] (linear positions, no M-RoPE).

use std::error::Error;

use crate::gguf::{GgmlType, GgufFile, MetadataValue};
use crate::inference::context::DispatchContext;
use crate::inference::kv_cache::KvCache;
use crate::inference::ops::{cache_io, elementwise, flash_attn, matmul, rms_norm, rope};
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::tokenizer::TokenizerBundle;

use super::{CacheDims, Model, ModelError};

const ARCH: &str = "gemma4";

#[derive(Debug, Clone)]
pub struct Gemma4Params {
    pub n_layer: u32,
    pub n_head: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub rms_eps: f32,
    pub embd_scale: f32,
    pub final_logit_softcap: f32,
    pub sliding_window: u32,
    // Sliding-window head config (n_rot == head_dim).
    pub head_dim_swa: u32,
    pub rope_base_swa: f32,
    // Global head config.
    pub head_dim_global: u32,
    pub rope_base_global: f32,
    // Per-layer: is layer `il` a sliding-window layer? (else global)
    pub swa: Vec<bool>,
    // Per-layer query-head KV count (8 for SWA, 1 for global here).
    pub n_head_kv: Vec<u32>,
    // Per-layer output scalar (`cur *= layer_output_scale[il]`), 1.0 if absent.
    pub layer_output_scale: Vec<f32>,
}

impl Gemma4Params {
    fn head_dim(&self, il: usize) -> u32 {
        if self.swa[il] {
            self.head_dim_swa
        } else {
            self.head_dim_global
        }
    }
    fn rope_base(&self, il: usize) -> f32 {
        if self.swa[il] {
            self.rope_base_swa
        } else {
            self.rope_base_global
        }
    }
    /// Q projection width = n_head · head_dim (≠ n_embd in gemma).
    fn q_dim(&self, il: usize) -> u32 {
        self.n_head * self.head_dim(il)
    }
    /// K/V projection width = n_head_kv · head_dim.
    fn kv_dim(&self, il: usize) -> u32 {
        self.n_head_kv[il] * self.head_dim(il)
    }
}

pub struct Gemma4BlockWeights {
    pub attn_norm: TensorView,
    pub wq: TensorView,
    pub wk: TensorView,
    /// `None` on global layers (V reuses the K projection).
    pub wv: Option<TensorView>,
    pub wo: TensorView,
    pub attn_q_norm: TensorView,
    pub attn_k_norm: TensorView,
    pub post_attn_norm: TensorView,
    pub ffn_norm: TensorView,
    pub ffn_gate: TensorView,
    pub ffn_up: TensorView,
    pub ffn_down: TensorView,
    pub post_ffw_norm: TensorView,
}

pub struct Gemma4Weights {
    pub token_embd: TensorView,
    pub blocks: Vec<Gemma4BlockWeights>,
    pub output_norm: TensorView,
    /// `None` ⇒ tied lm_head (uses `token_embd`).
    pub output: Option<TensorView>,
    /// Single `[head_dim_global/2]` freq-factor tensor for global layers.
    pub rope_freqs: Option<TensorView>,
}

pub struct Gemma4Model {
    pub params: Gemma4Params,
    pub weights: Gemma4Weights,
    pub handle: WeightsHandle,
    pub tokenizer: TokenizerBundle,
    /// Optional gemma4-assistant MTP draft (separate GGUF), attached via
    /// [`Model::attach_mtp_draft`] for speculative decoding.
    pub mtp_draft: Option<super::gemma4_assistant::Gemma4AssistantDraft>,
}

impl Gemma4Model {
    pub fn new(
        gguf: &GgufFile,
        handle: WeightsHandle,
        tokenizer: TokenizerBundle,
    ) -> Result<Self, Box<dyn Error>> {
        let params = parse_params(gguf)?;
        let weights = collect_weights(&handle, &params)?;
        Ok(Self {
            params,
            weights,
            handle,
            tokenizer,
            mtp_draft: None,
        })
    }

    /// Index of the base layer whose K/V the draft's sliding-window blocks
    /// borrow (the LAST sliding-window layer) and global blocks borrow (the
    /// LAST global layer) — per HF Gemma4Assistant's shared-KV design.
    fn mtp_kv_layer(&self, want_swa: bool) -> Option<usize> {
        (0..self.params.n_layer as usize)
            .rev()
            .find(|&il| self.params.swa[il] == want_swa)
    }
}

impl Model for Gemma4Model {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn vocab_size(&self) -> u32 {
        self.params.n_vocab
    }

    fn cache_dims(&self) -> CacheDims {
        // Representative (max) scalars for any uniform consumer; the real
        // per-layer dims come from `cache_per_layer_dims`.
        CacheDims {
            n_layer: self.params.n_layer,
            head_dim: self.params.head_dim_global.max(self.params.head_dim_swa),
            n_head_kv: self.params.n_head_kv.iter().copied().max().unwrap_or(1),
            n_head: self.params.n_head,
        }
    }

    fn cache_per_layer_dims(&self) -> Option<(Vec<u32>, Vec<u32>)> {
        let head_dims: Vec<u32> = (0..self.params.n_layer as usize)
            .map(|il| self.params.head_dim(il))
            .collect();
        Some((head_dims, self.params.n_head_kv.clone()))
    }

    fn weights(&self) -> &WeightsHandle {
        &self.handle
    }

    fn tokenizer(&self) -> &TokenizerBundle {
        &self.tokenizer
    }

    fn scratch_bytes_estimate(
        &self,
        n_ubatch: u32,
        max_seq_len: u32,
        _k_dtype: GgmlType,
        _v_dtype: GgmlType,
    ) -> u64 {
        let p = &self.params;
        let l = if n_ubatch == 0 {
            max_seq_len.max(1)
        } else {
            n_ubatch
        } as u64;
        let n_embd = p.n_embd as u64;
        let n_ff = p.n_ff as u64;
        let vocab = p.n_vocab as u64;
        let q_dim_max = (p.n_head * p.head_dim_global.max(p.head_dim_swa)) as u64;
        let kv_dim_max = p
            .n_head_kv
            .iter()
            .enumerate()
            .map(|(il, _)| p.kv_dim(il) as u64)
            .max()
            .unwrap_or(0);
        // Per-layer transient buffers (reclaimed each layer): residual-width
        // norms (x_norm, proj, proj_normed, x_norm2, down, down_normed),
        // Q-width (q, q_normed, q_roped, attn_out), KV-width (k, k_normed,
        // k_roped, v, v_normed), and the FFN width (gate, up, gate_gelu, hidden).
        let per_layer = (8 * n_embd + 4 * q_dim_max + 6 * kv_dim_max + 5 * n_ff) * l * 4;
        let residual = n_embd * l * 4;
        let mask = l * l * 4;
        // Single-seq epilogue computes one [vocab, 1] column, but batched serve
        // decode (`record_forward_batch`) allocates [vocab, B]. Budget 16
        // columns (2× the default `--parallel-max` 8) so the 256K-entry vocab
        // doesn't lean on the slack margin at high `--parallel`. The per-layer
        // [_, B] transients are covered many times over by the `l = n_ubatch`
        // sizing (B ≤ slots ≪ n_ubatch).
        let logits = vocab * 4 * 16;
        // Flash-attn prefill split-K partials (deepest split this context can
        // produce); mirrors llama.rs's accounting with the max head/q dims.
        let fa_walk = 8192u64;
        let fa_partials = if max_seq_len as u64 > fa_walk {
            let fa_k_num = (max_seq_len as u64).div_ceil(fa_walk);
            (p.head_dim_global.max(p.head_dim_swa) as u64 + 2) * l * p.n_head as u64 * fa_k_num * 4
        } else {
            0
        };
        let raw = per_layer + residual + mask + logits + fa_partials;
        raw + raw / 3 + (32 << 20)
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
            .forward_inner(
                ctx,
                cache,
                tokens,
                position_offset,
                compute_logits,
                false,
                None,
            )?
            .0)
    }

    fn supports_batch_decode(&self) -> bool {
        true
    }

    /// Batched decode: B sequences, one token each, in one forward. The dense
    /// ops (embedding+scale, norms, matmuls, RoPE, GeGLU FFN, per-layer output
    /// scale) process the `B`-wide token dimension exactly as the single-seq
    /// layer loop processes `L`; only the attention is per-sequence (own KV
    /// slab + length via `flash_attn::record_batched`, sliding-window applied
    /// analytically in-shader from each sequence's `kv_len`) and the K/V
    /// writes fan out one column per sequence into its slab. Per-layer dims
    /// (head_dim / n_head_kv / SWA window / rope base / freq-factors) vary as
    /// in [`Self::forward_inner`].
    fn record_forward_batch(
        &self,
        ctx: &mut DispatchContext,
        batch: &mut crate::inference::kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        use crate::inference::command::record_compute_barriers;
        let p = &self.params;
        let b = tokens.len() as u64;
        if b == 0 {
            return Err("record_forward_batch: empty batch".into());
        }
        if positions.len() != tokens.len() || slots.len() != tokens.len() {
            return Err("record_forward_batch: tokens/positions/slots length mismatch".into());
        }
        let hidden = p.n_embd as u64;
        let n_ff = p.n_ff as u64;
        let n_head = p.n_head as u64;
        let vocab = p.n_vocab as u64;
        let elem = 4u64;
        // Each sequence attends over [0, position + 1).
        let kv_lens: Vec<u32> = positions.iter().map(|&pos| pos + 1).collect();

        // ---- prologue: B token ids + B positions (one per sequence) ----
        let token_buf = ctx.alloc_scratch(b * 4)?;
        write_u32(ctx, token_buf, tokens)?;
        let positions_buf = ctx.alloc_scratch(b * 4)?;
        write_u32(ctx, positions_buf, positions)?;

        // Embedding lookup + ×√n_embd scale (text tokens only reach decode).
        let residual = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, b as u32, residual)?;
        elementwise::record_scale(ctx, residual, residual, p.embd_scale, 0.0)?;
        // Persistent per-forward DecodeDyn array for batched flash-attn — must
        // live above `layer_checkpoint` so per-layer `scratch_restore` cannot
        // reclaim it (the shader reads `kv_len` at execute time, after submit).
        let fa_dyn_range = crate::inference::decode_dyn::alloc_array(ctx, b as u32)?;
        let layer_checkpoint = ctx.scratch_checkpoint();

        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_checkpoint);

            let head_dim = p.head_dim(layer_idx) as u64;
            let n_head_kv = p.n_head_kv[layer_idx] as u64;
            let q_dim = p.q_dim(layer_idx) as u64;
            let kv_dim = p.kv_dim(layer_idx) as u64;
            let n_rot = head_dim as u32; // full rotation
            let rope_params = rope::RopeParams::llama_default(n_rot, p.rope_base(layer_idx));
            let freq_factors = if p.swa[layer_idx] {
                None
            } else {
                self.weights.rope_freqs.as_ref().map(|t| t.range())
            };
            let fa_params = flash_attn::FlashAttnParams {
                head_dim_k: head_dim as u32,
                head_dim_v: head_dim as u32,
                gqa_ratio: (n_head / n_head_kv).max(1) as u32,
                scale: 1.0, // gemma4: NO 1/sqrt(head_dim)
                // Sliding layers attend only to the most recent `sliding_window`
                // keys (analytic in-shader mask off each sequence's own kv_len);
                // global layers (swa[il]==false) use full causal (0).
                swa_window: if p.swa[layer_idx] {
                    p.sliding_window
                } else {
                    0
                },
            };

            // input norm
            let x_norm = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.attn_norm, x_norm, p.rms_eps)?;

            // Q/K/V projections (V only on SWA layers; global reuses K).
            let q = ctx.alloc_tensor([q_dim, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wq, x_norm, q)?;
            let k = ctx.alloc_tensor([kv_dim, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wk, x_norm, k)?;
            let v_proj = if let Some(wv) = block.wv {
                let v = ctx.alloc_tensor([kv_dim, b, 1, 1], GgmlType::F32)?;
                matmul::record_nofence(ctx, wv, x_norm, v)?;
                record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range(), v.range()]);
                v
            } else {
                record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range()]);
                k // global layer: V := K projection
            };

            // Per-head Q-norm / K-norm (over head_dim), then NEOX RoPE.
            let q_view = reshape_for_rope(q, head_dim, n_head, b);
            let q_normed = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            rms_norm::record_nofence(ctx, q_view, block.attn_q_norm, q_normed, p.rms_eps)?;
            let k_view = reshape_for_rope(k, head_dim, n_head_kv, b);
            let k_normed = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rms_norm::record_nofence(ctx, k_view, block.attn_k_norm, k_normed, p.rms_eps)?;
            // V-norm: weightless RMSNorm (no weight, no rope).
            let v_view = reshape_for_rope(v_proj, head_dim, n_head_kv, b);
            let v_normed = ctx.alloc_tensor(v_view.dims, GgmlType::F32)?;
            rms_norm::record_noweight_nofence(ctx, v_view, v_normed, p.rms_eps)?;
            record_compute_barriers(
                ctx.device,
                ctx.cmd,
                &[q_normed.range(), k_normed.range(), v_normed.range()],
            );

            let q_roped = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            let k_roped = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rope::record_neox_nofence(
                ctx,
                q_normed,
                positions_buf,
                q_roped,
                rope_params,
                freq_factors,
            )?;
            rope::record_neox_nofence(
                ctx,
                k_normed,
                positions_buf,
                k_roped,
                rope_params,
                freq_factors,
            )?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q_roped.range(), k_roped.range()]);

            // Per-sequence K/V cache writes (K post-rope, V normed): column s →
            // slot s's slab at its position, packed at this layer's natural
            // per-token stride (record_write derives it from the SOURCE dims).
            let k_natural = reshape_for_rope(k_roped, head_dim, n_head_kv, b);
            let v_natural = reshape_for_rope(v_normed, head_dim, n_head_kv, b);
            let col_stride = kv_dim * elem; // bytes between sequence columns
            for s in 0..tokens.len() {
                let k_col = column_view(k_natural, s as u64, col_stride, head_dim, n_head_kv);
                let v_col = column_view(v_natural, s as u64, col_stride, head_dim, n_head_kv);
                cache_io::record_write(
                    ctx,
                    k_col,
                    batch.slot_k_view(slots[s], layer_idx as u32),
                    positions[s],
                )?;
                cache_io::record_write(
                    ctx,
                    v_col,
                    batch.slot_v_view(slots[s], layer_idx as u32),
                    positions[s],
                )?;
            }

            // Batched attention: each sequence attends to its own slab
            // (slots[s]) and length; the K/V views bind all slabs, the flash
            // picks per sequence via DecodeDyn::slot. The `_dims` variants
            // carry this layer's real head_dim/n_head_kv (the slab packing) —
            // the uniform views stride at the allocation max, wrong for the
            // hybrid SWA/global dims.
            let q_attn = batched_q_attn_view(q_roped, head_dim, n_head, b, q_dim);
            let k_attn =
                batch.batched_k_attn_view_dims(layer_idx as u32, head_dim as u32, n_head_kv as u32);
            let v_attn =
                batch.batched_v_attn_view_dims(layer_idx as u32, head_dim as u32, n_head_kv as u32);
            let attn_out = ctx.alloc_tensor([q_dim, b, 1, 1], GgmlType::F32)?;
            flash_attn::record_batched(
                ctx,
                q_attn,
                k_attn,
                v_attn,
                attn_out,
                fa_params,
                &kv_lens,
                fa_dyn_range,
                Some(slots),
                /*query_lens=*/ None,
            )?;

            // O-proj → post_attention_norm → residual add.
            let proj = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wo, attn_out, proj)?;
            let proj_normed = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, proj, block.post_attn_norm, proj_normed, p.rms_eps)?;
            elementwise::record_add(ctx, residual, proj_normed, residual)?;

            // FFN: ffn_norm → GeGLU(gelu-tanh) → post_ffw_norm → residual add.
            let x_norm2 = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.ffn_norm, x_norm2, p.rms_eps)?;
            let gate = ctx.alloc_tensor([n_ff, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_gate, x_norm2, gate)?;
            let up = ctx.alloc_tensor([n_ff, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_up, x_norm2, up)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[gate.range(), up.range()]);
            let gate_gelu = ctx.alloc_tensor([n_ff, b, 1, 1], GgmlType::F32)?;
            elementwise::record_gelu(ctx, gate, gate_gelu)?;
            let ffn_hidden = ctx.alloc_tensor([n_ff, b, 1, 1], GgmlType::F32)?;
            elementwise::record_mul(ctx, gate_gelu, up, ffn_hidden)?;
            let down = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_down, ffn_hidden, down)?;
            let down_normed = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, down, block.post_ffw_norm, down_normed, p.rms_eps)?;
            elementwise::record_add(ctx, residual, down_normed, residual)?;

            // Per-layer output scalar (× layer_output_scale[il]).
            let scale = p.layer_output_scale[layer_idx];
            if scale != 1.0 {
                elementwise::record_scale(ctx, residual, residual, scale, 0.0)?;
            }
        }
        ctx.scratch_restore(layer_checkpoint);

        // Final norm + lm_head over ALL B columns (each column is its
        // sequence's last — and only — token this step), then the
        // final-logit softcap cap·tanh(x/cap) in place.
        let final_norm = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
        rms_norm::record(
            ctx,
            residual,
            self.weights.output_norm,
            final_norm,
            p.rms_eps,
        )?;
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let logits = ctx.alloc_tensor([vocab, b, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, final_norm, logits)?;
        if p.final_logit_softcap != 0.0 {
            let cap = p.final_logit_softcap;
            elementwise::record_scale(ctx, logits, logits, 1.0 / cap, 0.0)?;
            elementwise::record_tanh(ctx, logits, logits)?;
            elementwise::record_scale(ctx, logits, logits, cap, 0.0)?;
        }

        for (s, &pos) in positions.iter().enumerate() {
            batch.positions[slots[s] as usize] = pos + 1;
        }
        Ok(logits)
    }

    /// gemma4 image tokens use plain sequential 1D positions (no M-RoPE), so the
    /// engine must not apply a `rope_position_lag` after an image.
    fn image_uses_mrope(&self) -> bool {
        false
    }

    /// Prefill one chunk of an image-containing prompt: the `<|image>`-pad
    /// placeholder tokens in the chunk are replaced (post-embedding) by the
    /// vision encoder's output columns. Positions stay sequential.
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
        _prompt_pos0: u32,
        compute_logits: bool,
    ) -> Result<Option<TensorView>, Box<dyn Error>> {
        let n_embd = self.params.n_embd as usize;
        let n_tok = image_nx * image_ny;
        if image_embeddings.len() != n_embd * n_tok {
            return Err(format!(
                "gemma4 image embeddings len {} != n_embd {n_embd} * n_tok {n_tok} \
                 (vision projection_dim must equal text embedding_length)",
                image_embeddings.len()
            )
            .into());
        }
        let l = chunk_tokens.len();
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
            .forward_inner(
                ctx,
                cache,
                chunk_tokens,
                cache.position,
                compute_logits,
                false,
                splice,
            )?
            .0)
    }

    // ─── gemma4 MTP speculative-decode hooks (gemma4-assistant draft) ───

    fn supports_mtp_spec(&self) -> bool {
        self.mtp_draft.is_some()
    }

    fn attach_mtp_draft(
        &mut self,
        gguf: &GgufFile,
        handle: WeightsHandle,
    ) -> Result<(), Box<dyn Error>> {
        let draft = super::gemma4_assistant::Gemma4AssistantDraft::load(
            gguf,
            handle,
            self.params.n_embd,
            self.params.n_vocab,
        )?;
        self.mtp_draft = Some(draft);
        Ok(())
    }

    fn record_forward_full(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        full_logits: bool,
        _checkpoint: bool, // gemma4 has no SSM state → no per-position snapshots
    ) -> Result<crate::models::ForwardFullOut, Box<dyn Error>> {
        let (logits, residual) =
            self.forward_inner(ctx, cache, tokens, position_offset, true, full_logits, None)?;
        Ok(crate::models::ForwardFullOut { logits, residual })
    }

    fn record_mtp_seed(
        &self,
        _ctx: &mut DispatchContext,
        _cache: &mut KvCache,
        _hiddens: &[f32],
        _tokens: &[u32],
        _position_offset: u32,
    ) -> Result<(), Box<dyn Error>> {
        // The gemma4-assistant draft has no KV cache of its own — it
        // cross-attends to the base model's K/V — so there is nothing to seed.
        Ok(())
    }

    fn record_mtp_draft(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        h_last: &[f32],
        prev_token: u32,
        rel_pos: u32,
    ) -> Result<crate::models::MtpDraftOut, Box<dyn Error>> {
        self.mtp_draft_step(ctx, cache, h_last, prev_token, rel_pos)
    }
}

impl Gemma4Model {
    /// The shared forward body for text (`image_splice = None`) and image-chunk
    /// (`Some((columns, local_col_start))` overwrites that run of residual
    /// columns with the unscaled vision embeddings) prefill / decode.
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        compute_logits: bool,
        full_logits: bool,
        image_splice: Option<(&[f32], usize)>,
    ) -> Result<(Option<TensorView>, TensorView), Box<dyn Error>> {
        use crate::inference::command::{record_compute_barriers, record_global_barrier};
        let p = &self.params;
        let l = tokens.len() as u32;
        if l == 0 {
            return Err("empty prompt".into());
        }
        let hidden = p.n_embd as u64;
        let n_ff = p.n_ff as u64;

        if cache.position != position_offset {
            return Err(format!(
                "cache.position {} doesn't match caller-supplied position_offset {position_offset}",
                cache.position
            )
            .into());
        }
        let total_len = position_offset + l;
        let kv_len_u = total_len as u64;

        // ---- prologue: positions + mask + token id buffer ----
        let token_buf = ctx.alloc_scratch((l as u64) * 4)?;
        write_u32(ctx, token_buf, tokens)?;
        let positions_buf = ctx.alloc_scratch((l as u64) * 4)?;
        let positions: Vec<u32> = (position_offset..position_offset + l).collect();
        write_u32(ctx, positions_buf, &positions)?;

        let cache_direct = flash_attn::supports_pair(cache.config.k_dtype, cache.config.v_dtype);

        // Within-chunk causal mask (prefix is always-visible, shader-side).
        // NOTE: sliding-window layers attend over the whole prefix here — correct
        // only while total_len ≤ sliding_window. TODO: windowed mask for longer.
        let mask = if l > 1 {
            let m = ctx.alloc_tensor([l as u64, l as u64, 1, 1], GgmlType::F32)?;
            write_causal_mask(ctx, m, l)?;
            Some(m)
        } else {
            None
        };

        // ---- embedding lookup + ×√n_embd scale ----
        let residual = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, l, residual)?;
        elementwise::record_scale(ctx, residual, residual, p.embd_scale, 0.0)?;

        // Image splice: overwrite the `<|image>`-pad placeholder columns with the
        // (unscaled) vision embeddings. gemma scales only text-token embeddings
        // by √n_embd, so the overwrite replaces the just-scaled placeholder
        // values with the raw decoder-space vision columns.
        if let Some((sub, local_start)) = image_splice {
            let src = ctx.alloc_scratch(sub.len() as u64 * 4)?;
            write_f32(ctx, src, sub)?;
            record_global_barrier(ctx.device, ctx.cmd);
            unsafe {
                let copy = ash::vk::BufferCopy::default()
                    .src_offset(src.offset)
                    .dst_offset(residual.byte_offset + (local_start as u64) * hidden * 4)
                    .size(sub.len() as u64 * 4);
                ctx.device.device.cmd_copy_buffer(
                    ctx.cmd,
                    src.buffer,
                    residual.buffer,
                    std::slice::from_ref(&copy),
                );
            }
            record_global_barrier(ctx.device, ctx.cmd);
        }
        let layer_checkpoint = ctx.scratch_checkpoint();

        // ---- per-layer loop ----
        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_checkpoint);

            let head_dim = p.head_dim(layer_idx) as u64;
            let n_head = p.n_head as u64;
            let n_head_kv = p.n_head_kv[layer_idx] as u64;
            let q_dim = p.q_dim(layer_idx) as u64;
            let kv_dim = p.kv_dim(layer_idx) as u64;
            let n_rot = head_dim as u32; // full rotation
            let rope_params = rope::RopeParams::llama_default(n_rot, p.rope_base(layer_idx));
            let freq_factors = if p.swa[layer_idx] {
                None
            } else {
                self.weights.rope_freqs.as_ref().map(|t| t.range())
            };
            let fa_params = flash_attn::FlashAttnParams {
                head_dim_k: head_dim as u32,
                head_dim_v: head_dim as u32,
                gqa_ratio: (n_head / n_head_kv).max(1) as u32,
                scale: 1.0, // gemma4: NO 1/sqrt(head_dim)
                // Sliding layers attend only to the most recent `sliding_window`
                // keys; global layers (swa[il]==false) use full causal (0).
                swa_window: if p.swa[layer_idx] {
                    p.sliding_window
                } else {
                    0
                },
            };

            // input norm
            let x_norm = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.attn_norm, x_norm, p.rms_eps)?;

            // Q/K/V projections (V only on SWA layers; global reuses K).
            let q = ctx.alloc_tensor([q_dim, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wq, x_norm, q)?;
            let k = ctx.alloc_tensor([kv_dim, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wk, x_norm, k)?;
            let v_proj = if let Some(wv) = block.wv {
                let v = ctx.alloc_tensor([kv_dim, l as u64, 1, 1], GgmlType::F32)?;
                matmul::record_nofence(ctx, wv, x_norm, v)?;
                record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range(), v.range()]);
                v
            } else {
                record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range()]);
                k // global layer: V := K projection
            };

            // Per-head Q-norm / K-norm (over head_dim), then NEOX RoPE.
            let q_view = reshape_for_rope(q, head_dim, n_head, l as u64);
            let q_normed = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            rms_norm::record_nofence(ctx, q_view, block.attn_q_norm, q_normed, p.rms_eps)?;
            let k_view = reshape_for_rope(k, head_dim, n_head_kv, l as u64);
            let k_normed = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rms_norm::record_nofence(ctx, k_view, block.attn_k_norm, k_normed, p.rms_eps)?;
            // V-norm: weightless RMSNorm (no weight, no rope).
            let v_view = reshape_for_rope(v_proj, head_dim, n_head_kv, l as u64);
            let v_normed = ctx.alloc_tensor(v_view.dims, GgmlType::F32)?;
            rms_norm::record_noweight_nofence(ctx, v_view, v_normed, p.rms_eps)?;
            record_compute_barriers(
                ctx.device,
                ctx.cmd,
                &[q_normed.range(), k_normed.range(), v_normed.range()],
            );

            let q_roped = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            let k_roped = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rope::record_neox_nofence(
                ctx,
                q_normed,
                positions_buf,
                q_roped,
                rope_params,
                freq_factors,
            )?;
            rope::record_neox_nofence(
                ctx,
                k_normed,
                positions_buf,
                k_roped,
                rope_params,
                freq_factors,
            )?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q_roped.range(), k_roped.range()]);

            // KV cache write (K post-rope, V normed) in natural layout.
            let k_natural = reshape_for_rope(k_roped, head_dim, n_head_kv, l as u64);
            let v_natural = reshape_for_rope(v_normed, head_dim, n_head_kv, l as u64);
            cache_io::record_write_nofence(
                ctx,
                k_natural,
                cache.k_layers[layer_idx],
                position_offset,
            )?;
            cache_io::record_write(ctx, v_natural, cache.v_layers[layer_idx], position_offset)?;

            let (k_src, v_src) = if cache_direct {
                (
                    slice_cache_prefix(cache.k_layers[layer_idx], kv_len_u),
                    slice_cache_prefix(cache.v_layers[layer_idx], kv_len_u),
                )
            } else {
                (
                    cache_io::record_read(ctx, cache.k_layers[layer_idx], total_len)?,
                    cache_io::record_read(ctx, cache.v_layers[layer_idx], total_len)?,
                )
            };

            let q_perm = permute_to_attn(q_roped, head_dim, l as u64, n_head);
            let k_perm = permute_to_attn(k_src, head_dim, kv_len_u, n_head_kv);
            let v_perm = permute_to_attn(v_src, head_dim, kv_len_u, n_head_kv);

            let attn_out = ctx.alloc_tensor([q_dim, l as u64, 1, 1], GgmlType::F32)?;
            flash_attn::record(
                ctx, q_perm, k_perm, v_perm, mask, attn_out, fa_params, total_len,
            )?;

            // O-proj → post_attention_norm → residual add.
            let proj = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wo, attn_out, proj)?;
            let proj_normed = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, proj, block.post_attn_norm, proj_normed, p.rms_eps)?;
            elementwise::record_add(ctx, residual, proj_normed, residual)?;

            // FFN: ffn_norm → GeGLU(gelu-tanh) → post_ffw_norm → residual add.
            let x_norm2 = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.ffn_norm, x_norm2, p.rms_eps)?;
            let gate = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_gate, x_norm2, gate)?;
            let up = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_up, x_norm2, up)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[gate.range(), up.range()]);
            let gate_gelu = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_gelu(ctx, gate, gate_gelu)?;
            let ffn_hidden = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_mul(ctx, gate_gelu, up, ffn_hidden)?;
            let down = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_down, ffn_hidden, down)?;
            let down_normed = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, down, block.post_ffw_norm, down_normed, p.rms_eps)?;
            elementwise::record_add(ctx, residual, down_normed, residual)?;

            // Per-layer output scalar (× layer_output_scale[il]).
            let scale = p.layer_output_scale[layer_idx];
            if scale != 1.0 {
                elementwise::record_scale(ctx, residual, residual, scale, 0.0)?;
            }
        }
        ctx.scratch_restore(layer_checkpoint);

        if !compute_logits {
            cache_io::advance(cache, l);
            return Ok((None, residual));
        }

        // ---- final norm + lm_head + final-logit softcap ----
        // `full_logits` (MTP verify) computes logits for ALL L positions
        // (`[vocab, L]`); otherwise just the last token (`[vocab, 1]`).
        let elem_size = 4u64;
        let vocab = p.n_vocab as u64;
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let logits = if full_logits {
            let final_norm = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(
                ctx,
                residual,
                self.weights.output_norm,
                final_norm,
                p.rms_eps,
            )?;
            let lg = ctx.alloc_tensor([vocab, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, lm_head, final_norm, lg)?;
            lg
        } else {
            let residual_last = TensorView {
                buffer: residual.buffer,
                byte_offset: residual.byte_offset + (l as u64 - 1) * hidden * elem_size,
                byte_size: hidden * elem_size,
                dims: [hidden, 1, 1, 1],
                byte_stride: [
                    elem_size,
                    hidden * elem_size,
                    hidden * elem_size,
                    hidden * elem_size,
                ],
                element_stride: [1, hidden, hidden, hidden],
                dtype: residual.dtype,
            };
            let final_norm = ctx.alloc_tensor([hidden, 1, 1, 1], GgmlType::F32)?;
            rms_norm::record(
                ctx,
                residual_last,
                self.weights.output_norm,
                final_norm,
                p.rms_eps,
            )?;
            let last_logits = ctx.alloc_tensor([vocab, 1, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, lm_head, final_norm, last_logits)?;
            last_logits
        };

        // final_logit_softcap: cap·tanh(logits/cap), in place (any shape).
        if p.final_logit_softcap != 0.0 {
            let cap = p.final_logit_softcap;
            elementwise::record_scale(ctx, logits, logits, 1.0 / cap, 0.0)?;
            elementwise::record_tanh(ctx, logits, logits)?;
            elementwise::record_scale(ctx, logits, logits, cap, 0.0)?;
        }

        cache_io::advance(cache, l);
        Ok((Some(logits), residual))
    }

    /// One gemma4-assistant MTP draft step (`L=1`): given the base model's last
    /// pre-final-norm hidden `h_last` (`[n_embd]`) and the previous token, run
    /// the 4-block EAGLE draft — input `pre_projection(concat(√n_embd·embed(tok),
    /// h_last))`, blocks whose attention computes Q only and cross-attends the
    /// BASE K/V cache (last SWA / last global layer), then `post_projection` (the
    /// recurrence hidden) and the dense tied head (the draft's own `token_embd`).
    /// Returns the GPU-argmax draft token + the `[n_embd]` recurrence hidden.
    fn mtp_draft_step(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        h_last: &[f32],
        prev_token: u32,
        rel_pos: u32,
    ) -> Result<crate::models::MtpDraftOut, Box<dyn Error>> {
        use crate::inference::command::{record_compute_barrier, record_compute_barriers};
        let p = &self.params;
        let draft = self
            .mtp_draft
            .as_ref()
            .ok_or("record_mtp_draft: no gemma4-assistant draft attached")?;
        let dp = &draft.params;
        let dw = &draft.weights;
        let n_embd = p.n_embd as u64; // base hidden = draft output width (3840)
        let d_hidden = dp.n_embd as u64; // draft hidden (1024)
        let n_ff = dp.n_ff as u64;
        let n_head = dp.n_head as u64;
        let vocab = p.n_vocab as u64;
        if h_last.len() as u64 != n_embd {
            return Err(format!(
                "record_mtp_draft: h_last len {} != base n_embd {n_embd}",
                h_last.len()
            )
            .into());
        }
        // The draft cross-attends the committed base K/V over [0, kv_len).
        let kv_len = cache.position;
        let kv_len_u = kv_len as u64;
        if kv_len == 0 {
            return Err("record_mtp_draft: empty base KV cache".into());
        }
        let cache_direct = flash_attn::supports_pair(cache.config.k_dtype, cache.config.v_dtype);

        // ── input: inp_cat = [ √n_embd·embed(prev_token) | h_last ]  (2·n_embd) ──
        let inp_cat = ctx.alloc_tensor([2 * n_embd, 1, 1, 1], GgmlType::F32)?;
        let half = |off_elems: u64| -> TensorView {
            TensorView {
                buffer: inp_cat.buffer,
                byte_offset: inp_cat.byte_offset + off_elems * 4,
                byte_size: n_embd * 4,
                dims: [n_embd, 1, 1, 1],
                byte_stride: [4, n_embd * 4, n_embd * 4, n_embd * 4],
                element_stride: [1, n_embd, n_embd, n_embd],
                dtype: GgmlType::F32,
            }
        };
        let tok_buf = ctx.alloc_scratch(4)?;
        write_u32(ctx, tok_buf, &[prev_token])?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, tok_buf, 1, half(0))?;
        elementwise::record_scale(ctx, half(0), half(0), p.embd_scale, 0.0)?;
        write_f32(ctx, half(n_embd).range(), h_last)?;
        record_compute_barrier(ctx.device, ctx.cmd, inp_cat.range());

        // inpL = pre_projection @ inp_cat → draft hidden. Stable buffer, updated
        // in place by each block's output (L=1 → per-block temps stay tiny).
        let inpl = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, dw.pre_projection, inp_cat, inpl)?;

        let positions_buf = ctx.alloc_scratch(4)?;
        write_u32(ctx, positions_buf, &[rel_pos])?;

        // ── 4 draft blocks: Q-only attention over the borrowed base K/V ──
        for (il, block) in dw.blocks.iter().enumerate() {
            let head_dim = dp.head_dim(il) as u64;
            let q_dim = dp.q_dim(il) as u64;
            let n_rot = head_dim as u32;
            let rope_params = rope::RopeParams::llama_default(n_rot, dp.rope_base(il));
            let freq_factors = if dp.swa[il] {
                None
            } else {
                dw.rope_freqs.as_ref().map(|t| t.range())
            };
            // Borrow the base layer's K/V of the matching attention type.
            let il_kv = self
                .mtp_kv_layer(dp.swa[il])
                .ok_or("gemma4 MTP: base has no layer of the draft's attention type")?;
            let base_n_head_kv = p.n_head_kv[il_kv] as u64;
            let fa_params = flash_attn::FlashAttnParams {
                head_dim_k: head_dim as u32,
                head_dim_v: head_dim as u32,
                gqa_ratio: (n_head / base_n_head_kv).max(1) as u32,
                scale: 1.0, // gemma4: NO 1/sqrt(head_dim)
                swa_window: if dp.swa[il] { dp.sliding_window } else { 0 },
            };

            // input norm → Q proj → per-head Q-norm → NEOX RoPE.
            let x_norm = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, inpl, block.attn_norm, x_norm, dp.rms_eps)?;
            let q = ctx.alloc_tensor([q_dim, 1, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wq, x_norm, q)?;
            let q_view = reshape_for_rope(q, head_dim, n_head, 1);
            let q_normed = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            rms_norm::record_nofence(ctx, q_view, block.attn_q_norm, q_normed, dp.rms_eps)?;
            record_compute_barrier(ctx.device, ctx.cmd, q_normed.range());
            let q_roped = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            rope::record_neox_nofence(
                ctx,
                q_normed,
                positions_buf,
                q_roped,
                rope_params,
                freq_factors,
            )?;
            record_compute_barrier(ctx.device, ctx.cmd, q_roped.range());

            // K/V read-only from the base cache (no draft KV write).
            let (k_src, v_src) = if cache_direct {
                (
                    slice_cache_prefix(cache.k_layers[il_kv], kv_len_u),
                    slice_cache_prefix(cache.v_layers[il_kv], kv_len_u),
                )
            } else {
                (
                    cache_io::record_read(ctx, cache.k_layers[il_kv], kv_len)?,
                    cache_io::record_read(ctx, cache.v_layers[il_kv], kv_len)?,
                )
            };
            let q_perm = permute_to_attn(q_roped, head_dim, 1, n_head);
            let k_perm = permute_to_attn(k_src, head_dim, kv_len_u, base_n_head_kv);
            let v_perm = permute_to_attn(v_src, head_dim, kv_len_u, base_n_head_kv);
            let attn = ctx.alloc_tensor([q_dim, 1, 1, 1], GgmlType::F32)?;
            flash_attn::record(ctx, q_perm, k_perm, v_perm, None, attn, fa_params, kv_len)?;

            // O-proj → post_attn_norm → residual (attn_out = inpL + ·).
            let proj = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wo, attn, proj)?;
            let proj_normed = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, proj, block.post_attn_norm, proj_normed, dp.rms_eps)?;
            let attn_out = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
            elementwise::record_add(ctx, inpl, proj_normed, attn_out)?;

            // FFN: ffn_norm → GeGLU(gelu-tanh) → post_ffw_norm → residual.
            let x_norm2 = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, attn_out, block.ffn_norm, x_norm2, dp.rms_eps)?;
            let gate = ctx.alloc_tensor([n_ff, 1, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_gate, x_norm2, gate)?;
            let up = ctx.alloc_tensor([n_ff, 1, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_up, x_norm2, up)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[gate.range(), up.range()]);
            let gate_gelu = ctx.alloc_tensor([n_ff, 1, 1, 1], GgmlType::F32)?;
            elementwise::record_gelu(ctx, gate, gate_gelu)?;
            let ffn_hidden = ctx.alloc_tensor([n_ff, 1, 1, 1], GgmlType::F32)?;
            elementwise::record_mul(ctx, gate_gelu, up, ffn_hidden)?;
            let down = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_down, ffn_hidden, down)?;
            let down_normed = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, down, block.post_ffw_norm, down_normed, dp.rms_eps)?;
            // inpL = attn_out + down_normed (in place into the stable buffer).
            elementwise::record_add(ctx, attn_out, down_normed, inpl)?;
            let scale = dp.layer_output_scale[il];
            if scale != 1.0 {
                elementwise::record_scale(ctx, inpl, inpl, scale, 0.0)?;
            }
        }

        // ── output_norm → recurrence (post_projection) + dense tied LM head ──
        let h_inner = ctx.alloc_tensor([d_hidden, 1, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, inpl, dw.output_norm, h_inner, dp.rms_eps)?;
        // Recurrence hidden for the next step (base hidden space, n_embd).
        let h_post = ctx.alloc_tensor([n_embd, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, dw.post_projection, h_inner, h_post)?;
        // Dense head: the draft's own token_embd (d_hidden → vocab). Final-logit
        // softcap is monotonic and the draft only argmaxes → omitted (invariant).
        let logits = ctx.alloc_tensor([vocab, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, dw.token_embd, h_inner, logits)?;
        let draft_token = crate::inference::ops::sampler::record_greedy(ctx, logits)?;

        Ok(crate::models::MtpDraftOut {
            draft_token,
            block_out: h_post,
        })
    }
}

// ── view helpers (same as llama.rs; gemma's dims just vary per layer) ──

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
        element_stride: [1, head_dim, head_dim * n_heads, head_dim * n_heads * l],
        dtype: t.dtype,
    }
}

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
        element_stride: [1, head_dim * n_heads, head_dim, head_dim * n_heads * l],
        dtype: t.dtype,
    }
}

/// A single-column `[head_dim, n_head_kv, 1]` view of a `[head_dim, n_head_kv,
/// B]` tensor (sequence `s`), for the per-sequence KV cache write.
fn column_view(
    t: TensorView,
    s: u64,
    col_stride: u64,
    head_dim: u64,
    n_head_kv: u64,
) -> TensorView {
    let elem = t.byte_stride[0];
    TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset + s * col_stride,
        byte_size: head_dim * n_head_kv * elem,
        dims: [head_dim, n_head_kv, 1, 1],
        byte_stride: [
            elem,
            elem * head_dim,
            elem * head_dim * n_head_kv,
            elem * head_dim * n_head_kv,
        ],
        element_stride: [1, head_dim, head_dim * n_head_kv, head_dim * n_head_kv],
        dtype: t.dtype,
    }
}

/// Reinterpret a contiguous `[head_dim, n_head, B]` (post-RoPE) Q as the
/// `[head_dim, 1, n_head, B]` flash-attn batched-decode layout (one query row
/// per head per sequence; batch stride = the layer's q_dim, which in gemma is
/// n_head·head_dim ≠ n_embd).
fn batched_q_attn_view(
    t: TensorView,
    head_dim: u64,
    n_head: u64,
    b: u64,
    q_dim: u64,
) -> TensorView {
    let elem = t.byte_stride[0];
    TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset,
        byte_size: t.byte_size,
        dims: [head_dim, 1, n_head, b],
        byte_stride: [elem, elem * head_dim, elem * head_dim, elem * q_dim],
        element_stride: [1, head_dim, head_dim, q_dim],
        dtype: t.dtype,
    }
}

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

fn parse_params(gguf: &GgufFile) -> Result<Gemma4Params, Box<dyn Error>> {
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
    let u32_or = |k: &'static str, d: u32| -> u32 { gguf.get(k).and_then(coerce_u32).unwrap_or(d) };
    let f32_or = |k: &'static str, d: f32| -> f32 { gguf.get(k).and_then(coerce_f32).unwrap_or(d) };

    let n_layer = u32_key("gemma4.block_count")?;
    let n_head = u32_key("gemma4.attention.head_count")?;
    let n_embd = u32_key("gemma4.embedding_length")?;
    let n_ff = u32_key("gemma4.feed_forward_length")?;
    let n_ctx_train = u32_or("gemma4.context_length", 131072);
    let rms_eps = f32_or("gemma4.attention.layer_norm_rms_epsilon", 1e-6);
    let final_logit_softcap = f32_or("gemma4.final_logit_softcapping", 0.0);
    let sliding_window = u32_or("gemma4.attention.sliding_window", 0);

    let head_dim_global = u32_or("gemma4.attention.key_length", n_embd / n_head);
    let head_dim_swa = u32_or("gemma4.attention.key_length_swa", head_dim_global);
    let rope_base_global = f32_or("gemma4.rope.freq_base", 1_000_000.0);
    let rope_base_swa = f32_or("gemma4.rope.freq_base_swa", 10_000.0);

    let n_vocab = u32_key("gemma4.vocab_size").or_else(|_| derive_vocab(gguf))?;

    // Per-layer arrays.
    let swa = read_bool_array(gguf, "gemma4.attention.sliding_window_pattern", n_layer)?;
    let n_head_kv = read_u32_array(gguf, "gemma4.attention.head_count_kv", n_layer)?;

    // Per-layer output scalar (optional; default 1.0).
    let layer_output_scale: Vec<f32> = (0..n_layer)
        .map(|i| {
            read_scalar_f32(gguf, &format!("blk.{i}.layer_output_scale.weight")).unwrap_or(1.0)
        })
        .collect();

    Ok(Gemma4Params {
        n_layer,
        n_head,
        n_embd,
        n_ff,
        n_vocab,
        n_ctx_train,
        rms_eps,
        embd_scale: (n_embd as f32).sqrt(),
        final_logit_softcap,
        sliding_window,
        head_dim_swa,
        rope_base_swa,
        head_dim_global,
        rope_base_global,
        swa,
        n_head_kv,
        layer_output_scale,
    })
}

fn derive_vocab(gguf: &GgufFile) -> Result<u32, Box<dyn Error>> {
    // Fall back to the token_embd tensor's row count when the metadata key
    // is absent (gemma4 GGUFs omit gemma4.vocab_size).
    gguf.tensor("token_embd.weight")
        .map(|t| t.dims[1] as u32)
        .ok_or_else(|| ModelError::MissingMetadata("gemma4.vocab_size").into())
}

fn collect_weights(
    handle: &WeightsHandle,
    params: &Gemma4Params,
) -> Result<Gemma4Weights, Box<dyn Error>> {
    let view = |name: &str| -> Result<TensorView, Box<dyn Error>> {
        handle
            .view(name)
            .map_err(|_| ModelError::MissingTensor(name.to_string()).into())
    };

    let token_embd = view("token_embd.weight")?;
    let output_norm = view("output_norm.weight")?;
    let output = handle.view("output.weight").ok();
    let rope_freqs = handle.view("rope_freqs.weight").ok();

    let mut blocks = Vec::with_capacity(params.n_layer as usize);
    for i in 0..params.n_layer {
        blocks.push(Gemma4BlockWeights {
            attn_norm: view(&format!("blk.{i}.attn_norm.weight"))?,
            wq: view(&format!("blk.{i}.attn_q.weight"))?,
            wk: view(&format!("blk.{i}.attn_k.weight"))?,
            wv: handle.view(&format!("blk.{i}.attn_v.weight")).ok(),
            wo: view(&format!("blk.{i}.attn_output.weight"))?,
            attn_q_norm: view(&format!("blk.{i}.attn_q_norm.weight"))?,
            attn_k_norm: view(&format!("blk.{i}.attn_k_norm.weight"))?,
            post_attn_norm: view(&format!("blk.{i}.post_attention_norm.weight"))?,
            ffn_norm: view(&format!("blk.{i}.ffn_norm.weight"))?,
            ffn_gate: view(&format!("blk.{i}.ffn_gate.weight"))?,
            ffn_up: view(&format!("blk.{i}.ffn_up.weight"))?,
            ffn_down: view(&format!("blk.{i}.ffn_down.weight"))?,
            post_ffw_norm: view(&format!("blk.{i}.post_ffw_norm.weight"))?,
        });
    }

    Ok(Gemma4Weights {
        token_embd,
        blocks,
        output_norm,
        output,
        rope_freqs,
    })
}

pub(crate) fn read_u32_array(
    gguf: &GgufFile,
    key: &'static str,
    n_layer: u32,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let v = gguf.get(key).ok_or(ModelError::MissingMetadata(key))?;
    let items = match v {
        MetadataValue::Array(items) => items,
        other => {
            return Err(ModelError::BadMetadata {
                key,
                detail: format!("expected array, got {other:?}"),
            }
            .into());
        }
    };
    if items.len() != n_layer as usize {
        return Err(ModelError::BadMetadata {
            key,
            detail: format!("expected {n_layer} entries, got {}", items.len()),
        }
        .into());
    }
    items
        .iter()
        .map(|it| {
            coerce_u32(it).ok_or_else(|| {
                ModelError::BadMetadata {
                    key,
                    detail: format!("non-u32 entry {it:?}"),
                }
                .into()
            })
        })
        .collect()
}

pub(crate) fn read_bool_array(
    gguf: &GgufFile,
    key: &'static str,
    n_layer: u32,
) -> Result<Vec<bool>, Box<dyn Error>> {
    let v = gguf.get(key).ok_or(ModelError::MissingMetadata(key))?;
    let items = match v {
        MetadataValue::Array(items) => items,
        other => {
            return Err(ModelError::BadMetadata {
                key,
                detail: format!("expected array, got {other:?}"),
            }
            .into());
        }
    };
    if items.len() != n_layer as usize {
        return Err(ModelError::BadMetadata {
            key,
            detail: format!("expected {n_layer} entries, got {}", items.len()),
        }
        .into());
    }
    items
        .iter()
        .map(|it| match it {
            MetadataValue::Bool(b) => Ok(*b),
            other => Err(ModelError::BadMetadata {
                key,
                detail: format!("non-bool entry {other:?}"),
            }
            .into()),
        })
        .collect()
}

/// Read a single F32 (or F16) scalar weight directly from the GGUF mmap (small
/// per-layer scalars like `layer_output_scale` that we want as host constants).
pub(crate) fn read_scalar_f32(gguf: &GgufFile, name: &str) -> Option<f32> {
    let info = gguf.tensor(name)?;
    let bytes = gguf.tensor_data(name)?;
    match info.ggml_type {
        GgmlType::F32 if bytes.len() >= 4 => {
            Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        GgmlType::F16 if bytes.len() >= 2 => {
            Some(f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        }
        _ => None,
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        (mant as f32 / 1024.0) * 2f32.powi(-14)
    } else if exp == 31 {
        f32::INFINITY
    } else {
        (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

pub(crate) fn coerce_u32(v: &MetadataValue) -> Option<u32> {
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

pub(crate) fn coerce_f32(v: &MetadataValue) -> Option<f32> {
    Some(match v {
        MetadataValue::F32(x) => *x,
        MetadataValue::F64(x) => *x as f32,
        _ => return None,
    })
}
