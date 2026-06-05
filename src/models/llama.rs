//! LLaMA-architecture model. Loads architecture parameters and per-layer
//! weight handles from a GGUF, then implements [`Model::record_forward`]
//! against the inference dispatch primitives.
//!
//! Forward pass — exact op sequence per llama.cpp's `build_llama` in
//! `src/llama-graph.cpp`. MVP: single forward pass, no KV cache, F16
//! weights, F32 activations throughout.

use std::error::Error;

use crate::gguf::{GgmlType, GgufFile, MetadataValue};
use crate::inference::context::DispatchContext;
use crate::inference::kv_cache::{KvCache, is_turbo};
use crate::inference::ops::turbo_wht::WhtDir;
use crate::inference::ops::{cache_io, elementwise, flash_attn, matmul, rms_norm, rope, turbo_wht};
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::tokenizer::TokenizerBundle;

use super::{CacheDims, Model, ModelError};

#[derive(Debug, Clone)]
pub struct LlamaParams {
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub rope_dim: u32,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
}

impl LlamaParams {
    pub fn head_dim(&self) -> u32 {
        self.n_embd / self.n_head
    }
    pub fn n_embd_kv(&self) -> u32 {
        self.head_dim() * self.n_head_kv
    }
}

pub struct LlamaBlockWeights {
    pub attn_norm: TensorView,
    pub wq: TensorView,
    pub wk: TensorView,
    pub wv: TensorView,
    pub wo: TensorView,
    pub ffn_norm: TensorView,
    pub ffn_gate: TensorView,
    pub ffn_up: TensorView,
    pub ffn_down: TensorView,
}

pub struct LlamaWeights {
    pub token_embd: TensorView,
    pub blocks: Vec<LlamaBlockWeights>,
    pub output_norm: TensorView,
    /// `None` ⇒ tied weights: lm_head uses `token_embd`.
    pub output: Option<TensorView>,
}

pub struct LlamaModel {
    pub params: LlamaParams,
    pub weights: LlamaWeights,
    pub handle: WeightsHandle,
    #[allow(dead_code)]
    pub tokenizer: TokenizerBundle,
}

impl LlamaModel {
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
        })
    }
}

impl Model for LlamaModel {
    fn arch(&self) -> &'static str {
        "llama"
    }

    fn vocab_size(&self) -> u32 {
        self.params.n_vocab
    }

    fn cache_dims(&self) -> CacheDims {
        CacheDims {
            n_layer: self.params.n_layer,
            head_dim: self.params.head_dim(),
            n_head_kv: self.params.n_head_kv,
            n_head: self.params.n_head,
        }
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
        k_dtype: GgmlType,
        v_dtype: GgmlType,
    ) -> u64 {
        let p = &self.params;
        // Per-pass token count: bounded by n_ubatch, or the whole context when
        // chunking is disabled (n_ubatch == 0 ⇒ a single full-prompt pass).
        let l = if n_ubatch == 0 {
            max_seq_len.max(1)
        } else {
            n_ubatch
        } as u64;
        let hidden = p.n_embd as u64;
        let n_kv = p.n_embd_kv() as u64;
        let n_ff = p.n_ff as u64;
        let vocab = p.n_vocab as u64;
        // Per-layer transient buffers accumulate within a layer (reclaimed only
        // at the next layer's scratch_restore): ≈ 7×hidden + 3×n_kv + 4×n_ff
        // columns, each [_, l] F32.
        let per_layer = (7 * hidden + 3 * n_kv + 4 * n_ff) * l * 4;
        let residual = hidden * l * 4; // persistent across the layer loop
        let mask = l * l * 4; // within-chunk only
        let logits = vocab * 4; // last token only
        // Heterogeneous K/V caches materialize the [0, ctx) prefix to F32 per
        // layer (cache_io::record_read); homogeneous caches bind directly.
        let staging = if k_dtype != v_dtype {
            2 * p.head_dim() as u64 * p.n_head_kv as u64 * max_seq_len as u64 * 4
        } else {
            0
        };
        // Flash-attn prefill split-K partials: a chunk attending to a long KV
        // prefix splits the KV across `k_num` workgroups, each writing a
        // (head_dim_v+2)·l·n_head partial slice (one buffer alloc'd from scratch
        // per FA call, reclaimed at the next layer's scratch_restore). Without
        // this the deep-prefill split would overflow scratch. Size for the
        // deepest split this context can produce; keep `fa_walk` in sync with
        // `flash_attn::prefill_fa_kv_walk()`.
        let fa_walk = 8192u64;
        let fa_partials = if max_seq_len as u64 > fa_walk {
            let fa_k_num = (max_seq_len as u64).div_ceil(fa_walk);
            (p.head_dim() as u64 + 2) * l * p.n_head as u64 * fa_k_num * 4
        } else {
            0
        };
        let raw = per_layer + residual + mask + logits + staging + fa_partials;
        raw + raw / 3 + (32 << 20) // +33% headroom + 32 MiB slack
    }

    fn record_forward(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        compute_logits: bool,
    ) -> Result<Option<crate::inference::weights::TensorView>, Box<dyn Error>> {
        let p = &self.params;
        let l = tokens.len() as u32;
        if l == 0 {
            return Err("empty prompt".into());
        }
        let hidden = p.n_embd as u64;
        let head_dim = p.head_dim() as u64;
        let n_kv_embd = p.n_embd_kv() as u64;
        let n_ff = p.n_ff as u64;

        if cache.position != position_offset {
            return Err(format!(
                "cache.position {} doesn't match caller-supplied position_offset {position_offset}",
                cache.position
            )
            .into());
        }
        let total_len = position_offset + l; // KV-context length after this step
        let kv_len_u = total_len as u64;

        // ---- prologue: positions + mask + token id buffer ----
        let token_buf = ctx.alloc_scratch((l as u64) * 4)?;
        write_u32(ctx, token_buf, tokens)?;

        let positions_buf = ctx.alloc_scratch((l as u64) * 4)?;
        let positions: Vec<u32> = (position_offset..position_offset + l).collect();
        write_u32(ctx, positions_buf, &positions)?;

        // flash_attn binds the cache layer directly (no copy, no dequant) using
        // its dtype-specialized variant whenever it has a variant for this (K, V)
        // pair — all symmetric pairs and the exposed asymmetric ones. Unsupported
        // asymmetric pairs (rare) fall back to materialize-then-attend.
        let cache_direct = flash_attn::supports_pair(cache.config.k_dtype, cache.config.v_dtype);

        // Mask is always F32 regardless of cache dtype (the shader's
        // `data_m` binding is F32 across every variant since `e41661f`).
        // Single-token decode (l == 1) needs no mask: the one query sits at
        // the newest position, so every KV slot is causally visible (the
        // whole row is 0). Skip the O(total_len) host-side mask build per
        // step — flash_attn runs that case with MASK_ENABLE=0.
        let mask = if l > 1 {
            // Within-chunk mask only: [l × l] (not [kv_len × l]). The cached
            // prefix [0, position_offset) is always visible and is handled
            // shader-side via mask_kv_offset, so the mask is O(l²) regardless
            // of context length.
            let m = ctx.alloc_tensor([l as u64, l as u64, 1, 1], GgmlType::F32)?;
            write_causal_mask(ctx, m, l, GgmlType::F32)?;
            Some(m)
        } else {
            None
        };

        // ---- embedding lookup ----
        // Residual is one persistent slot, reused across layers via in-place
        // adds. Everything else allocated below the loop checkpoint is
        // reclaimed on `scratch_restore` between layers.
        let residual = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, l, residual)?;
        let layer_checkpoint = ctx.scratch_checkpoint();

        let rope_params = rope::RopeParams::llama_default(p.rope_dim, p.rope_freq_base);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = (p.n_head / p.n_head_kv).max(1);
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim as u32,
            head_dim_v: head_dim as u32,
            gqa_ratio,
            scale,
            swa_window: 0,
        };

        // ---- per-layer loop ----
        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_checkpoint);

            // x_norm = rms_norm(residual) * attn_norm
            let x_norm = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.attn_norm, x_norm, p.rms_eps)?;

            // Q/K/V read the same x_norm and write disjoint outputs — fan
            // out three dispatches with no inter-barriers, then fence all
            // three ranges in one vkCmdPipelineBarrier before RoPE reads.
            let q = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wq, x_norm, q)?;
            let k = ctx.alloc_tensor([n_kv_embd, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wk, x_norm, k)?;
            let v = ctx.alloc_tensor([n_kv_embd, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wv, x_norm, v)?;
            crate::inference::command::record_compute_barriers(
                ctx.device,
                ctx.cmd,
                &[q.range(), k.range(), v.range()],
            );

            // RoPE on Q and K (separate scratch dst). Same fan-out
            // pattern: both nofence, then one combined barrier.
            let q_view = reshape_for_rope(q, head_dim, p.n_head as u64, l as u64);
            let k_view = reshape_for_rope(k, head_dim, p.n_head_kv as u64, l as u64);
            let q_roped = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            let k_roped = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rope::record_nofence(ctx, q_view, positions_buf, q_roped, rope_params)?;
            rope::record_nofence(ctx, k_view, positions_buf, k_roped, rope_params)?;
            crate::inference::command::record_compute_barriers(
                ctx.device,
                ctx.cmd,
                &[q_roped.range(), k_roped.range()],
            );

            // K, V (post-RoPE for K, raw for V) in natural
            // [head_dim, n_head_kv, L] layout for cache write.
            // (reshape_for_rope produces exactly that layout.)
            let k_natural = reshape_for_rope(k_roped, head_dim, p.n_head_kv as u64, l as u64);
            let v_natural = reshape_for_rope(v, head_dim, p.n_head_kv as u64, l as u64);
            // K and V write to disjoint cache buffers; V's trailing global
            // barrier covers both before flash_attn reads.
            cache_io::record_write_nofence(
                ctx,
                k_natural,
                cache.k_layers[layer_idx],
                position_offset,
            )?;
            cache_io::record_write(ctx, v_natural, cache.v_layers[layer_idx], position_offset)?;

            // Source the K/V views fed to flash_attn:
            //   - F32 / F16 cache: bind cache layer directly (zero copy).
            //   - BF16 / quants:   materialize the [0, total_len) prefix
            //     into F32 scratch (transient — reclaimed on next
            //     layer's `scratch_restore`).
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

            // TurboQuant: forward-WHT the query when K is turbo (K/V are stored
            // WHT-rotated; <WHT Q, WHT K> = <Q,K>). head_dim must be % 128 == 0.
            let q_for_attn = if is_turbo(cache.k_layers[layer_idx].dtype) {
                let qw = ctx.alloc_tensor(q_roped.dims, GgmlType::F32)?;
                turbo_wht::record(ctx, q_roped, qw, WhtDir::Forward)?;
                qw
            } else {
                q_roped
            };

            // Permute Q to [head_dim, L, n_head] and K/V to
            // [head_dim, total_len, n_head_kv] (flash_attn input layout).
            let q_perm = permute_to_attn(q_for_attn, head_dim, l as u64, p.n_head as u64);
            let k_perm = permute_to_attn(k_src, head_dim, kv_len_u, p.n_head_kv as u64);
            let v_perm = permute_to_attn(v_src, head_dim, kv_len_u, p.n_head_kv as u64);

            // attn_out = flash_attn(Q, K, V, mask) → [hidden, L]
            let attn_out_raw = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            flash_attn::record(
                ctx,
                q_perm,
                k_perm,
                v_perm,
                mask,
                attn_out_raw,
                fa_params,
                total_len,
            )?;
            // (mask is Option<TensorView>: Some for prefill chunks, None for
            // single-token decode — see the prologue.)
            // TurboQuant: inverse-WHT the output (in V's rotated basis) when V is turbo.
            let attn_out = if is_turbo(cache.v_layers[layer_idx].dtype) {
                let ao = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
                turbo_wht::record(ctx, attn_out_raw, ao, WhtDir::Inverse)?;
                ao
            } else {
                attn_out_raw
            };

            // proj = wo @ attn_out → [hidden, L]
            let proj = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wo, attn_out, proj)?;

            // residual += proj (in-place into the persistent residual slot)
            elementwise::record_add(ctx, residual, proj, residual)?;

            // x_norm = rms_norm(residual) * ffn_norm
            let x_norm2 = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.ffn_norm, x_norm2, p.rms_eps)?;

            // ffn_gate and ffn_up both read x_norm2 and write disjoint
            // tensors — fan out, then fence both ranges in one barrier.
            let gate = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_gate, x_norm2, gate)?;
            let up = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_up, x_norm2, up)?;
            crate::inference::command::record_compute_barriers(
                ctx.device,
                ctx.cmd,
                &[gate.range(), up.range()],
            );
            // gate = silu(gate)
            let gate_silu = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_silu(ctx, gate, gate_silu)?;
            // ffn_hidden = gate * up
            let ffn_hidden = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_mul(ctx, gate_silu, up, ffn_hidden)?;
            // down = ffn_down @ ffn_hidden → [hidden, L]
            let down = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_down, ffn_hidden, down)?;

            // residual += down (in-place)
            elementwise::record_add(ctx, residual, down, residual)?;
        }
        ctx.scratch_restore(layer_checkpoint);

        // Intermediate prefill ubatches only populate the KV cache — skip the
        // final norm + lm_head entirely (no logits needed until the last
        // ubatch). cache.position is still advanced below.
        if !compute_logits {
            cache_io::advance(cache, l);
            return Ok(None);
        }

        // ---- final norm + lm_head (last token only) ----
        // We sample from the final position only, so normalize + project just
        // the last token's residual. Full-batch logits would burn
        // n_vocab × L bytes of scratch (~262MB at L=512, vocab=128k — exceeds
        // the scratch region on its own, which is the long-prompt prefill OOM).
        // Slicing residual to the last token via a strided TensorView lets both
        // rms_norm and the lm_head matmul run with L=1. (Mirrors qwen35moe.rs.)
        let elem_size = 4u64;
        let vocab = p.n_vocab as u64;
        let residual_last = crate::inference::weights::TensorView {
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

        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let last_logits = ctx.alloc_tensor([vocab, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, final_norm, last_logits)?;

        // Advance cache position for the next call. (All cache writes were
        // already recorded above; the GPU executes them in order before the
        // logits readback.)
        cache_io::advance(cache, l);
        Ok(Some(last_logits))
    }

    fn supports_unified(&self) -> bool {
        true
    }

    /// Batched decode: B sequences, one token each, in one forward. The dense
    /// ops (embedding, RMSNorm, all matmuls, RoPE, FFN) process the `B`-wide
    /// token dimension unchanged; only the attention is per-sequence (own KV
    /// slab + length via `flash_attn::record_batched`) and the K/V writes fan
    /// out one column per sequence into its slab.
    fn record_forward_batch(
        &self,
        ctx: &mut DispatchContext,
        batch: &mut crate::inference::kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<crate::inference::weights::TensorView, Box<dyn Error>> {
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
        let head_dim = p.head_dim() as u64;
        let n_kv_embd = p.n_embd_kv() as u64;
        let n_ff = p.n_ff as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv as u64;
        let vocab = p.n_vocab as u64;
        let elem = 4u64;
        // Each sequence attends over [0, position + 1).
        let kv_lens: Vec<u32> = positions.iter().map(|&pos| pos + 1).collect();

        // ---- prologue: B token ids + B positions (one per sequence) ----
        let token_buf = ctx.alloc_scratch(b * 4)?;
        write_u32(ctx, token_buf, tokens)?;
        let positions_buf = ctx.alloc_scratch(b * 4)?;
        write_u32(ctx, positions_buf, positions)?;

        let residual = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, b as u32, residual)?;
        // Persistent per-forward DecodeDyn array for batched flash-attn — must
        // live above `layer_checkpoint` so per-layer `scratch_restore` cannot
        // reclaim it (the shader reads `kv_len` at execute time, after submit).
        let fa_dyn_range = crate::inference::decode_dyn::alloc_array(ctx, b as u32)?;
        let layer_checkpoint = ctx.scratch_checkpoint();

        let rope_params = rope::RopeParams::llama_default(p.rope_dim, p.rope_freq_base);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = (p.n_head / p.n_head_kv).max(1);
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim as u32,
            head_dim_v: head_dim as u32,
            gqa_ratio,
            scale,
            swa_window: 0,
        };

        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_checkpoint);

            let x_norm = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.attn_norm, x_norm, p.rms_eps)?;

            let q = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wq, x_norm, q)?;
            let k = ctx.alloc_tensor([n_kv_embd, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wk, x_norm, k)?;
            let v = ctx.alloc_tensor([n_kv_embd, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wv, x_norm, v)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range(), v.range()]);

            let q_view = reshape_for_rope(q, head_dim, n_head, b);
            let k_view = reshape_for_rope(k, head_dim, n_head_kv, b);
            let q_roped = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            let k_roped = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rope::record_nofence(ctx, q_view, positions_buf, q_roped, rope_params)?;
            rope::record_nofence(ctx, k_view, positions_buf, k_roped, rope_params)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q_roped.range(), k_roped.range()]);

            // Per-sequence K/V cache writes: column s → slot s's slab at its position.
            let k_natural = reshape_for_rope(k_roped, head_dim, n_head_kv, b);
            let v_natural = reshape_for_rope(v, head_dim, n_head_kv, b);
            let col_stride = head_dim * n_head_kv * elem; // bytes between sequence columns
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

            // Batched attention: each sequence attends to its own slab (slots[s])
            // and length; the K/V views bind all slabs, the flash picks per
            // sequence via DecodeDyn::slot.
            let q_attn = batched_q_attn_view(q_roped, head_dim, n_head, b, hidden);
            let k_attn = batch.batched_k_attn_view(layer_idx as u32);
            let v_attn = batch.batched_v_attn_view(layer_idx as u32);
            let attn_out = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
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

            let proj = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wo, attn_out, proj)?;
            elementwise::record_add(ctx, residual, proj, residual)?;

            let x_norm2 = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.ffn_norm, x_norm2, p.rms_eps)?;
            let gate = ctx.alloc_tensor([n_ff, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_gate, x_norm2, gate)?;
            let up = ctx.alloc_tensor([n_ff, b, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_up, x_norm2, up)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[gate.range(), up.range()]);
            let gate_silu = ctx.alloc_tensor([n_ff, b, 1, 1], GgmlType::F32)?;
            elementwise::record_silu(ctx, gate, gate_silu)?;
            let ffn_hidden = ctx.alloc_tensor([n_ff, b, 1, 1], GgmlType::F32)?;
            elementwise::record_mul(ctx, gate_silu, up, ffn_hidden)?;
            let down = ctx.alloc_tensor([hidden, b, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_down, ffn_hidden, down)?;
            elementwise::record_add(ctx, residual, down, residual)?;
        }
        ctx.scratch_restore(layer_checkpoint);

        // Final norm + lm_head over ALL B columns (each column is its
        // sequence's last — and only — token this step).
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

        for (s, &pos) in positions.iter().enumerate() {
            batch.positions[slots[s] as usize] = pos + 1;
        }
        Ok(logits)
    }

    /// Unified varlen forward (M5): see the trait doc. `tokens`/`positions` are
    /// the flat `[N_total]` packed stream; `seq_lens[s]` is sequence `s`'s token
    /// count this step (prefill chunk or 1 for decode). Mirrors
    /// [`Self::record_forward_batch`] but the token dimension is `N_total` and
    /// attention is varlen-causal (Phase-1 `record_batched` with `query_lens`).
    fn record_forward_unified(
        &self,
        ctx: &mut DispatchContext,
        batch: &mut crate::inference::kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        seq_lens: &[u32],
        slots: &[u32],
    ) -> Result<crate::inference::weights::TensorView, Box<dyn Error>> {
        use crate::inference::command::{record_compute_barriers, record_global_barrier};
        let p = &self.params;
        let b = seq_lens.len();
        let n_total = tokens.len() as u64;
        if b == 0 || n_total == 0 {
            return Err("record_forward_unified: empty batch".into());
        }
        if positions.len() != tokens.len() || slots.len() != b {
            return Err(
                "record_forward_unified: tokens/positions/seq_lens/slots length mismatch".into(),
            );
        }
        if seq_lens.iter().map(|&l| l as u64).sum::<u64>() != n_total {
            return Err("record_forward_unified: sum(seq_lens) != tokens.len()".into());
        }
        let hidden = p.n_embd as u64;
        let head_dim = p.head_dim() as u64;
        let n_kv_embd = p.n_embd_kv() as u64;
        let n_ff = p.n_ff as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv as u64;
        let vocab = p.n_vocab as u64;
        let elem = 4u64;

        // Per-sequence flat offsets (prefix sum), causal lengths, query lengths.
        let q_starts: Vec<u64> = seq_lens
            .iter()
            .scan(0u64, |a, &l| {
                let s = *a;
                *a += l as u64;
                Some(s)
            })
            .collect();
        // base_s = the seq's first token's absolute position; kv after this step.
        let kv_lens: Vec<u32> = (0..b)
            .map(|s| positions[q_starts[s] as usize] + seq_lens[s])
            .collect();
        let query_lens: Vec<u32> = seq_lens.to_vec();

        // ---- prologue: N_total token ids + N_total flat positions ----
        let token_buf = ctx.alloc_scratch(n_total * 4)?;
        write_u32(ctx, token_buf, tokens)?;
        let positions_buf = ctx.alloc_scratch(n_total * 4)?;
        write_u32(ctx, positions_buf, positions)?;

        let residual = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(
            ctx,
            self.weights.token_embd,
            token_buf,
            n_total as u32,
            residual,
        )?;
        let fa_dyn_range = crate::inference::decode_dyn::alloc_array(ctx, b as u32)?;
        let layer_checkpoint = ctx.scratch_checkpoint();

        let rope_params = rope::RopeParams::llama_default(p.rope_dim, p.rope_freq_base);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = (p.n_head / p.n_head_kv).max(1);
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim as u32,
            head_dim_v: head_dim as u32,
            gqa_ratio,
            scale,
            swa_window: 0,
        };

        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_checkpoint);

            let x_norm = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.attn_norm, x_norm, p.rms_eps)?;

            let q = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wq, x_norm, q)?;
            let k = ctx.alloc_tensor([n_kv_embd, n_total, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wk, x_norm, k)?;
            let v = ctx.alloc_tensor([n_kv_embd, n_total, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wv, x_norm, v)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range(), v.range()]);

            let q_view = reshape_for_rope(q, head_dim, n_head, n_total);
            let k_view = reshape_for_rope(k, head_dim, n_head_kv, n_total);
            let q_roped = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            let k_roped = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rope::record_nofence(ctx, q_view, positions_buf, q_roped, rope_params)?;
            rope::record_nofence(ctx, k_view, positions_buf, k_roped, rope_params)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q_roped.range(), k_roped.range()]);

            // Per-sequence K/V write: seq s's L_s-token chunk (flat columns
            // [q_start, q_start+L)) lands in slab slots[s] at base_s.
            let k_natural = reshape_for_rope(k_roped, head_dim, n_head_kv, n_total);
            let v_natural = reshape_for_rope(v, head_dim, n_head_kv, n_total);
            let tok_stride = head_dim * n_head_kv * elem; // bytes per token column
            for s in 0..b {
                let l = seq_lens[s] as u64;
                let off = q_starts[s] * tok_stride;
                let chunk = |t: TensorView| -> TensorView {
                    TensorView {
                        byte_offset: t.byte_offset + off,
                        byte_size: l * tok_stride,
                        dims: [head_dim, n_head_kv, l, 1],
                        byte_stride: [elem, elem * head_dim, tok_stride, tok_stride * l],
                        element_stride: [
                            1,
                            head_dim,
                            head_dim * n_head_kv,
                            head_dim * n_head_kv * l,
                        ],
                        ..t
                    }
                };
                let base_pos = positions[q_starts[s] as usize];
                cache_io::record_write(
                    ctx,
                    chunk(k_natural),
                    batch.slot_k_view(slots[s], layer_idx as u32),
                    base_pos,
                )?;
                cache_io::record_write(
                    ctx,
                    chunk(v_natural),
                    batch.slot_v_view(slots[s], layer_idx as u32),
                    base_pos,
                )?;
            }

            // Varlen attention: each sequence's L_s query rows attend causally
            // over its own slab; flat Q/out, in-shader causal mask.
            let q_attn = permute_to_attn(q_roped, head_dim, n_total, n_head);
            let k_attn = batch.batched_k_attn_view(layer_idx as u32);
            let v_attn = batch.batched_v_attn_view(layer_idx as u32);
            let attn_out = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
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
                Some(&query_lens),
            )?;

            let proj = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wo, attn_out, proj)?;
            elementwise::record_add(ctx, residual, proj, residual)?;

            let x_norm2 = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.ffn_norm, x_norm2, p.rms_eps)?;
            let gate = ctx.alloc_tensor([n_ff, n_total, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_gate, x_norm2, gate)?;
            let up = ctx.alloc_tensor([n_ff, n_total, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_up, x_norm2, up)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[gate.range(), up.range()]);
            let gate_silu = ctx.alloc_tensor([n_ff, n_total, 1, 1], GgmlType::F32)?;
            elementwise::record_silu(ctx, gate, gate_silu)?;
            let ffn_hidden = ctx.alloc_tensor([n_ff, n_total, 1, 1], GgmlType::F32)?;
            elementwise::record_mul(ctx, gate_silu, up, ffn_hidden)?;
            let down = ctx.alloc_tensor([hidden, n_total, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_down, ffn_hidden, down)?;
            elementwise::record_add(ctx, residual, down, residual)?;
        }
        ctx.scratch_restore(layer_checkpoint);

        // Gather each sequence's LAST-token column (flat index q_start+L-1) into
        // a packed [hidden, B] tensor, then norm + lm_head only on those B
        // columns (avoids an N_total-wide lm_head). The sample at column s is
        // sequence s's next token (valid iff s just finished prefill / decodes).
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
                    ctx.cmd,
                    residual.buffer,
                    last_hidden.buffer,
                    std::slice::from_ref(&copy),
                );
            }
        }
        record_global_barrier(ctx.device, ctx.cmd);

        let final_norm = ctx.alloc_tensor([hidden, b as u64, 1, 1], GgmlType::F32)?;
        rms_norm::record(
            ctx,
            last_hidden,
            self.weights.output_norm,
            final_norm,
            p.rms_eps,
        )?;
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let logits = ctx.alloc_tensor([vocab, b as u64, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, final_norm, logits)?;

        for s in 0..b {
            batch.positions[slots[s] as usize] = kv_lens[s];
        }
        Ok(logits)
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
/// per head per sequence; batch stride = hidden).
fn batched_q_attn_view(
    t: TensorView,
    head_dim: u64,
    n_head: u64,
    b: u64,
    hidden: u64,
) -> TensorView {
    let elem = t.byte_stride[0];
    TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset,
        byte_size: t.byte_size,
        dims: [head_dim, 1, n_head, b],
        byte_stride: [elem, elem * head_dim, elem * head_dim, elem * hidden],
        element_stride: [1, head_dim, head_dim, hidden],
        dtype: t.dtype,
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

fn write_causal_mask(
    ctx: &mut DispatchContext,
    mask: TensorView,
    l: u32,
    dtype: GgmlType,
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch region not host-visible")?;
    let l = l as usize;
    // Within-chunk causal mask only: an [l × l] lower triangle, row-major
    // [l rows][l cols]. Query row i attends to within-chunk kv column jc iff
    // jc <= i (both at absolute position position_offset + index, so the
    // offset cancels). The always-visible cached prefix carries no entries —
    // the shader treats columns < mask_kv_offset as visible.
    let n = l * l;
    let unmasked = |i: usize, jc: usize| jc <= i;
    match dtype {
        GgmlType::F32 => {
            let mut buf: Vec<f32> = vec![0.0; n];
            for i in 0..l {
                for jc in 0..l {
                    buf[i * l + jc] = if unmasked(i, jc) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    };
                }
            }
            unsafe {
                let dst = host_ptr.add(mask.byte_offset as usize) as *mut f32;
                std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
            }
        }
        GgmlType::F16 => {
            // F16 bit patterns: +0.0 = 0x0000, -inf = 0xFC00.
            let mut buf: Vec<u16> = vec![0u16; n];
            for i in 0..l {
                for jc in 0..l {
                    buf[i * l + jc] = if unmasked(i, jc) { 0x0000 } else { 0xFC00 };
                }
            }
            unsafe {
                let dst = host_ptr.add(mask.byte_offset as usize) as *mut u16;
                std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
            }
        }
        other => return Err(format!("mask dtype {other:?} not supported").into()),
    }
    Ok(())
}

/// Take the first `total_len` token positions of a cache layer's K (or V)
/// view, leaving strides intact (the prefix is contiguous in the cache's
/// natural `[head_dim, n_head_kv, max_seq_len]` layout, so byte_stride[2] is
/// the right per-token stride for the smaller slice too).
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

/// Reshape Q (or K) from matmul-output `[n_embd, L]` to RoPE-input
/// `[head_dim, n_head, L]`. Strides recomputed under contiguous assumption.
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

/// Permute from RoPE layout `[head_dim, n_heads, L]` (contiguous) to
/// `[head_dim, L, n_heads]` (non-contiguous view — same memory, different
/// strides). Used to feed Q/K/V into flash_attn.
fn permute_to_attn(t: TensorView, head_dim: u64, l: u64, n_heads: u64) -> TensorView {
    let elem = t.byte_stride[0];
    TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset,
        byte_size: t.byte_size,
        dims: [head_dim, l, n_heads, 1],
        // memory[d, h, t] = t * (head_dim * n_heads) + h * head_dim + d
        // we want view[d, t, h] = same offset
        // nb0 = 1, nb1 (per-t) = head_dim * n_heads, nb2 (per-h) = head_dim
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

fn parse_params(gguf: &GgufFile) -> Result<LlamaParams, Box<dyn Error>> {
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
    let f32_key = |k: &'static str| -> Result<f32, Box<dyn Error>> {
        let v = gguf.get(k).ok_or(ModelError::MissingMetadata(k))?;
        coerce_f32(v).ok_or_else(|| {
            ModelError::BadMetadata {
                key: k,
                detail: format!("expected float, got {v:?}"),
            }
            .into()
        })
    };

    let n_layer = u32_key("llama.block_count")?;
    let n_head = u32_key("llama.attention.head_count")?;
    let n_head_kv = u32_key("llama.attention.head_count_kv").unwrap_or(n_head);
    let n_embd = u32_key("llama.embedding_length")?;
    let n_ff = u32_key("llama.feed_forward_length")?;
    let n_ctx_train = u32_key("llama.context_length")?;
    let rope_dim = u32_key("llama.rope.dimension_count").unwrap_or(n_embd / n_head);
    let rope_freq_base = f32_key("llama.rope.freq_base").unwrap_or(10000.0);
    let rms_eps = f32_key("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

    let n_vocab = u32_key("llama.vocab_size")?;

    Ok(LlamaParams {
        n_layer,
        n_head,
        n_head_kv,
        n_embd,
        n_ff,
        n_vocab,
        n_ctx_train,
        rope_dim,
        rope_freq_base,
        rms_eps,
    })
}

fn collect_weights(
    handle: &WeightsHandle,
    params: &LlamaParams,
) -> Result<LlamaWeights, Box<dyn Error>> {
    let view = |name: &str| -> Result<TensorView, Box<dyn Error>> {
        handle
            .view(name)
            .map_err(|_| ModelError::MissingTensor(name.to_string()).into())
    };

    let token_embd = view("token_embd.weight")?;
    let output_norm = view("output_norm.weight")?;
    let output = handle.view("output.weight").ok();

    let mut blocks = Vec::with_capacity(params.n_layer as usize);
    for i in 0..params.n_layer {
        blocks.push(LlamaBlockWeights {
            attn_norm: view(&format!("blk.{i}.attn_norm.weight"))?,
            wq: view(&format!("blk.{i}.attn_q.weight"))?,
            wk: view(&format!("blk.{i}.attn_k.weight"))?,
            wv: view(&format!("blk.{i}.attn_v.weight"))?,
            wo: view(&format!("blk.{i}.attn_output.weight"))?,
            ffn_norm: view(&format!("blk.{i}.ffn_norm.weight"))?,
            ffn_gate: view(&format!("blk.{i}.ffn_gate.weight"))?,
            ffn_up: view(&format!("blk.{i}.ffn_up.weight"))?,
            ffn_down: view(&format!("blk.{i}.ffn_down.weight"))?,
        });
    }

    Ok(LlamaWeights {
        token_embd,
        blocks,
        output_norm,
        output,
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

fn coerce_f32(v: &MetadataValue) -> Option<f32> {
    Some(match v {
        MetadataValue::F32(x) => *x,
        MetadataValue::F64(x) => *x as f32,
        _ => return None,
    })
}
