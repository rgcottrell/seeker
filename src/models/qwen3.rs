//! Qwen3 (`general.architecture == "qwen3"`) — dense text decoder. A standard
//! GQA transformer matching llama.cpp's `build_qwen3`:
//!
//!   * **Decoupled head_dim**: `head_dim = attention.key_length` (128), which is
//!     NOT `n_embd / n_head` — so the Q projection is `n_head·head_dim` wide
//!     (2048 for the 0.6B), wider than `n_embd` (1024); K/V are
//!     `n_head_kv·head_dim` (1024).
//!   * **Per-head Q-norm / K-norm** (RMSNorm over `head_dim`, learned weight)
//!     applied before RoPE. No V-norm.
//!   * **NEOX RoPE** with full rotation (`n_rot == head_dim`), θ = `rope.freq_base`
//!     (1e6).
//!   * **SwiGLU** MLP (silu gate), plain pre-norm residual blocks (no post-attn /
//!     post-ffn norms, no embedding scale, no per-layer output scale, no softcap,
//!     no sliding window — those are gemma4-specific).
//!   * Attention `scale = 1/√head_dim`. Tied lm_head (no `output.weight`).
//!
//! This is essentially [`super::gemma4`]'s `forward_inner` with the gemma extras
//! removed and uniform per-layer dims. The pre-`output_norm` hidden returned by
//! [`Model::record_forward_full`] is the hook `seeker embedding` pools over.

use std::error::Error;

use crate::gguf::{GgmlType, GgufFile};
use crate::inference::context::DispatchContext;
use crate::inference::kv_cache::KvCache;
use crate::inference::ops::{cache_io, elementwise, flash_attn, matmul, rms_norm, rope};
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::tokenizer::TokenizerBundle;

use super::gemma4::{coerce_f32, coerce_u32};
use super::{CacheDims, Model, ModelError};

const ARCH: &str = "qwen3";

#[derive(Debug, Clone)]
pub struct Qwen3Params {
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    /// `attention.key_length` (128) — decoupled from `n_embd / n_head`.
    pub head_dim: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// RoPE rotation dimension (== `head_dim` for Qwen3).
    pub rope_dim: u32,
}

impl Qwen3Params {
    /// Q projection width = `n_head · head_dim` (≠ `n_embd`).
    fn q_dim(&self) -> u32 {
        self.n_head * self.head_dim
    }
    /// K/V projection width = `n_head_kv · head_dim`.
    fn kv_dim(&self) -> u32 {
        self.n_head_kv * self.head_dim
    }
}

pub struct Qwen3BlockWeights {
    pub attn_norm: TensorView,
    pub wq: TensorView,
    pub attn_q_norm: TensorView,
    pub wk: TensorView,
    pub attn_k_norm: TensorView,
    pub wv: TensorView,
    pub wo: TensorView,
    pub ffn_norm: TensorView,
    pub ffn_gate: TensorView,
    pub ffn_up: TensorView,
    pub ffn_down: TensorView,
}

pub struct Qwen3Weights {
    pub token_embd: TensorView,
    pub blocks: Vec<Qwen3BlockWeights>,
    pub output_norm: TensorView,
    /// `None` ⇒ tied lm_head (uses `token_embd`). Qwen3-Embedding has no `output.weight`.
    pub output: Option<TensorView>,
}

pub struct Qwen3Model {
    pub params: Qwen3Params,
    pub weights: Qwen3Weights,
    pub handle: WeightsHandle,
    pub tokenizer: TokenizerBundle,
}

impl Qwen3Model {
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

impl Model for Qwen3Model {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn vocab_size(&self) -> u32 {
        self.params.n_vocab
    }

    fn cache_dims(&self) -> CacheDims {
        CacheDims {
            n_layer: self.params.n_layer,
            head_dim: self.params.head_dim, // 128 (key_length), NOT n_embd/n_head
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
        _k_dtype: GgmlType,
        _v_dtype: GgmlType,
        max_batch: u32,
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
        let q_dim = p.q_dim() as u64;
        let kv_dim = p.kv_dim() as u64;
        // Per-layer transient buffers (reclaimed each layer): residual-width
        // (x_norm, proj, x_norm2, down), Q-width (q, q_normed, q_roped, attn_out),
        // KV-width (k, k_normed, k_roped, v), FFN-width (gate, up, gate_silu, hidden).
        let per_layer = (4 * n_embd + 4 * q_dim + 4 * kv_dim + 4 * n_ff) * l * 4;
        let residual = n_embd * l * 4;
        let mask = l * l * 4;
        // One column per sequence: [vocab, 1] single-seq, [vocab, B] batched.
        let logits = vocab * 4 * max_batch.max(1) as u64;
        // Flash-attn prefill split-K partials (deepest split this context produces).
        let fa_walk = 8192u64;
        let fa_partials = if max_seq_len as u64 > fa_walk {
            let fa_k_num = (max_seq_len as u64).div_ceil(fa_walk);
            (p.head_dim as u64 + 2) * l * p.n_head as u64 * fa_k_num * 4
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

    fn record_forward_full(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        full_logits: bool,
        _checkpoint: bool, // qwen3 has no SSM state → no per-position snapshots
    ) -> Result<crate::models::ForwardFullOut, Box<dyn Error>> {
        let (logits, residual) =
            self.forward_inner(ctx, cache, tokens, position_offset, true, full_logits, None)?;
        Ok(crate::models::ForwardFullOut { logits, residual })
    }

    fn record_forward_embed_batch(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        seq_lens: &[u32],
    ) -> Result<TensorView, Box<dyn Error>> {
        if seq_lens.is_empty() || tokens.is_empty() {
            return Err("record_forward_embed_batch: empty batch".into());
        }
        if seq_lens.iter().map(|&l| l as usize).sum::<usize>() != tokens.len() {
            return Err("record_forward_embed_batch: seq_lens don't sum to tokens.len()".into());
        }
        // Packed-flat prefill at position 0; the block-diagonal mask isolates
        // each text, so this returns the same residual as prefilling each text
        // alone. compute_logits=false → residual only (the embedding hook).
        let (_logits, residual) =
            self.forward_inner(ctx, cache, tokens, 0, false, false, Some(seq_lens))?;
        Ok(residual)
    }

    fn supports_embed_batch(&self) -> bool {
        true
    }
}

impl Qwen3Model {
    /// Shared forward body. Returns `(optional logits, pre-output_norm residual
    /// [n_embd, L])`. The residual is the embedding hook for `seeker embedding`.
    #[allow(clippy::too_many_arguments)] // high-arity by nature (flags + varlen plan)
    fn forward_inner(
        &self,
        ctx: &mut DispatchContext,
        cache: &mut KvCache,
        tokens: &[u32],
        position_offset: u32,
        compute_logits: bool,
        full_logits: bool,
        // `Some(seq_lens)` = batched embedding: `tokens` packs N independent
        // texts; each restarts RoPE at 0 and attends only within itself
        // (block-diagonal causal mask). Requires `position_offset == 0` and
        // `!compute_logits` (residual-only). `None` = the normal single causal
        // sequence.
        embed_seq_lens: Option<&[u32]>,
    ) -> Result<(Option<TensorView>, TensorView), Box<dyn Error>> {
        use crate::inference::command::record_compute_barriers;
        let p = &self.params;
        let l = tokens.len() as u32;
        if l == 0 {
            return Err("empty prompt".into());
        }
        let hidden = p.n_embd as u64;
        let head_dim = p.head_dim as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv as u64;
        let q_dim = p.q_dim() as u64;
        let kv_dim = p.kv_dim() as u64;
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
        // Batched embedding restarts each text's positions at 0; otherwise the
        // contiguous [position_offset, position_offset + l) sequence.
        let positions: Vec<u32> = match embed_seq_lens {
            Some(lens) => lens.iter().flat_map(|&ls| 0..ls).collect(),
            None => (position_offset..position_offset + l).collect(),
        };
        write_u32(ctx, positions_buf, &positions)?;

        let cache_direct = flash_attn::supports_pair(cache.config.k_dtype, cache.config.v_dtype);

        // Mask: block-diagonal causal for a packed embedding batch (each text
        // sees only its own [start, i] prefix), else the within-chunk causal
        // triangle (the cached prefix is always-visible shader-side).
        let mask = if let Some(lens) = embed_seq_lens {
            let m = ctx.alloc_tensor([l as u64, l as u64, 1, 1], GgmlType::F32)?;
            write_block_diagonal_causal_mask(ctx, m, lens)?;
            Some(m)
        } else if l > 1 {
            let m = ctx.alloc_tensor([l as u64, l as u64, 1, 1], GgmlType::F32)?;
            write_causal_mask(ctx, m, l)?;
            Some(m)
        } else {
            None
        };

        // ---- embedding lookup (no √n_embd scale — that is gemma-specific) ----
        let residual = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, l, residual)?;
        let layer_checkpoint = ctx.scratch_checkpoint();

        let rope_params = rope::RopeParams::llama_default(p.rope_dim, p.rope_freq_base);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim as u32,
            head_dim_v: head_dim as u32,
            gqa_ratio: (n_head / n_head_kv).max(1) as u32,
            scale,
            swa_window: 0,
        };

        // ---- per-layer loop ----
        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_checkpoint);

            // input norm
            let x_norm = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.attn_norm, x_norm, p.rms_eps)?;

            // Q/K/V projections (separate V; widths use the decoupled head_dim).
            let q = ctx.alloc_tensor([q_dim, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wq, x_norm, q)?;
            let k = ctx.alloc_tensor([kv_dim, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wk, x_norm, k)?;
            let v = ctx.alloc_tensor([kv_dim, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.wv, x_norm, v)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range(), v.range()]);

            // Per-head Q-norm / K-norm (RMSNorm over head_dim), then NEOX RoPE.
            // V is used raw (no norm, no rope).
            let q_view = reshape_for_rope(q, head_dim, n_head, l as u64);
            let q_normed = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            rms_norm::record_nofence(ctx, q_view, block.attn_q_norm, q_normed, p.rms_eps)?;
            let k_view = reshape_for_rope(k, head_dim, n_head_kv, l as u64);
            let k_normed = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rms_norm::record_nofence(ctx, k_view, block.attn_k_norm, k_normed, p.rms_eps)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q_normed.range(), k_normed.range()]);

            let q_roped = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            let k_roped = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rope::record_neox_nofence(ctx, q_normed, positions_buf, q_roped, rope_params, None)?;
            rope::record_neox_nofence(ctx, k_normed, positions_buf, k_roped, rope_params, None)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q_roped.range(), k_roped.range()]);

            // KV cache write: K post-rope, V raw, both in natural layout.
            let k_natural = reshape_for_rope(k_roped, head_dim, n_head_kv, l as u64);
            let v_natural = reshape_for_rope(v, head_dim, n_head_kv, l as u64);
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

            // attn_out is q_dim-wide (n_head · head_dim_v), NOT n_embd.
            let attn_out = ctx.alloc_tensor([q_dim, l as u64, 1, 1], GgmlType::F32)?;
            flash_attn::record(
                ctx, q_perm, k_perm, v_perm, mask, attn_out, fa_params, total_len,
            )?;

            // O-proj (wo: q_dim → n_embd) → residual add. No post-attn norm.
            let proj = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wo, attn_out, proj)?;
            elementwise::record_add(ctx, residual, proj, residual)?;

            // FFN: ffn_norm → SwiGLU(silu) → residual add. No post-ffn norm.
            let x_norm2 = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.ffn_norm, x_norm2, p.rms_eps)?;
            let gate = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_gate, x_norm2, gate)?;
            let up = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, block.ffn_up, x_norm2, up)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[gate.range(), up.range()]);
            let gate_silu = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_silu(ctx, gate, gate_silu)?;
            let ffn_hidden = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_mul(ctx, gate_silu, up, ffn_hidden)?;
            let down = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_down, ffn_hidden, down)?;
            elementwise::record_add(ctx, residual, down, residual)?;
        }
        ctx.scratch_restore(layer_checkpoint);

        if !compute_logits {
            cache_io::advance(cache, l);
            return Ok((None, residual));
        }

        // ---- final norm + tied lm_head (no softcap) ----
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

        cache_io::advance(cache, l);
        Ok((Some(logits), residual))
    }
}

// ── file-local helpers (copies of the gemma4/llama private helpers) ──

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

/// Block-diagonal causal mask for a packed embedding batch: query row `i`
/// (in text `s`) attends key `jc` iff `jc` is in the same text AND `jc <= i`.
/// Cross-text entries are `-inf`, so the packed forward is identical to
/// prefilling each text alone. Same `[N_total, N_total]` row-major layout +
/// 0/-inf convention as [`write_causal_mask`] (a single-text batch reduces to
/// exactly the causal triangle).
fn write_block_diagonal_causal_mask(
    ctx: &mut DispatchContext,
    mask: TensorView,
    seq_lens: &[u32],
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch region not host-visible")?;
    let n_total: usize = seq_lens.iter().map(|&l| l as usize).sum();
    let mut buf: Vec<f32> = vec![f32::NEG_INFINITY; n_total * n_total];
    let mut start = 0usize;
    for &ls in seq_lens {
        let ls = ls as usize;
        // Within text block [start, start+ls): lower-triangular (causal).
        for r in 0..ls {
            let i = start + r;
            let row = i * n_total;
            for c in 0..=r {
                buf[row + start + c] = 0.0;
            }
        }
        start += ls;
    }
    unsafe {
        let dst = host_ptr.add(mask.byte_offset as usize) as *mut f32;
        std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
    }
    Ok(())
}

fn parse_params(gguf: &GgufFile) -> Result<Qwen3Params, Box<dyn Error>> {
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

    let bad = |key: &'static str, detail: String| -> Box<dyn Error> {
        ModelError::BadMetadata { key, detail }.into()
    };

    let n_layer = u32_key("qwen3.block_count")?;
    let n_head = u32_key("qwen3.attention.head_count")?;
    if n_head == 0 {
        return Err(bad("qwen3.attention.head_count", "must be > 0".into()));
    }
    let n_head_kv = u32_or("qwen3.attention.head_count_kv", n_head);
    if n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
        return Err(bad(
            "qwen3.attention.head_count_kv",
            format!("must be > 0 and divide head_count ({n_head}), got {n_head_kv}"),
        ));
    }
    let n_embd = u32_key("qwen3.embedding_length")?;
    let n_ff = u32_key("qwen3.feed_forward_length")?;
    let n_ctx_train = u32_or("qwen3.context_length", 32768);
    let rms_eps = f32_or("qwen3.attention.layer_norm_rms_epsilon", 1e-6);
    // `n_head > 0` is guaranteed above, so the `n_embd / n_head` default is safe.
    let head_dim = u32_or("qwen3.attention.key_length", n_embd / n_head);
    if head_dim == 0 {
        return Err(bad("qwen3.attention.key_length", "must be > 0".into()));
    }
    let rope_freq_base = f32_or("qwen3.rope.freq_base", 1_000_000.0);
    let rope_dim = u32_or("qwen3.rope.dimension_count", head_dim);
    if rope_dim == 0 || rope_dim > head_dim {
        return Err(bad(
            "qwen3.rope.dimension_count",
            format!("must be in 1..={head_dim} (head_dim), got {rope_dim}"),
        ));
    }
    let n_vocab = u32_key("qwen3.vocab_size").or_else(|_| derive_vocab(gguf))?;

    Ok(Qwen3Params {
        n_layer,
        n_head,
        n_head_kv,
        n_embd,
        n_ff,
        n_vocab,
        n_ctx_train,
        head_dim,
        rms_eps,
        rope_freq_base,
        rope_dim,
    })
}

fn derive_vocab(gguf: &GgufFile) -> Result<u32, Box<dyn Error>> {
    gguf.tensor("token_embd.weight")
        .map(|t| t.dims[1] as u32)
        .ok_or_else(|| ModelError::MissingMetadata("qwen3.vocab_size").into())
}

fn collect_weights(
    handle: &WeightsHandle,
    params: &Qwen3Params,
) -> Result<Qwen3Weights, Box<dyn Error>> {
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
        blocks.push(Qwen3BlockWeights {
            attn_norm: view(&format!("blk.{i}.attn_norm.weight"))?,
            wq: view(&format!("blk.{i}.attn_q.weight"))?,
            attn_q_norm: view(&format!("blk.{i}.attn_q_norm.weight"))?,
            wk: view(&format!("blk.{i}.attn_k.weight"))?,
            attn_k_norm: view(&format!("blk.{i}.attn_k_norm.weight"))?,
            wv: view(&format!("blk.{i}.attn_v.weight"))?,
            wo: view(&format!("blk.{i}.attn_output.weight"))?,
            ffn_norm: view(&format!("blk.{i}.ffn_norm.weight"))?,
            ffn_gate: view(&format!("blk.{i}.ffn_gate.weight"))?,
            ffn_up: view(&format!("blk.{i}.ffn_up.weight"))?,
            ffn_down: view(&format!("blk.{i}.ffn_down.weight"))?,
        });
    }

    Ok(Qwen3Weights {
        token_embd,
        blocks,
        output_norm,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_decoupled_head_dim() {
        // Hand-construct params as parse_params would for Qwen3-Embedding-0.6B.
        let p = Qwen3Params {
            n_layer: 28,
            n_head: 16,
            n_head_kv: 8,
            n_embd: 1024,
            n_ff: 3072,
            n_vocab: 151936,
            n_ctx_train: 32768,
            head_dim: 128,
            rms_eps: 1e-6,
            rope_freq_base: 1e6,
            rope_dim: 128,
        };
        assert_eq!(p.q_dim(), 2048, "q = n_head*head_dim, NOT n_embd");
        assert_eq!(p.kv_dim(), 1024);
        assert_ne!(
            p.head_dim,
            p.n_embd / p.n_head,
            "head_dim is decoupled (128 != 64)"
        );
    }
}
