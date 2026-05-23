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
use crate::inference::ops::{elementwise, flash_attn, matmul, rms_norm, rope};
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::tokenizer::TokenizerBundle;

use super::{Model, ModelError};

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

    fn weights(&self) -> &WeightsHandle {
        &self.handle
    }

    fn tokenizer(&self) -> &TokenizerBundle {
        &self.tokenizer
    }

    fn record_forward(
        &self,
        ctx: &mut DispatchContext,
        tokens: &[u32],
    ) -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        let p = &self.params;
        let l = tokens.len() as u32;
        if l == 0 {
            return Err("empty prompt".into());
        }
        let hidden = p.n_embd as u64;
        let head_dim = p.head_dim() as u64;
        let n_kv_embd = p.n_embd_kv() as u64;
        let n_ff = p.n_ff as u64;

        // ---- prologue: positions + mask + token id buffer ----
        let token_buf = ctx.alloc_scratch((l as u64) * 4)?;
        write_u32(ctx, token_buf, tokens)?;

        let positions_buf = ctx.alloc_scratch((l as u64) * 4)?;
        let positions: Vec<u32> = (0..l).collect();
        write_u32(ctx, positions_buf, &positions)?;

        // Causal mask [L, L] in F32 (matches the f32_f32 flash_attn variant):
        // 0 if j <= i else -inf.
        let mask = ctx.alloc_tensor([l as u64, l as u64, 1, 1], GgmlType::F32)?;
        write_causal_mask(ctx, mask, l)?;

        // ---- embedding lookup ----
        let mut residual = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(
            ctx,
            self.weights.token_embd,
            token_buf,
            l,
            residual,
        )?;

        let rope_params = rope::RopeParams::llama_default(p.rope_dim, p.rope_freq_base);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = (p.n_head / p.n_head_kv).max(1);
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim as u32,
            head_dim_v: head_dim as u32,
            gqa_ratio,
            scale,
        };

        // ---- per-layer loop ----
        for block in self.weights.blocks.iter() {
            // x_norm = rms_norm(residual) * attn_norm
            let x_norm = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.attn_norm, x_norm, p.rms_eps)?;

            // Q = wq @ x_norm  → [n_embd, L]
            let q = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wq, x_norm, q)?;
            // K = wk @ x_norm  → [n_embd_kv, L]
            let k = ctx.alloc_tensor([n_kv_embd, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wk, x_norm, k)?;
            // V = wv @ x_norm  → [n_embd_kv, L]
            let v = ctx.alloc_tensor([n_kv_embd, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wv, x_norm, v)?;

            // RoPE on Q and K (in-place via separate scratch dst).
            let q_view = reshape_for_rope(q, head_dim, p.n_head as u64, l as u64);
            let k_view = reshape_for_rope(k, head_dim, p.n_head_kv as u64, l as u64);
            let q_roped = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
            let k_roped = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
            rope::record(ctx, q_view, positions_buf, q_roped, rope_params)?;
            rope::record(ctx, k_view, positions_buf, k_roped, rope_params)?;

            // Permute Q to [head_dim, L, n_head] and K/V to [head_dim, L, n_head_kv].
            let q_perm = permute_to_attn(q_roped, head_dim, l as u64, p.n_head as u64);
            let k_perm = permute_to_attn(k_roped, head_dim, l as u64, p.n_head_kv as u64);
            let v_perm = permute_to_attn(
                reshape_for_rope(v, head_dim, p.n_head_kv as u64, l as u64),
                head_dim,
                l as u64,
                p.n_head_kv as u64,
            );

            // attn_out = flash_attn(Q, K, V, mask) → [hidden, L]
            let attn_out = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            flash_attn::record(ctx, q_perm, k_perm, v_perm, mask, attn_out, fa_params)?;

            // proj = wo @ attn_out → [hidden, L]
            let proj = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.wo, attn_out, proj)?;

            // residual += proj
            let new_residual = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_add(ctx, residual, proj, new_residual)?;
            residual = new_residual;

            // x_norm = rms_norm(residual) * ffn_norm
            let x_norm2 = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            rms_norm::record(ctx, residual, block.ffn_norm, x_norm2, p.rms_eps)?;

            // gate = ffn_gate @ x_norm  → [n_ff, L]
            let gate = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_gate, x_norm2, gate)?;
            // up = ffn_up @ x_norm  → [n_ff, L]
            let up = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_up, x_norm2, up)?;
            // gate = silu(gate)
            let gate_silu = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_silu(ctx, gate, gate_silu)?;
            // ffn_hidden = gate * up
            let ffn_hidden = ctx.alloc_tensor([n_ff, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_mul(ctx, gate_silu, up, ffn_hidden)?;
            // down = ffn_down @ ffn_hidden → [hidden, L]
            let down = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            matmul::record(ctx, block.ffn_down, ffn_hidden, down)?;

            // residual += down
            let new_residual = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
            elementwise::record_add(ctx, residual, down, new_residual)?;
            residual = new_residual;
        }

        // ---- final norm + lm_head ----
        let final_norm = ctx.alloc_tensor([hidden, l as u64, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, residual, self.weights.output_norm, final_norm, p.rms_eps)?;

        // Compute logits for all positions, then return the slice for the
        // last token. (We could matmul only the last column by slicing
        // final_norm into [hidden, 1], but it requires the dst byte_offset
        // to be `min_storage_buffer_offset_alignment`-aligned and adds
        // little for the small batch sizes the MVP supports.)
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let all_logits = ctx.alloc_tensor([p.n_vocab as u64, l as u64, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, final_norm, all_logits)?;

        let elem_size = 4u64;
        let vocab = p.n_vocab as u64;
        let logits_range = crate::inference::buffer::BufferRange {
            buffer: all_logits.buffer,
            offset: all_logits.byte_offset + (l as u64 - 1) * vocab * elem_size,
            size: vocab * elem_size,
        };
        Ok(logits_range)
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
        for j in 0..l {
            buf[i * l + j] = if j <= i { 0.0 } else { f32::NEG_INFINITY };
        }
    }
    unsafe {
        let dst = host_ptr.add(mask.byte_offset as usize) as *mut f32;
        std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
    }
    Ok(())
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
        byte_stride: [elem, elem * head_dim, elem * head_dim * n_heads, elem * head_dim * n_heads * l],
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
        byte_stride: [elem, elem * head_dim * n_heads, elem * head_dim, elem * head_dim * n_heads * l],
        element_stride: [1, head_dim * n_heads, head_dim, head_dim * n_heads * l],
        dtype: t.dtype,
    }
}

fn parse_params(gguf: &GgufFile) -> Result<LlamaParams, Box<dyn Error>> {
    let u32_key = |k: &'static str| -> Result<u32, Box<dyn Error>> {
        let v = gguf
            .get(k)
            .ok_or(ModelError::MissingMetadata(k))?;
        coerce_u32(v).ok_or_else(|| {
            ModelError::BadMetadata {
                key: k,
                detail: format!("expected unsigned int, got {v:?}"),
            }
            .into()
        })
    };
    let f32_key = |k: &'static str| -> Result<f32, Box<dyn Error>> {
        let v = gguf
            .get(k)
            .ok_or(ModelError::MissingMetadata(k))?;
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
