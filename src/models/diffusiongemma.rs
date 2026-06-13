//! DiffusionGemma (`general.architecture == "diffusion-gemma"`) — a block
//! **text-diffusion** MoE on a Gemma-4 backbone (Google's diffusion Gemma 4;
//! e.g. `diffusiongemma-26B-A4B-it`). Unlike every other model here it is
//! **non-autoregressive**: generation fills a fixed-length "canvas" of
//! `canvas_length` tokens and iteratively denoises it (see
//! [`crate::inference::diffusion`]). A single bidirectional forward over
//! `[prompt | canvas]` (the UNIFIED phase) reproduces llama.cpp's
//! `build_diffusion_gemma` result.
//!
//! Three things are **region-aware**, split at `P = n_tokens − canvas_length`
//! (canvas = the last `canvas_length` positions), matching
//! `diffusion-gemma.cpp`:
//!   1. **input embeddings** — prompt = `embed·√n_embd`; canvas =
//!      `rms_norm_noscale(embed·√n_embd [+ self-conditioning])`.
//!   2. **per-layer scalar** — prompt × `enc_layer_output_scale`; canvas ×
//!      `layer_output_scale`.
//!   3. **attention mask** — prompt queries are causal over the prompt only
//!      (SWA-clipped on sliding layers); canvas queries are bidirectional over
//!      all prompt+canvas (global layers see all prompt, sliding layers only
//!      the last `n_swa−1` prompt positions).
//!
//! The Gemma-4 backbone is otherwise identical to [`super::gemma4`]: hybrid
//! per-layer SWA(head_dim 256)/global(head_dim 512) attention, per-head
//! Q/K-norm + weightless V-norm, attention scale 1.0, NEOX RoPE (global layers
//! use `rope_freqs`), sandwich norms, ×√n_embd embeddings, final-logit softcap.
//! The FFN is the Gemma-4 **dual** block: a dense shared MLP **plus** a
//! 128-expert MoE (the "A4B"), combined and post-normed (see [`Self::ffn_moe`]).

use std::error::Error;

use crate::gguf::{GgmlType, GgufFile};
use crate::inference::command::{record_compute_barriers, record_global_barrier};
use crate::inference::context::DispatchContext;
use crate::inference::kv_cache::KvCache;
use crate::inference::ops::{elementwise, flash_attn, matmul, moe, rms_norm, rope};
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::tokenizer::TokenizerBundle;

use super::gemma4::{coerce_f32, coerce_u32, read_bool_array, read_scalar_f32, read_u32_array};
use super::{CacheDims, DiffusionScInput, Model, ModelError};

const ARCH: &str = "diffusion-gemma";

#[derive(Debug, Clone)]
pub struct DiffusiongemmaParams {
    pub n_layer: u32,
    pub n_head: u32,
    pub n_embd: u32,
    /// Dense shared-MLP feed-forward length (`feed_forward_length`).
    pub n_ff: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub rms_eps: f32,
    pub embd_scale: f32,
    pub final_logit_softcap: f32,
    pub sliding_window: u32,
    pub head_dim_swa: u32,
    pub rope_base_swa: f32,
    pub head_dim_global: u32,
    pub rope_base_global: f32,
    /// Per-layer: is layer `il` a sliding-window layer? (else global)
    pub swa: Vec<bool>,
    pub n_head_kv: Vec<u32>,
    // ── MoE ──
    pub n_expert: u32,
    pub n_expert_used: u32,
    /// Per-expert feed-forward length (`expert_feed_forward_length`).
    pub expert_ff: u32,
    // ── diffusion ──
    /// Fixed canvas (block) length the denoiser generates per pass.
    pub canvas_length: u32,
    // ── per-layer region scalars (host constants) ──
    /// `× layer_output_scale[il]` applied to the canvas (decoder) region.
    pub out_scale: Vec<f32>,
    /// `× enc_layer_output_scale[il]` applied to the prompt (encoder) region.
    pub enc_out_scale: Vec<f32>,
    /// Whether the self-conditioning gated-MLP tensors are present.
    pub has_sc: bool,
}

impl DiffusiongemmaParams {
    pub(crate) fn head_dim(&self, il: usize) -> u32 {
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
    /// Q projection width = n_head · head_dim.
    fn q_dim(&self, il: usize) -> u32 {
        self.n_head * self.head_dim(il)
    }
    /// K/V projection width = n_head_kv · head_dim.
    fn kv_dim(&self, il: usize) -> u32 {
        self.n_head_kv[il] * self.head_dim(il)
    }
}

/// Per-layer attention + dual-FFN (dense shared MLP + 128-expert MoE) weights.
pub struct DiffusiongemmaBlockWeights {
    pub attn_norm: TensorView,
    pub wq: TensorView,
    pub wk: TensorView,
    /// `None` on global layers (V reuses the K projection).
    pub wv: Option<TensorView>,
    pub wo: TensorView,
    pub attn_q_norm: TensorView,
    pub attn_k_norm: TensorView,
    pub post_attn_norm: TensorView,
    // ── dense shared MLP ──
    pub ffn_norm: TensorView,
    pub ffn_gate: TensorView,
    pub ffn_up: TensorView,
    pub ffn_down: TensorView,
    pub ffn_post_norm_1: TensorView,
    // ── MoE ──
    /// Pre-norm applied to `attn_out` for the MoE *expert* input.
    pub ffn_pre_norm_2: TensorView,
    /// Post-norm applied to the summed MoE output.
    pub ffn_post_norm_2: TensorView,
    /// Router projection `[n_embd, n_expert]` (F32).
    pub ffn_gate_inp: TensorView,
    /// Per-channel router-input scale `[n_embd]` (F32).
    pub ffn_gate_inp_s: TensorView,
    /// Combined gate+up experts `[n_embd, 2·expert_ff, n_expert]`.
    pub ffn_gate_up_exps: TensorView,
    /// Down experts `[expert_ff, n_embd, n_expert]`.
    pub ffn_down_exps: TensorView,
    /// Per-expert down scale `[n_expert]` (F32), folded into the routing weights.
    pub ffn_down_exps_s: TensorView,
    /// Final post-FFN norm (combined dense+MoE) before the residual add.
    pub ffn_post_norm: TensorView,
}

/// Self-conditioning gated MLP (global; optional). Reads the previous
/// denoising step's canvas logits and adds a soft-embedding correction.
pub struct DiffusionScWeights {
    pub pre_norm: TensorView,
    pub gate: TensorView,
    pub up: TensorView,
    pub down: TensorView,
}

pub struct DiffusiongemmaWeights {
    pub token_embd: TensorView,
    pub blocks: Vec<DiffusiongemmaBlockWeights>,
    pub output_norm: TensorView,
    /// `None` ⇒ tied lm_head (uses `token_embd`).
    pub output: Option<TensorView>,
    /// Single `[head_dim_global/2]` freq-factor tensor for global layers.
    pub rope_freqs: Option<TensorView>,
    /// Self-conditioning gated MLP (present when `has_sc`).
    pub sc: Option<DiffusionScWeights>,
}

pub struct DiffusiongemmaModel {
    pub params: DiffusiongemmaParams,
    pub weights: DiffusiongemmaWeights,
    pub handle: WeightsHandle,
    pub tokenizer: TokenizerBundle,
}

impl DiffusiongemmaModel {
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

impl Model for DiffusiongemmaModel {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn vocab_size(&self) -> u32 {
        self.params.n_vocab
    }

    fn cache_dims(&self) -> CacheDims {
        // The UNIFIED diffusion forward does not use a persistent KV cache, but
        // the engine still queries representative dims at setup.
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

    fn diffusion_canvas_length(&self) -> Option<u32> {
        Some(self.params.canvas_length)
    }

    fn scratch_bytes_estimate(
        &self,
        n_ubatch: u32,
        max_seq_len: u32,
        _k_dtype: GgmlType,
        _v_dtype: GgmlType,
        _max_batch: u32,
    ) -> u64 {
        let p = &self.params;
        // The UNIFIED forward processes the whole [prompt|canvas] in one pass;
        // size for the largest such N this engine is configured to allow.
        let l = if n_ubatch == 0 {
            max_seq_len.max(1)
        } else {
            n_ubatch
        } as u64;
        let n_embd = p.n_embd as u64;
        let n_ff = p.n_ff as u64;
        let vocab = p.n_vocab as u64;
        let q_dim_max = (p.n_head * p.head_dim_global.max(p.head_dim_swa)) as u64;
        let kv_dim_max = (0..p.n_layer as usize)
            .map(|il| p.kv_dim(il) as u64)
            .max()
            .unwrap_or(0);
        let ff_exp = p.expert_ff as u64;
        let n_used = p.n_expert_used as u64;
        // Dense per-layer norms/proj/Q/KV/FFN transients (cf. gemma4) ...
        let per_layer = (10 * n_embd + 4 * q_dim_max + 6 * kv_dim_max + 5 * n_ff) * l * 4;
        // ... plus the MoE expert buffers [2·ff_exp, n_used, L] (gate_up) and
        // [ff_exp, n_used, L] (glu), and the router logits [n_expert, L].
        let moe = (3 * ff_exp * n_used + p.n_expert as u64) * l * 4;
        let residual = n_embd * l * 4;
        // Two region masks [N, N] (global + SWA), F32.
        let mask = 2 * l * l * 4;
        // lm_head computes only the canvas columns: [vocab, canvas_length].
        let logits = vocab * (p.canvas_length as u64).max(1) * 4;
        let raw = per_layer + moe + residual + mask + logits;
        raw + raw / 3 + (64 << 20)
    }

    fn record_forward(
        &self,
        _ctx: &mut DispatchContext,
        _cache: &mut KvCache,
        _tokens: &[u32],
        _position_offset: u32,
        _compute_logits: bool,
    ) -> Result<Option<TensorView>, Box<dyn Error>> {
        // diffusion-gemma is non-autoregressive: generation goes through the
        // diffusion denoiser ([`crate::inference::diffusion`]) which drives
        // [`Self::record_forward_diffusion`], not this single-seq path.
        Err("diffusion-gemma uses diffusion generation; call record_forward_diffusion".into())
    }

    fn record_forward_diffusion(
        &self,
        ctx: &mut DispatchContext,
        tokens: &[u32],
        n_prompt: u32,
        sc: Option<DiffusionScInput>,
    ) -> Result<TensorView, Box<dyn Error>> {
        self.forward_unified(ctx, tokens, n_prompt, sc)
    }
}

fn parse_params(gguf: &GgufFile) -> Result<DiffusiongemmaParams, Box<dyn Error>> {
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

    let n_layer = u32_key("diffusion-gemma.block_count")?;
    let n_head = u32_key("diffusion-gemma.attention.head_count")?;
    let n_embd = u32_key("diffusion-gemma.embedding_length")?;
    let n_ff = u32_key("diffusion-gemma.feed_forward_length")?;
    let n_ctx_train = u32_or("diffusion-gemma.context_length", 131072);
    let rms_eps = f32_or("diffusion-gemma.attention.layer_norm_rms_epsilon", 1e-6);
    let final_logit_softcap = f32_or("diffusion-gemma.final_logit_softcapping", 0.0);
    let sliding_window = u32_or("diffusion-gemma.attention.sliding_window", 0);

    let head_dim_global = u32_or("diffusion-gemma.attention.key_length", n_embd / n_head);
    let head_dim_swa = u32_or("diffusion-gemma.attention.key_length_swa", head_dim_global);
    let rope_base_global = f32_or("diffusion-gemma.rope.freq_base", 1_000_000.0);
    let rope_base_swa = f32_or("diffusion-gemma.rope.freq_base_swa", 10_000.0);

    let n_vocab = u32_key("diffusion-gemma.vocab_size").or_else(|_| derive_vocab(gguf))?;

    let swa = read_bool_array(
        gguf,
        "diffusion-gemma.attention.sliding_window_pattern",
        n_layer,
    )?;
    let n_head_kv = read_u32_array(gguf, "diffusion-gemma.attention.head_count_kv", n_layer)?;

    let n_expert = u32_or("diffusion-gemma.expert_count", 0);
    let n_expert_used = u32_or("diffusion-gemma.expert_used_count", 0);
    let expert_ff = u32_or("diffusion-gemma.expert_feed_forward_length", 0);

    let canvas_length = u32_key("diffusion.canvas_length")?;
    if canvas_length == 0 {
        return Err("diffusion-gemma requires a positive diffusion.canvas_length".into());
    }

    let out_scale: Vec<f32> = (0..n_layer)
        .map(|i| {
            read_scalar_f32(gguf, &format!("blk.{i}.layer_output_scale.weight")).unwrap_or(1.0)
        })
        .collect();
    let enc_out_scale: Vec<f32> = (0..n_layer)
        .map(|i| {
            read_scalar_f32(gguf, &format!("blk.{i}.enc_layer_output_scale.weight")).unwrap_or(1.0)
        })
        .collect();

    let has_sc = gguf.tensor("self_cond_gate.weight").is_some();

    Ok(DiffusiongemmaParams {
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
        n_expert,
        n_expert_used,
        expert_ff,
        canvas_length,
        out_scale,
        enc_out_scale,
        has_sc,
    })
}

impl DiffusiongemmaModel {
    /// Bidirectional UNIFIED forward over `[prompt | canvas]`. `tokens` is the
    /// full `P + C` sequence (prompt then canvas), `n_prompt = P`. Returns the
    /// **canvas** logits `[vocab, C]` (after final-logit softcap). No KV cache:
    /// the whole sequence is re-forwarded each denoising step (perf phase 4
    /// adds the prompt-KV cache).
    fn forward_unified(
        &self,
        ctx: &mut DispatchContext,
        tokens: &[u32],
        n_prompt: u32,
        sc: Option<DiffusionScInput>,
    ) -> Result<TensorView, Box<dyn Error>> {
        let p = &self.params;
        let n = tokens.len() as u32;
        if n == 0 {
            return Err("diffusion forward: empty sequence".into());
        }
        let big_p = n_prompt.min(n);
        let canvas = n - big_p;
        if canvas == 0 {
            return Err("diffusion forward: empty canvas (n_prompt >= n_tokens)".into());
        }
        let hidden = p.n_embd as u64;
        let nu = n as u64;
        let eps = p.rms_eps;

        // ---- prologue: tokens, positions, region masks ----
        let token_buf = ctx.alloc_scratch(nu * 4)?;
        write_u32(ctx, token_buf, tokens)?;
        let positions_buf = ctx.alloc_scratch(nu * 4)?;
        let positions: Vec<u32> = (0..n).collect();
        write_u32(ctx, positions_buf, &positions)?;

        // Two region masks (global + SWA), each [N, N] F32 (additive 0 / -inf).
        let mask_global = ctx.alloc_tensor([nu, nu, 1, 1], GgmlType::F32)?;
        write_region_mask(ctx, mask_global, big_p, n, p.sliding_window, false)?;
        let mask_swa = ctx.alloc_tensor([nu, nu, 1, 1], GgmlType::F32)?;
        write_region_mask(ctx, mask_swa, big_p, n, p.sliding_window, true)?;

        // ---- embedding × √n_embd, then canvas region → rms_norm(noscale)[+SC] ----
        let residual = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, self.weights.token_embd, token_buf, n, residual)?;
        elementwise::record_scale(ctx, residual, residual, p.embd_scale, 0.0)?;
        self.canvas_embed(ctx, residual, big_p, canvas, sc)?;

        let layer_cp = ctx.scratch_checkpoint();
        for (il, block) in self.weights.blocks.iter().enumerate() {
            ctx.scratch_restore(layer_cp);
            self.attention(
                ctx,
                block,
                residual,
                positions_buf,
                mask_global,
                mask_swa,
                il,
                n,
            )?;
            self.ffn_moe(ctx, block, residual, il, n)?;
            // Region-aware per-layer scalar: prompt × enc, canvas × dec.
            let enc = p.enc_out_scale[il];
            let dec = p.out_scale[il];
            if big_p > 0 {
                let prompt = col_slice(residual, 0, big_p as u64);
                elementwise::record_scale(ctx, prompt, prompt, enc, 0.0)?;
            }
            let cv = col_slice(residual, big_p as u64, canvas as u64);
            elementwise::record_scale(ctx, cv, cv, dec, 0.0)?;
        }
        ctx.scratch_restore(layer_cp);

        // ---- final norm + lm_head over the CANVAS columns only → [vocab, C] ----
        let vocab = p.n_vocab as u64;
        let lm_head = self.weights.output.unwrap_or(self.weights.token_embd);
        let canvas_res = col_slice(residual, big_p as u64, canvas as u64);
        let final_norm = ctx.alloc_tensor([hidden, canvas as u64, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, canvas_res, self.weights.output_norm, final_norm, eps)?;
        let logits = ctx.alloc_tensor([vocab, canvas as u64, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, lm_head, final_norm, logits)?;

        if p.final_logit_softcap != 0.0 {
            let cap = p.final_logit_softcap;
            elementwise::record_scale(ctx, logits, logits, 1.0 / cap, 0.0)?;
            elementwise::record_tanh(ctx, logits, logits)?;
            elementwise::record_scale(ctx, logits, logits, cap, 0.0)?;
        }
        Ok(logits)
    }

    /// Canvas input embedding: `rms_norm_noscale(scaled_embed [+ self-cond])`.
    /// Applied in place to columns `[P, P+C)` of `residual` (the prompt columns
    /// keep the plain scaled embedding). `sc` adds the self-conditioning
    /// correction before the norm (phase 3).
    fn canvas_embed(
        &self,
        ctx: &mut DispatchContext,
        residual: TensorView,
        big_p: u32,
        canvas: u32,
        sc: Option<DiffusionScInput>,
    ) -> Result<(), Box<dyn Error>> {
        let hidden = self.params.n_embd as u64;
        let canvas_view = col_slice(residual, big_p as u64, canvas as u64);
        // Self-conditioning (phase 3) adds its correction into `canvas_view`
        // before the norm; without it the canvas is just rms_norm_noscale.
        if let Some(sc) = sc {
            self.self_condition(ctx, canvas_view, canvas, sc)?;
        }
        // rms_norm(noscale) into a temp, then copy back into the canvas columns.
        let normed = ctx.alloc_tensor([hidden, canvas as u64, 1, 1], GgmlType::F32)?;
        rms_norm::record_noweight(ctx, canvas_view, normed, self.params.rms_eps)?;
        copy_buffer(
            ctx,
            normed.range(),
            canvas_view.byte_offset,
            canvas_view.buffer,
        );
        Ok(())
    }

    /// Attention block (UNIFIED, region mask): in-place `residual += post_attn_norm(O·attn)`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        ctx: &mut DispatchContext,
        block: &DiffusiongemmaBlockWeights,
        residual: TensorView,
        positions_buf: crate::inference::buffer::BufferRange,
        mask_global: TensorView,
        mask_swa: TensorView,
        il: usize,
        n: u32,
    ) -> Result<(), Box<dyn Error>> {
        let p = &self.params;
        let hidden = p.n_embd as u64;
        let nu = n as u64;
        let eps = p.rms_eps;
        let head_dim = p.head_dim(il) as u64;
        let n_head = p.n_head as u64;
        let n_head_kv = p.n_head_kv[il] as u64;
        let q_dim = p.q_dim(il) as u64;
        let kv_dim = p.kv_dim(il) as u64;
        let n_rot = head_dim as u32; // full rotation
        let rope_params = rope::RopeParams::llama_default(n_rot, p.rope_base(il));
        let freq_factors = if p.swa[il] {
            None
        } else {
            self.weights.rope_freqs.as_ref().map(|t| t.range())
        };
        let fa_params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim as u32,
            head_dim_v: head_dim as u32,
            gqa_ratio: (n_head / n_head_kv).max(1) as u32,
            scale: 1.0,    // gemma backbone: NO 1/sqrt(head_dim)
            swa_window: 0, // SWA baked into the explicit region mask
            ring_depth: 0,
        };
        let mask = if p.swa[il] { mask_swa } else { mask_global };

        let x_norm = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, residual, block.attn_norm, x_norm, eps)?;
        ctx.tap(&format!("attn_norm-{il}"), x_norm)?;

        let q = ctx.alloc_tensor([q_dim, nu, 1, 1], GgmlType::F32)?;
        matmul::record_nofence(ctx, block.wq, x_norm, q)?;
        let k = ctx.alloc_tensor([kv_dim, nu, 1, 1], GgmlType::F32)?;
        matmul::record_nofence(ctx, block.wk, x_norm, k)?;
        let v_proj = if let Some(wv) = block.wv {
            let v = ctx.alloc_tensor([kv_dim, nu, 1, 1], GgmlType::F32)?;
            matmul::record_nofence(ctx, wv, x_norm, v)?;
            record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range(), v.range()]);
            v
        } else {
            record_compute_barriers(ctx.device, ctx.cmd, &[q.range(), k.range()]);
            k // global layer: V := K projection
        };

        // Per-head Q-norm / K-norm, weightless V-norm, then NEOX RoPE on Q/K.
        let q_view = reshape_for_rope(q, head_dim, n_head, nu);
        let q_normed = ctx.alloc_tensor(q_view.dims, GgmlType::F32)?;
        rms_norm::record_nofence(ctx, q_view, block.attn_q_norm, q_normed, eps)?;
        let k_view = reshape_for_rope(k, head_dim, n_head_kv, nu);
        let k_normed = ctx.alloc_tensor(k_view.dims, GgmlType::F32)?;
        rms_norm::record_nofence(ctx, k_view, block.attn_k_norm, k_normed, eps)?;
        let v_view = reshape_for_rope(v_proj, head_dim, n_head_kv, nu);
        let v_normed = ctx.alloc_tensor(v_view.dims, GgmlType::F32)?;
        rms_norm::record_noweight_nofence(ctx, v_view, v_normed, eps)?;
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

        let q_perm = permute_to_attn(q_roped, head_dim, nu, n_head);
        let k_perm = permute_to_attn(k_roped, head_dim, nu, n_head_kv);
        let v_perm = permute_to_attn(v_normed, head_dim, nu, n_head_kv);
        let attn_out = ctx.alloc_tensor([q_dim, nu, 1, 1], GgmlType::F32)?;
        flash_attn::record(
            ctx,
            q_perm,
            k_perm,
            v_perm,
            Some(mask),
            attn_out,
            fa_params,
            n,
        )?;

        // O-proj → post_attention_norm → residual add.
        let proj = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, block.wo, attn_out, proj)?;
        let proj_normed = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, proj, block.post_attn_norm, proj_normed, eps)?;
        elementwise::record_add(ctx, residual, proj_normed, residual)?;
        ctx.tap(&format!("attn_out-{il}"), residual)?;
        Ok(())
    }

    /// Dense shared MLP + 128-expert MoE on the post-attention residual
    /// `attn_out` (= `residual`), then `ffn_post_norm` + residual add — updates
    /// `residual` in place. Mirrors llama.cpp's `gemma4_build_ffn_moe`.
    fn ffn_moe(
        &self,
        ctx: &mut DispatchContext,
        block: &DiffusiongemmaBlockWeights,
        residual: TensorView,
        il: usize,
        n: u32,
    ) -> Result<(), Box<dyn Error>> {
        let p = &self.params;
        let hidden = p.n_embd as u64;
        let n_ff = p.n_ff as u64;
        let ff_exp = p.expert_ff as u64;
        let nu = n as u64;
        let eps = p.rms_eps;
        let n_experts = p.n_expert;
        let n_used = p.n_expert_used;

        // ── dense shared MLP: rms(ffn_norm) → gelu·gate*up → down → rms(post_1) ──
        let dx = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, residual, block.ffn_norm, dx, eps)?;
        let dgate = ctx.alloc_tensor([n_ff, nu, 1, 1], GgmlType::F32)?;
        matmul::record_nofence(ctx, block.ffn_gate, dx, dgate)?;
        let dup = ctx.alloc_tensor([n_ff, nu, 1, 1], GgmlType::F32)?;
        matmul::record_nofence(ctx, block.ffn_up, dx, dup)?;
        record_compute_barriers(ctx.device, ctx.cmd, &[dgate.range(), dup.range()]);
        let dgelu = ctx.alloc_tensor([n_ff, nu, 1, 1], GgmlType::F32)?;
        elementwise::record_gelu(ctx, dgate, dgelu)?;
        let dh = ctx.alloc_tensor([n_ff, nu, 1, 1], GgmlType::F32)?;
        elementwise::record_mul(ctx, dgelu, dup, dh)?;
        let dense = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, block.ffn_down, dh, dense)?;
        let dense_normed = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, dense, block.ffn_post_norm_1, dense_normed, eps)?;

        // ── router: rms_noscale(attn_out)·(1/√n_embd)·gate_inp_s → gate_inp ──
        let r_in = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        rms_norm::record_noweight(ctx, residual, r_in, eps)?;
        let inv_sqrt = 1.0f32 / (p.n_embd as f32).sqrt();
        elementwise::record_scale(ctx, r_in, r_in, inv_sqrt, 0.0)?;
        let gate_inp_s = broadcast_col(block.ffn_gate_inp_s, nu);
        elementwise::record_mul(ctx, r_in, gate_inp_s, r_in)?;
        let gate_logits = ctx.alloc_tensor([n_experts as u64, nu, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, block.ffn_gate_inp, r_in, gate_logits)?;
        ctx.tap(&format!("ffn_moe_logits-{il}"), gate_logits)?;

        // topk → ids + weights, then fold the per-expert down scale into weights.
        let ids = ctx.alloc_scratch((n_experts as u64) * nu * 4)?;
        let weights_buf = ctx.alloc_scratch((n_used as u64) * nu * 4)?;
        moe::record_topk_moe(
            ctx,
            gate_logits,
            weights_buf,
            ids,
            moe::TopkMoeParams {
                n_experts,
                n_expert_used: n_used,
                gating_func: moe::GATING_SOFTMAX,
                with_norm: true,
            },
        )?;
        moe::record_moe_expert_weight_scale(
            ctx,
            ids,
            block.ffn_down_exps_s,
            weights_buf,
            n_used,
            n_experts,
            n,
        )?;

        // ── experts: rms(pre_2) → gate_up matvec_id → geglu → fused down ──
        let ex = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, residual, block.ffn_pre_norm_2, ex, eps)?;
        let gate_up = ctx.alloc_tensor([2 * ff_exp, n_used as u64, nu, 1], GgmlType::F32)?;
        moe::record_matvec_q5k_id(ctx, block.ffn_gate_up_exps, ex, ids, gate_up, n_used)?;
        let ffn_h = ctx.alloc_tensor([ff_exp, n_used as u64, nu, 1], GgmlType::F32)?;
        elementwise::record_geglu_fused(ctx, gate_up, ffn_h)?;
        let routed = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        moe::record_moe_down_q8_0(
            ctx,
            block.ffn_down_exps,
            ffn_h,
            ids,
            weights_buf,
            routed,
            n_used,
        )?;
        let moe_normed = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, routed, block.ffn_post_norm_2, moe_normed, eps)?;

        // ── combine dense + moe → rms(ffn_post_norm) → residual add ──
        let combined = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        elementwise::record_add(ctx, dense_normed, moe_normed, combined)?;
        let combined_normed = ctx.alloc_tensor([hidden, nu, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, combined, block.ffn_post_norm, combined_normed, eps)?;
        elementwise::record_add(ctx, residual, combined_normed, residual)?;
        ctx.tap(&format!("ffn_out-{il}"), residual)?;
        Ok(())
    }

    /// Self-conditioning correction added into the canvas columns before the
    /// canvas norm (phase 3). Stub for phase 2 (no SC).
    fn self_condition(
        &self,
        _ctx: &mut DispatchContext,
        _canvas: TensorView,
        _canvas_len: u32,
        _sc: DiffusionScInput,
    ) -> Result<(), Box<dyn Error>> {
        Err("diffusion self-conditioning not yet implemented".into())
    }
}

fn derive_vocab(gguf: &GgufFile) -> Result<u32, Box<dyn Error>> {
    gguf.tensor("token_embd.weight")
        .map(|t| t.dims[1] as u32)
        .ok_or_else(|| ModelError::MissingMetadata("diffusion-gemma.vocab_size").into())
}

fn collect_weights(
    handle: &WeightsHandle,
    params: &DiffusiongemmaParams,
) -> Result<DiffusiongemmaWeights, Box<dyn Error>> {
    let view = |name: &str| -> Result<TensorView, Box<dyn Error>> {
        handle
            .view(name)
            .map_err(|_| ModelError::MissingTensor(name.to_string()).into())
    };

    let token_embd = view("token_embd.weight")?;
    let output_norm = view("output_norm.weight")?;
    let output = handle.view("output.weight").ok();
    let rope_freqs = handle.view("rope_freqs.weight").ok();

    let sc = if params.has_sc {
        Some(DiffusionScWeights {
            pre_norm: view("self_cond_pre_norm.weight")?,
            gate: view("self_cond_gate.weight")?,
            up: view("self_cond_up.weight")?,
            down: view("self_cond_down.weight")?,
        })
    } else {
        None
    };

    let mut blocks = Vec::with_capacity(params.n_layer as usize);
    for i in 0..params.n_layer {
        blocks.push(DiffusiongemmaBlockWeights {
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
            ffn_post_norm_1: view(&format!("blk.{i}.post_ffw_norm_1.weight"))?,
            ffn_pre_norm_2: view(&format!("blk.{i}.pre_ffw_norm_2.weight"))?,
            ffn_post_norm_2: view(&format!("blk.{i}.post_ffw_norm_2.weight"))?,
            ffn_gate_inp: view(&format!("blk.{i}.ffn_gate_inp.weight"))?,
            ffn_gate_inp_s: view(&format!("blk.{i}.ffn_gate_inp.scale"))?,
            ffn_gate_up_exps: view(&format!("blk.{i}.ffn_gate_up_exps.weight"))?,
            ffn_down_exps: view(&format!("blk.{i}.ffn_down_exps.weight"))?,
            ffn_down_exps_s: view(&format!("blk.{i}.ffn_down_exps.scale"))?,
            ffn_post_norm: view(&format!("blk.{i}.post_ffw_norm.weight"))?,
        });
    }

    Ok(DiffusiongemmaWeights {
        token_embd,
        blocks,
        output_norm,
        output,
        rope_freqs,
        sc,
    })
}

// ── forward helpers ───────────────────────────────────────────────────────

/// Columns `[c0, c0+count)` of a contiguous `[hidden, N]` F32 tensor.
fn col_slice(t: TensorView, c0: u64, count: u64) -> TensorView {
    let col_stride = t.byte_stride[1];
    TensorView {
        byte_offset: t.byte_offset + c0 * col_stride,
        byte_size: count * col_stride,
        dims: [t.dims[0], count, 1, 1],
        ..t
    }
}

/// Broadcast a `[hidden, 1]` (per-channel) vector across `n` columns (col
/// stride 0) for an element-wise `mul` against a `[hidden, n]` tensor.
fn broadcast_col(t: TensorView, n: u64) -> TensorView {
    TensorView {
        dims: [t.dims[0], n, 1, 1],
        byte_stride: [t.byte_stride[0], 0, 0, 0],
        element_stride: [t.element_stride[0], 0, 0, 0],
        ..t
    }
}

/// `[q_dim, L]` projection → `[head_dim, n_heads, L]` (RoPE/per-head layout).
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

/// `[head_dim, n_heads, L]` → `[head_dim, L, n_heads]` (flash-attn layout).
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

/// Region-aware additive attention mask `[N, N]` (row-major `[query, key]`,
/// `0` = visible, `-inf` = masked), matching llama.cpp's
/// `llm_graph_input_attn_diffusion::set_input`. Prompt queries (`q < P`) are
/// causal over the prompt only (SWA-clipped on sliding layers); canvas queries
/// (`q >= P`) are bidirectional over all prompt+canvas, with sliding layers
/// reaching only the last `n_swa-1` prompt positions.
fn write_region_mask(
    ctx: &mut DispatchContext,
    mask: TensorView,
    big_p: u32,
    n: u32,
    n_swa: u32,
    swa: bool,
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch region not host-visible")?;
    let buf = region_mask_values(big_p, n, n_swa, swa);
    unsafe {
        let dst = host_ptr.add(mask.byte_offset as usize) as *mut f32;
        std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
    }
    Ok(())
}

/// Pure `[N, N]` row-major (`[query, key]`) additive mask values (`0` visible,
/// `-inf` masked). Split out from [`write_region_mask`] so the region logic is
/// unit-testable without a GPU.
fn region_mask_values(big_p: u32, n: u32, n_swa: u32, swa: bool) -> Vec<f32> {
    let nn = n as usize;
    let p = big_p as i64;
    let n_swa = n_swa as i64;
    let canvas_prompt_lo = p - n_swa + 1;
    let mut buf: Vec<f32> = vec![f32::NEG_INFINITY; nn * nn];
    for q in 0..nn {
        let q_is_canvas = q as i64 >= p;
        for k in 0..nn {
            let k_is_canvas = k as i64 >= p;
            let mut allow = if q_is_canvas {
                if swa {
                    k_is_canvas || (k as i64 >= canvas_prompt_lo)
                } else {
                    true
                }
            } else {
                !k_is_canvas && (k <= q)
            };
            // STANDARD SWA clip on prompt (causal) queries.
            if allow && swa && !q_is_canvas && (q as i64 - k as i64 >= n_swa) {
                allow = false;
            }
            if allow {
                buf[q * nn + k] = 0.0;
            }
        }
    }
    buf
}

/// GPU buffer→buffer copy (`src_range` → `dst_buffer` at `dst_offset`), fenced
/// both sides. Used to write the canvas-normed embedding back into the residual.
fn copy_buffer(
    ctx: &mut DispatchContext,
    src: crate::inference::buffer::BufferRange,
    dst_offset: u64,
    dst_buffer: ash::vk::Buffer,
) {
    record_global_barrier(ctx.device, ctx.cmd);
    unsafe {
        let copy = ash::vk::BufferCopy::default()
            .src_offset(src.offset)
            .dst_offset(dst_offset)
            .size(src.size);
        ctx.device.device.cmd_copy_buffer(
            ctx.cmd,
            src.buffer,
            dst_buffer,
            std::slice::from_ref(&copy),
        );
    }
    record_global_barrier(ctx.device, ctx.cmd);
}

#[cfg(test)]
mod tests {
    use super::region_mask_values;

    fn visible(m: &[f32], n: usize, q: usize, k: usize) -> bool {
        m[q * n + k] == 0.0
    }

    #[test]
    fn global_mask_prompt_causal_canvas_bidirectional() {
        // P=3 prompt, C=2 canvas, N=5, global layer (large window → no clip).
        let n = 5;
        let m = region_mask_values(3, n as u32, 1024, false);
        // Prompt query q=1: causal over prompt only (k=0,1 visible; 2,3,4 not).
        assert!(visible(&m, n, 1, 0) && visible(&m, n, 1, 1));
        assert!(!visible(&m, n, 1, 2) && !visible(&m, n, 1, 3) && !visible(&m, n, 1, 4));
        // Prompt query never sees the canvas.
        assert!(!visible(&m, n, 0, 3) && !visible(&m, n, 2, 4));
        // Canvas query q=4 (global): sees everything (all prompt + all canvas).
        for k in 0..n {
            assert!(visible(&m, n, 4, k), "canvas query should see key {k}");
        }
    }

    #[test]
    fn swa_mask_clips_prompt_window_and_canvas_reach() {
        // P=5 prompt, C=2 canvas, N=7, sliding window n_swa=2.
        let n = 7;
        let m = region_mask_values(5, n as u32, 2, true);
        // Prompt query q=4, window 2 → only k in (q-2, q] = {3,4} visible.
        assert!(visible(&m, n, 4, 3) && visible(&m, n, 4, 4));
        assert!(!visible(&m, n, 4, 2) && !visible(&m, n, 4, 1));
        // Canvas query q=6 (sliding): sees all canvas (k=5,6) + last n_swa-1=1
        // prompt position (canvas_prompt_lo = 5-2+1 = 4 → k>=4 of the prompt).
        assert!(visible(&m, n, 6, 4) && visible(&m, n, 6, 5) && visible(&m, n, 6, 6));
        assert!(!visible(&m, n, 6, 3) && !visible(&m, n, 6, 0));
    }
}
