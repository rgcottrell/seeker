//! `gemma4-assistant` — the MTP / EAGLE-style **draft model** for gemma4
//! speculative decoding. Unlike qwen35moe's self-speculation (a single NextN
//! block living inside the base GGUF), gemma4 ships its draft as a *separate*
//! small GGUF (`MTP/gemma-4-12B-it-MTP-*.gguf`, `general.architecture ==
//! "gemma4-assistant"`) that is paired with the base model at runtime.
//!
//! Shape (from the unsloth MTP GGUF, confirmed by inspect):
//!   * 4 transformer blocks (`nextn_predict_layers = 4`), draft hidden
//!     `embedding_length = 1024`, output `embedding_length_out = 3840` (== base
//!     `gemma4.embedding_length`), `feed_forward_length = 8192`.
//!   * Same hybrid attention as gemma4: `sliding_window_pattern =
//!     [t,t,t,false]`, `key_length 512 / key_length_swa 256`, `head_count 16`,
//!     `head_count_kv [8,8,8,1]`, rope `freq_base 1e6 / 1e4`.
//!   * **No `attn_k` / `attn_v`** — the draft borrows the BASE model's K/V
//!     (EAGLE-style; `shared_kv_layers = 4`). Each block has `attn_q`,
//!     `attn_q_norm`, `attn_output`, the GeGLU FFN, and the gemma4 sandwich
//!     norms + `layer_output_scale`.
//!   * `nextn.pre_projection` `[7680, 1024]` (= 2·3840 → 1024) folds the base
//!     hidden + token embedding into the draft hidden; `nextn.post_projection`
//!     `[1024, 3840]` lifts the draft hidden back to base space, then
//!     `output_norm` + the BASE tied lm_head produce draft logits.
//!
//! This module currently implements **loading + validation** (the
//! separate-draft-GGUF plumbing). The draft forward + engine wiring land in a
//! follow-up step; see the gemma4 MTP plan.

use std::error::Error;

use crate::gguf::GgufFile;
use crate::inference::weights::{TensorView, WeightsHandle};

use super::ModelError;
use super::gemma4::{coerce_f32, coerce_u32, read_bool_array, read_scalar_f32, read_u32_array};

pub const ARCH: &str = "gemma4-assistant";

#[derive(Debug, Clone)]
pub struct Gemma4AssistantParams {
    pub n_layer: u32,
    pub n_head: u32,
    /// Draft hidden width (1024) — distinct from the base model's hidden.
    pub n_embd: u32,
    /// Output width the draft projects back to (== base `gemma4.embedding_length`).
    pub n_embd_out: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub rms_eps: f32,
    pub sliding_window: u32,
    pub head_dim_swa: u32,
    pub rope_base_swa: f32,
    pub head_dim_global: u32,
    pub rope_base_global: f32,
    /// Per-layer: is layer `il` a sliding-window layer? (else global)
    pub swa: Vec<bool>,
    /// Per-layer query-head KV count (8 SWA, 1 global) — describes how the
    /// draft's Q attends over the borrowed base K/V.
    pub n_head_kv: Vec<u32>,
    /// Per-layer output scalar (`cur *= layer_output_scale[il]`).
    pub layer_output_scale: Vec<f32>,
}

impl Gemma4AssistantParams {
    pub fn head_dim(&self, il: usize) -> u32 {
        if self.swa[il] {
            self.head_dim_swa
        } else {
            self.head_dim_global
        }
    }
    pub fn rope_base(&self, il: usize) -> f32 {
        if self.swa[il] {
            self.rope_base_swa
        } else {
            self.rope_base_global
        }
    }
    /// Q projection width = n_head · head_dim.
    pub fn q_dim(&self, il: usize) -> u32 {
        self.n_head * self.head_dim(il)
    }
}

pub struct Gemma4AssistantBlockWeights {
    pub attn_norm: TensorView,
    pub wq: TensorView,
    pub attn_q_norm: TensorView,
    pub wo: TensorView,
    pub post_attn_norm: TensorView,
    pub ffn_norm: TensorView,
    pub ffn_gate: TensorView,
    pub ffn_up: TensorView,
    pub ffn_down: TensorView,
    pub post_ffw_norm: TensorView,
}

pub struct Gemma4AssistantWeights {
    /// Draft's own `[n_embd(1024), vocab]` token embedding (input side).
    pub token_embd: TensorView,
    /// `[7680, 1024]` — folds (base hidden ⊕ token embedding) → draft hidden.
    pub pre_projection: TensorView,
    /// `[1024, 3840]` — lifts draft hidden back to base hidden space.
    pub post_projection: TensorView,
    pub output_norm: TensorView,
    /// Global-layer rope freq-factors (`[head_dim_global/2]`).
    pub rope_freqs: Option<TensorView>,
    pub blocks: Vec<Gemma4AssistantBlockWeights>,
}

/// A loaded gemma4 draft model (its own GGUF + uploaded weights), paired with a
/// base gemma4 model for speculative decoding.
pub struct Gemma4AssistantDraft {
    pub params: Gemma4AssistantParams,
    pub weights: Gemma4AssistantWeights,
    pub handle: WeightsHandle,
}

impl Gemma4AssistantDraft {
    /// Load + validate the draft against the base model's hidden width and
    /// vocabulary. `handle` is the draft GGUF's uploaded weights.
    pub fn load(
        gguf: &GgufFile,
        handle: WeightsHandle,
        base_n_embd: u32,
        base_n_vocab: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let arch = gguf
            .architecture()
            .ok_or(ModelError::MissingMetadata("general.architecture"))?;
        if arch != ARCH {
            return Err(ModelError::BadMetadata {
                key: "general.architecture",
                detail: format!("draft model must be `{ARCH}`, got `{arch}`"),
            }
            .into());
        }

        let params = parse_params(gguf)?;
        let weights = collect_weights(&handle, &params)?;

        // Cross-checks vs the base model it will drive.
        if params.n_embd_out != base_n_embd {
            return Err(ModelError::BadMetadata {
                key: "gemma4-assistant.embedding_length_out",
                detail: format!(
                    "draft output width {} != base hidden {base_n_embd}",
                    params.n_embd_out
                ),
            }
            .into());
        }
        if params.n_vocab != base_n_vocab {
            return Err(ModelError::BadMetadata {
                key: "gemma4-assistant.vocab",
                detail: format!(
                    "draft vocab {} != base vocab {base_n_vocab}",
                    params.n_vocab
                ),
            }
            .into());
        }
        // post_projection lifts draft hidden → base hidden; its output row count
        // must equal base hidden (so the base lm_head consumes it).
        let post_out = weights.post_projection.dims[1] as u32;
        if post_out != base_n_embd {
            return Err(ModelError::BadMetadata {
                key: "nextn.post_projection",
                detail: format!("post_projection output {post_out} != base hidden {base_n_embd}"),
            }
            .into());
        }

        Ok(Self {
            params,
            weights,
            handle,
        })
    }

    /// One-line load summary for logging.
    pub fn summary(&self) -> String {
        format!(
            "gemma4-assistant draft: {} blocks, hidden {} → {}, ff {}, heads {}, \
             kv/layer {:?}, swa {:?}, {} tensors",
            self.params.n_layer,
            self.params.n_embd,
            self.params.n_embd_out,
            self.params.n_ff,
            self.params.n_head,
            self.params.n_head_kv,
            self.params.swa,
            self.handle.views.len(),
        )
    }
}

fn parse_params(gguf: &GgufFile) -> Result<Gemma4AssistantParams, Box<dyn Error>> {
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

    let n_layer = u32_key("gemma4-assistant.block_count")?;
    let n_head = u32_key("gemma4-assistant.attention.head_count")?;
    let n_embd = u32_key("gemma4-assistant.embedding_length")?;
    let n_embd_out = u32_key("gemma4-assistant.embedding_length_out")?;
    let n_ff = u32_key("gemma4-assistant.feed_forward_length")?;
    let n_ctx_train = u32_or("gemma4-assistant.context_length", 131072);
    let rms_eps = f32_or("gemma4-assistant.attention.layer_norm_rms_epsilon", 1e-6);
    let sliding_window = u32_or("gemma4-assistant.attention.sliding_window", 0);

    let head_dim_global = u32_or("gemma4-assistant.attention.key_length", n_embd / n_head);
    let head_dim_swa = u32_or("gemma4-assistant.attention.key_length_swa", head_dim_global);
    let rope_base_global = f32_or("gemma4-assistant.rope.freq_base", 1_000_000.0);
    let rope_base_swa = f32_or("gemma4-assistant.rope.freq_base_swa", 10_000.0);

    // No explicit vocab key — derive from the draft token_embd row count.
    let n_vocab = gguf
        .tensor("token_embd.weight")
        .map(|t| t.dims[1] as u32)
        .ok_or(ModelError::MissingTensor("token_embd.weight".to_string()))?;

    let swa = read_bool_array(
        gguf,
        "gemma4-assistant.attention.sliding_window_pattern",
        n_layer,
    )?;
    let n_head_kv = read_u32_array(gguf, "gemma4-assistant.attention.head_count_kv", n_layer)?;

    let layer_output_scale: Vec<f32> = (0..n_layer)
        .map(|i| {
            read_scalar_f32(gguf, &format!("blk.{i}.layer_output_scale.weight")).unwrap_or(1.0)
        })
        .collect();

    Ok(Gemma4AssistantParams {
        n_layer,
        n_head,
        n_embd,
        n_embd_out,
        n_ff,
        n_vocab,
        n_ctx_train,
        rms_eps,
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

fn collect_weights(
    handle: &WeightsHandle,
    params: &Gemma4AssistantParams,
) -> Result<Gemma4AssistantWeights, Box<dyn Error>> {
    let view = |name: &str| -> Result<TensorView, Box<dyn Error>> {
        handle
            .view(name)
            .map_err(|_| ModelError::MissingTensor(name.to_string()).into())
    };

    let token_embd = view("token_embd.weight")?;
    let pre_projection = view("nextn.pre_projection.weight")?;
    let post_projection = view("nextn.post_projection.weight")?;
    let output_norm = view("output_norm.weight")?;
    let rope_freqs = handle.view("rope_freqs.weight").ok();

    let mut blocks = Vec::with_capacity(params.n_layer as usize);
    for i in 0..params.n_layer {
        blocks.push(Gemma4AssistantBlockWeights {
            attn_norm: view(&format!("blk.{i}.attn_norm.weight"))?,
            wq: view(&format!("blk.{i}.attn_q.weight"))?,
            attn_q_norm: view(&format!("blk.{i}.attn_q_norm.weight"))?,
            wo: view(&format!("blk.{i}.attn_output.weight"))?,
            post_attn_norm: view(&format!("blk.{i}.post_attention_norm.weight"))?,
            ffn_norm: view(&format!("blk.{i}.ffn_norm.weight"))?,
            ffn_gate: view(&format!("blk.{i}.ffn_gate.weight"))?,
            ffn_up: view(&format!("blk.{i}.ffn_up.weight"))?,
            ffn_down: view(&format!("blk.{i}.ffn_down.weight"))?,
            post_ffw_norm: view(&format!("blk.{i}.post_ffw_norm.weight"))?,
        });
    }

    Ok(Gemma4AssistantWeights {
        token_embd,
        pre_projection,
        post_projection,
        output_norm,
        rope_freqs,
        blocks,
    })
}
