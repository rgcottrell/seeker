//! LLaMA-architecture model. Loads architecture parameters and per-layer
//! weight handles from a GGUF, then implements [`Model::record_forward`]
//! against the inference dispatch primitives.

use std::error::Error;

use crate::gguf::{GgufFile, MetadataValue};
use crate::inference::context::DispatchContext;
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

    fn record_forward(
        &self,
        _ctx: &mut DispatchContext,
        _tokens: &[u32],
    ) -> Result<(), Box<dyn Error>> {
        // TODO: record per-block dispatch sequence using inference::ops::* helpers.
        Err("LlamaModel::record_forward not yet implemented".into())
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

    // Vocab size: prefer the explicit metadata field; fall back to tokens
    // array length read elsewhere — for LlamaParams we want the value the
    // matmul shapes use, which is `llama.vocab_size` when present.
    let n_vocab = u32_key("llama.vocab_size").or_else(|_| {
        // Older GGUFs omit it — derive from token_embd shape later. For now,
        // require it (SmolLM2 has it).
        Err::<u32, Box<dyn Error>>(ModelError::MissingMetadata("llama.vocab_size").into())
    })?;

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
    let output = handle.view("output.weight").ok(); // tied if absent

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
