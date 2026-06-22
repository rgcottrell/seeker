//! llama-server native (non-OpenAI / non-Anthropic) endpoint DTOs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// /embeddings (llama-server native) — request reuses the OpenAI
// `EmbeddingRequest` (it carries both `input` and `content`).
// ---------------------------------------------------------------------------

/// One element of the native bare-array embeddings response. `embedding` is a
/// **2D** array (`[[...]]`): per-token rows for `--pooling none`, else one row.
#[derive(Debug, Serialize)]
pub struct NativeEmbeddingObject {
    pub index: u32,
    pub embedding: Value,
}

// ---------------------------------------------------------------------------
// /completion (llama-server native)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub prompt: Option<Value>,
    #[serde(default)]
    pub n_predict: Option<i32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub repeat_last_n: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub cache_prompt: Option<bool>,
    #[serde(default)]
    pub id_slot: Option<i32>,
}

impl CompletionRequest {
    /// Translate sampling fields into a `SamplerConfig`, filling missing fields
    /// from the server's CLI-provided `base` defaults. `logit_bias` always
    /// inherits the CLI base (not exposed per-request).
    pub fn sampler_config(
        &self,
        base: &crate::inference::sample::SamplerConfig,
    ) -> crate::inference::sample::SamplerConfig {
        crate::inference::sample::SamplerConfig {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_k: self.top_k.unwrap_or(base.top_k),
            top_p: self.top_p.unwrap_or(base.top_p),
            min_p: self.min_p.unwrap_or(base.min_p),
            presence_penalty: self.presence_penalty.unwrap_or(base.presence_penalty),
            frequency_penalty: self.frequency_penalty.unwrap_or(base.frequency_penalty),
            repeat_penalty: self.repeat_penalty.unwrap_or(base.repeat_penalty),
            penalty_last_n: self.repeat_last_n.unwrap_or(base.penalty_last_n),
            seed: self.seed.unwrap_or(base.seed),
            logit_bias: base.logit_bias.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub content: String,
    pub stop: bool,
    pub model: String,
    pub tokens_predicted: u32,
    pub tokens_evaluated: u32,
}

// ---------------------------------------------------------------------------
// /infill
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct InfillRequest {
    #[serde(default)]
    pub input_prefix: Option<String>,
    #[serde(default)]
    pub input_suffix: Option<String>,
    #[serde(default)]
    pub input_extra: Option<Value>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub n_predict: Option<i32>,
    #[serde(default)]
    pub stream: Option<bool>,
}

// ---------------------------------------------------------------------------
// /tokenize, /detokenize
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct TokenizeRequest {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub add_special: Option<bool>,
    #[serde(default)]
    pub with_pieces: Option<bool>,
}

/// `tokens` is either a bare id array or, when `with_pieces` is set, an array
/// of `{id, piece}` objects — matching llama-server's two response shapes.
#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Value,
}

impl TokenizeResponse {
    pub fn ids(ids: Vec<u32>) -> Self {
        Self {
            tokens: Value::Array(ids.into_iter().map(Value::from).collect()),
        }
    }

    pub fn pieces(pairs: Vec<(u32, String)>) -> Self {
        let arr = pairs
            .into_iter()
            .map(|(id, piece)| serde_json::json!({"id": id, "piece": piece}))
            .collect();
        Self {
            tokens: Value::Array(arr),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct DetokenizeRequest {
    #[serde(default)]
    pub tokens: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct DetokenizeResponse {
    pub content: String,
}

// ---------------------------------------------------------------------------
// /apply-template
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct ApplyTemplateRequest {
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
    #[serde(default)]
    pub tools: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct ApplyTemplateResponse {
    pub prompt: String,
}

// ---------------------------------------------------------------------------
// /props (GET + POST)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct PropsResponse {
    pub model_path: Option<String>,
    pub chat_template: Option<String>,
    pub build_info: &'static str,
    pub default_generation_settings: Value,
    pub total_slots: u32,
}

#[derive(Debug, Default, Deserialize)]
pub struct PropsUpdateRequest {
    #[serde(default)]
    pub chat_template: Option<String>,
}

// ---------------------------------------------------------------------------
// /slots
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct SlotState {
    pub id: u32,
    pub is_processing: bool,
    pub prompt: String,
}

// ---------------------------------------------------------------------------
// /lora-adapters
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct LoraAdapter {
    pub id: u32,
    pub path: String,
    pub scale: f32,
}
