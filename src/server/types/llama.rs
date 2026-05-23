//! llama-server native (non-OpenAI / non-Anthropic) endpoint DTOs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const STUB_MODEL: &str = "seeker-stub";

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
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub cache_prompt: Option<bool>,
    #[serde(default)]
    pub id_slot: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub content: String,
    pub stop: bool,
    pub model: String,
    pub tokens_predicted: u32,
    pub tokens_evaluated: u32,
}

impl CompletionResponse {
    pub fn stub() -> Self {
        Self {
            content: super::openai::STUB_TEXT.to_string(),
            stop: true,
            model: STUB_MODEL.to_string(),
            tokens_predicted: 0,
            tokens_evaluated: 0,
        }
    }
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

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
}

impl TokenizeResponse {
    pub fn stub() -> Self {
        Self { tokens: vec![0] }
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

impl DetokenizeResponse {
    pub fn stub() -> Self {
        Self {
            content: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// /embedding (singular — native llama-server form, separate from OpenAI)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct LlamaEmbeddingRequest {
    #[serde(default)]
    pub content: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct LlamaEmbeddingResponse {
    pub embedding: Vec<f32>,
}

impl LlamaEmbeddingResponse {
    pub fn stub() -> Self {
        Self {
            embedding: vec![0.0; 8],
        }
    }
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

impl ApplyTemplateResponse {
    pub fn stub() -> Self {
        Self {
            prompt: String::new(),
        }
    }
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

impl PropsResponse {
    pub fn stub() -> Self {
        Self {
            model_path: None,
            chat_template: None,
            build_info: "seeker-stub",
            default_generation_settings: Value::Object(serde_json::Map::new()),
            total_slots: 0,
        }
    }
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
