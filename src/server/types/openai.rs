//! OpenAI-compatible request / response DTOs.
//!
//! Request types are deliberately permissive: unknown fields are accepted
//! (via `#[serde(default)]` on every option) so callers can hand us their
//! full payload and have it round-trip cleanly until inference is wired up.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::Usage;

pub const STUB_TEXT: &str = "[stub] seeker serve has no inference backend wired up yet";
pub const STUB_MODEL: &str = "seeker-stub";

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    // ─── Sampling (OpenAI + llama-server extensions) ──────────────────
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// llama-server extension. OpenAI itself doesn't expose top_k.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// llama-server extension.
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// llama-server extension; OpenAI doesn't expose a multiplicative
    /// repetition penalty.
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    /// llama-server extension. Defaults to 64 when present.
    #[serde(default)]
    pub repeat_last_n: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

impl ChatCompletionRequest {
    /// Translate the request's sampling fields into a `SamplerConfig`,
    /// filling unspecified fields from the Qwen3-recommended defaults.
    /// llama-server-flavored: presence/frequency penalties have llama.cpp
    /// semantics, NOT OpenAI's additive [-2, 2] semantics.
    pub fn sampler_config(&self) -> crate::inference::sample::SamplerConfig {
        let d = crate::inference::sample::SamplerConfig::default();
        crate::inference::sample::SamplerConfig {
            temperature: self.temperature.unwrap_or(d.temperature),
            top_k: self.top_k.unwrap_or(d.top_k),
            top_p: self.top_p.unwrap_or(d.top_p),
            min_p: self.min_p.unwrap_or(d.min_p),
            presence_penalty: self.presence_penalty.unwrap_or(d.presence_penalty),
            frequency_penalty: self.frequency_penalty.unwrap_or(d.frequency_penalty),
            repeat_penalty: self.repeat_penalty.unwrap_or(d.repeat_penalty),
            penalty_last_n: self.repeat_last_n.unwrap_or(d.penalty_last_n),
            seed: self.seed.unwrap_or(d.seed),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
}

impl ChatCompletionResponse {
    pub fn stub(req: &ChatCompletionRequest) -> Self {
        Self {
            id: "chatcmpl-seeker-stub".to_string(),
            object: "chat.completion",
            created: 0,
            model: req.model.clone().unwrap_or_else(|| STUB_MODEL.to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Value::String(STUB_TEXT.to_string()),
                    name: None,
                    tool_call_id: None,
                },
                finish_reason: "stop",
            }],
            usage: Usage::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/completions
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: Option<Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub suffix: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: &'static str,
    pub logprobs: Option<Value>,
}

impl CompletionResponse {
    pub fn stub(req: &CompletionRequest) -> Self {
        Self {
            id: "cmpl-seeker-stub".to_string(),
            object: "text_completion",
            created: 0,
            model: req.model.clone().unwrap_or_else(|| STUB_MODEL.to_string()),
            choices: vec![CompletionChoice {
                index: 0,
                text: STUB_TEXT.to_string(),
                finish_reason: "stop",
                logprobs: None,
            }],
            usage: Usage::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/embeddings
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct EmbeddingRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u32>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingItem>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingItem {
    pub object: &'static str,
    pub index: u32,
    pub embedding: Vec<f32>,
}

impl EmbeddingResponse {
    pub fn stub(req: &EmbeddingRequest) -> Self {
        let inputs = match &req.input {
            Some(Value::Array(arr)) => arr.len().max(1),
            _ => 1,
        };
        let data = (0..inputs as u32)
            .map(|i| EmbeddingItem {
                object: "embedding",
                index: i,
                embedding: vec![0.0; 8],
            })
            .collect();
        Self {
            object: "list",
            data,
            model: req.model.clone().unwrap_or_else(|| STUB_MODEL.to_string()),
            usage: Usage::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<Model>,
}

#[derive(Debug, Serialize)]
pub struct Model {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

impl ModelListResponse {
    pub fn stub() -> Self {
        Self {
            object: "list",
            data: vec![Model {
                id: STUB_MODEL.to_string(),
                object: "model",
                created: 0,
                owned_by: "seeker".to_string(),
            }],
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/rerank
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct RerankRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub documents: Vec<Value>,
    #[serde(default)]
    pub top_n: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RerankItem>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct RerankItem {
    pub index: u32,
    pub relevance_score: f32,
}

impl RerankResponse {
    pub fn stub(req: &RerankRequest) -> Self {
        let results = req
            .documents
            .iter()
            .enumerate()
            .map(|(i, _)| RerankItem {
                index: i as u32,
                relevance_score: 0.0,
            })
            .collect();
        Self {
            model: req.model.clone().unwrap_or_else(|| STUB_MODEL.to_string()),
            results,
            usage: Usage::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/audio/transcriptions
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

impl TranscriptionResponse {
    pub fn stub() -> Self {
        Self {
            text: STUB_TEXT.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// /v1/responses (OpenAI's newer "Responses" API)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub model: String,
    pub status: &'static str,
    pub output: Vec<ResponsesOutputItem>,
    pub usage: ResponsesUsage,
}

#[derive(Debug, Serialize)]
pub struct ResponsesOutputItem {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub content: Vec<ResponsesContentBlock>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesContentBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Default, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

impl ResponsesResponse {
    pub fn stub(req: &ResponsesRequest) -> Self {
        Self {
            id: "resp_seeker_stub".to_string(),
            object: "response",
            created_at: 0,
            model: req.model.clone().unwrap_or_else(|| STUB_MODEL.to_string()),
            status: "completed",
            output: vec![ResponsesOutputItem {
                id: "msg_seeker_stub".to_string(),
                kind: "message",
                role: "assistant",
                content: vec![ResponsesContentBlock {
                    kind: "output_text",
                    text: STUB_TEXT.to_string(),
                }],
            }],
            usage: ResponsesUsage::default(),
        }
    }
}
