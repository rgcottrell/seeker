//! Anthropic Messages API DTOs. snake_case throughout — Anthropic's API
//! does not use the `rename_all = "snake_case"` pattern's quirks, since
//! every field name we care about is already snake_case. ContentBlock is
//! an enum tagged by `type` so it round-trips text vs tool_use cleanly.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Default, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<MessagesInputMessage>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessagesInputMessage {
    pub role: String,
    pub content: Value,
}

/// Output content blocks. Tagged by `type` per Anthropic's wire format.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: &'static str,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

// ---------------------------------------------------------------------------
// /v1/messages/count_tokens
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct CountTokensRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<MessagesInputMessage>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub tools: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct CountTokensResponse {
    pub input_tokens: u32,
}
