//! Shared structs reused across multiple API surfaces.

use serde::{Deserialize, Serialize};

/// OpenAI-style token accounting block (Anthropic uses its own shape — see
/// [`crate::server::types::anthropic::AnthropicUsage`]).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Generic error envelope. Mirrors the OpenAI shape (`{"error": {...}}`) so
/// SDKs that bubble these up don't need a special branch for us.
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}
