//! OpenAI-compatible request / response DTOs.
//!
//! Request types are deliberately permissive: unknown fields are accepted
//! (via `#[serde(default)]` on every option) so callers can hand us their
//! full payload and have it round-trip cleanly.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::Usage;

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
    /// Translate the request's sampling fields into a `SamplerConfig`, filling
    /// unspecified fields from the server's CLI-provided `base` defaults.
    /// llama-server-flavored: presence/frequency penalties have llama.cpp
    /// semantics, NOT OpenAI's additive [-2, 2] semantics. `logit_bias` and the
    /// EOS handling always inherit the CLI base (not exposed per-request).
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

impl CompletionRequest {
    /// Build a `SamplerConfig` from the CLI `base`, overriding only the fields
    /// the OpenAI completion request exposes (temperature).
    pub fn sampler_config(
        &self,
        base: &crate::inference::sample::SamplerConfig,
    ) -> crate::inference::sample::SamplerConfig {
        crate::inference::sample::SamplerConfig {
            temperature: self.temperature.unwrap_or(base.temperature),
            ..base.clone()
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::sample::SamplerConfig;

    fn base() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.3,
            top_k: 11,
            top_p: 0.5,
            min_p: 0.1,
            seed: 99,
            logit_bias: vec![(7, 1.0)],
            ..SamplerConfig::default()
        }
    }

    #[test]
    fn chat_request_overrides_base_only_where_present() {
        // Empty request → every field inherits the CLI base (incl. logit_bias).
        let cfg = ChatCompletionRequest::default().sampler_config(&base());
        assert_eq!(cfg.temperature, 0.3);
        assert_eq!(cfg.top_k, 11);
        assert_eq!(cfg.seed, 99);
        assert_eq!(cfg.logit_bias, vec![(7, 1.0)]);

        // A request value overrides just that field.
        let req = ChatCompletionRequest {
            temperature: Some(1.5),
            seed: Some(42),
            ..Default::default()
        };
        let cfg = req.sampler_config(&base());
        assert_eq!(cfg.temperature, 1.5); // overridden
        assert_eq!(cfg.seed, 42); // overridden
        assert_eq!(cfg.top_p, 0.5); // inherited
        assert_eq!(cfg.logit_bias, vec![(7, 1.0)]); // always inherited
    }

    #[test]
    fn completion_request_overrides_temperature_only() {
        let req = CompletionRequest {
            temperature: Some(0.0),
            ..Default::default()
        };
        let cfg = req.sampler_config(&base());
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_k, 11); // inherited from base
    }
}
