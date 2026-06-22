//! Error-response builders. OpenAI / llama-native routes use the
//! `{error:{message,type,code}}` envelope ([`super::types::common::ApiError`]);
//! Anthropic routes use `{type:"error", error:{type,message}}`. SDKs branch on
//! these shapes, so each surface returns its own.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use super::types::common::{ApiError, ApiErrorDetail};

/// OpenAI / llama-style error response.
pub fn openai(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    let body = ApiError {
        error: ApiErrorDetail {
            message: message.into(),
            kind: kind.to_string(),
            code: None,
        },
    };
    (status, Json(body)).into_response()
}

/// Anthropic-style error response.
pub fn anthropic(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    let body = json!({"type": "error", "error": {"type": kind, "message": message.into()}});
    (status, Json(body)).into_response()
}

const NO_MODEL_MSG: &str =
    "seeker serve has no model loaded — start it with `--model PATH` (or `--hf-repo`)";

/// 503 for OpenAI / llama generation endpoints when no model is loaded.
pub fn no_model_openai() -> Response {
    openai(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        NO_MODEL_MSG,
    )
}

/// 503 for Anthropic generation endpoints when no model is loaded.
pub fn no_model_anthropic() -> Response {
    anthropic(StatusCode::SERVICE_UNAVAILABLE, "api_error", NO_MODEL_MSG)
}

/// 400 invalid-request (OpenAI shape).
pub fn bad_request(message: impl Into<String>) -> Response {
    openai(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// 501 for endpoints with no analogue on a causal-LM logits engine.
pub fn not_supported(message: impl Into<String>) -> Response {
    openai(StatusCode::NOT_IMPLEMENTED, "not_supported", message)
}

/// 500 from a worker `GenEvent::Error` mid-request (OpenAI shape).
pub fn internal(message: impl Into<String>) -> Response {
    openai(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
}
