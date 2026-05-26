//! llama-server native handlers: completion, infill, tokenize, detokenize,
//! embedding, apply-template.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::chat_template::{self, ChatMessage};
use crate::server::state::AppState;
use crate::server::types::llama::{
    ApplyTemplateRequest, ApplyTemplateResponse, CompletionRequest, CompletionResponse,
    DetokenizeRequest, DetokenizeResponse, InfillRequest, LlamaEmbeddingRequest,
    LlamaEmbeddingResponse, TokenizeRequest, TokenizeResponse,
};

pub async fn completion(Json(_req): Json<CompletionRequest>) -> impl IntoResponse {
    Json(CompletionResponse::stub())
}

pub async fn infill(Json(_req): Json<InfillRequest>) -> impl IntoResponse {
    Json(CompletionResponse::stub())
}

pub async fn tokenize(Json(_req): Json<TokenizeRequest>) -> impl IntoResponse {
    Json(TokenizeResponse::stub())
}

pub async fn detokenize(Json(_req): Json<DetokenizeRequest>) -> impl IntoResponse {
    Json(DetokenizeResponse::stub())
}

pub async fn embedding(Json(_req): Json<LlamaEmbeddingRequest>) -> impl IntoResponse {
    Json(LlamaEmbeddingResponse::stub())
}

pub async fn apply_template(
    State(state): State<AppState>,
    Json(req): Json<ApplyTemplateRequest>,
) -> Result<Json<ApplyTemplateResponse>, (StatusCode, String)> {
    let Some(template) = state.chat_template() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "seeker serve has no model loaded — pass `--model PATH` on startup".into(),
        ));
    };
    let messages = convert_messages(&req.messages).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let prompt = chat_template::render(
        template,
        &messages,
        req.add_generation_prompt.unwrap_or(true),
        state.bos_token().unwrap_or(""),
        state.eos_token().unwrap_or(""),
        /* enable_thinking = */ true,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(ApplyTemplateResponse { prompt }))
}

/// `serde_json::Value` → typed `ChatMessage`. Accepts the common
/// `{role, content}` shape that llama-server and OpenAI use. Tools and
/// multi-modal content are dropped for now; if a request needs them,
/// surface a clearer error and bail.
fn convert_messages(raw: &[serde_json::Value]) -> Result<Vec<ChatMessage>, String> {
    raw.iter()
        .map(|v| {
            let role = v
                .get("role")
                .and_then(|r| r.as_str())
                .ok_or_else(|| "message missing string `role`".to_string())?;
            let content = v
                .get("content")
                .and_then(|c| c.as_str())
                .ok_or_else(|| "message `content` must be a plain string for now".to_string())?;
            Ok(ChatMessage {
                role: role.to_string(),
                content: content.to_string(),
            })
        })
        .collect()
}
