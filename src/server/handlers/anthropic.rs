//! Anthropic Messages API handlers.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::server::convert::{self, anthropic_to_chat};
use crate::server::error;
use crate::server::inference::{collect, GenConfig, GenOutput};
use crate::server::state::AppState;
use crate::server::stream::{anthropic_messages_stream, gen_id};
use crate::server::types::anthropic::{
    AnthropicUsage, ContentBlock, CountTokensRequest, CountTokensResponse, MessagesRequest,
    MessagesResponse,
};

pub async fn messages(State(state): State<AppState>, Json(req): Json<MessagesRequest>) -> Response {
    let Some(handle) = state.inference() else {
        return error::no_model_anthropic();
    };
    let messages = match anthropic_to_chat(&req.system, &req.messages) {
        Ok(m) => m,
        Err(e) => return error::anthropic(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let tokens = match convert::render_and_encode(&state, messages) {
        Ok(t) => t,
        Err(e) => return error::anthropic(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    // Anthropic exposes only temperature / top_p / top_k; the rest inherit the
    // CLI base (no min_p / penalties / seed in the request).
    let mut sampler = state.default_sampler().clone();
    if let Some(t) = req.temperature {
        sampler.temperature = t;
    }
    if let Some(p) = req.top_p {
        sampler.top_p = p;
    }
    if let Some(k) = req.top_k {
        sampler.top_k = k;
    }
    let config = GenConfig {
        sampler,
        max_tokens: req.max_tokens.unwrap_or(state.default_max_tokens()),
        stop: req.stop_sequences.clone().unwrap_or_default(),
        ignore_eos: state.default_ignore_eos(),
    };
    let rx = match handle.start(tokens, config).await {
        Ok(rx) => rx,
        Err(e) => return error::anthropic(StatusCode::SERVICE_UNAVAILABLE, "api_error", e),
    };
    let model = state.model_id().to_string();
    if req.stream.unwrap_or(false) {
        Sse::new(anthropic_messages_stream(rx, model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match collect(rx).await {
            Ok(out) => Json(messages_response(model, out)).into_response(),
            Err(e) => error::anthropic(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e),
        }
    }
}

fn messages_response(model: String, out: GenOutput) -> MessagesResponse {
    MessagesResponse {
        id: gen_id("msg"),
        kind: "message",
        role: "assistant",
        model,
        content: vec![ContentBlock::Text { text: out.text }],
        stop_reason: out.stop_reason.anthropic_reason(),
        stop_sequence: out.stop_reason.matched_sequence().map(|s| s.to_string()),
        usage: AnthropicUsage {
            input_tokens: out.prompt_tokens,
            output_tokens: out.completion_tokens,
        },
    }
}

pub async fn count_tokens(
    State(state): State<AppState>,
    Json(req): Json<CountTokensRequest>,
) -> Response {
    if state.tokenizer().is_none() {
        return error::no_model_anthropic();
    }
    let messages = match anthropic_to_chat(&req.system, &req.messages) {
        Ok(m) => m,
        Err(e) => return error::anthropic(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    match convert::render_and_encode(&state, messages) {
        Ok(tokens) => Json(CountTokensResponse {
            input_tokens: tokens.len() as u32,
        })
        .into_response(),
        Err(e) => error::anthropic(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}
