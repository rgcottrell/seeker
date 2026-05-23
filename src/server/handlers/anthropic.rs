//! Anthropic Messages API handlers.

use axum::response::sse::{KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;

use crate::server::stream::anthropic_stub_stream;
use crate::server::types::anthropic::{
    CountTokensRequest, CountTokensResponse, MessagesRequest, MessagesResponse, STUB_MODEL,
};

pub async fn messages(Json(req): Json<MessagesRequest>) -> axum::response::Response {
    if req.stream.unwrap_or(false) {
        let model = req.model.clone().unwrap_or_else(|| STUB_MODEL.to_string());
        Sse::new(anthropic_stub_stream(model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(MessagesResponse::stub(&req)).into_response()
    }
}

pub async fn count_tokens(Json(_req): Json<CountTokensRequest>) -> impl IntoResponse {
    Json(CountTokensResponse::stub())
}
