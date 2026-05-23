//! llama-server native handlers: completion, infill, tokenize, detokenize,
//! embedding, apply-template.

use axum::response::IntoResponse;
use axum::Json;

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

pub async fn apply_template(Json(_req): Json<ApplyTemplateRequest>) -> impl IntoResponse {
    Json(ApplyTemplateResponse::stub())
}
