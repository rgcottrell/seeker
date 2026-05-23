//! OpenAI-compatible handlers (chat, completions, embeddings, models, rerank,
//! responses, audio).

use axum::response::sse::{KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;

use crate::server::stream::openai_stub_stream;
use crate::server::types::openai::{
    ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse,
    EmbeddingRequest, EmbeddingResponse, ModelListResponse, RerankRequest, RerankResponse,
    ResponsesRequest, ResponsesResponse, TranscriptionResponse, STUB_MODEL,
};

pub async fn chat_completions(Json(req): Json<ChatCompletionRequest>) -> axum::response::Response {
    if req.stream.unwrap_or(false) {
        let model = req.model.clone().unwrap_or_else(|| STUB_MODEL.to_string());
        Sse::new(openai_stub_stream(model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(ChatCompletionResponse::stub(&req)).into_response()
    }
}

pub async fn completions(Json(req): Json<CompletionRequest>) -> impl IntoResponse {
    Json(CompletionResponse::stub(&req))
}

pub async fn embeddings(Json(req): Json<EmbeddingRequest>) -> impl IntoResponse {
    Json(EmbeddingResponse::stub(&req))
}

pub async fn models() -> impl IntoResponse {
    Json(ModelListResponse::stub())
}

pub async fn rerank(Json(req): Json<RerankRequest>) -> impl IntoResponse {
    Json(RerankResponse::stub(&req))
}

pub async fn responses(Json(req): Json<ResponsesRequest>) -> impl IntoResponse {
    Json(ResponsesResponse::stub(&req))
}

/// `multipart/form-data` audio transcription. We don't parse the file payload
/// — just return the canned stub text so SDKs that expect a JSON response
/// don't choke.
pub async fn audio_transcriptions() -> impl IntoResponse {
    Json(TranscriptionResponse::stub())
}
