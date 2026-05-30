//! Build the full `axum::Router` for the seeker HTTP surface. Public so
//! callers (the CLI shim, future tests, future embedded uses) can mount the
//! same set of routes without going through `run`.

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use super::handlers::{anthropic, llama, ops, openai};
use super::state::AppState;

/// Build the router with all stub endpoints wired in.
///
/// `cors` toggles a permissive `CorsLayer` (off by default). The
/// `TraceLayer` is always on — it just emits structured tracing events,
/// which our existing `tracing_subscriber` setup already filters.
pub fn build_router(cors: bool, state: AppState) -> Router {
    let mut app = Router::new()
        // -------------------- ops --------------------
        .route("/health", get(ops::health))
        .route("/v1/health", get(ops::health))
        .route("/props", get(ops::props_get).post(ops::props_post))
        .route("/slots", get(ops::slots))
        .route("/metrics", get(ops::metrics))
        .route(
            "/lora-adapters",
            get(ops::lora_adapters_get).post(ops::lora_adapters_post),
        )
        // -------------------- OpenAI compat --------------------
        .route("/models", get(openai::models))
        .route("/v1/models", get(openai::models))
        .route("/chat/completions", post(openai::chat_completions))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/responses", post(openai::responses))
        .route("/v1/responses", post(openai::responses))
        .route("/embeddings", post(openai::embeddings))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/rerank", post(openai::rerank))
        .route("/reranking", post(openai::rerank))
        .route("/v1/rerank", post(openai::rerank))
        .route("/v1/reranking", post(openai::rerank))
        .route("/audio/transcriptions", post(openai::audio_transcriptions))
        .route("/v1/audio/transcriptions", post(openai::audio_transcriptions))
        // -------------------- Anthropic --------------------
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        // -------------------- llama-server native --------------------
        .route("/completion", post(llama::completion))
        .route("/completions", post(llama::completion))
        .route("/infill", post(llama::infill))
        .route("/tokenize", post(llama::tokenize))
        .route("/detokenize", post(llama::detokenize))
        .route("/embedding", post(llama::embedding))
        .route("/apply-template", post(llama::apply_template));

    if cors {
        app = app.layer(CorsLayer::permissive());
    }
    app.layer(TraceLayer::new_for_http()).with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for `oneshot`

    /// Drive one request through the no-model router and return the status.
    async fn status(method: &str, uri: &str, body: &str) -> StatusCode {
        let app = build_router(false, AppState::default());
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        app.oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn health_and_models_ok_without_model() {
        assert_eq!(status("GET", "/health", "").await, StatusCode::OK);
        assert_eq!(status("GET", "/v1/models", "").await, StatusCode::OK);
        assert_eq!(status("GET", "/props", "").await, StatusCode::OK);
    }

    #[tokio::test]
    async fn generation_endpoints_503_without_model() {
        for uri in ["/v1/chat/completions", "/v1/completions", "/completion", "/v1/messages"] {
            assert_eq!(
                status("POST", uri, "{}").await,
                StatusCode::SERVICE_UNAVAILABLE,
                "{uri} should 503 without a model"
            );
        }
    }

    #[tokio::test]
    async fn unsupported_endpoints_501() {
        for uri in ["/v1/embeddings", "/embedding", "/v1/rerank", "/v1/audio/transcriptions"] {
            assert_eq!(
                status("POST", uri, "{}").await,
                StatusCode::NOT_IMPLEMENTED,
                "{uri} should 501"
            );
        }
    }
}
