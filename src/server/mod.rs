//! HTTP server library. The CLI subcommand `seeker serve` is a thin shim
//! over [`run`]; tests and future embedded users build the same router via
//! [`router::build_router`].
//!
//! Generation endpoints drive the GPU engine through a dedicated worker thread
//! ([`inference`]); the wire types and routes match what llama.cpp's
//! `llama-server` exposes (OpenAI + Anthropic + native). Endpoints with no
//! analogue on a causal-LM logits engine (embeddings, rerank, audio) return a
//! clear 501.

pub mod config;
pub mod convert;
pub mod error;
pub mod handlers;
pub mod inference;
pub mod router;
pub mod state;
pub mod static_assets;
pub mod stream;
pub mod types;

use std::error::Error;

pub use config::ServerConfig;
pub use router::build_router;
pub use state::{AppState, AppStateInit};

/// Bind to `(host, port)` and serve until the process is interrupted.
pub async fn run(config: ServerConfig) -> Result<(), Box<dyn Error>> {
    let app = build_router(config.cors, config.app_state.clone());
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let bound = listener.local_addr()?;
    tracing::info!(address = %bound, "seeker serve listening");
    axum::serve(listener, app).await?;
    Ok(())
}
