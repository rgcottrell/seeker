//! HTTP server library. The CLI subcommand `seeker serve` is a thin shim
//! over [`run`]; tests and future embedded users build the same router via
//! [`router::build_router`].
//!
//! Every endpoint is presently a stub that returns shape-correct, deterministic
//! placeholder responses. Real inference will land later by replacing the body
//! of each handler in `handlers/*` — the wire types and routes already match
//! what llama.cpp's `llama-server` exposes (OpenAI + Anthropic + native).

pub mod config;
pub mod handlers;
pub mod router;
pub mod state;
pub mod stream;
pub mod types;

use std::error::Error;

pub use config::ServerConfig;
pub use router::build_router;
pub use state::AppState;

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
