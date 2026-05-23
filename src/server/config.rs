//! Runtime configuration for the embedded HTTP server. Built by the CLI
//! (or any other caller) and handed to [`crate::server::run`].

use super::state::AppState;

#[derive(Clone, Default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cors: bool,
    /// Shared per-request state — chat template, special tokens, and
    /// (eventually) the loaded model.
    pub app_state: AppState,
}

impl ServerConfig {
    pub fn defaults() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 11434,
            cors: false,
            app_state: AppState::default(),
        }
    }
}
