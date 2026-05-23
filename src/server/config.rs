//! Runtime configuration for the embedded HTTP server. Built by the CLI
//! (or any other caller) and handed to [`crate::server::run`].

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 11434,
            cors: false,
        }
    }
}
