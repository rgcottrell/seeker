//! `seeker serve` — start the HTTP server. Thin CLI shim over
//! [`crate::server::run`]: parse flags, build a [`ServerConfig`], hand off.

use std::error::Error;
use std::path::PathBuf;

use clap::Args;

use crate::gguf::GgufFile;
use crate::server::{run as server_run, AppState, ServerConfig};
use crate::tokenizer::build_tokenizer;

#[derive(Args)]
pub struct ServeArgs {
    /// Address to bind. Use 0.0.0.0 to accept connections from other hosts.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// TCP port to listen on. Defaults to 11434 (Ollama-style).
    #[arg(long, default_value_t = 11434)]
    port: u16,

    /// Attach a permissive CORS layer (off by default for safety in dev).
    #[arg(long)]
    cors: bool,

    /// Path to a local .gguf model file. Only the tokenizer + chat
    /// template are loaded today — needed by `/apply-template`. Full
    /// inference lands when the OpenAI / llama-server completion
    /// handlers stop returning stubs.
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,

    /// Extra key/value pairs merged into the chat-template rendering context
    /// for `/apply-template`, as a JSON object string, e.g.
    /// `'{"enable_thinking":false}'`. Keys override the built-in context
    /// variables. Mirrors llama.cpp's `--chat-template-kwargs`.
    #[arg(long = "chat-template-kwargs", value_parser = crate::chat_template::parse_template_kwargs)]
    chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

pub async fn run(args: ServeArgs) -> Result<(), Box<dyn Error>> {
    let app_state = match args.model.as_ref() {
        Some(path) => {
            let gguf = GgufFile::open(path)?;
            let bundle = build_tokenizer(&gguf)?;
            tracing::info!(
                template_present = bundle.chat_template.is_some(),
                "loaded tokenizer for serve state",
            );
            AppState::new(
                bundle.chat_template,
                bundle.bos_token,
                bundle.eos_token,
                args.chat_template_kwargs.clone().unwrap_or_default(),
            )
        }
        None => AppState::default(),
    };
    server_run(ServerConfig {
        host: args.host,
        port: args.port,
        cors: args.cors,
        app_state,
    })
    .await
}
