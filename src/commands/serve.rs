//! `seeker serve` — start the HTTP server. Thin CLI shim over
//! [`crate::server::run`]: parse flags, build a [`ServerConfig`], hand off.

use std::error::Error;

use clap::Args;

use crate::server::{run as server_run, ServerConfig};

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
}

pub async fn run(args: ServeArgs) -> Result<(), Box<dyn Error>> {
    server_run(ServerConfig {
        host: args.host,
        port: args.port,
        cors: args.cors,
    })
    .await
}
