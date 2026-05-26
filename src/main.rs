use std::error::Error;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use crate::commands::bench::{self, BenchArgs};
use crate::commands::chat::{self, ChatArgs};
use crate::commands::detokenize::{self, DetokenizeArgs};
use crate::commands::download::{self, DownloadArgs};
use crate::commands::inspect::{self, InspectArgs};
use crate::commands::run::{self as run_cmd, RunArgs};
use crate::commands::serve::{self, ServeArgs};
use crate::commands::tokenize::{self, TokenizeArgs};

mod chat_template;
mod commands;
#[allow(dead_code)]
mod gguf;
#[allow(dead_code)]
mod runtime_flags;
#[allow(dead_code)]
mod inference;
#[allow(dead_code)]
mod models;
#[allow(dead_code)]
mod server;
mod tokenizer;

#[allow(dead_code)]
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

#[derive(Parser)]
#[command(name = "seeker", version, about = "Vulkan compute toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download a model file from a Hugging Face repository.
    Download(DownloadArgs),
    /// Dump the header, metadata, and tensor table of a GGUF file.
    Inspect(InspectArgs),
    /// Encode text using the tokenizer embedded in a GGUF model.
    Tokenize(TokenizeArgs),
    /// Decode token ids using the tokenizer embedded in a GGUF model.
    Detokenize(DetokenizeArgs),
    /// Start an HTTP server that stubs out llama-server's full API surface.
    Serve(ServeArgs),
    /// Interactive chat REPL against a model's embedded tokenizer (stubbed).
    Chat(ChatArgs),
    /// Run a single forward pass and print the predicted next token.
    Run(RunArgs),
    /// Benchmark prefill + decode tok/s; optionally dump first-token logits.
    Bench(BenchArgs),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse_from(rewrite_shorts(std::env::args()));
    match cli.command {
        Command::Download(args) => download::run(args).await,
        Command::Inspect(args) => inspect::run(args).await,
        Command::Tokenize(args) => tokenize::run(args).await,
        Command::Detokenize(args) => detokenize::run(args).await,
        Command::Serve(args) => serve::run(args).await,
        Command::Chat(args) => chat::run(args).await,
        Command::Run(args) => run_cmd::run(args).await,
        Command::Bench(args) => bench::run(args).await,
    }
}

/// Rewrite single-dash multi-character shortcuts (`-hf`, `-hfr`, `-hff`, `-hft`) to
/// their `--hf-*` equivalents before handing argv to clap, which only supports
/// single-character shorts. Handles both `-hf foo` and `-hf=foo` forms.
fn rewrite_shorts<I, S>(args: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    args.into_iter()
        .map(|a| {
            let a: String = a.into();
            let (head, tail) = match a.split_once('=') {
                Some((h, t)) => (h, Some(t)),
                None => (a.as_str(), None),
            };
            let canonical = match head {
                "-hf" | "-hfr" => Some("--hf-repo"),
                "-hff" => Some("--hf-file"),
                "-hft" => Some("--hf-token"),
                _ => None,
            };
            match (canonical, tail) {
                (Some(c), Some(t)) => format!("{c}={t}"),
                (Some(c), None) => c.to_string(),
                (None, _) => a,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::rewrite_shorts;

    #[test]
    fn rewrites_space_form() {
        let rewritten = rewrite_shorts([
            "seeker", "download", "-hf", "org/name", "-hff", "x.gguf", "-hft", "tk",
        ]);
        assert_eq!(
            rewritten,
            vec![
                "seeker",
                "download",
                "--hf-repo",
                "org/name",
                "--hf-file",
                "x.gguf",
                "--hf-token",
                "tk",
            ]
        );
    }

    #[test]
    fn rewrites_equals_form_and_hfr_alias() {
        let rewritten = rewrite_shorts(["seeker", "download", "-hfr=org/name:Q4_K_M"]);
        assert_eq!(rewritten, vec!["seeker", "download", "--hf-repo=org/name:Q4_K_M"]);
    }

    #[test]
    fn leaves_unrelated_args_alone() {
        let rewritten = rewrite_shorts([
            "seeker", "download", "--hf-repo", "org/name", "--offline", "-help",
        ]);
        assert_eq!(
            rewritten,
            vec!["seeker", "download", "--hf-repo", "org/name", "--offline", "-help"]
        );
    }
}
