//! `seeker tokenize` — encode text to token IDs using the tokenizer embedded
//! in a GGUF file. Model selection mirrors `inspect`: either a Hugging Face
//! repo (with the `--hf-*` flags) or `-m/--model PATH`. The text comes from
//! `--text` or, when that's omitted, from stdin.

use std::error::Error;
use std::io::{IsTerminal, Read};
use std::path::PathBuf;

use clap::Args;

use crate::commands::download::{HfResolveArgs, resolve_hf};
use crate::gguf::GgufFile;
use crate::tokenizer::{TokenizerBundle, build_tokenizer};

#[derive(Args)]
pub struct TokenizeArgs {
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]". (short: -hf, -hfr)
    #[arg(
        long = "hf-repo",
        required_unless_present = "model",
        conflicts_with = "model"
    )]
    hf_repo: Option<String>,

    /// Specific file to tokenize within the repo. (short: -hff)
    #[arg(long = "hf-file", requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,

    /// HF auth token (defaults to HF_TOKEN env / ~/.cache/huggingface/token). (short: -hft)
    #[arg(long = "hf-token", requires = "hf_repo", conflicts_with = "model")]
    hf_token: Option<String>,

    /// Resolve files from the local cache only; never hit the network.
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    offline: bool,

    /// Path to a local .gguf model file.
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,

    /// Text to tokenize. Falls back to stdin if omitted.
    #[arg(long)]
    text: Option<String>,

    /// Prepend BOS / append EOS (forced; overrides the GGUF default).
    #[arg(long)]
    add_special: bool,

    /// Print only token ids, one per line. Default also prints the
    /// human-readable token string in a second tab-separated column.
    #[arg(long)]
    ids_only: bool,
}

pub async fn run(args: TokenizeArgs) -> Result<(), Box<dyn Error>> {
    let path = resolve_model_path(&args).await?;
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;

    let text = read_text(args.text)?;
    let add_special = args.add_special || bundle.add_bos_default || bundle.add_eos_default;

    let TokenizerBundle {
        tokenizer,
        model_kind,
        ..
    } = bundle;
    tracing::debug!(model = %model_kind, add_special, "encoding");

    let encoding = tokenizer
        .encode(text.as_str(), add_special)
        .map_err(|e| format!("tokenizer encode failed: {e}"))?;

    let ids = encoding.get_ids();
    let toks = encoding.get_tokens();

    if args.ids_only {
        for id in ids {
            println!("{id}");
        }
    } else {
        for (id, tok) in ids.iter().zip(toks) {
            println!("{id}\t{tok}");
        }
    }
    Ok(())
}

async fn resolve_model_path(args: &TokenizeArgs) -> Result<PathBuf, Box<dyn Error>> {
    match (args.hf_repo.clone(), args.model.clone()) {
        (Some(repo), None) => Ok(resolve_hf(
            &HfResolveArgs {
                repo,
                file: args.hf_file.clone(),
                token: args.hf_token.clone(),
                offline: args.offline,
            },
            false,
        )
        .await?
        .main),
        (None, Some(model)) => Ok(model),
        _ => unreachable!("clap group invariant"),
    }
}

/// Pull text from `--text` first, otherwise drain stdin. Refuse silently when
/// stdin is a TTY and no flag was given — that's almost always a user mistake.
fn read_text(flag: Option<String>) -> Result<String, Box<dyn Error>> {
    if let Some(t) = flag {
        return Ok(t);
    }
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return Err("no input: pass --text \"...\" or pipe text via stdin".into());
    }
    let mut buf = String::new();
    stdin.lock().read_to_string(&mut buf)?;
    // Trim a single trailing newline so `echo "hi" | seeker tokenize ...` works as expected.
    if buf.ends_with('\n') {
        buf.pop();
        if buf.ends_with('\r') {
            buf.pop();
        }
    }
    Ok(buf)
}
