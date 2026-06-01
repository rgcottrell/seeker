//! `seeker detokenize` — decode token IDs to text using the tokenizer embedded
//! in a GGUF file. IDs come from `--ids` (whitespace or comma separated) or,
//! when that's omitted, from stdin. Model selection mirrors `tokenize`.

use std::error::Error;
use std::io::{IsTerminal, Read};
use std::path::PathBuf;

use clap::Args;

use crate::commands::download::{HfResolveArgs, resolve_hf};
use crate::gguf::GgufFile;
use crate::tokenizer::build_tokenizer;

#[derive(Args)]
pub struct DetokenizeArgs {
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]". (short: -hf, -hfr)
    #[arg(
        long = "hf-repo",
        required_unless_present = "model",
        conflicts_with = "model"
    )]
    hf_repo: Option<String>,

    /// Specific file to use within the repo. (short: -hff)
    #[arg(long = "hf-file", requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,

    /// HF auth token. (short: -hft)
    #[arg(long = "hf-token", requires = "hf_repo", conflicts_with = "model")]
    hf_token: Option<String>,

    /// Resolve files from the local cache only; never hit the network.
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    offline: bool,

    /// Path to a local .gguf model file.
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,

    /// Token ids, whitespace- or comma-separated: `--ids "1 23 456"`.
    /// Falls back to stdin if omitted.
    #[arg(long)]
    ids: Option<String>,

    /// Don't strip special tokens (BOS/EOS/etc.) from the decoded output.
    #[arg(long)]
    keep_special: bool,
}

pub async fn run(args: DetokenizeArgs) -> Result<(), Box<dyn Error>> {
    let path = resolve_model_path(&args).await?;
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;

    let raw = read_ids_source(args.ids)?;
    let ids = parse_ids(&raw)?;
    tracing::debug!(model = %bundle.model_kind, n = ids.len(), "decoding");

    let text = bundle
        .tokenizer
        .decode(&ids, !args.keep_special)
        .map_err(|e| format!("tokenizer decode failed: {e}"))?;
    println!("{text}");
    Ok(())
}

async fn resolve_model_path(args: &DetokenizeArgs) -> Result<PathBuf, Box<dyn Error>> {
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

fn read_ids_source(flag: Option<String>) -> Result<String, Box<dyn Error>> {
    if let Some(s) = flag {
        return Ok(s);
    }
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return Err("no input: pass --ids \"1 2 3\" or pipe ids via stdin".into());
    }
    let mut buf = String::new();
    stdin.lock().read_to_string(&mut buf)?;
    Ok(buf)
}

/// Accepts whitespace and commas as separators so the user can paste either
/// `1 2 3` or `1,2,3` or even `[1, 2, 3]` (brackets are ignored).
fn parse_ids(raw: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut out = Vec::new();
    for tok in raw.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let id: u32 = t.parse().map_err(|_| format!("invalid token id `{t}`"))?;
        out.push(id);
    }
    if out.is_empty() {
        return Err("no token ids found in input".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::parse_ids;

    #[test]
    fn parses_whitespace_and_commas() {
        assert_eq!(parse_ids("1 2 3").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_ids("1,2,3").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_ids("[1, 2, 3]").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_ids(" 42 \n 99\n").unwrap(), vec![42, 99]);
    }

    #[test]
    fn rejects_non_numeric() {
        assert!(parse_ids("1 abc 3").is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_ids("").is_err());
        assert!(parse_ids("   ").is_err());
    }
}
