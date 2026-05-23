//! `seeker chat` — interactive REPL stub. Model selection mirrors `inspect`
//! / `tokenize` (either `--hf-*` or `-m/--model PATH`). Real inference isn't
//! wired up yet — each prompt gets a canned assistant reply — but the model
//! is fully loaded and each prompt runs through its embedded GGUF tokenizer
//! so file-missing errors surface at startup and the user sees real token
//! counts.

use std::error::Error;
use std::fs;
use std::io::{BufRead, IsTerminal};
use std::path::{Path, PathBuf};

use clap::Args;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::{GgufFile, MetadataValue};
use crate::tokenizer::{build_tokenizer, TokenizerBundle};

const STUB_REPLY: &str = "[stub] seeker chat has no inference backend wired up yet";

const BANNER: &str = r#"███████ ███████ ███████ ██  ██ ███████ ██████
██      ██      ██      ██ ██  ██      ██   ██
███████ █████   █████   ████   █████   ██████
     ██ ██      ██      ██ ██  ██      ██   ██
███████ ███████ ███████ ██  ██ ███████ ██   ██"#;

#[derive(Args)]
pub struct ChatArgs {
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]". (short: -hf, -hfr)
    #[arg(long = "hf-repo", required_unless_present = "model", conflicts_with = "model")]
    hf_repo: Option<String>,

    /// Specific file within the repo. (short: -hff)
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

    /// Skip reading and writing the line-history file.
    #[arg(long)]
    no_history: bool,

    /// Override the history-file location. Defaults to an OS-appropriate
    /// path under the user's data directory.
    #[arg(long)]
    history_file: Option<PathBuf>,
}

pub async fn run(args: ChatArgs) -> Result<(), Box<dyn Error>> {
    let path = resolve_model_path(&args).await?;
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;

    let history = if args.no_history {
        None
    } else {
        args.history_file
            .clone()
            .or_else(default_history_path)
    };

    if std::io::stdin().is_terminal() {
        run_interactive(&bundle, &gguf, &path, history.as_deref())
    } else {
        run_piped(&bundle)
    }
}

async fn resolve_model_path(args: &ChatArgs) -> Result<PathBuf, Box<dyn Error>> {
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

/// Best-effort CPU / SoC name for the banner. macOS hits `sysctl`, Linux
/// reads `/proc/cpuinfo`, everything else falls back to OS + arch labels.
fn device_name() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    return s;
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("model name") {
                    if let Some(v) = rest.split(':').nth(1) {
                        return v.trim().to_string();
                    }
                }
            }
        }
    }
    format!("{} ({})", std::env::consts::OS, std::env::consts::ARCH)
}

/// Coerce any numeric GGUF metadata value into a `u64`. GGUF doesn't pin
/// down which width different writers use for the same key (block_count
/// shows up as both U32 and U64 in the wild), so we accept everything that
/// fits.
fn read_metadata_u64(g: &GgufFile, key: &str) -> Option<u64> {
    match g.get(key)? {
        MetadataValue::U8(n) => Some(*n as u64),
        MetadataValue::U16(n) => Some(*n as u64),
        MetadataValue::U32(n) => Some(*n as u64),
        MetadataValue::U64(n) => Some(*n),
        MetadataValue::I8(n) if *n >= 0 => Some(*n as u64),
        MetadataValue::I16(n) if *n >= 0 => Some(*n as u64),
        MetadataValue::I32(n) if *n >= 0 => Some(*n as u64),
        MetadataValue::I64(n) if *n >= 0 => Some(*n as u64),
        _ => None,
    }
}

fn format_arch_line(g: &GgufFile) -> String {
    let arch = g.architecture().unwrap_or("(unknown)");
    let mut parts: Vec<String> = vec![arch.to_string()];
    let key = |k: &str| format!("{arch}.{k}");
    if let Some(n) = read_metadata_u64(g, &key("block_count")) {
        parts.push(format!("{n} layers"));
    }
    let heads = read_metadata_u64(g, &key("attention.head_count"));
    let heads_kv = read_metadata_u64(g, &key("attention.head_count_kv"));
    match (heads, heads_kv) {
        (Some(h), Some(kv)) if h != kv => parts.push(format!("{h} heads ({kv} KV)")),
        (Some(h), _) => parts.push(format!("{h} heads")),
        (None, Some(kv)) => parts.push(format!("({kv} KV)")),
        _ => {}
    }
    if let Some(n) = read_metadata_u64(g, &key("embedding_length")) {
        parts.push(format!("hidden {n}"));
    }
    parts.join(", ")
}

fn format_ctx_line(g: &GgufFile) -> String {
    let arch = g.architecture().unwrap_or("(unknown)");
    match read_metadata_u64(g, &format!("{arch}.context_length")) {
        Some(n) => format!("{n} tokens"),
        None => "(unknown)".to_string(),
    }
}

fn print_banner(gguf: &GgufFile, path: &Path) {
    let model_file = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>");

    println!("{BANNER}");
    println!();
    println!("version : {}", env!("CARGO_PKG_VERSION"));
    println!("device  : {}", device_name());
    println!("model   : {model_file}");
    println!("arch    : {}", format_arch_line(gguf));
    println!("ctx     : {}", format_ctx_line(gguf));
    println!();
    println!("commands:");
    println!("  /clear            clear conversation history and KV cache");
    println!("  /read <path>      read a UTF-8 text file as a user message");
    println!("  /exit, Ctrl+D     exit");
    println!();
}

fn default_history_path() -> Option<PathBuf> {
    if let Some(d) = dirs::data_dir() {
        return Some(d.join("seeker").join("history"));
    }
    dirs::home_dir().map(|h| h.join(".seeker_history"))
}

/// Populated as the conversation accumulates. Fields are unread in the stub
/// pass — they're here so the future inference path can read prior turns
/// without reshaping the loop.
#[allow(dead_code)]
#[derive(Clone)]
struct ChatTurn {
    role: &'static str,
    content: String,
}

fn run_interactive(
    bundle: &TokenizerBundle,
    gguf: &GgufFile,
    path: &Path,
    history: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let mut editor = DefaultEditor::new()?;
    if let Some(p) = history {
        if let Err(e) = editor.load_history(p) {
            tracing::debug!(path = %p.display(), error = %e, "history load");
        }
    }

    print_banner(gguf, path);

    let mut transcript: Vec<ChatTurn> = Vec::new();
    let add_special = bundle.add_bos_default || bundle.add_eos_default;

    loop {
        match editor.readline("> ") {
            Ok(raw) => {
                let _ = editor.add_history_entry(&raw);
                let line = raw.trim();
                if line.is_empty() {
                    continue;
                }
                if let Some(cmd) = line.strip_prefix('/') {
                    if cmd == "exit" {
                        break;
                    } else if cmd == "clear" {
                        transcript.clear();
                        println!("(transcript cleared)");
                    } else if let Some(arg) = cmd.strip_prefix("read") {
                        handle_read(bundle, &mut transcript, add_special, arg.trim());
                    } else {
                        println!("unknown command: /{cmd}");
                    }
                    continue;
                }
                handle_prompt(bundle, &mut transcript, add_special, line, None);
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => break,
            Err(e) => return Err(Box::new(e)),
        }
    }

    if let Some(p) = history {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Err(e) = editor.save_history(p) {
            tracing::debug!(path = %p.display(), error = %e, "history save");
        }
    }
    Ok(())
}

fn run_piped(bundle: &TokenizerBundle) -> Result<(), Box<dyn Error>> {
    let mut transcript: Vec<ChatTurn> = Vec::new();
    let add_special = bundle.add_bos_default || bundle.add_eos_default;

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let raw = line?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        handle_prompt(bundle, &mut transcript, add_special, trimmed, None);
    }
    Ok(())
}

fn handle_prompt(
    bundle: &TokenizerBundle,
    transcript: &mut Vec<ChatTurn>,
    add_special: bool,
    text: &str,
    source: Option<&str>,
) {
    match bundle.tokenizer.encode(text, add_special) {
        Ok(encoding) => match source {
            Some(p) => eprintln!("[{} tokens from {p}]", encoding.get_ids().len()),
            None => eprintln!("[{} tokens]", encoding.get_ids().len()),
        },
        Err(e) => eprintln!("[tokenize failed: {e}]"),
    }
    transcript.push(ChatTurn {
        role: "user",
        content: text.to_string(),
    });
    transcript.push(ChatTurn {
        role: "assistant",
        content: STUB_REPLY.to_string(),
    });
    println!("assistant: {STUB_REPLY}");
}

fn handle_read(
    bundle: &TokenizerBundle,
    transcript: &mut Vec<ChatTurn>,
    add_special: bool,
    path: &str,
) {
    if path.is_empty() {
        println!("usage: /read <path>");
        return;
    }
    match fs::read_to_string(path) {
        Ok(content) => {
            let trimmed = content.trim();
            if trimmed.is_empty() {
                println!("(file is empty: {path})");
                return;
            }
            handle_prompt(bundle, transcript, add_special, trimmed, Some(path));
        }
        Err(e) => println!("/read failed: {e}"),
    }
}
