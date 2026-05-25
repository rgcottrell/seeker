//! `seeker chat` — interactive REPL. Loads the model, renders each turn
//! through the GGUF's embedded chat template, and decodes via the GPU
//! sampler chain. Model selection mirrors `inspect` / `tokenize` (either
//! `--hf-*` or `-m/--model PATH`). KV cache persists across turns; the
//! sampler's RNG / recent-token state survives `/clear`.

use std::error::Error;
use std::fs;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use clap::Args;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::chat_template::{self, ChatMessage};
use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::{GgmlType, GgufFile, MetadataValue};
use crate::inference::kv_cache::{parse_dtype, KvCacheConfig};
use crate::inference::sample::{Sampler, SamplerConfig};
use crate::inference::Engine;
use crate::tokenizer::build_tokenizer;

const SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

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

    /// Max tokens per assistant reply.
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,

    /// KV-cache budget for the whole conversation, in tokens.
    #[arg(long = "ctx-size", default_value_t = 4096)]
    ctx_size: u32,

    /// KV cache K dtype. One of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1.
    #[arg(long = "cache-type-k", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_k: GgmlType,

    /// KV cache V dtype. Same legal values as --cache-type-k.
    #[arg(long = "cache-type-v", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_v: GgmlType,

    // ─── Sampling ───────────────────────────────────────────────────────
    /// Sampling temperature. 0 → greedy argmax. (llama.cpp default: 0.8)
    #[arg(long = "temp", alias = "temperature", default_value_t = 0.8)]
    temperature: f32,

    /// Top-K filter (0 = disabled, full vocab). (llama.cpp default: 40)
    #[arg(long = "top-k", default_value_t = 40)]
    top_k: u32,

    /// Top-P (nucleus) filter (1.0 = disabled).
    #[arg(long = "top-p", default_value_t = 0.95)]
    top_p: f32,

    /// Min-P filter (0.0 = disabled). (llama.cpp default: 0.05)
    #[arg(long = "min-p", default_value_t = 0.05)]
    min_p: f32,

    /// Presence penalty (subtract from any repeated-token logit; 0.0 = off).
    #[arg(long = "presence-penalty", default_value_t = 0.0)]
    presence_penalty: f32,

    /// Frequency penalty (subtract count×p from repeated-token logits; 0.0 = off).
    #[arg(long = "frequency-penalty", default_value_t = 0.0)]
    frequency_penalty: f32,

    /// Repetition penalty (multiply/divide repeated logits; 1.0 = off).
    #[arg(long = "repeat-penalty", alias = "repetition-penalty", default_value_t = 1.0)]
    repeat_penalty: f32,

    /// How many trailing tokens contribute to penalties.
    #[arg(long = "penalty-last-n", default_value_t = 64)]
    penalty_last_n: usize,

    /// RNG seed for stochastic sampling.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn parse_dtype_arg(s: &str) -> Result<GgmlType, String> {
    parse_dtype(s)
}

impl ChatArgs {
    fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_penalty: self.repeat_penalty,
            penalty_last_n: self.penalty_last_n,
            seed: self.seed,
        }
    }
}

pub async fn run(args: ChatArgs) -> Result<(), Box<dyn Error>> {
    let path = resolve_model_path(&args).await?;
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;
    let chat_template = bundle.chat_template.clone().ok_or_else(|| -> Box<dyn Error> {
        "model has no `tokenizer.chat_template` — use `seeker run` for base completions".into()
    })?;

    let engine = Engine::new(SCRATCH_BYTES)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened");
    let weights = engine.upload_weights(&gguf)?;
    let model = crate::models::open(&gguf, weights, bundle)?;

    let cache_config = KvCacheConfig {
        k_dtype: args.cache_type_k,
        v_dtype: args.cache_type_v,
        max_seq_len: args.ctx_size,
    };
    let dims = model.cache_dims();
    let cache = engine.allocate_kv_cache(
        dims.n_layer,
        dims.head_dim,
        dims.n_head_kv,
        cache_config,
    )?;

    let sampler = Sampler::new(args.sampler_config());

    // Stop on the GGUF-declared EOS. For chat-tuned models this is
    // usually `<|im_end|>` (or equivalent); for non-chat models we
    // already errored above.
    let mut eos_ids: Vec<u32> = Vec::new();
    if let Some(id) = model.tokenizer().eos_id {
        eos_ids.push(id);
    }

    let mut session = ChatSession {
        engine,
        model,
        cache,
        sampler,
        messages: Vec::new(),
        prior_tokens: Vec::new(),
        chat_template,
        eos_ids,
        max_tokens: args.max_tokens,
    };

    let history = if args.no_history {
        None
    } else {
        args.history_file
            .clone()
            .or_else(default_history_path)
    };

    if std::io::stdin().is_terminal() {
        run_interactive(&mut session, &gguf, &path, history.as_deref())
    } else {
        run_piped(&mut session)
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

/// All conversation state plus the GPU resources needed to advance it.
/// One per REPL session; persists across turns.
struct ChatSession {
    engine: Engine,
    model: Box<dyn crate::models::Model>,
    cache: crate::inference::kv_cache::KvCache,
    sampler: Sampler,
    /// Conversation so far, in the order the chat template iterates.
    messages: Vec<ChatMessage>,
    /// Token IDs currently in the KV cache. Used for prefix-matching
    /// between turns so we re-prefill only the divergent suffix.
    prior_tokens: Vec<u32>,
    chat_template: String,
    /// Tokens that terminate an assistant reply (GGUF `eos_token_id`).
    eos_ids: Vec<u32>,
    max_tokens: u32,
}

impl ChatSession {
    /// Push a user turn, render + tokenize, decode the assistant reply,
    /// and store the turn. `on_text` fires once per sampled token with the
    /// newly-emitted byte slice — used by the REPL to stream output as it's
    /// generated. Returns the full assistant reply (without trailing EOS
    /// markers) so callers that just want the final string can ignore the
    /// callback.
    fn handle_user_message(
        &mut self,
        text: &str,
        mut on_text: impl FnMut(&str),
    ) -> Result<String, Box<dyn Error>> {
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: text.to_string(),
        });

        let bundle = self.model.tokenizer();
        let bos = bundle.bos_token.as_deref().unwrap_or("");
        let eos = bundle.eos_token.as_deref().unwrap_or("");
        let rendered = chat_template::render(
            &self.chat_template,
            &self.messages,
            /* add_generation_prompt = */ true,
            bos,
            eos,
        )?;

        // Tokenize the full conversation. `add_special_tokens=false`: the
        // template already includes any BOS markers it wants.
        let enc = bundle
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| format!("tokenize failed: {e}"))?;
        let new_tokens: Vec<u32> = enc.get_ids().to_vec();

        // Prefix reuse: keep the cache prefix that still matches.
        let common = self
            .prior_tokens
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        if common > self.cache.position as usize {
            // Shouldn't happen — prior_tokens tracks what's in the cache —
            // but be defensive.
            return Err(format!(
                "cache/prior_tokens drift: common={common} cache.position={}",
                self.cache.position
            )
            .into());
        }
        self.cache.position = common as u32;
        let delta: Vec<u32> = new_tokens[common..].to_vec();
        if delta.is_empty() {
            // Pathological: user typed nothing new after template render
            // (e.g. empty content). Force at least the assistant opener
            // by feeding one rendered token, otherwise the model has no
            // logits to sample from.
            return Err("nothing new to feed after template render".into());
        }

        if (common + delta.len()) as u32 > self.cache.config.max_seq_len {
            return Err(format!(
                "conversation length {} exceeds --ctx-size {}",
                common + delta.len(),
                self.cache.config.max_seq_len
            )
            .into());
        }

        // Prefill + decode in one loop. forward_sampled records the model
        // forward then the sampler chain, returning the next token id.
        // After each non-EOS token we cumulative-decode and emit the
        // newly-completed byte slice. Cumulative-decode is the only
        // reliable pattern: a single BPE token may be a fragment of a
        // multi-byte UTF-8 char or a whitespace-managing prefix, but the
        // running prefix always decodes cleanly with `skip_special_tokens=true`.
        let mut step_tokens = delta;
        let mut assistant_tokens: Vec<u32> = Vec::new();
        let mut printed_len: usize = 0;
        loop {
            if (self.cache.position as usize + step_tokens.len() + 1) as u32
                > self.cache.config.max_seq_len
            {
                break;
            }
            let cache = &mut self.cache;
            let model = &self.model;
            let token = self.engine.forward_sampled(
                model.weights(),
                &mut self.sampler,
                |ctx| model.record_forward(ctx, cache, &step_tokens, cache.position),
            )?;
            if self.eos_ids.contains(&token) {
                // Don't emit EOS — but it IS now in the cache (the model
                // wrote K/V for it). Track that in prior_tokens so the
                // next render's prefix-match accounts for it.
                assistant_tokens.push(token);
                break;
            }
            assistant_tokens.push(token);

            // Stream the new bytes. `decode` errors are rare (the tokenizer
            // should always succeed on its own outputs); on the off-chance
            // we hit one, skip the emit and try again next iteration —
            // whatever bytes were withheld will reappear in the cumulative
            // decode once the run is decodable.
            if let Ok(text) = self
                .model
                .tokenizer()
                .tokenizer
                .decode(&assistant_tokens, /* skip_special_tokens = */ true)
            {
                if text.len() > printed_len {
                    on_text(&text[printed_len..]);
                    printed_len = text.len();
                }
            }

            if assistant_tokens.len() as u32 >= self.max_tokens {
                break;
            }
            step_tokens = vec![token];
        }

        // Final reply — same decode used for streaming, but we want the
        // canonical string for storage. (When EOS was sampled the per-loop
        // emit skipped it, so this matches what the user saw.)
        let reply = self
            .model
            .tokenizer()
            .tokenizer
            .decode(&assistant_tokens, /* skip_special_tokens = */ true)
            .map_err(|e| format!("decode failed: {e}"))?;

        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: reply.clone(),
        });

        // The cache holds K/V for every token whose forward we *fed* —
        // that's the rendered prompt plus all but the most-recently
        // sampled token (which was the *output* of the last forward,
        // not an input, so it hasn't been encoded yet). prior_tokens
        // mirrors what the cache holds so prefix-matching next turn
        // lines up with cache.position.
        self.prior_tokens = new_tokens;
        if !assistant_tokens.is_empty() {
            let n = assistant_tokens.len();
            self.prior_tokens.extend(&assistant_tokens[..n - 1]);
        }

        Ok(reply)
    }

    /// Reset everything that ties to the current conversation; keep the
    /// model / engine / sampler RNG so deterministic seeds still
    /// reproduce after a `/clear`.
    fn clear(&mut self) {
        self.messages.clear();
        self.prior_tokens.clear();
        self.cache.reset();
        self.sampler.reset_recent();
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

fn run_interactive(
    session: &mut ChatSession,
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
                        session.clear();
                        println!("(conversation cleared)");
                    } else if let Some(arg) = cmd.strip_prefix("read") {
                        handle_read(session, arg.trim());
                    } else {
                        println!("unknown command: /{cmd}");
                    }
                    continue;
                }
                emit_reply(session, line);
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

fn run_piped(session: &mut ChatSession) -> Result<(), Box<dyn Error>> {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let raw = line?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        emit_reply(session, trimmed);
    }
    Ok(())
}

fn emit_reply(session: &mut ChatSession, line: &str) {
    print!("assistant: ");
    let _ = std::io::stdout().flush();
    let result = session.handle_user_message(line, |delta| {
        print!("{delta}");
        let _ = std::io::stdout().flush();
    });
    match result {
        Ok(_) => println!(),
        Err(e) => println!("\n[error: {e}]"),
    }
}

fn handle_read(session: &mut ChatSession, path: &str) {
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
            eprintln!("[{} bytes from {path}]", trimmed.len());
            emit_reply(session, trimmed);
        }
        Err(e) => println!("/read failed: {e}"),
    }
}
