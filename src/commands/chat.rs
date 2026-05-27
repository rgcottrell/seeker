//! `seeker chat` — interactive REPL. Loads the model, renders each turn
//! through the GGUF's embedded chat template, and decodes via the GPU
//! sampler chain. Model selection mirrors `inspect` / `tokenize` (either
//! `--hf-*` or `-m/--model PATH`). KV cache persists across turns; the
//! sampler's RNG / recent-token state survives `/clear`.

use std::borrow::Cow;
use std::error::Error;
use std::fs;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use clap::Args;
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::{Completer, Editor, Helper, Hinter, Validator};

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

    /// Extra key/value pairs merged into the chat-template rendering context,
    /// as a JSON object string, e.g. `'{"enable_thinking":false}'`. Keys
    /// override the built-in context variables. This is how reasoning /
    /// "thinking" mode is controlled (e.g. Qwen3's `enable_thinking`) — there
    /// is no dedicated flag; absent a kwarg, the template's own default
    /// applies. Mirrors llama.cpp's `--chat-template-kwargs`.
    #[arg(long = "chat-template-kwargs", value_parser = chat_template::parse_template_kwargs)]
    chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
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
    let mut cache = engine.allocate_kv_cache(
        dims.n_layer,
        dims.head_dim,
        dims.n_head_kv,
        cache_config,
    )?;
    if let Some(ssm) = model.ssm_state_dims() {
        cache.allocate_ssm_state(
            &engine.device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
        tracing::info!(
            n_ssm_layers = ssm.n_ssm_layers,
            conv_state_floats = ssm.conv_state_floats,
            gdn_state_floats = ssm.gdn_state_floats,
            "ssm state allocated",
        );
    }

    let sampler = Sampler::new(args.sampler_config());

    // Stop on the GGUF-declared EOS. For chat-tuned models this is
    // usually `<|im_end|>` (or equivalent); for non-chat models we
    // already errored above.
    let mut eos_ids: Vec<u32> = Vec::new();
    if let Some(id) = model.tokenizer().eos_id {
        eos_ids.push(id);
    }

    // Reasoning-model think markers, if present (single special tokens for
    // Qwen3-style models). Used to split reasoning from the final answer.
    let think_open_id = model.tokenizer().tokenizer.token_to_id("<think>");
    let think_close_id = model.tokenizer().tokenizer.token_to_id("</think>");

    let mut session = ChatSession {
        engine,
        model,
        cache,
        sampler,
        messages: Vec::new(),
        prior_tokens: Vec::new(),
        chat_template,
        eos_ids,
        think_open_id,
        think_close_id,
        max_tokens: args.max_tokens,
        template_kwargs: args.chat_template_kwargs.clone().unwrap_or_default(),
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
    /// `<think>` / `</think>` token ids, when the model has them. Used to
    /// split a reply's reasoning from its final answer (for coloring and for
    /// storing `reasoning_content` separately). `None` for non-reasoning
    /// models, in which case the whole reply is treated as the final answer.
    think_open_id: Option<u32>,
    think_close_id: Option<u32>,
    max_tokens: u32,
    /// Extra template-context variables from `--chat-template-kwargs`,
    /// merged into every render (override the built-in variables). Carries
    /// switches like `enable_thinking` when the user sets them.
    template_kwargs: serde_json::Map<String, serde_json::Value>,
}

/// Timing for one reply, used for the `[ Prompt … | Generation … ]` line.
/// `prompt_tokens` is the prefill suffix actually fed this turn (after
/// prefix-reuse), timed by `prefill_secs`; `decode_tokens` is the
/// autoregressive steps after the first, timed by `decode_secs`.
struct ReplyStats {
    prompt_tokens: usize,
    prefill_secs: f64,
    decode_tokens: usize,
    decode_secs: f64,
}

/// Which part of a streamed reply a piece belongs to — drives its color.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Segment {
    Thinking,
    Final,
}

// ANSI styling, brightness-separated so it reads for any color vision: user
// input is bold (bright) cyan, reasoning is dim/faint, the final answer is the
// terminal's normal default. User input is colored via the rustyline
// highlighter (`ChatHelper`); the streamed reply uses these directly.
const C_USER: &str = "\x1b[1;36m"; // bold cyan
const C_THINK: &str = "\x1b[2m"; // dim / faint
const C_RESET: &str = "\x1b[0m";

/// Color is emitted only to a real terminal, and suppressed when `NO_COLOR`
/// is set (see https://no-color.org).
fn color_enabled() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

/// True when the rendered prompt left a `<think>` block open (the assistant
/// opener starts inside thinking), so the model's first generated tokens are
/// reasoning even though the opening tag isn't part of the stream.
fn prompt_opens_think(rendered: &str) -> bool {
    match (rendered.rfind("<think>"), rendered.rfind("</think>")) {
        (Some(open), Some(close)) => open > close,
        (Some(_), None) => true,
        _ => false,
    }
}

/// rustyline line-editor helper: colors the prompt and the text being typed
/// in the user color. Completion / hints / validation are the default no-ops.
#[derive(Completer, Helper, Hinter, Validator)]
struct ChatHelper;

impl Highlighter for ChatHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if line.is_empty() || !color_enabled() {
            Cow::Borrowed(line)
        } else {
            Cow::Owned(format!("{C_USER}{line}{C_RESET}"))
        }
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        if color_enabled() {
            Cow::Owned(format!("{C_USER}{prompt}{C_RESET}"))
        } else {
            Cow::Borrowed(prompt)
        }
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        // Re-highlight on every edit so the whole input line stays colored.
        color_enabled()
    }
}

impl ChatSession {
    /// Push a user turn, render + tokenize, decode the assistant reply, and
    /// store the turn. `on_text` fires once per sampled token with the
    /// newly-emitted byte slice — the REPL streams output through it. The
    /// full reply is stored in `self.messages`; the return value carries
    /// per-turn timing for the stats line.
    fn handle_user_message(
        &mut self,
        text: &str,
        mut on_text: impl FnMut(&str, Segment),
    ) -> Result<ReplyStats, Box<dyn Error>> {
        self.messages.push(ChatMessage::user(text));

        let bundle = self.model.tokenizer();
        let bos = bundle.bos_token.as_deref().unwrap_or("");
        let eos = bundle.eos_token.as_deref().unwrap_or("");
        let rendered = chat_template::render(
            &self.chat_template,
            &self.messages,
            /* add_generation_prompt = */ true,
            bos,
            eos,
            &self.template_kwargs,
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
        // After each non-EOS token we step a `DecodeStream` and emit any
        // newly-completed UTF-8 chars. The streaming decoder is the right
        // tool here: when a multi-byte char (emoji, CJK) is split across
        // two BPE byte tokens, a bulk `decode` of the running prefix
        // substitutes `\u{fffd}` for the in-flight bytes and then
        // *retcons* it into the real char when the completing token
        // arrives — naive byte-length diffing slices into the middle of
        // that real char and panics.
        let mut step_tokens = delta;
        let prompt_tokens = step_tokens.len(); // prefill suffix fed this turn
        let mut prefill_secs = 0.0f64;
        let mut decode_secs = 0.0f64;
        let mut forwards = 0usize;
        let mut assistant_tokens: Vec<u32> = Vec::new();
        // Reasoning state: the prompt's assistant opener may already be inside
        // a `<think>` block (Qwen3 with thinking on), so seed from it; flips
        // are then driven by the think token ids below.
        let mut in_think = prompt_opens_think(&rendered);
        let mut think_close_at: Option<usize> = None;
        let mut stream = self
            .model
            .tokenizer()
            .tokenizer
            .decode_stream(/* skip_special_tokens = */ true);
        loop {
            if (self.cache.position as usize + step_tokens.len() + 1) as u32
                > self.cache.config.max_seq_len
            {
                break;
            }
            let cache = &mut self.cache;
            let model = &self.model;
            let t0 = std::time::Instant::now();
            let token = self.engine.forward_sampled(
                model.weights(),
                &mut self.sampler,
                |ctx| model.record_forward(ctx, cache, &step_tokens, cache.position),
            )?;
            // Forward 0 is the prefill (N = prompt_tokens); the rest are
            // single-token decode steps. Time them separately, like llama.cpp.
            let dt = t0.elapsed().as_secs_f64();
            if forwards == 0 {
                prefill_secs = dt;
            } else {
                decode_secs += dt;
            }
            forwards += 1;
            if self.eos_ids.contains(&token) {
                // Don't emit EOS — but it IS now in the cache (the model
                // wrote K/V for it). Track that in prior_tokens so the
                // next render's prefix-match accounts for it.
                assistant_tokens.push(token);
                break;
            }
            assistant_tokens.push(token);

            // Token-level think transitions detected by id (the boundary is
            // only reliable here, not in the decoded text). The marker tokens
            // belong to the reasoning region, so a piece is dimmed if we were
            // in think before this token OR still are after it — that keeps a
            // visible `</think>` dimmed with the reasoning rather than the
            // answer.
            let was_in_think = in_think;
            if Some(token) == self.think_open_id {
                in_think = true;
            } else if Some(token) == self.think_close_id {
                in_think = false;
                think_close_at = Some(assistant_tokens.len() - 1);
            }
            let seg = if was_in_think || in_think {
                Segment::Thinking
            } else {
                Segment::Final
            };

            // Stream emit: `step` buffers partial UTF-8 internally and
            // returns `Some(piece)` only once one or more complete chars
            // are ready. Errors are rare (the tokenizer succeeds on its
            // own outputs); on the off-chance we hit one, skip the emit
            // — the next step's output will catch up.
            if let Ok(Some(piece)) = stream.step(token) {
                on_text(&piece, seg);
            }

            if assistant_tokens.len() as u32 >= self.max_tokens {
                break;
            }
            step_tokens = vec![token];
        }

        // Split the reply into final answer + reasoning at the `</think>`
        // token (when one was emitted). `decode` with skip_special_tokens
        // drops the think markers and EOS from each half. Storing reasoning
        // in its own field lets reasoning templates (Qwen3) decide per-turn
        // whether to re-include it — by default they drop it from older turns
        // (override with `--chat-template-kwargs '{"preserve_thinking":true}'`).
        let tok = &self.model.tokenizer().tokenizer;
        let decode = |ids: &[u32]| {
            tok.decode(ids, /* skip_special_tokens = */ true)
                .map_err(|e| format!("decode failed: {e}"))
        };
        let (content, reasoning_content) = match think_close_at {
            Some(c) => {
                let reasoning = decode(&assistant_tokens[..c])?;
                let answer = decode(&assistant_tokens[c + 1..])?;
                let reasoning = reasoning.trim();
                let r = (!reasoning.is_empty()).then(|| reasoning.to_string());
                (answer, r)
            }
            // No closed think block (non-reasoning model, thinking disabled, or
            // the budget ran out mid-thought): keep the whole reply as the
            // answer so nothing is lost.
            None => (decode(&assistant_tokens)?, None),
        };

        self.messages
            .push(ChatMessage::assistant(content, reasoning_content));

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

        Ok(ReplyStats {
            prompt_tokens,
            prefill_secs,
            // The prefill forward emits the first token; the rest are decode.
            decode_tokens: forwards.saturating_sub(1),
            decode_secs,
        })
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
    let mut editor = Editor::new()?;
    editor.set_helper(Some(ChatHelper));
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
    println!(); // blank line between the input and the reply
    let _ = std::io::stdout().flush();
    let color = color_enabled();
    let result = session.handle_user_message(line, |delta, seg| {
        // Reasoning is dimmed; the final answer uses the normal default color.
        if color && seg == Segment::Thinking {
            print!("{C_THINK}{delta}{C_RESET}");
        } else {
            print!("{delta}");
        }
        let _ = std::io::stdout().flush();
    });
    match result {
        Ok(stats) => {
            println!(); // terminate the streamed reply line
            println!(); // blank line between the reply and the stats
            print_stats(&stats);
            println!(); // blank line before the next prompt
        }
        Err(e) => println!("\n[error: {e}]\n"),
    }
}

/// `[ Prompt: X t/s | Generation: Y t/s ]`, dimmed when stdout is a TTY.
fn print_stats(stats: &ReplyStats) {
    let prompt_tps = stats.prompt_tokens as f64 / stats.prefill_secs.max(1e-9);
    let gen_tps = if stats.decode_secs > 0.0 {
        stats.decode_tokens as f64 / stats.decode_secs
    } else {
        0.0
    };
    let line = format!("[ Prompt: {prompt_tps:.1} t/s | Generation: {gen_tps:.1} t/s ]");
    if std::io::stdout().is_terminal() {
        println!("\x1b[2m{line}\x1b[0m"); // ANSI dim
    } else {
        println!("{line}");
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

#[cfg(test)]
mod tests {
    use super::prompt_opens_think;

    #[test]
    fn prompt_opens_think_detects_open_block() {
        // Qwen3 thinking-on: assistant opener ends with an unclosed `<think>`.
        assert!(prompt_opens_think("<|im_start|>assistant\n<think>\n"));
        // Open then closed again (thinking disabled → empty block) is closed.
        assert!(!prompt_opens_think("<think>\n\n</think>\n\n"));
        // A prior closed turn followed by a new open block is open.
        assert!(prompt_opens_think("<think>\na\n</think>\nb<|im_start|>assistant\n<think>\n"));
        // No think markers at all (e.g. Llama) → closed.
        assert!(!prompt_opens_think("<|im_start|>assistant\n"));
    }
}
