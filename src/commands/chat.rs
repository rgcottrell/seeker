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
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Args;
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Completer, Editor, Helper, Hinter};

use crate::chat_template::{self, ChatMessage};
use crate::commands::chat_cache;
use crate::commands::download;
use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::{GgmlType, GgufFile, MetadataValue};
use crate::inference::kv_cache::{parse_dtype, KvCacheConfig};
use crate::inference::sample::{Sampler, SamplerConfig};
use crate::inference::Engine;
use crate::tokenizer::build_tokenizer;
use crate::vision::encoder::{HostWeights, VisionEncoder};

/// The media placeholder (`mtmd_default_marker()`). `/image` prepends it to the
/// user turn; `render_prompt` splits the rendered string on it to place the
/// vision block where the chat template put the content. See `commands::run`.
const MEDIA_MARKER: &str = "<__media__>";

/// A vision tower built on first `/image` and kept for the session. `vision`
/// owns the uploaded mmproj weights (the encoder's tensor views borrow nothing
/// — they hold GPU buffer handles kept valid by `vision`); `host_weights` is the
/// CPU-side patch-embed/pos-embd copy the encoder needs for the pos-embd resize.
struct VisionCtx {
    vision: crate::vision::VisionModel,
    encoder: VisionEncoder,
    host_weights: HostWeights,
}

/// An image encoded through the vision tower: `[proj_dim, n_tok]` host f32
/// (column = merged token) plus its merged-grid dims. Spliced into the decoder
/// residual at the `<|image_pad|>` rows during the image turn's prefill.
#[derive(Clone)]
struct EncodedImage {
    embeddings: Vec<f32>,
    nx: usize,
    ny: usize,
    n_tok: usize,
}

/// Set by the SIGINT watcher when the user presses Ctrl+C *during* generation
/// (in interactive mode). The decode loop polls it between tokens and stops
/// the current reply, returning control to the prompt instead of letting the
/// default SIGINT kill the process. At the readline prompt rustyline runs the
/// terminal in raw mode (ISIG off), so Ctrl+C there surfaces as
/// `ReadlineError::Interrupted` rather than a signal — this flag is only ever
/// set mid-generation.
static GENERATION_CANCELLED: AtomicBool = AtomicBool::new(false);

/// Install a SIGINT handler that flips [`GENERATION_CANCELLED`] so an
/// in-progress reply can be interrupted (interactive mode only — piped mode
/// keeps the default kill-on-Ctrl+C). Uses a `recv()` loop so it re-arms for
/// every turn, not just the first interrupt. A failure to install is logged
/// and ignored (Ctrl+C then falls back to the default action).
#[cfg(unix)]
fn spawn_interrupt_watcher() {
    use tokio::signal::unix::{signal, SignalKind};
    match signal(SignalKind::interrupt()) {
        Ok(mut sigint) => {
            tokio::spawn(async move {
                while sigint.recv().await.is_some() {
                    GENERATION_CANCELLED.store(true, Ordering::SeqCst);
                }
            });
        }
        Err(e) => tracing::debug!(error = %e, "could not install SIGINT handler"),
    }
}
#[cfg(not(unix))]
fn spawn_interrupt_watcher() {}

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

    /// Multi-line input: Enter inserts a newline and a line ending with `\`
    /// submits. Without this, the default is the inverse — a trailing `\`
    /// continues to the next line and a bare Enter submits. (llama.cpp's
    /// `--multiline-input`.)
    #[arg(long = "multiline-input")]
    multiline_input: bool,

    /// Override the history-file location. Defaults to an OS-appropriate
    /// path under the user's data directory.
    #[arg(long)]
    history_file: Option<PathBuf>,

    /// Persist the conversation's KV cache (+ SSM state) and messages to this
    /// file, and resume from it on the next run with the same model — skipping
    /// the prefill of the restored prefix. (llama.cpp's `--prompt-cache`.)
    #[arg(long = "prompt-cache")]
    prompt_cache: Option<PathBuf>,

    /// Load `--prompt-cache` but don't write it back on exit (read-only).
    #[arg(long = "prompt-cache-ro", requires = "prompt_cache")]
    prompt_cache_ro: bool,

    /// Max tokens per assistant reply.
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,

    /// KV-cache budget for the whole conversation, in tokens.
    #[arg(long = "ctx-size", default_value_t = 4096)]
    ctx_size: u32,

    /// When the context fills, drop the oldest conversation turns and continue
    /// instead of stopping (matches llama.cpp's opt-in context shift). Off by
    /// default: the reply is truncated cleanly at the limit.
    #[arg(long = "context-shift")]
    context_shift: bool,

    /// Conversation turns to pin (in addition to the system prompt) when
    /// `--context-shift` drops old history. Counts whole user→assistant
    /// exchanges, not raw tokens — seeker re-renders through the chat template
    /// rather than slicing KV cells, so eviction is message-granular.
    #[arg(long = "keep", default_value_t = 0)]
    keep: u32,

    // ─── Batch limits (llama.cpp parity) ────────────────────────────────
    /// Logical batch size (max tokens per submit). Validation-only in this
    /// single-sequence engine; `--ubatch-size` is the memory-relevant knob.
    #[arg(short = 'b', long = "batch-size", default_value_t = 2048)]
    batch_size: u32,

    /// Physical micro-batch size: prefill is split into ≤ this many tokens
    /// per GPU pass so scratch memory stays bounded on long prompts.
    /// 0 = unbounded (single pass). (short: -ub)
    #[arg(long = "ubatch-size", default_value_t = 512)]
    ubatch_size: u32,

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

    /// How many trailing tokens contribute to penalties. `-1` = the whole
    /// context (`--ctx-size`); `0` = disabled. (llama.cpp's `--repeat-last-n`.)
    #[arg(long = "penalty-last-n", default_value_t = 64, allow_hyphen_values = true)]
    penalty_last_n: i32,

    /// Never stop on an end-of-generation token; generate until `--max-tokens`.
    /// (llama.cpp's `--ignore-eos`.)
    #[arg(long = "ignore-eos")]
    ignore_eos: bool,

    /// Bias a token's logit, repeatable. Format `ID(+/-)BIAS` (llama.cpp
    /// style) or `ID=BIAS`, e.g. `--logit-bias 15043+2.0 --logit-bias
    /// 128009-inf` (boost 15043, ban 128009). `-inf` bans, `+inf` forces.
    #[arg(long = "logit-bias", value_parser = parse_logit_bias)]
    logit_bias: Vec<(u32, f32)>,

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

    /// System prompt, prepended as the conversation's first (system) message.
    /// Overrides any default the chat template would inject, persists across
    /// `/clear`, and can be changed mid-session with `/system <text>`.
    /// (short: -sys)
    #[arg(long = "system-prompt")]
    system_prompt: Option<String>,

    /// Read the system prompt from a UTF-8 text file instead of the CLI.
    /// (short: -sysf)
    #[arg(long = "system-prompt-file", conflicts_with = "system_prompt")]
    system_prompt_file: Option<PathBuf>,

    /// Do not load the matching mmproj vision projector.
    #[arg(long = "no-mmproj")]
    no_mmproj: bool,
}

fn parse_dtype_arg(s: &str) -> Result<GgmlType, String> {
    parse_dtype(s)
}

/// Parse one `--logit-bias` entry: `ID=BIAS` or llama.cpp's `ID(+/-)BIAS`
/// (the sign is part of the bias). BIAS may be `inf` / `-inf` to force / ban.
/// Shared with `seeker serve` (same `--logit-bias` flag).
pub(crate) fn parse_logit_bias(s: &str) -> Result<(u32, f32), String> {
    let (id_str, bias_str) = if let Some((a, b)) = s.split_once('=') {
        (a, b)
    } else {
        // Leading digits are the token id; the remainder (with its sign) is
        // the bias — e.g. "15043+2.0", "128009-inf".
        let split = s
            .find(|c: char| !c.is_ascii_digit())
            .ok_or_else(|| format!("--logit-bias {s:?}: expected ID followed by +/-BIAS"))?;
        if split == 0 {
            return Err(format!("--logit-bias {s:?}: missing token id"));
        }
        (&s[..split], &s[split..])
    };
    let id: u32 = id_str
        .parse()
        .map_err(|_| format!("--logit-bias {s:?}: invalid token id {id_str:?}"))?;
    let bias: f32 = bias_str
        .parse()
        .map_err(|_| format!("--logit-bias {s:?}: invalid bias {bias_str:?}"))?;
    Ok((id, bias))
}

impl ChatArgs {
    fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig::from_cli(
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.presence_penalty,
            self.frequency_penalty,
            self.repeat_penalty,
            self.penalty_last_n,
            self.ctx_size,
            self.seed,
            self.logit_bias.clone(),
        )
    }
}

pub async fn run(args: ChatArgs) -> Result<(), Box<dyn Error>> {
    let resolved = resolve_model_path(&args).await?;
    let path = resolved.main.clone();
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;
    let chat_template = bundle.chat_template.clone().ok_or_else(|| -> Box<dyn Error> {
        "model has no `tokenizer.chat_template` — use `seeker run` for base completions".into()
    })?;

    let mut engine = Engine::new(args.ubatch_size, args.batch_size)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened");
    let weights = engine.upload_weights(&gguf)?;
    let model = crate::models::open(&gguf, weights, bundle, /*spec_enabled=*/ false)?;

    // The mmproj vision sidecar (if resolved and not `--no-mmproj`). The vision
    // tower is built lazily on the first `/image` (see `ChatSession::attach_image`)
    // so a text-only session never uploads the projector.
    let mmproj_path = if args.no_mmproj { None } else { resolved.mmproj.clone() };

    // Size the scratch (compute buffer) for this model + n_ubatch (and the
    // full ctx for heterogeneous caches), replacing the Engine::new
    // placeholder. An `/image` turn grows this on demand (image prefill is
    // single-pass).
    let scratch_bytes = model.scratch_bytes_estimate(
        args.ubatch_size,
        args.ctx_size,
        args.cache_type_k,
        args.cache_type_v,
    );
    engine.allocate_scratch(scratch_bytes)?;

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

    // Stop on any end-of-generation token (EOS / EOT / EOM / well-known turn
    // terminators), matching llama.cpp's `llama_token_is_eog`. A chat-tuned
    // model whose turn terminator (`<|im_end|>`) differs from its EOS would
    // otherwise never stop and burn to --max-tokens every reply.
    // `--ignore-eos` drops the whole stop set, so generation runs to
    // --max-tokens (or a /clear / Ctrl+C).
    let eos_ids: Vec<u32> = if args.ignore_eos {
        Vec::new()
    } else {
        model.tokenizer().eog_ids.clone()
    };

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
        context_shift: args.context_shift,
        keep_turns: args.keep,
        template_kwargs: args.chat_template_kwargs.clone().unwrap_or_default(),
        mmproj_path,
        vision_ctx: None,
        pending_image: None,
        image: None,
        scratch_bytes,
    };

    // Seed an optional system prompt (CLI flag or file) as messages[0].
    if let Some(text) = resolve_system_prompt(&args)? {
        session.set_system_prompt(text);
    }

    // Resume from `--prompt-cache` if it exists and matches this model — this
    // restores the saved KV/SSM + tokens + messages, overriding the fresh
    // system prompt above. A missing file is a normal first run; a corrupt or
    // mismatched file is warned about and ignored (fresh start).
    let arch = gguf.architecture().unwrap_or("unknown").to_string();
    if let Some(p) = &args.prompt_cache {
        match chat_cache::load(p, &arch, &mut session.cache) {
            Ok(Some((tokens, messages))) => {
                let turns = messages.iter().filter(|m| m.role != "system").count();
                session.prior_tokens = tokens;
                session.messages = messages;
                println!("(resumed {turns} turn(s) from {})", p.display());
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("{e}"),
        }
    }

    let history = if args.no_history {
        None
    } else {
        args.history_file
            .clone()
            .or_else(default_history_path)
    };

    let result = if std::io::stdin().is_terminal() {
        // Ctrl+C during a reply should stop that reply, not the program.
        spawn_interrupt_watcher();
        run_interactive(&mut session, &gguf, &path, history.as_deref(), args.multiline_input)
    } else {
        run_piped(&mut session)
    };

    // Persist the session for next time (best-effort; never fail the run).
    if let Some(p) = &args.prompt_cache {
        if !args.prompt_cache_ro {
            match chat_cache::save(p, &arch, &session.cache, &session.prior_tokens, &session.messages) {
                Ok(()) => tracing::info!(path = %p.display(), "prompt cache saved"),
                Err(e) => tracing::warn!("prompt-cache save failed: {e}"),
            }
        }
    }
    result
}

/// The system prompt to seed, from `--system-prompt` or `--system-prompt-file`
/// (clap makes them mutually exclusive). The file's trailing newline is
/// trimmed so an editor's stray `\n` doesn't leak into the prompt.
fn resolve_system_prompt(args: &ChatArgs) -> Result<Option<String>, Box<dyn Error>> {
    if let Some(s) = &args.system_prompt {
        return Ok(Some(s.clone()));
    }
    if let Some(path) = &args.system_prompt_file {
        let s = fs::read_to_string(path)
            .map_err(|e| format!("--system-prompt-file {}: {e}", path.display()))?;
        return Ok(Some(s.trim_end().to_string()));
    }
    Ok(None)
}

/// Resolve the main model path (and any matching mmproj sidecar) from either a
/// local `-m PATH` or an HF repo. For the local path, scans the model's
/// directory for an mmproj GGUF (unless `--no-mmproj`); for HF, asks
/// [`resolve_hf`] to fetch the sidecar too.
async fn resolve_model_path(args: &ChatArgs) -> Result<download::Resolved, Box<dyn Error>> {
    match (args.hf_repo.clone(), args.model.clone()) {
        (Some(repo), None) => Ok(resolve_hf(
            &HfResolveArgs {
                repo,
                file: args.hf_file.clone(),
                token: args.hf_token.clone(),
                offline: args.offline,
            },
            !args.no_mmproj,
        )
        .await?),
        (None, Some(model)) => {
            let mmproj = if args.no_mmproj {
                None
            } else {
                download::find_sidecar_mmproj(&model)
            };
            Ok(download::Resolved { main: model, mmproj })
        }
        _ => unreachable!("clap group invariant"),
    }
}

/// Conservative scratch estimate for the vision tower's single forward (mirrors
/// `commands::run` / `server`). [`VisionEncoder::encode_image`] reclaims each
/// stage's scratch between layers (checkpoint/restore), so the working set is
/// the persistent residual carriers + RoPE positions plus the single largest
/// stage — O(n_pos), NOT O(n_layer · n_pos). The per-token high-water across
/// the stages is ~28k floats; budget 32k for margin (copy ops, alignment).
/// Floored at 64 MiB.
fn vision_scratch_estimate(pimg: &crate::vision::preprocess::PreprocessedImage) -> u64 {
    let n_pos = (pimg.grid_w as u64) * (pimg.grid_h as u64);
    (32_000u64 * n_pos * 4).max(64 << 20)
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
    /// When true, evict the oldest turns to fit instead of stopping at the
    /// context limit (`--context-shift`).
    context_shift: bool,
    /// Leading turns pinned from eviction, beyond the system prompt (`--keep`).
    keep_turns: u32,
    /// Extra template-context variables from `--chat-template-kwargs`,
    /// merged into every render (override the built-in variables). Carries
    /// switches like `enable_thinking` when the user sets them.
    template_kwargs: serde_json::Map<String, serde_json::Value>,
    /// Path to the mmproj vision sidecar, if one was resolved and `--no-mmproj`
    /// wasn't passed. The vision tower ([`VisionCtx`]) is built lazily from it on
    /// the first `/image` (so a text-only session never uploads the projector).
    mmproj_path: Option<PathBuf>,
    /// The lazily-built vision tower (None until the first `/image`).
    vision_ctx: Option<VisionCtx>,
    /// An image encoded by `/image` but not yet attached to a message — moved to
    /// `image` (and the marker prepended) when the next user turn is sent.
    pending_image: Option<EncodedImage>,
    /// The single image committed to this conversation (its `<__media__>` marker
    /// lives in one user message). `None` for a text-only conversation. The
    /// first cut supports one image per conversation; `/clear` drops it.
    image: Option<EncodedImage>,
    /// Bytes the scratch region is currently sized for, so image prefills (which
    /// run single-pass, unlike chunked text prefill) can grow it on demand
    /// without shrinking it back.
    scratch_bytes: u64,
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
    /// True when the user interrupted the reply with Ctrl+C (partial output).
    interrupted: bool,
    /// True when the reply was truncated by hitting the context limit (the
    /// default no-`--context-shift` behavior).
    ctx_full: bool,
    /// Oldest turn pairs dropped by `--context-shift` before this reply.
    shifted_turns: usize,
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

/// rustyline line-editor helper: colors the prompt/input and decides when a
/// line is complete (multi-line input). Completion / hints are no-ops.
#[derive(Completer, Helper, Hinter)]
struct ChatHelper {
    /// `--multiline-input`: when true Enter inserts a newline and a trailing
    /// `\` submits; when false (default) a trailing `\` continues to the next
    /// line and a bare Enter submits.
    multiline: bool,
}

impl Validator for ChatHelper {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        // XOR: default mode is "incomplete iff trailing backslash"; multiline
        // mode flips it to "incomplete unless trailing backslash".
        let incomplete = self.multiline ^ ctx.input().ends_with('\\');
        Ok(if incomplete {
            ValidationResult::Incomplete
        } else {
            ValidationResult::Valid(None)
        })
    }
}

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

/// Collapse line-continuation markers from a (possibly multi-line) raw readline
/// result into the message text. In default mode a `\` right before a newline
/// was a "continue" marker → drop the `\`, keep the newline. In multiline mode
/// the single trailing `\` was the "submit" marker → drop it (intermediate
/// newlines are real). Literal backslashes elsewhere are left untouched.
fn join_continuations(raw: &str, multiline: bool) -> String {
    if multiline {
        raw.strip_suffix('\\').unwrap_or(raw).to_string()
    } else {
        raw.replace("\\\n", "\n")
    }
}

impl ChatSession {
    /// Push a user turn and generate the assistant reply via [`generate`]. On
    /// any error the just-pushed user turn is rolled back so a failed turn
    /// never leaves a dangling message in the history.
    fn handle_user_message(
        &mut self,
        text: &str,
        on_text: impl FnMut(&str, Segment),
    ) -> Result<ReplyStats, Box<dyn Error>> {
        // Attach a pending `/image` to this turn: the user content carries the
        // media marker at its head (like llama-mtmd-cli) and the encoded image
        // becomes the conversation's committed image. Moved, not cloned.
        let had_pending = self.pending_image.is_some();
        let content = if had_pending {
            format!("{MEDIA_MARKER}{text}")
        } else {
            text.to_string()
        };
        if had_pending {
            self.image = self.pending_image.take();
        }
        self.messages.push(ChatMessage::user(content));
        let result = self.generate(on_text);
        if result.is_err() {
            self.messages.pop();
            if had_pending {
                // Fully undo the attach so the image can be retried next turn.
                self.pending_image = self.image.take();
            }
        }
        result
    }

    /// Drop the last assistant reply and generate a fresh one from the same
    /// prompt (`/regen`). The conversation prompt is already fully in the
    /// cache, so [`generate`] re-feeds just its last token to get new logits;
    /// with a non-zero temperature the advanced RNG yields a different sample.
    /// On error the dropped reply is restored.
    fn regenerate(
        &mut self,
        on_text: impl FnMut(&str, Segment),
    ) -> Result<ReplyStats, Box<dyn Error>> {
        if self.messages.last().map(|m| m.role.as_str()) != Some("assistant") {
            return Err("nothing to regenerate yet".into());
        }
        let prev = self.messages.pop().expect("last message is assistant");
        let result = self.generate(on_text);
        if result.is_err() {
            self.messages.push(prev);
        }
        result
    }

    /// Render the current conversation with a generation prompt and tokenize
    /// it (`add_special_tokens=false` — the template emits any BOS it wants).
    /// Returns the rendered string (the reasoning seed reads it), the token ids,
    /// and — when an image is attached — the local index of the first
    /// `<|image_pad|>` token (where the vision-tower embeddings splice in). The
    /// rendered string carries the `<__media__>` marker (in the image turn's
    /// content); we split on it and replace it with
    /// `<|vision_start|><|image_pad|>×n_tok<|vision_end|>`. Shared by `generate`
    /// and `evict_to_fit`.
    #[allow(clippy::type_complexity)] // (rendered, tokens, image_start) is clearer inline than a named alias
    fn render_prompt(&self) -> Result<(String, Vec<u32>, Option<usize>), Box<dyn Error>> {
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
        let encode = |text: &str| -> Result<Vec<u32>, Box<dyn Error>> {
            Ok(bundle
                .tokenizer
                .encode(text, false)
                .map_err(|e| format!("tokenize failed: {e}"))?
                .get_ids()
                .to_vec())
        };
        let Some(img) = &self.image else {
            let tokens = encode(&rendered)?;
            return Ok((rendered, tokens, None));
        };
        // One image: split on the marker and splice the vision block.
        let (before, after) = rendered.split_once(MEDIA_MARKER).ok_or(
            "conversation has an image but the rendered prompt has no <__media__> marker \
             (chat template dropped it?)",
        )?;
        let tid = |s: &str| -> Result<u32, Box<dyn Error>> {
            bundle.tokenizer.token_to_id(s).ok_or_else(|| {
                format!("tokenizer has no {s} token — this model is not vision-capable").into()
            })
        };
        let mut tokens = encode(before)?;
        tokens.push(tid("<|vision_start|>")?);
        let image_start = tokens.len();
        tokens.resize(tokens.len() + img.n_tok, tid("<|image_pad|>")?);
        tokens.push(tid("<|vision_end|>")?);
        tokens.extend(encode(after)?);
        Ok((rendered, tokens, Some(image_start)))
    }

    /// `--context-shift`: drop the oldest *whole* user→assistant turn pairs
    /// until the rendered prompt fits in `ctx_size - max_tokens` (the reply
    /// needs that headroom, or it would just hit the limit immediately). A
    /// leading system message plus the first `keep_turns` exchanges are pinned,
    /// and the in-flight (last) user turn is never evicted. Whole-pair eviction
    /// keeps the post-pin sequence starting on a user turn, so templates that
    /// assert role alternation don't choke.
    ///
    /// On any drop, the cache + prior_tokens are reset so the next render
    /// re-prefills the survivors from position 0 — the only SSM/GDN-safe rewind
    /// (recurrent state has no per-position undo; see `KvCache::reset`).
    /// Returns the number of turn pairs dropped; a no-op (and zero cost) when
    /// the shift is off or the conversation already fits.
    fn evict_to_fit(&mut self) -> Result<usize, Box<dyn Error>> {
        // Context-shift eviction is disabled while an image is attached: dropping
        // turns would move the image's position (and could evict the image turn
        // itself), which the single-image first cut doesn't track. The ctx-full
        // guards in `generate` still apply.
        if !self.context_shift || self.image.is_some() {
            return Ok(0);
        }
        let budget = self
            .cache
            .config
            .max_seq_len
            .saturating_sub(self.max_tokens) as usize;
        let has_system = matches!(self.messages.first(), Some(m) if m.role == "system");
        let pinned = has_system as usize + 2 * self.keep_turns as usize;
        let mut dropped = 0usize;
        loop {
            if self.render_prompt()?.1.len() <= budget {
                break;
            }
            // Need a full (user, assistant) pair after the pinned prefix that
            // isn't the in-flight (last) user turn. If only the pinned prefix
            // plus the current turn remain, we can't shrink further — fall
            // through to the clean-stop guards in `generate`.
            if pinned + 2 >= self.messages.len() {
                break;
            }
            self.messages.drain(pinned..pinned + 2);
            dropped += 1;
        }
        if dropped > 0 {
            self.cache.reset();
            self.prior_tokens.clear();
        }
        Ok(dropped)
    }

    /// Render the current `messages` (which must end at a user turn) with a
    /// generation prompt, prefill the divergent suffix, decode the reply, and
    /// push it as an assistant turn. `on_text` fires once per sampled token
    /// with the newly-emitted byte slice — the REPL streams output through it.
    /// Callers own `messages` rollback: this never pops on error.
    ///
    /// Prefix reuse keeps the cached prefix that still matches `prior_tokens`.
    /// For **reasoning models** this prefix is shorter than it looks: the
    /// generation prompt's `<think>…</think>` block is in the cache, but the
    /// template drops it when re-rendering that turn as history, which shifts
    /// the previous answer and invalidates its K/V — so the answer (and the
    /// new turn) re-prefill every turn. That re-prefill is correctness-required
    /// (re-injecting the stripped reasoning would corrupt what the model sees),
    /// not a bug; it's logged at debug when it happens. Preserving thinking via
    /// `--chat-template-kwargs` trades context growth for full prefix reuse.
    fn generate(
        &mut self,
        mut on_text: impl FnMut(&str, Segment),
    ) -> Result<ReplyStats, Box<dyn Error>> {
        // `--context-shift`: drop the oldest turns to fit before rendering.
        // When it drops anything it resets the cache + prior_tokens, so the
        // render below re-prefills the survivors from scratch (SSM-safe).
        let shifted_turns = self.evict_to_fit()?;
        let (rendered, new_tokens, image_start) = self.render_prompt()?;

        // Prefix reuse: keep the cache prefix that still matches.
        let mut common0 = self
            .prior_tokens
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        // If an image is attached, the prefill that (re)feeds its vision block
        // must run through `forward_image_sampled`, which needs the WHOLE block
        // in the delta. The identical `<|image_pad|>` ids mean prefix reuse
        // normally stops before the block or sails past it; only an edited cache
        // could leave the boundary strictly inside the pads — rewind to the
        // block start there so the full image is re-prefilled, never half.
        match (&self.image, image_start) {
            (Some(img), Some(s)) if common0 > s && common0 < s + img.n_tok => common0 = s,
            _ => {}
        }
        if common0 > self.cache.position as usize {
            // Shouldn't happen — prior_tokens tracks what's in the cache.
            return Err(format!(
                "cache/prior_tokens drift: common={common0} cache.position={}",
                self.cache.position
            )
            .into());
        }
        // The divergent suffix to (re)feed. If the whole prompt is already
        // cached (a `/regen`, or an identically-rendered turn) `delta` would be
        // empty and we'd have no logits to sample from — rewind one position
        // and re-feed the last prompt token instead.
        let (common, delta): (usize, Vec<u32>) = if common0 < new_tokens.len() {
            (common0, new_tokens[common0..].to_vec())
        } else if let Some(&last) = new_tokens.last() {
            (new_tokens.len() - 1, vec![last])
        } else {
            return Err("empty prompt — nothing to generate".into());
        };
        // Surface the re-prefill that was otherwise silent: when we can reuse
        // less than the whole cached prefix, some computed K/V is discarded and
        // re-fed. Expected on reasoning models (the dropped <think> block, see
        // the doc above) and after edited/regenerated history; a perfectly
        // matching turn reuses everything and stays quiet.
        let cached = self.prior_tokens.len();
        if common < cached {
            tracing::debug!(
                reused = common,
                discarded = cached - common,
                reprefilled = delta.len(),
                "prefix reuse fell short of the cached prefix — re-prefilling \
                 (reasoning <think>-drop, /regen, or edited history)"
            );
        }
        self.cache.position = common as u32;

        if (common + delta.len()) as u32 > self.cache.config.max_seq_len {
            return Err(format!(
                "conversation is {} tokens but --ctx-size is {} — /clear, raise \
                 --ctx-size, or restart with --context-shift",
                common + delta.len(),
                self.cache.config.max_seq_len
            )
            .into());
        }

        // Does this prefill delta contain the image's vision block? Only when
        // the reuse boundary is at/before the first pad — otherwise the block is
        // already cached (its embeddings were spliced when first prefilled) and
        // the normal text path applies. When it is in the delta, the prefill
        // runs single-pass through `forward_image_sampled` (no chunking), so grow
        // the scratch to fit the whole delta first.
        let image_in_delta = matches!((&self.image, image_start), (Some(_), Some(s)) if common <= s);
        // Local pad offset only when the block is in the delta (else `s < common`
        // for a cached image would underflow this usize).
        let img_start_in_delta = image_start.filter(|_| image_in_delta).map(|s| s - common);
        let image_dims = self.image.as_ref().map(|i| (i.nx, i.ny));
        if image_in_delta {
            let need = self.model.scratch_bytes_estimate(
                /*n_ubatch=*/ 0,
                (common + delta.len()) as u32,
                self.cache.config.k_dtype,
                self.cache.config.v_dtype,
            );
            if need > self.scratch_bytes {
                self.engine.allocate_scratch(need)?;
                self.scratch_bytes = need;
            }
        }
        // Borrow the embeddings for the (single) image prefill forward below.
        // A slice ref (not a clone) — the [proj_dim,n_tok] buffer can be MBs.
        let image_embeds: Option<&[f32]> = if image_in_delta {
            self.image.as_ref().map(|i| i.embeddings.as_slice())
        } else {
            None
        };

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
        // Clear any stale interrupt so only a Ctrl+C during *this* reply counts.
        let mut cancelled = false;
        let mut ctx_full = false;
        GENERATION_CANCELLED.store(false, Ordering::SeqCst);
        loop {
            // Ctrl+C during generation (interactive mode): stop here and keep
            // whatever was produced so far as the reply. forward_sampled waits
            // on its fence each call, so breaking between tokens leaves the
            // cache/position consistent — the partial turn is stored normally.
            if GENERATION_CANCELLED.load(Ordering::SeqCst) {
                cancelled = true;
                break;
            }
            if (self.cache.position as usize + step_tokens.len() + 1) as u32
                > self.cache.config.max_seq_len
            {
                // Out of context room. forwards>0 → mid-reply truncation (a
                // clean stop, partial reply kept); forwards==0 is the
                // can't-even-start case handled after the loop.
                ctx_full = forwards > 0;
                break;
            }
            let cache = &mut self.cache;
            let model = &self.model;
            let t0 = std::time::Instant::now();
            let position = cache.position;
            // Forward 0 of an image turn splices the vision embeddings + uses the
            // qwen-vl M-RoPE positions (single-pass). Every other forward — the
            // rest of the prefill delta and all decode steps — is the normal
            // text path; the cache's `rope_position_lag` (set during the image
            // prefill) keeps their positions continuous past the image.
            let token = if forwards == 0 && image_in_delta {
                let (nx, ny) = image_dims.expect("image_in_delta ⇒ dims");
                self.engine.forward_image_sampled(
                    &**model,
                    cache,
                    &step_tokens,
                    image_embeds.expect("image_in_delta ⇒ embeddings"),
                    img_start_in_delta.expect("image_in_delta ⇒ start"),
                    nx,
                    ny,
                    &mut self.sampler,
                )?
            } else {
                self.engine.forward_sampled(
                    &**model,
                    cache,
                    &step_tokens,
                    position,
                    &mut self.sampler,
                )?
            };
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

        // No forward ran: the prompt filled the context window exactly
        // (common + delta == --ctx-size), so the loop-top budget guard broke
        // before the prefill — there's no slot to sample even the first token.
        // Roll back the just-pushed user turn and report it. Otherwise we'd
        // store an empty reply and set `prior_tokens` longer than
        // `cache.position`, tripping the drift guard on every later turn and
        // wedging the session until `/clear`.
        if forwards == 0 {
            // No forward ran: either the prompt exactly filled the context
            // window (the loop-top budget guard broke before the prefill) or
            // the user interrupted in the instant before it started. Return
            // Err so the caller rolls back; no partial turn is stored.
            if cancelled {
                return Err("interrupted before any output".into());
            }
            return Err(format!(
                "conversation is {} tokens but --ctx-size is {} (no room to \
                 generate) — /clear, raise --ctx-size, or restart with \
                 --context-shift",
                common + prompt_tokens,
                self.cache.config.max_seq_len
            )
            .into());
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
            // Think block was opened (the prompt seeded it on, or the model
            // emitted `<think>`) but never closed — generation stopped
            // mid-thought (EOS or max_tokens before `</think>`). The whole
            // reply is reasoning, not an answer; store it as `reasoning_content`
            // so the next turn's template splits it correctly instead of
            // re-rendering raw chain-of-thought as the assistant's final answer
            // (and re-prefixing it into the cache).
            None if in_think => {
                let reasoning = decode(&assistant_tokens)?;
                let reasoning = reasoning.trim();
                let r = (!reasoning.is_empty()).then(|| reasoning.to_string());
                (String::new(), r)
            }
            // No think block at all (non-reasoning model or thinking disabled):
            // the whole reply is the answer.
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
            interrupted: cancelled,
            ctx_full,
            shifted_turns,
        })
    }

    /// Build the vision tower from the mmproj sidecar on first `/image` and keep
    /// it for the session. Errors if the model has no mmproj (or `--no-mmproj`).
    fn ensure_vision(&mut self) -> Result<(), Box<dyn Error>> {
        if self.vision_ctx.is_some() {
            return Ok(());
        }
        let path = self.mmproj_path.clone().ok_or(
            "no vision model available — this model has no mmproj sidecar \
             (or it was disabled with --no-mmproj)",
        )?;
        tracing::info!(path = ?path, "loading vision tower for /image");
        let gguf = GgufFile::open(&path)?;
        let weights = self.engine.upload_weights(&gguf)?;
        let cfg = crate::vision::parse_config(&gguf)?;
        // The encoder copies its tensor views out of `weights` (no borrow held),
        // so moving `weights` into the VisionModel below keeps them valid.
        let encoder = VisionEncoder::new(
            &weights,
            cfg.n_embd as usize,
            cfg.patch_size as usize,
            cfg.n_head as usize,
            cfg.n_ff as usize,
            cfg.n_layer as usize,
            cfg.eps,
        )?;
        let host_weights = HostWeights::from_gguf(&gguf)?;
        let vision = crate::vision::VisionModel { config: cfg, weights };
        self.vision_ctx = Some(VisionCtx { vision, encoder, host_weights });
        Ok(())
    }

    /// Encode `path` through the vision tower (GPU) and stage it as the pending
    /// image for the next user turn. Returns the merged grid `(nx, ny, n_tok)`
    /// for the confirmation line. Errors if an image is already attached.
    fn attach_image(&mut self, path: &Path) -> Result<(usize, usize, usize), Box<dyn Error>> {
        if self.image.is_some() {
            return Err("this conversation already has an image — /clear to start over \
                        (one image per conversation for now)"
                .into());
        }
        self.ensure_vision()?;
        let cfg = self.vision_ctx.as_ref().expect("ensure_vision built it").vision.config.clone();
        let pcfg = crate::vision::preprocess::PreprocessConfig::qwen3vl_default(
            cfg.patch_size,
            cfg.spatial_merge_size,
            cfg.image_mean,
            cfg.image_std,
        );
        let pimg = crate::vision::preprocess::preprocess(path, &pcfg)?;
        let merge = cfg.spatial_merge_size as usize;
        let (nx, ny) = (pimg.grid_w as usize / merge, pimg.grid_h as usize / merge);
        let n_tok = pimg.n_tokens as usize;

        // Grow the scratch for the vision tower's working set, then encode.
        let need = vision_scratch_estimate(&pimg);
        if need > self.scratch_bytes {
            self.engine.allocate_scratch(need)?;
            self.scratch_bytes = need;
        }
        let vc = self.vision_ctx.as_ref().expect("ensure_vision built it");
        let (encoder, host_weights, weights) = (&vc.encoder, &vc.host_weights, &vc.vision.weights);
        let embeddings = crate::vision::encoder::encode_image_chunked(
            &mut self.engine,
            weights,
            encoder,
            &pimg,
            host_weights,
        )?;
        self.pending_image = Some(EncodedImage { embeddings, nx, ny, n_tok });
        Ok((nx, ny, n_tok))
    }

    /// Set (or replace) the leading system message, then reset the cache and
    /// prior-token tracking so the next turn re-prefills the whole conversation
    /// from the new prompt. The full reset is required because the prefix
    /// changes at position 0 — and the SSM/GDN recurrent state has no partial
    /// rewind, so it must rebuild from scratch (see `KvCache::reset`).
    fn set_system_prompt(&mut self, text: impl Into<String>) {
        let msg = ChatMessage::system(text);
        match self.messages.first() {
            Some(m) if m.role == "system" => self.messages[0] = msg,
            _ => self.messages.insert(0, msg),
        }
        self.prior_tokens.clear();
        self.cache.reset();
    }

    /// The current system prompt, if one is set (the leading system message).
    fn system_prompt(&self) -> Option<&str> {
        match self.messages.first() {
            Some(m) if m.role == "system" => Some(&m.content),
            _ => None,
        }
    }

    /// Reset everything that ties to the current conversation; keep the
    /// model / engine / sampler RNG so deterministic seeds still reproduce
    /// after a `/clear`. A system prompt is preserved across the clear (like
    /// llama.cpp) — only the user/assistant turns are dropped.
    fn clear(&mut self) {
        let system = self.system_prompt().map(|s| ChatMessage::system(s));
        self.messages.clear();
        if let Some(s) = system {
            self.messages.push(s);
        }
        self.prior_tokens.clear();
        self.cache.reset();
        self.sampler.reset_recent();
        // Drop any attached/pending image (the built vision tower is kept — it's
        // reusable for the next `/image`).
        self.image = None;
        self.pending_image = None;
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
    println!("available commands:");
    println!("  /exit or Ctrl+C     stop or exit");
    println!("  /regen              regenerate the last response");
    println!("  /system [text]      show or set the system prompt");
    println!("  /clear              clear the chat history");
    println!("  /read <file>        add a text file");
    println!("  /glob <pattern>     add text files using globbing pattern");
    println!("  /image <file>       attach an image to your next message (VL models)");
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
    multiline: bool,
) -> Result<(), Box<dyn Error>> {
    let mut editor = Editor::new()?;
    editor.set_helper(Some(ChatHelper { multiline }));
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
                let joined = join_continuations(&raw, multiline);
                let line = joined.trim();
                if line.is_empty() {
                    continue;
                }
                if let Some(cmd) = line.strip_prefix('/') {
                    if cmd == "exit" {
                        break;
                    } else if cmd == "regen" {
                        emit_regen(session);
                    } else if let Some(arg) = cmd.strip_prefix("system") {
                        let text = arg.trim();
                        if text.is_empty() {
                            match session.system_prompt() {
                                Some(s) => println!("system prompt:\n{s}"),
                                None => println!("(no system prompt set)"),
                            }
                        } else {
                            session.set_system_prompt(text);
                            println!("(system prompt set)");
                        }
                    } else if cmd == "clear" {
                        session.clear();
                        println!("(conversation cleared)");
                    } else if let Some(arg) = cmd.strip_prefix("read") {
                        handle_read(session, arg.trim());
                    } else if let Some(arg) = cmd.strip_prefix("glob") {
                        handle_glob(session, arg.trim());
                    } else if let Some(arg) = cmd.strip_prefix("image") {
                        handle_image(session, arg.trim());
                    } else {
                        println!("unknown command: /{cmd}");
                    }
                    continue;
                }
                emit_reply(session, line);
            }
            // Ctrl+D, or Ctrl+C at the prompt, exits ("stop or exit"). A Ctrl+C
            // *during* a reply is caught by the SIGINT watcher — it stops the
            // reply and returns here — so it never surfaces as Interrupted.
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
    run_turn(session, |s, cb| s.handle_user_message(line, cb));
}

fn emit_regen(session: &mut ChatSession) {
    run_turn(session, |s, cb| s.regenerate(cb));
}

/// Shared streaming + stats/error reporting for a generated turn. `produce`
/// runs the generation, streaming each token's text through the supplied
/// callback (reasoning dimmed, final answer in the default color).
fn run_turn(
    session: &mut ChatSession,
    produce: impl FnOnce(
        &mut ChatSession,
        &mut dyn FnMut(&str, Segment),
    ) -> Result<ReplyStats, Box<dyn Error>>,
) {
    println!(); // blank line between the input and the reply
    let _ = std::io::stdout().flush();
    let color = color_enabled();
    let mut on_text = |delta: &str, seg: Segment| {
        if color && seg == Segment::Thinking {
            print!("{C_THINK}{delta}{C_RESET}");
        } else {
            print!("{delta}");
        }
        let _ = std::io::stdout().flush();
    };
    let result = produce(session, &mut on_text);
    match result {
        Ok(stats) => {
            println!(); // terminate the streamed reply line
            if stats.shifted_turns > 0 {
                println!("[context shift: dropped {} oldest turn(s)]", stats.shifted_turns);
            }
            if stats.interrupted {
                println!("[interrupted]");
            }
            if stats.ctx_full {
                println!("[context full — reply truncated at --ctx-size; /clear, raise --ctx-size, or use --context-shift]");
            }
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

/// Expand a leading `~` / `~/` to `$HOME`. The chat REPL parses its own command
/// arguments, so — unlike a shell — it must do tilde expansion itself; otherwise
/// `/image ~/pic.png` tries to open a file literally named `~/pic.png`. Only the
/// bare `~` and `~/…` forms are handled (the `~user` form needs a passwd lookup
/// and is rare); anything else, or an unset `$HOME`, is returned unchanged.
fn expand_tilde(path: &str) -> String {
    match std::env::var("HOME") {
        Ok(home) if path == "~" => home,
        Ok(home) if path.starts_with("~/") => {
            format!("{}/{}", home.trim_end_matches('/'), &path[2..])
        }
        _ => path.to_string(),
    }
}

fn handle_read(session: &mut ChatSession, path: &str) {
    if path.is_empty() {
        println!("usage: /read <path>");
        return;
    }
    let path = &expand_tilde(path);
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

/// `/image <file>`: encode an image through the vision tower and stage it for
/// the next user message (the message then carries the `<__media__>` marker, so
/// the vision block is spliced into that turn's prefill). Vision models only.
fn handle_image(session: &mut ChatSession, arg: &str) {
    // Tolerate surrounding quotes (paths with spaces pasted with quotes), then
    // expand a leading `~` (the REPL isn't a shell, so it must do this itself).
    let path = expand_tilde(arg.trim().trim_matches('"').trim_matches('\''));
    if path.is_empty() {
        println!("usage: /image <path-to-image>");
        return;
    }
    match session.attach_image(Path::new(&path)) {
        Ok((nx, ny, n_tok)) => println!(
            "(image attached: {nx}×{ny} merged grid, {n_tok} tokens — sent with your next message)"
        ),
        Err(e) => println!("/image failed: {e}"),
    }
}

/// `/glob <pattern>`: read every text file matching the shell glob, concatenate
/// them (each under a `===== path =====` header), and submit the result as one
/// user message — handy for dropping a directory of sources into the chat.
/// Non-UTF-8 / unreadable matches are skipped with a note.
fn handle_glob(session: &mut ChatSession, pattern: &str) {
    if pattern.is_empty() {
        println!("usage: /glob <pattern>");
        return;
    }
    let pattern = &expand_tilde(pattern);
    let paths = match glob::glob(pattern) {
        Ok(paths) => paths,
        Err(e) => {
            println!("/glob: invalid pattern: {e}");
            return;
        }
    };
    let mut combined = String::new();
    let mut n = 0usize;
    for entry in paths {
        let path = match entry {
            Ok(p) if p.is_file() => p,
            Ok(_) => continue, // directory match — skip
            Err(e) => {
                eprintln!("[glob skip: {e}]");
                continue;
            }
        };
        match fs::read_to_string(&path) {
            Ok(content) if !content.trim().is_empty() => {
                combined.push_str(&format!("===== {} =====\n", path.display()));
                combined.push_str(content.trim_end());
                combined.push_str("\n\n");
                n += 1;
            }
            Ok(_) => {} // empty file — skip silently
            Err(e) => eprintln!("[skip {}: {e}]", path.display()),
        }
    }
    if n == 0 {
        println!("(no readable text files matched: {pattern})");
        return;
    }
    eprintln!("[{n} file(s), {} bytes from {pattern}]", combined.len());
    emit_reply(session, combined.trim());
}

#[cfg(test)]
mod tests {
    use super::{expand_tilde, join_continuations, parse_logit_bias, prompt_opens_think};

    #[test]
    fn expand_tilde_forms() {
        // SAFETY: single-threaded within this test; no other test reads $HOME.
        unsafe { std::env::set_var("HOME", "/home/bob") };
        assert_eq!(expand_tilde("~/map.jpg"), "/home/bob/map.jpg");
        assert_eq!(expand_tilde("~"), "/home/bob");
        assert_eq!(expand_tilde("~/a/b.png"), "/home/bob/a/b.png");
        // Non-leading or unrelated `~` is left alone, as are absolute/relative paths.
        assert_eq!(expand_tilde("/abs/path"), "/abs/path");
        assert_eq!(expand_tilde("rel/path"), "rel/path");
        assert_eq!(expand_tilde("dir/~/x"), "dir/~/x");
        assert_eq!(expand_tilde("~user/x"), "~user/x");
        // Trailing slash on $HOME doesn't double up.
        unsafe { std::env::set_var("HOME", "/home/bob/") };
        assert_eq!(expand_tilde("~/map.jpg"), "/home/bob/map.jpg");
    }

    #[test]
    fn join_continuations_default_mode() {
        // A backslash right before a newline is a continue marker → drop it,
        // keep the newline; other backslashes are literal.
        assert_eq!(join_continuations("foo\\\nbar", false), "foo\nbar");
        assert_eq!(join_continuations("a\\\nb\\\nc", false), "a\nb\nc");
        assert_eq!(join_continuations("one line", false), "one line");
        assert_eq!(join_continuations("C:\\path", false), "C:\\path");
        assert_eq!(join_continuations("mid\\slash", false), "mid\\slash");
    }

    #[test]
    fn join_continuations_multiline_mode() {
        // Enter inserted the newlines; only the trailing submit-backslash drops.
        assert_eq!(join_continuations("foo\nbar\nbaz\\", true), "foo\nbar\nbaz");
        assert_eq!(join_continuations("just this\\", true), "just this");
        assert_eq!(join_continuations("no marker", true), "no marker");
    }

    #[test]
    fn parse_logit_bias_forms() {
        assert_eq!(parse_logit_bias("15043+2.0"), Ok((15043, 2.0)));
        assert_eq!(parse_logit_bias("7-1.5"), Ok((7, -1.5)));
        assert_eq!(parse_logit_bias("5=1.5"), Ok((5, 1.5)));
        assert_eq!(parse_logit_bias("128009-inf"), Ok((128009, f32::NEG_INFINITY)));
        assert_eq!(parse_logit_bias("9=inf"), Ok((9, f32::INFINITY)));
        // Missing id / non-numeric id / no bias → errors, not panics.
        assert!(parse_logit_bias("+1").is_err());
        assert!(parse_logit_bias("abc").is_err());
        assert!(parse_logit_bias("12").is_err());
    }

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
