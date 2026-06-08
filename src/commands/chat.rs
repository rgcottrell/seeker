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
use rustyline::completion::{Completer, FilenameCompleter, Pair, unescape};
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{CompletionType, Config, Context, Editor, Helper, Hinter};

use crate::chat_template::{self, ChatMessage};
use crate::commands::chat_cache;
use crate::commands::download;
use crate::commands::download::{HfResolveArgs, resolve_hf};
use crate::gguf::{GgmlType, GgufFile, MetadataValue};
use crate::inference::Engine;
use crate::inference::budget;
use crate::inference::kv_cache::{
    KvCacheConfig, estimate_kv_bytes, estimate_kv_bytes_uniform, estimate_ssm_bytes, parse_dtype,
};
use crate::inference::sample::{GgufSamplingDefaults, Sampler, SamplerConfig};
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
    /// The transformer-tower encoder — `None` for the gemma4uv "no-tower"
    /// projector (which has no vision blocks; [`attach_image`] runs the light
    /// [`encode_image_gemma4`](crate::vision::encoder::encode_image_gemma4)
    /// pipeline instead).
    encoder: Option<VisionEncoder>,
    /// CPU-side weights for the tower encoder's pos-embd resize — `None` for
    /// gemma4uv (its encoder needs no host-side copy).
    host_weights: Option<HostWeights>,
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

/// Audio encoded through the gemma4ua projector: `[proj_dim, n_tok]` host f32
/// (column = 40 ms frame). Spliced into the decoder residual at the `<|audio|>`
/// rows during the audio turn's prefill, exactly like [`EncodedImage`].
#[derive(Clone)]
struct EncodedAudio {
    embeddings: Vec<f32>,
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
    use tokio::signal::unix::{SignalKind, signal};
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
    #[arg(
        long = "hf-repo",
        required_unless_present = "model",
        conflicts_with = "model"
    )]
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

    /// Speculative-decode draft model and max draft tokens per step
    /// (`--spec-draft-model` / `--spec-draft-hf` / `--spec-draft-n-max`). A draft
    /// with `n_max > 0` enables MTP speculative decoding for replies.
    #[command(flatten)]
    spec: crate::commands::download::SpecDraftArgs,

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

    /// Max tokens per assistant reply. llama.cpp's `n_predict` defaults to -1
    /// (unbounded — stops only at an EOG token or when the context fills). We
    /// keep a finite cap (it also reserves reply headroom for `--context-shift`
    /// budgeting) but set it generously so coherent replies aren't cut off.
    #[arg(long, default_value_t = 2048)]
    max_tokens: u32,

    /// KV-cache budget for the whole conversation, in tokens. `0` (the default)
    /// uses the model's full trained context length — llama.cpp's `-c 0`
    /// convention; pass a smaller value to cap KV memory.
    #[arg(long = "ctx-size", default_value_t = 0)]
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

    /// KV cache K dtype. One of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1
    /// turbo2 turbo3 turbo4. K and V may differ (asymmetric cache). The turbo*
    /// (TurboQuant) quants require head_dim % 128 == 0.
    #[arg(long = "cache-type-k", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_k: GgmlType,

    /// KV cache V dtype. Same legal values as --cache-type-k.
    #[arg(long = "cache-type-v", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_v: GgmlType,

    // ─── Sampling ───────────────────────────────────────────────────────
    // Unset → the GGUF's `general.sampling.*` default if present, else the
    // built-in fallback (temp 0.8, top-k 40, top-p 0.95, min-p 0.05,
    // repeat-penalty 1.0, penalty-last-n 64). An explicit flag always wins.
    /// Sampling temperature. 0 → greedy argmax. (default: GGUF, else 0.8)
    #[arg(long = "temp", alias = "temperature")]
    temperature: Option<f32>,

    /// Top-K filter (0 = disabled, full vocab). (default: GGUF, else 40)
    #[arg(long = "top-k")]
    top_k: Option<u32>,

    /// Top-P (nucleus) filter (1.0 = disabled). (default: GGUF, else 0.95)
    #[arg(long = "top-p")]
    top_p: Option<f32>,

    /// Min-P filter (0.0 = disabled). (default: GGUF, else 0.05)
    #[arg(long = "min-p")]
    min_p: Option<f32>,

    /// Presence penalty (subtract from any repeated-token logit; 0.0 = off).
    #[arg(long = "presence-penalty", default_value_t = 0.0)]
    presence_penalty: f32,

    /// Frequency penalty (subtract count×p from repeated-token logits; 0.0 = off).
    #[arg(long = "frequency-penalty", default_value_t = 0.0)]
    frequency_penalty: f32,

    /// Repetition penalty (multiply/divide repeated logits; 1.0 = off).
    /// (default: GGUF, else 1.0)
    #[arg(long = "repeat-penalty", alias = "repetition-penalty")]
    repeat_penalty: Option<f32>,

    /// How many trailing tokens contribute to penalties. `-1` = the whole
    /// context (`--ctx-size`); `0` = disabled. (default: GGUF, else 64;
    /// llama.cpp's `--repeat-last-n`.)
    #[arg(long = "penalty-last-n", allow_hyphen_values = true)]
    penalty_last_n: Option<i32>,

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
    /// Resolve the sampler knobs with precedence **CLI flag → GGUF
    /// `general.sampling.*` → built-in fallback** (matching llama.cpp's
    /// `common/common.cpp`). `gg` carries the GGUF-embedded defaults.
    fn sampler_config(&self, gg: &GgufSamplingDefaults, ctx_size: u32) -> SamplerConfig {
        use crate::inference::sample as s;
        SamplerConfig::from_cli(
            self.temperature
                .or(gg.temperature)
                .unwrap_or(s::DEFAULT_TEMPERATURE),
            self.top_k.or(gg.top_k).unwrap_or(s::DEFAULT_TOP_K),
            self.top_p.or(gg.top_p).unwrap_or(s::DEFAULT_TOP_P),
            self.min_p.or(gg.min_p).unwrap_or(s::DEFAULT_MIN_P),
            self.presence_penalty,
            self.frequency_penalty,
            self.repeat_penalty
                .or(gg.repeat_penalty)
                .unwrap_or(s::DEFAULT_REPEAT_PENALTY),
            self.penalty_last_n
                .or(gg.penalty_last_n)
                .unwrap_or(s::DEFAULT_PENALTY_LAST_N),
            ctx_size,
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
    let chat_template = bundle
        .chat_template
        .clone()
        .ok_or_else(|| -> Box<dyn Error> {
            "model has no `tokenizer.chat_template` — use `seeker run` for base completions".into()
        })?;

    let mut engine = Engine::new(args.ubatch_size, args.batch_size)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened");
    let weights = engine.upload_weights(&gguf)?;
    let mut model = crate::models::open(&gguf, weights, bundle, args.spec.spec_draft_n_max > 0)?;

    // Optional MTP/EAGLE draft model for speculative decoding — a separate
    // gemma4 `gemma4-assistant` GGUF (local `--spec-draft-model` or HF
    // `--spec-draft-hf`, downloaded if absent). qwen35moe self-spec needs no
    // draft (its NextN head loads from the base GGUF via `spec_enabled` above).
    let draft_path = download::resolve_spec_draft(
        args.spec.spec_draft_model.clone(),
        args.spec.spec_draft_hf.clone(),
        args.hf_token.clone(),
        args.offline,
    )
    .await?;
    if let Some(draft_path) = &draft_path {
        let draft_gguf = GgufFile::open(draft_path)?;
        let draft_weights = engine.upload_weights(&draft_gguf)?;
        model.attach_mtp_draft(&draft_gguf, draft_weights)?;
        tracing::info!(path = ?draft_path, "attached MTP draft model");
    }
    // Speculative decoding is active when a draft head is available (separate
    // draft attached, or qwen NextN loaded) and `--spec-draft-n-max > 0`.
    let spec_n_max = if model.supports_mtp_spec() {
        args.spec.spec_draft_n_max
    } else {
        0
    };

    // Speculative decode's verify writes up to `n_max + 1` lookahead K/V per
    // step before truncating, so reserve that headroom on top of the context
    // window or the final verify writes past the cache and hangs the GPU.
    let spec_headroom = if spec_n_max > 0 { spec_n_max + 1 } else { 0 };

    // `--ctx-size 0` (the default) means "use the model's full trained context"
    // (llama.cpp's `-c 0`); fall back to 4096 if the model omits the metadata.
    let mut ctx_size = if args.ctx_size == 0 {
        gguf.trained_ctx_len().unwrap_or(4096)
    } else {
        args.ctx_size
    };
    let requested_ctx = ctx_size;

    // Auto-fit the context to GPU memory when it was left unset (`--ctx-size 0`
    // → trained max): pick the largest ctx whose weights + KV + scratch fit live
    // free memory, so a dense model with a 256K default starts instead of
    // wedging the device. An explicit `--ctx-size` is honored verbatim (the
    // holistic preflight in `allocate_kv_cache*` fail-fasts if it doesn't fit).
    if args.ctx_size == 0 && budget::fit_enabled() {
        let dims = model.cache_dims();
        let per_layer = model.cache_per_layer_dims();
        let align = engine
            .device
            .limits
            .min_storage_buffer_offset_alignment
            .max(1);
        let weights_bytes = model.weights().total_bytes;
        let ssm_bytes = model
            .ssm_state_dims()
            .map(|d| estimate_ssm_bytes(&d, align))
            .unwrap_or(0);
        // Weights are already resident; SSM state is ctx-independent — net both
        // out of the budget up front so the search varies only KV + scratch.
        let hb = budget::kv_heap_budget(&engine.device, 0.9);
        let usable = hb.usable_for_new(weights_bytes).saturating_sub(ssm_bytes);
        let cost_at = |ctx: u32| -> u64 {
            let cfg = KvCacheConfig {
                k_dtype: args.cache_type_k,
                v_dtype: args.cache_type_v,
                max_seq_len: ctx + spec_headroom,
                n_head: dims.n_head,
            };
            let kv = match &per_layer {
                Some((hd, nkv)) => estimate_kv_bytes(hd, nkv, &cfg, align),
                None => estimate_kv_bytes_uniform(
                    dims.n_layer,
                    dims.head_dim,
                    dims.n_head_kv,
                    &cfg,
                    align,
                ),
            };
            let scratch = model.scratch_bytes_estimate(
                args.ubatch_size,
                ctx,
                args.cache_type_k,
                args.cache_type_v,
            );
            kv + scratch
        };
        match budget::fit_ctx(
            ctx_size,
            budget::fit_min_ctx().min(ctx_size),
            usable,
            cost_at,
        ) {
            Ok(c) => {
                if c < ctx_size {
                    tracing::warn!(
                        requested = ctx_size,
                        chosen = c,
                        "ctx auto-reduced to fit GPU memory (--fit); pass --ctx-size to override \
                         or SEEKER_FIT=0 to disable"
                    );
                }
                ctx_size = c;
            }
            Err(e) => {
                const GIB: f64 = (1u64 << 30) as f64;
                return Err(format!(
                    "model weights ({:.1} GiB) + min KV/scratch at ctx {} don't fit GPU memory: \
                     need {:.1} GiB but only {:.1} GiB usable — use a smaller --cache-type-k/v or \
                     free memory",
                    weights_bytes as f64 / GIB,
                    e.floor,
                    e.need as f64 / GIB,
                    e.usable as f64 / GIB,
                )
                .into());
            }
        }
    }
    tracing::info!(ctx_size, "context window");

    // The mmproj vision sidecar (if resolved and not `--no-mmproj`). The vision
    // tower is built lazily on the first `/image` (see `ChatSession::attach_image`)
    // so a text-only session never uploads the projector.
    let mmproj_path = if args.no_mmproj {
        None
    } else {
        resolved.mmproj.clone()
    };

    // Size the scratch (compute buffer) for this model + n_ubatch (and the
    // full ctx for heterogeneous caches), replacing the Engine::new
    // placeholder. An `/image` turn grows this on demand (image prefill is
    // single-pass).
    let scratch_bytes = model.scratch_bytes_estimate(
        args.ubatch_size,
        ctx_size,
        args.cache_type_k,
        args.cache_type_v,
    );
    engine.allocate_scratch(scratch_bytes)?;

    let dims = model.cache_dims();
    let cache_config = KvCacheConfig {
        k_dtype: args.cache_type_k,
        v_dtype: args.cache_type_v,
        max_seq_len: ctx_size + spec_headroom,
        n_head: dims.n_head,
    };
    let mut cache = match model.cache_per_layer_dims() {
        Some((hd, nkv)) => engine.allocate_kv_cache_per_layer(&hd, &nkv, cache_config)?,
        None => {
            engine.allocate_kv_cache(dims.n_layer, dims.head_dim, dims.n_head_kv, cache_config)?
        }
    };
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
        // Per-position SSM checkpoint buffers for speculative decode (qwen35moe
        // NextN). Without these, `decode_speculative`'s verify advances the GDN
        // recurrent state through all N+1 draft positions but never rolls it
        // back to the accepted length (finalize is gated on these snapshots), so
        // a hybrid model's output drifts — worse with larger --spec-draft-n-max.
        if spec_n_max > 0 {
            let max_snapshots = spec_n_max.clamp(1, 8) + 1;
            cache.allocate_ssm_snapshots(&engine.device, &ssm, max_snapshots)?;
        }
    }

    // Startup memory breakdown (the `llama_memory_breakdown_print` analog):
    // weights / KV / scratch / SSM vs the heap, plus the chosen vs requested
    // context. Uses the *actual* allocated KV bytes (`cache.kv_bytes()`).
    {
        let align = engine
            .device
            .limits
            .min_storage_buffer_offset_alignment
            .max(1);
        let ssm = model
            .ssm_state_dims()
            .map(|d| estimate_ssm_bytes(&d, align))
            .unwrap_or(0);
        let proj = budget::MemoryProjection {
            weights: model.weights().total_bytes,
            scratch: scratch_bytes,
            kv: cache.kv_bytes(),
            ssm,
            prefix_pool: 0,
        };
        budget::log_breakdown(
            &proj,
            &budget::kv_heap_budget(&engine.device, 0.9),
            requested_ctx,
            ctx_size,
            1,
        );
    }

    // GGUF-embedded sampling defaults (`general.sampling.*`) seed the sampler,
    // overridden by any explicit CLI flag — matching llama.cpp.
    let gg_sampling = GgufSamplingDefaults::from_gguf(&gguf);
    if !gg_sampling.is_empty() {
        tracing::info!(?gg_sampling, "using GGUF general.sampling.* defaults");
    }
    let sampler_config = args.sampler_config(&gg_sampling, ctx_size);
    tracing::info!(
        temperature = sampler_config.temperature,
        top_k = sampler_config.top_k,
        top_p = sampler_config.top_p,
        min_p = sampler_config.min_p,
        repeat_penalty = sampler_config.repeat_penalty,
        "resolved sampler",
    );
    let sampler = Sampler::new(sampler_config);

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
        spec_n_max,
        context_shift: args.context_shift,
        keep_turns: args.keep,
        template_kwargs: args.chat_template_kwargs.clone().unwrap_or_default(),
        mmproj_path,
        vision_ctx: None,
        pending_image: None,
        image: None,
        audio_cfg: None,
        pending_audio: None,
        audio: None,
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
        args.history_file.clone().or_else(default_history_path)
    };

    let result = if std::io::stdin().is_terminal() {
        // Ctrl+C during a reply should stop that reply, not the program.
        spawn_interrupt_watcher();
        run_interactive(
            &mut session,
            &gguf,
            &path,
            history.as_deref(),
            args.multiline_input,
        )
    } else {
        run_piped(&mut session)
    };

    // Persist the session for next time (best-effort; never fail the run).
    if let Some(p) = &args.prompt_cache
        && !args.prompt_cache_ro
    {
        match chat_cache::save(
            p,
            &arch,
            &session.cache,
            &session.prior_tokens,
            &session.messages,
        ) {
            Ok(()) => tracing::info!(path = %p.display(), "prompt cache saved"),
            Err(e) => tracing::warn!("prompt-cache save failed: {e}"),
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
            Ok(download::Resolved {
                main: model,
                mmproj,
            })
        }
        _ => unreachable!("clap group invariant"),
    }
}

/// Conservative scratch estimate for the vision tower's single forward (mirrors
/// `commands::run` / `server`). [`VisionEncoder::encode_image`] reclaims each
/// stage's scratch between layers (checkpoint/restore), so the working set is
/// the persistent residual carriers + RoPE positions plus the single largest
/// stage — O(n_pos), NOT O(n_layer · n_pos). The per-token high-water across
/// the stages is ~28k floats; budget 40k for margin (copy ops, alignment, and
/// the long-KV flash-attn split-K partials ~3k floats/token). Floored at 64 MiB.
fn vision_scratch_estimate(pimg: &crate::vision::preprocess::PreprocessedImage) -> u64 {
    let n_pos = (pimg.grid_w as u64) * (pimg.grid_h as u64);
    (40_000u64 * n_pos * 4).max(64 << 20)
}

/// Generous scratch estimate for the gemma4ua audio encoder's working set
/// (input + normed `[frame, n_tok]` + projection `[proj_dim, n_tok]` + matmul
/// temps), sized per audio token like [`vision_scratch_estimate`].
fn audio_scratch_estimate(n_tok: usize) -> u64 {
    (40_000u64 * n_tok as u64 * 4).max(64 << 20)
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
    /// Max MTP draft tokens per speculative step (0 ⇒ spec-decode off). Active
    /// only when the model has a draft head (`model.supports_mtp_spec()`); the
    /// KV cache reserves `spec_n_max + 1` lookahead headroom when nonzero.
    spec_n_max: u32,
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
    /// The gemma4ua audio config, parsed from the mmproj alongside the vision
    /// config when the projector is loaded (`None` until then, or if the mmproj
    /// has no audio encoder). The audio path reuses the vision tower's uploaded
    /// mmproj [`WeightsHandle`](crate::inference::weights::WeightsHandle).
    audio_cfg: Option<crate::audio::AudioConfig>,
    /// Audio encoded by `/audio` but not yet attached — moved to `audio` (and the
    /// marker prepended) when the next user turn is sent. Mirrors `pending_image`.
    pending_audio: Option<EncodedAudio>,
    /// The single audio clip committed to this conversation. Mutually exclusive
    /// with `image` in the first cut (one media item per conversation); `/clear`
    /// drops it.
    audio: Option<EncodedAudio>,
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

/// rustyline line-editor helper: colors the prompt/input, decides when a line
/// is complete (multi-line input), and Tab-completes filesystem paths for the
/// path-argument commands. Hints are a no-op.
#[derive(Helper, Hinter)]
struct ChatHelper {
    /// `--multiline-input`: when true Enter inserts a newline and a trailing
    /// `\` submits; when false (default) a trailing `\` continues to the next
    /// line and a bare Enter submits.
    multiline: bool,
    /// Filesystem completer used only inside a [`PATH_COMMANDS`] argument. It
    /// expands `~/` to read the home dir but leaves the `~` literal in the
    /// inserted text (and backslash-escapes spaces / shell break chars) —
    /// submit-time [`normalize_path_arg`] undoes that and expands the `~`.
    completer: FilenameCompleter,
}

/// Commands whose sole argument is a filesystem path. Typing one of these
/// followed by whitespace puts the line into filename-completion mode (Tab
/// expands a partial path, like llama-cli); plain chat and other commands get
/// no completion, so Tab stays inert mid-conversation.
const PATH_COMMANDS: [&str; 4] = ["/read", "/glob", "/image", "/audio"];

/// True when `pos` sits in the path-argument region of a [`PATH_COMMANDS`]
/// line — i.e. `^\s*/(read|glob|image|audio)[ \t]…`. Gates filename completion so it
/// only fires where a path is expected. The separator is a literal space/tab,
/// not any whitespace: in `--multiline-input` mode the buffer can hold a `\n`
/// right after the command, and we don't want completion firing once the
/// cursor has moved onto a fresh continuation line.
fn in_path_arg(line: &str, pos: usize) -> bool {
    let head = line[..pos].trim_start();
    PATH_COMMANDS.iter().any(|cmd| {
        head.strip_prefix(cmd)
            .is_some_and(|rest| rest.starts_with([' ', '\t']))
    })
}

impl Completer for ChatHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        if in_path_arg(line, pos) {
            // Delegate to rustyline's FilenameCompleter: space is a word-break
            // char so it isolates just the path token at the cursor (the
            // leading `/image ` is excluded), then lists matching entries. With
            // the default `with-dirs` feature it reads `~/…` from $HOME while
            // keeping `~` literal in the replacement.
            self.completer.complete(line, pos, ctx)
        } else {
            Ok((pos, Vec::new()))
        }
    }
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
        // Attach a pending `/image` or `/audio` to this turn: the user content
        // carries the media marker at its head (like llama-mtmd-cli) and the
        // encoded media becomes the conversation's committed item. Moved, not
        // cloned. Image and audio are mutually exclusive in the first cut.
        let had_image = self.pending_image.is_some();
        let had_audio = self.pending_audio.is_some();
        let had_pending = had_image || had_audio;
        let content = if had_pending {
            format!("{MEDIA_MARKER}{text}")
        } else {
            text.to_string()
        };
        if had_image {
            self.image = self.pending_image.take();
        }
        if had_audio {
            self.audio = self.pending_audio.take();
        }
        self.messages.push(ChatMessage::user(content));
        let result = self.generate(on_text);
        if result.is_err() {
            self.messages.pop();
            // Fully undo the attach so the media can be retried next turn.
            if had_image {
                self.pending_image = self.image.take();
            }
            if had_audio {
                self.pending_audio = self.audio.take();
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
    /// soft-placeholder token (where the encoder embeddings splice in). The
    /// rendered string carries the `<__media__>` marker (in the image turn's
    /// content); we split on it and replace it with the projector's image block
    /// (`<|vision_start|><|image_pad|>×n_tok<|vision_end|>` for qwen-style towers,
    /// `<|image><|image|>×n_tok<image|>` for gemma4uv). Shared by `generate`
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
        // One audio clip: split on the marker and splice the gemma4ua block
        // `<|audio><|audio|>×n_tok<audio|>` (begin / per-frame placeholder / end,
        // matching llama.cpp mtmd). The placeholder rows are overwritten by the
        // audio embeddings, so only their count + surrounding markers matter.
        if let Some(audio) = &self.audio {
            let (before, after) = rendered.split_once(MEDIA_MARKER).ok_or(
                "conversation has audio but the rendered prompt has no <__media__> marker \
                 (chat template dropped it?)",
            )?;
            let tid = |s: &str| -> Result<u32, Box<dyn Error>> {
                bundle.tokenizer.token_to_id(s).ok_or_else(|| {
                    format!("tokenizer has no {s} token — this model is not audio-capable").into()
                })
            };
            let mut tokens = encode(before)?;
            tokens.push(tid("<|audio>")?);
            let audio_start = tokens.len();
            tokens.resize(tokens.len() + audio.n_tok, tid("<|audio|>")?);
            tokens.push(tid("<audio|>")?);
            tokens.extend(encode(after)?);
            return Ok((rendered, tokens, Some(audio_start)));
        }
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
        // Image markers differ per projector (begin, per-token soft placeholder,
        // end). The placeholder rows are overwritten by the vision embeddings, so
        // only their count + the surrounding markers matter for the decoder.
        let is_gemma4 = self.vision_ctx.as_ref().is_some_and(|vc| {
            vc.vision.config.projector_type == crate::vision::ProjectorType::Gemma4Uv
        });
        let (vstart, vpad, vend) = if is_gemma4 {
            ("<|image>", "<|image|>", "<image|>")
        } else {
            ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
        };
        let mut tokens = encode(before)?;
        tokens.push(tid(vstart)?);
        let image_start = tokens.len();
        tokens.resize(tokens.len() + img.n_tok, tid(vpad)?);
        tokens.push(tid(vend)?);
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
        if !self.context_shift || self.image.is_some() || self.audio.is_some() {
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
        let (rendered, new_tokens, media_start) = self.render_prompt()?;
        // The one committed media item (image XOR audio) and its placeholder-token
        // count — both modalities splice via the same path, differing only in the
        // forward call below.
        let media_n_tok: Option<usize> = self
            .image
            .as_ref()
            .map(|i| i.n_tok)
            .or_else(|| self.audio.as_ref().map(|a| a.n_tok));

        // Prefix reuse: keep the cache prefix that still matches.
        let mut common0 = self
            .prior_tokens
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        // If media is attached, the prefill that (re)feeds its block must run
        // through `forward_{image,audio}_sampled`, which needs the WHOLE block in
        // the delta. The identical placeholder ids mean prefix reuse normally
        // stops before the block or sails past it; only an edited cache could
        // leave the boundary strictly inside the placeholders — rewind to the
        // block start there so the full media is re-prefilled, never half.
        if let (Some(s), Some(n)) = (media_start, media_n_tok)
            && common0 > s
            && common0 < s + n
        {
            common0 = s;
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

        // Does this prefill delta contain the media (image/audio) block? Only
        // when the reuse boundary is at/before the first placeholder — otherwise
        // the block is already cached (its embeddings were spliced when first
        // prefilled) and the normal text path applies. When it is in the delta,
        // the prefill runs through `forward_{image,audio}_sampled`, so grow the
        // scratch to fit the whole delta first.
        let media_in_delta = matches!(media_start, Some(s) if common <= s);
        // Local placeholder offset only when the block is in the delta (else
        // `s < common` for a cached media item would underflow this usize).
        let media_start_in_delta = media_start.filter(|_| media_in_delta).map(|s| s - common);
        // Image grid dims (None ⇒ the committed media is audio, fed as 1×N).
        let image_dims = self.image.as_ref().map(|i| (i.nx, i.ny));
        if media_in_delta {
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
        // Borrow the embeddings for the (single) media prefill forward below.
        // A slice ref (not a clone) — the [proj_dim,n_tok] buffer can be MBs.
        let media_embeds: Option<&[f32]> = if media_in_delta {
            self.image
                .as_ref()
                .map(|i| i.embeddings.as_slice())
                .or_else(|| self.audio.as_ref().map(|a| a.embeddings.as_slice()))
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
        let prompt_tokens = delta.len(); // prefill suffix fed this turn
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

        // Speculative-decode (MTP) path: enabled for plain-text turns when a
        // draft head is available and `--spec-draft-n-max > 0`. Prefill via the
        // hidden-exposing forward, then draft+verify blocks. Media turns and the
        // no-draft case fall through to the single-token loop unchanged.
        if self.spec_n_max > 0 && !media_in_delta {
            let t0 = std::time::Instant::now();
            let (logits, residual) = self.engine.forward_full_readback(
                &*self.model,
                &mut self.cache,
                &delta,
                common as u32,
                /* full_logits = */ false,
            )?;
            // Seed the draft head's KV from the prompt hiddens (qwen NextN; a
            // no-op for gemma4, whose draft cross-attends the base K/V).
            let hsz = residual.len() / delta.len();
            if delta.len() >= 2 {
                self.engine.run_mtp_seed(
                    &*self.model,
                    &mut self.cache,
                    &residual[0..(delta.len() - 1) * hsz],
                    &delta[1..],
                    common as u32,
                )?;
            }
            prefill_secs = t0.elapsed().as_secs_f64();
            let mut h_last = residual[(delta.len() - 1) * hsz..delta.len() * hsz].to_vec();
            let first = self.sampler.sample_one(&logits);
            self.sampler.accept(first);
            let mut last_token = first;

            // Per-emitted-token tail (EOS / push / `<think>` transition / UTF-8
            // stream emit / max-tokens) shared by the first token and each
            // verified+accepted token. Captures only locals, so the
            // `decode_speculative` call (borrowing `self.engine`/`cache`/
            // `sampler`) doesn't conflict; scoped so the &mut captures release
            // before the epilogue reads `assistant_tokens`. Returns true = STOP.
            {
                let eos_ids = self.eos_ids.clone();
                let max_tokens = self.max_tokens;
                let think_open = self.think_open_id;
                let think_close = self.think_close_id;
                let n_max = self.spec_n_max;
                let mut emit = |token: u32| -> bool {
                    if eos_ids.contains(&token) {
                        assistant_tokens.push(token);
                        return true;
                    }
                    assistant_tokens.push(token);
                    let was = in_think;
                    if Some(token) == think_open {
                        in_think = true;
                    } else if Some(token) == think_close {
                        in_think = false;
                        think_close_at = Some(assistant_tokens.len() - 1);
                    }
                    let seg = if was || in_think {
                        Segment::Thinking
                    } else {
                        Segment::Final
                    };
                    if let Ok(Some(piece)) = stream.step(token) {
                        on_text(&piece, seg);
                    }
                    assistant_tokens.len() as u32 >= max_tokens
                };

                let t_dec = std::time::Instant::now();
                if !emit(first) {
                    loop {
                        if GENERATION_CANCELLED.load(Ordering::SeqCst) {
                            cancelled = true;
                            break;
                        }
                        // Reserve the verify's `n_max + 1` lookahead within the
                        // cache capacity (sized with that headroom).
                        if self.cache.position + n_max + 1 > self.cache.config.max_seq_len {
                            ctx_full = true;
                            break;
                        }
                        let position = self.cache.position;
                        let out = self.engine.decode_speculative(
                            &*self.model,
                            &mut self.cache,
                            last_token,
                            &h_last,
                            position,
                            &mut self.sampler,
                            n_max,
                        )?;
                        let mut stop = false;
                        for &tk in &out.emitted {
                            if emit(tk) {
                                stop = true;
                                break;
                            }
                        }
                        last_token = out.last_token;
                        h_last = out.h_last;
                        if stop {
                            break;
                        }
                    }
                }
                decode_secs = t_dec.elapsed().as_secs_f64();
            }
            // One "forward" per emitted token, for the epilogue's stats and its
            // `forwards == 0` guard (mirrors the single-token loop's accounting).
            forwards = assistant_tokens.len();
        } else {
            let mut step_tokens = delta;
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
                // Forward 0 of a media turn splices the encoder embeddings over the
                // placeholder rows. Image uses the (possibly M-RoPE) image path; audio
                // (image_dims == None) uses the 1×N audio path. Every other forward —
                // the rest of the prefill delta and all decode steps — is the normal
                // text path.
                let token = if forwards == 0 && media_in_delta {
                    let start = media_start_in_delta.expect("media_in_delta ⇒ start");
                    let embeds = media_embeds.expect("media_in_delta ⇒ embeddings");
                    if let Some((nx, ny)) = image_dims {
                        self.engine.forward_image_sampled(
                            &**model,
                            cache,
                            &step_tokens,
                            embeds,
                            start,
                            nx,
                            ny,
                            &mut self.sampler,
                        )?
                    } else {
                        let n = media_n_tok.expect("media_in_delta ⇒ audio n_tok");
                        self.engine.forward_audio_sampled(
                            &**model,
                            cache,
                            &step_tokens,
                            embeds,
                            start,
                            n,
                            &mut self.sampler,
                        )?
                    }
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
        } // end non-speculative decode branch

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

    /// Load the mmproj sidecar on first `/image` or `/audio` and keep it for the
    /// session. One upload serves both modalities: it builds the vision tower
    /// ([`VisionCtx`]) and parses the gemma4ua audio config into `audio_cfg` (the
    /// audio encoder reuses the vision tower's uploaded weights). Errors if the
    /// model has no mmproj (or `--no-mmproj`).
    fn ensure_mmproj(&mut self) -> Result<(), Box<dyn Error>> {
        if self.vision_ctx.is_some() {
            return Ok(());
        }
        let path = self.mmproj_path.clone().ok_or(
            "no multimodal projector available — this model has no mmproj sidecar \
             (or it was disabled with --no-mmproj)",
        )?;
        tracing::info!(path = ?path, "loading mmproj projector");
        let gguf = GgufFile::open(&path)?;
        let weights = self.engine.upload_weights(&gguf)?;
        // Parse the audio encoder config if the mmproj carries one (gemma4 is
        // "any-to-any": vision + audio share this single weight upload).
        self.audio_cfg = crate::audio::parse_config(&gguf).ok();
        let cfg = crate::vision::parse_config(&gguf)?;
        // The gemma4uv projector has no transformer tower — skip the tower
        // encoder and its host-side weights; `attach_image` runs the light
        // `encode_image_gemma4` pipeline for it.
        let (encoder, host_weights) =
            if cfg.projector_type == crate::vision::ProjectorType::Gemma4Uv {
                (None, None)
            } else {
                // The encoder copies its tensor views out of `weights` (no borrow
                // held), so moving `weights` into the VisionModel below keeps them
                // valid.
                let encoder = VisionEncoder::new(
                    &weights,
                    cfg.n_embd as usize,
                    cfg.patch_size as usize,
                    cfg.n_head as usize,
                    cfg.n_ff as usize,
                    cfg.n_layer as usize,
                    cfg.eps,
                )?;
                (Some(encoder), Some(HostWeights::from_gguf(&gguf)?))
            };
        let vision = crate::vision::VisionModel {
            config: cfg,
            weights,
        };
        self.vision_ctx = Some(VisionCtx {
            vision,
            encoder,
            host_weights,
        });
        Ok(())
    }

    /// Encode `path` through the vision tower (GPU) and stage it as the pending
    /// image for the next user turn. Returns the merged grid `(nx, ny, n_tok)`
    /// for the confirmation line. Errors if an image is already attached.
    fn attach_image(&mut self, path: &Path) -> Result<(usize, usize, usize), Box<dyn Error>> {
        if self.image.is_some() {
            return Err(
                "this conversation already has an image — /clear to start over \
                        (one image per conversation for now)"
                    .into(),
            );
        }
        if self.audio.is_some() {
            return Err(
                "this conversation already has audio — /clear to start over \
                        (image and audio can't be mixed in one conversation yet)"
                    .into(),
            );
        }
        self.ensure_mmproj()?;
        let cfg = self
            .vision_ctx
            .as_ref()
            .expect("ensure_mmproj built it")
            .vision
            .config
            .clone();
        let is_gemma4 = cfg.projector_type == crate::vision::ProjectorType::Gemma4Uv;
        let pcfg = if is_gemma4 {
            crate::vision::preprocess::PreprocessConfig::gemma4_default(
                cfg.patch_size,
                cfg.n_merge,
                cfg.image_mean,
                cfg.image_std,
            )
        } else {
            crate::vision::preprocess::PreprocessConfig::qwen3vl_default(
                cfg.patch_size,
                cfg.spatial_merge_size,
                cfg.image_mean,
                cfg.image_std,
            )
        };
        let pimg = crate::vision::preprocess::preprocess(path, &pcfg)?;

        // Grow the scratch for the encoder's working set, then encode.
        let need = vision_scratch_estimate(&pimg);
        if need > self.scratch_bytes {
            self.engine.allocate_scratch(need)?;
            self.scratch_bytes = need;
        }
        let vc = self.vision_ctx.as_ref().expect("ensure_mmproj built it");
        let weights = &vc.vision.weights;
        let (embeddings, nx, ny, n_tok) = if is_gemma4 {
            // No-tower projector: the light gemma4uv embed pipeline returns its
            // own merged grid (npx·npy = n_tok).
            let (emb, _ntok, npx, npy) = crate::vision::encoder::encode_image_gemma4(
                &mut self.engine,
                weights,
                &cfg,
                &pimg,
            )?;
            (emb, npx, npy, npx * npy)
        } else {
            let merge = cfg.spatial_merge_size as usize;
            let (nx, ny) = (pimg.grid_w as usize / merge, pimg.grid_h as usize / merge);
            let encoder = vc.encoder.as_ref().expect("non-gemma4 tower encoder");
            let host_weights = vc.host_weights.as_ref().expect("non-gemma4 host weights");
            let embeddings = crate::vision::encoder::encode_image_chunked(
                &mut self.engine,
                weights,
                encoder,
                &pimg,
                host_weights,
            )?;
            (embeddings, nx, ny, pimg.n_tokens as usize)
        };
        self.pending_image = Some(EncodedImage {
            embeddings,
            nx,
            ny,
            n_tok,
        });
        Ok((nx, ny, n_tok))
    }

    /// Decode `path` to 16 kHz mono, encode it through the gemma4ua audio
    /// projector (GPU), and stage it as the pending audio for the next user turn.
    /// Returns `n_tok` (40 ms frames) for the confirmation line. Errors if audio
    /// or an image is already attached (one media item per conversation for now).
    fn attach_audio(&mut self, path: &Path) -> Result<usize, Box<dyn Error>> {
        if self.audio.is_some() {
            return Err(
                "this conversation already has audio — /clear to start over \
                        (one audio clip per conversation for now)"
                    .into(),
            );
        }
        if self.image.is_some() {
            return Err(
                "this conversation already has an image — /clear to start over \
                        (image and audio can't be mixed in one conversation yet)"
                    .into(),
            );
        }
        self.ensure_mmproj()?;
        let cfg = self
            .audio_cfg
            .clone()
            .ok_or("this model's mmproj has no audio encoder")?;

        // Decode (CPU) to 16 kHz mono f32 before touching the GPU.
        let samples = crate::audio::decode::decode_audio_file(path)?;
        let n_tok = samples.len().div_ceil(cfg.frame_size as usize);

        // Grow scratch for the encoder working set, then encode on the GPU. The
        // gemma4ua encoder allocates input + normed [frame, n_tok] and the
        // projection output [proj_dim, n_tok]; size generously like vision.
        let need = audio_scratch_estimate(n_tok);
        if need > self.scratch_bytes {
            self.engine.allocate_scratch(need)?;
            self.scratch_bytes = need;
        }
        let weights = &self
            .vision_ctx
            .as_ref()
            .expect("ensure_mmproj built it")
            .vision
            .weights;
        let (embeddings, n_tok) =
            crate::audio::encoder::encode_audio_gemma4(&mut self.engine, weights, &cfg, &samples)?;
        self.pending_audio = Some(EncodedAudio { embeddings, n_tok });
        Ok(n_tok)
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
        let system = self.system_prompt().map(ChatMessage::system);
        self.messages.clear();
        if let Some(s) = system {
            self.messages.push(s);
        }
        self.prior_tokens.clear();
        self.cache.reset();
        self.sampler.reset_recent();
        // Drop any attached/pending media (the loaded mmproj projector + parsed
        // audio config are kept — reusable for the next `/image` or `/audio`).
        self.image = None;
        self.pending_image = None;
        self.audio = None;
        self.pending_audio = None;
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
                if let Some(rest) = line.strip_prefix("model name")
                    && let Some(v) = rest.split(':').nth(1)
                {
                    return v.trim().to_string();
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

fn format_ctx_line(g: &GgufFile, ctx_size: u32) -> String {
    // Show the effective window; note the model's trained max when capped below it.
    match g.trained_ctx_len() {
        Some(max) if max > ctx_size => format!("{ctx_size} tokens (model max {max})"),
        _ => format!("{ctx_size} tokens"),
    }
}

fn print_banner(gguf: &GgufFile, path: &Path, ctx_size: u32) {
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
    println!("ctx     : {}", format_ctx_line(gguf, ctx_size));
    println!();
    println!("available commands:");
    println!("  /exit or Ctrl+C     stop or exit");
    println!("  /regen              regenerate the last response");
    println!("  /system [text]      show or set the system prompt");
    println!("  /clear              clear the chat history");
    println!("  /read <file>        add a text file");
    println!("  /glob <pattern>     add text files using globbing pattern");
    println!("  /image <file>       attach an image to your next message (VL models)");
    println!("  /audio <file>       attach audio to your next message (audio models)");
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
    // `List` completion = bash/readline style (complete to the longest common
    // prefix, list candidates when ambiguous), matching llama-cli's feel; the
    // rustyline default is `Circular` (cycle through matches on each Tab).
    let config = Config::builder()
        .completion_type(CompletionType::List)
        .build();
    let mut editor = Editor::with_config(config)?;
    editor.set_helper(Some(ChatHelper {
        multiline,
        completer: FilenameCompleter::new(),
    }));
    if let Some(p) = history
        && let Err(e) = editor.load_history(p)
    {
        tracing::debug!(path = %p.display(), error = %e, "history load");
    }

    print_banner(gguf, path, session.cache.config.max_seq_len);

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
                    } else if let Some(arg) = cmd.strip_prefix("audio") {
                        handle_audio(session, arg.trim());
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
                println!(
                    "[context shift: dropped {} oldest turn(s)]",
                    stats.shifted_turns
                );
            }
            if stats.interrupted {
                println!("[interrupted]");
            }
            if stats.ctx_full {
                println!(
                    "[context full — reply truncated at --ctx-size; /clear, raise --ctx-size, or use --context-shift]"
                );
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

/// Normalize a (possibly Tab-completed) literal-path argument into a real
/// filesystem path. rustyline's `FilenameCompleter` inserts paths in shell
/// style — break chars (spaces, `(`, `$`, …) backslash-escaped, or completed
/// inside quotes — but the REPL isn't a shell, so a literal-path command must
/// undo that itself: drop surrounding quotes, remove the escaping backslashes,
/// then expand a leading `~`. Used by `/read` and `/image`. `/glob` does NOT
/// use this — there a `\` is meaningful to the glob crate (a glob escape), so
/// it keeps backslashes and only strips quotes.
fn normalize_path_arg(arg: &str) -> String {
    let arg = arg.trim().trim_matches('"').trim_matches('\'');
    let unescaped = unescape(arg, Some('\\'));
    expand_tilde(&unescaped)
}

fn handle_read(session: &mut ChatSession, path: &str) {
    if path.is_empty() {
        println!("usage: /read <path>");
        return;
    }
    let path = &normalize_path_arg(path);
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
    // Unquote / unescape a Tab-completed path and expand a leading `~` (the
    // REPL isn't a shell, so it must do this itself). See `normalize_path_arg`.
    let path = normalize_path_arg(arg);
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

/// `/audio <file>`: decode + encode an audio clip through the gemma4ua projector
/// and stage it for the next user message (the message then carries the
/// `<__media__>` marker, so the audio block is spliced into that turn's prefill).
/// Audio-capable models only (gemma4 mmproj with an audio encoder).
fn handle_audio(session: &mut ChatSession, arg: &str) {
    // Unquote / unescape a Tab-completed path and expand a leading `~`, as for
    // `/image` (see `normalize_path_arg`).
    let path = normalize_path_arg(arg);
    if path.is_empty() {
        println!("usage: /audio <path-to-audio-file>");
        return;
    }
    match session.attach_audio(Path::new(&path)) {
        Ok(n_tok) => println!(
            "(audio attached: {n_tok} tokens, ~{}s — sent with your next message)",
            n_tok / 25
        ),
        Err(e) => println!("/audio failed: {e}"),
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
    // Strip surrounding quotes a Tab-completion may have left, but keep any
    // backslashes — here `\` is a glob escape (e.g. `foo\*.txt` for a literal
    // `*`), so unescaping like the literal-path commands would corrupt the
    // pattern. A completed `\ ` is a glob-literal space either way.
    let pattern = &expand_tilde(pattern.trim_matches('"').trim_matches('\''));
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
    use super::{
        expand_tilde, in_path_arg, join_continuations, normalize_path_arg, parse_logit_bias,
        prompt_opens_think,
    };

    #[test]
    fn in_path_arg_gates_completion() {
        // Fires once a path command is followed by a space/tab, anywhere the
        // cursor sits past that separator.
        assert!(in_path_arg("/image ", 7));
        assert!(in_path_arg("/image ~/p", 10));
        assert!(in_path_arg("/read foo", 9));
        assert!(in_path_arg("/read\tfoo", 9)); // tab separator
        assert!(in_path_arg("  /glob *.png", 13)); // leading space tolerated
        // Not yet at the argument: still typing the command name, or no space.
        assert!(!in_path_arg("/imag", 5));
        assert!(!in_path_arg("/image", 6));
        assert!(!in_path_arg("/imagefoo ", 10)); // not a real command
        // A newline is NOT a path separator: in multiline mode the buffer can
        // hold `/image\n` once the cursor moves to the next line — don't fire.
        assert!(!in_path_arg("/image\n", 7));
        // Plain chat and non-path commands never complete.
        assert!(!in_path_arg("hello world", 11));
        assert!(!in_path_arg("/system you are", 15));
    }

    #[test]
    fn normalize_path_arg_reverses_completion() {
        // SAFETY: single-threaded within this test; no other test reads $HOME.
        unsafe { std::env::set_var("HOME", "/home/bob") };
        // Backslash-escaped break chars (what FilenameCompleter inserts for an
        // unquoted path) are unescaped back to the real name.
        assert_eq!(normalize_path_arg("report\\ 2024.txt"), "report 2024.txt");
        assert_eq!(normalize_path_arg("data\\(1\\).png"), "data(1).png");
        // Surrounding quotes (a quote-completed path) are stripped, including
        // the lone leading quote a completion inside an unclosed quote leaves.
        assert_eq!(normalize_path_arg("\"My Docs/a.txt"), "My Docs/a.txt");
        assert_eq!(normalize_path_arg("'My Docs/a.txt'"), "My Docs/a.txt");
        // Tilde still expands, after unescaping.
        assert_eq!(
            normalize_path_arg("~/My\\ Pics/a.png"),
            "/home/bob/My Pics/a.png"
        );
        // A plain path is untouched.
        assert_eq!(normalize_path_arg("/abs/path.png"), "/abs/path.png");
    }

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
        assert_eq!(
            parse_logit_bias("128009-inf"),
            Ok((128009, f32::NEG_INFINITY))
        );
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
        assert!(prompt_opens_think(
            "<think>\na\n</think>\nb<|im_start|>assistant\n<think>\n"
        ));
        // No think markers at all (e.g. Llama) → closed.
        assert!(!prompt_opens_think("<|im_start|>assistant\n"));
    }
}
