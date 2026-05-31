//! `seeker serve` — start the HTTP server. Loads the model (same selection as
//! `chat`), spawns the GPU inference worker, and serves the OpenAI / Anthropic /
//! llama-native API surface. The CLI sampling flags become the per-request
//! defaults that individual API calls override.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;

use crate::commands::chat::parse_logit_bias;
use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::{GgmlType, GgufFile};
use crate::inference::kv_cache::parse_dtype;
use crate::inference::sample::SamplerConfig;
use crate::server::inference::{InferenceHandle, WorkerConfig};
use crate::server::{run as server_run, AppState, AppStateInit, ServerConfig};
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

    // ─── Model selection (mirrors chat; all optional — serve can start with
    //     no model for /health + /apply-template) ──────────────────────────
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]". (short: -hf, -hfr)
    #[arg(long = "hf-repo", conflicts_with = "model")]
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

    // ─── Runtime ────────────────────────────────────────────────────────
    /// Max tokens per reply, and the default ceiling for requests that omit
    /// `max_tokens` / `n_predict`.
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,

    /// KV-cache budget per request, in tokens.
    #[arg(long = "ctx-size", default_value_t = 4096)]
    ctx_size: u32,

    /// Logical batch size (max tokens per submit). Validation-only in this
    /// single-sequence engine; `--ubatch-size` is the memory-relevant knob.
    #[arg(short = 'b', long = "batch-size", default_value_t = 2048)]
    batch_size: u32,

    /// Physical micro-batch size: prefill is split into ≤ this many tokens
    /// per GPU pass so scratch memory stays bounded on long prompts.
    /// 0 = unbounded (single pass). (short: -ub)
    #[arg(long = "ubatch-size", default_value_t = 512)]
    ubatch_size: u32,

    /// Number of independent KV-cache slots (llama.cpp's `--parallel`). Each
    /// slot is a full `--ctx-size` cache, so total KV(+SSM) memory is N× the
    /// per-slot size (printed at startup). `1` = a single cache. Raise it so
    /// interleaved sessions — e.g. concurrent subagents — each keep their cache
    /// warm and reuse it instead of re-prefilling on every switch. (short: -np)
    #[arg(long = "parallel", default_value_t = 1)]
    parallel: u32,

    /// KV cache K dtype. One of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1.
    #[arg(long = "cache-type-k", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_k: GgmlType,

    /// KV cache V dtype. Same legal values as --cache-type-k.
    #[arg(long = "cache-type-v", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_v: GgmlType,

    // ─── Sampling (per-request defaults) ────────────────────────────────
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

    /// Never stop on an end-of-generation token; generate until the limit.
    /// (llama.cpp's `--ignore-eos`.)
    #[arg(long = "ignore-eos")]
    ignore_eos: bool,

    /// Bias a token's logit, repeatable. Format `ID(+/-)BIAS` or `ID=BIAS`,
    /// e.g. `--logit-bias 15043+2.0 --logit-bias 128009-inf`. `-inf` bans,
    /// `+inf` forces. Applies to every request.
    #[arg(long = "logit-bias", value_parser = parse_logit_bias)]
    logit_bias: Vec<(u32, f32)>,

    /// RNG seed for stochastic sampling.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    // ─── Prompt / template ──────────────────────────────────────────────
    /// System prompt, injected as the leading system message for chat/messages
    /// requests that carry no system message of their own. (short: -sys)
    #[arg(long = "system-prompt")]
    system_prompt: Option<String>,

    /// Read the system prompt from a UTF-8 text file instead of the CLI.
    /// (short: -sysf)
    #[arg(long = "system-prompt-file", conflicts_with = "system_prompt")]
    system_prompt_file: Option<PathBuf>,

    /// Extra key/value pairs merged into the chat-template rendering context,
    /// as a JSON object string, e.g. `'{"enable_thinking":false}'`. Mirrors
    /// llama.cpp's `--chat-template-kwargs`.
    #[arg(long = "chat-template-kwargs", value_parser = crate::chat_template::parse_template_kwargs)]
    chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

fn parse_dtype_arg(s: &str) -> Result<GgmlType, String> {
    parse_dtype(s)
}

impl ServeArgs {
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

pub async fn run(args: ServeArgs) -> Result<(), Box<dyn Error>> {
    let app_state = match resolve_model_path(&args).await? {
        Some(path) => build_loaded_state(&args, path).await?,
        None => {
            tracing::warn!(
                "serve started with no model — generation endpoints will return 503; \
                 pass --model PATH or --hf-repo to enable inference"
            );
            AppState::default()
        }
    };
    server_run(ServerConfig {
        host: args.host,
        port: args.port,
        cors: args.cors,
        app_state,
    })
    .await
}

/// Open the model for the handler-side tokenizer/template, spawn the GPU
/// inference worker, and wait for it to report ready (failing fast on a bad
/// model / missing device, exactly like `seeker chat`).
async fn build_loaded_state(args: &ServeArgs, path: PathBuf) -> Result<AppState, Box<dyn Error>> {
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;
    tracing::info!(
        template_present = bundle.chat_template.is_some(),
        "loaded tokenizer for serve",
    );

    let n_slots = args.parallel.max(1);
    let (handle, ready) = InferenceHandle::spawn(WorkerConfig {
        model_path: path.clone(),
        n_ubatch: args.ubatch_size,
        n_batch: args.batch_size,
        ctx_size: args.ctx_size,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        n_slots,
    });
    match ready.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(format!("failed to load model: {e}").into()),
        Err(_) => return Err("inference worker exited during startup".into()),
    }

    let model_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    Ok(AppState::new(AppStateInit {
        tokenizer: Arc::new(bundle),
        inference: handle,
        template_kwargs: args.chat_template_kwargs.clone().unwrap_or_default(),
        default_sampler: args.sampler_config(),
        default_max_tokens: args.max_tokens,
        default_ignore_eos: args.ignore_eos,
        default_system_prompt: resolve_system_prompt(args)?,
        ctx_size: args.ctx_size,
        n_slots,
        model_id,
        model_path: path.display().to_string(),
    }))
}

/// Model path from `--model` or `--hf-*`; `None` when neither is given (serve
/// then runs without inference).
async fn resolve_model_path(args: &ServeArgs) -> Result<Option<PathBuf>, Box<dyn Error>> {
    match (args.hf_repo.clone(), args.model.clone()) {
        (Some(repo), None) => Ok(Some(
            resolve_hf(
                &HfResolveArgs {
                    repo,
                    file: args.hf_file.clone(),
                    token: args.hf_token.clone(),
                    offline: args.offline,
                },
                false,
            )
            .await?
            .main,
        )),
        (None, Some(model)) => Ok(Some(model)),
        (None, None) => Ok(None),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with model/hf_repo"),
    }
}

/// The CLI system prompt, from `--system-prompt` or `--system-prompt-file`.
fn resolve_system_prompt(args: &ServeArgs) -> Result<Option<String>, Box<dyn Error>> {
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
