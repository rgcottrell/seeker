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
use crate::commands::download::{HfResolveArgs, resolve_hf};
use crate::gguf::{GgmlType, GgufFile};
use crate::inference::kv_cache::parse_dtype;
use crate::inference::sample::{GgufSamplingDefaults, SamplerConfig};
use crate::server::inference::{InferenceHandle, WorkerConfig};
use crate::server::{AppState, AppStateInit, ServerConfig, run as server_run};
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

    /// KV-cache budget per slot, in tokens. `0` (the default) uses the model's
    /// full trained context length — llama.cpp's `-c 0` convention. Each of the
    /// `--parallel` slots is a full `--ctx-size` cache, so lower it to cap total
    /// KV memory.
    #[arg(long = "ctx-size", default_value_t = 0)]
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
    /// per-slot size (printed at startup). Concurrent subagents each keep their
    /// cache warm and reuse it instead of re-prefilling on every switch.
    /// `0` (the default) = **auto**: fit as many slots as memory allows, capped
    /// at `--parallel-max`. `1` = a single cache. (short: -np)
    #[arg(long = "parallel", default_value_t = 0)]
    parallel: u32,

    /// Upper bound for `--parallel 0` (auto). The per-slot full-context slab is
    /// large, so the auto path caps the count for the handful-of-subagents
    /// target; raise it (and/or lower `--ctx-size`) for more concurrency.
    #[arg(long = "parallel-max", default_value_t = 8)]
    parallel_max: u32,

    /// Fraction of device memory the `--parallel 0` auto path may budget for KV
    /// slots, after weights + scratch. Leaves headroom for the OS / transient
    /// image scratch on the unified-memory APU.
    #[arg(long = "mem-fraction", default_value_t = 0.9)]
    mem_fraction: f32,

    /// Run in embedding-only mode (like llama-server's `--embeddings`): the
    /// `/embeddings` (native) and `/v1/embeddings` (OpenAI) endpoints serve pooled
    /// vectors and generation is disabled. Requires an embedding model (one with
    /// an `output_norm.weight`, e.g. Qwen3-Embedding).
    #[arg(long = "embeddings")]
    embeddings: bool,

    /// Pooling for embedding mode. Defaults to the model's GGUF `pooling_type`
    /// (Qwen3-Embedding = last).
    #[arg(long = "pooling", value_enum)]
    pooling: Option<crate::inference::embed::Pooling>,

    /// Embedding normalization (llama.cpp `--embd-normalize`): -1 none, 0 max-abs,
    /// 1 taxicab/L1, 2 euclidean/L2 (default), p>2 p-norm. Per-request overridable.
    #[arg(
        long = "embd-normalize",
        default_value_t = 2,
        allow_negative_numbers = true
    )]
    embd_normalize: i32,

    /// KV cache K dtype. One of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1
    /// (quant + turbo* require the per-block BatchKvCache path).
    #[arg(long = "cache-type-k", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_k: GgmlType,

    /// KV cache V dtype. Same legal values as --cache-type-k.
    #[arg(long = "cache-type-v", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_v: GgmlType,

    // ─── Sampling (per-request defaults) ────────────────────────────────
    // Unset → the GGUF's `general.sampling.*` default if present, else the
    // built-in fallback (temp 0.8, top-k 40, top-p 0.95, min-p 0.05,
    // repeat-penalty 1.0, penalty-last-n 64). An explicit flag always wins; a
    // per-request API field overrides the resolved server default in turn.
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

    /// Do not load the matching mmproj vision projector (serve text-only even
    /// for a VL model). By default the sidecar is loaded so chat requests can
    /// carry images (OpenAI `image_url` content).
    #[arg(long = "no-mmproj")]
    no_mmproj: bool,

    /// Speculative-decode draft model and max draft tokens per step
    /// (`--spec-draft-model` / `--spec-draft-hf` / `--spec-draft-n-max`). With a
    /// draft and `n_max > 0`, a SINGLE active request decodes speculatively
    /// (concurrent requests fall back to plain batched decode).
    #[command(flatten)]
    spec: crate::commands::download::SpecDraftArgs,

    #[command(flatten)]
    diffusion: crate::commands::chat::DiffusionArgs,
}

fn parse_dtype_arg(s: &str) -> Result<GgmlType, String> {
    parse_dtype(s)
}

impl ServeArgs {
    /// Resolve the server's default sampler with precedence **CLI flag → GGUF
    /// `general.sampling.*` → built-in fallback** (matching llama.cpp). A
    /// per-request API field overrides the resolved default in turn.
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

    // diffusion-gemma (non-autoregressive) re-forwards [prompt|canvas] each step
    // and never uses the KV cache; bound its context (the trained 256K would
    // size a giant unused per-slot cache) and run requests sequentially.
    let is_diffusion = gguf.architecture() == Some("diffusion-gemma");

    // `--ctx-size 0` (the default) means "use the model's full trained context"
    // (llama.cpp's `-c 0`); fall back to 4096 if the model omits the metadata.
    // Each of the `--parallel` slots gets a full `ctx_size` cache.
    let ctx_size = if args.ctx_size == 0 {
        if is_diffusion {
            4096
        } else {
            gguf.trained_ctx_len().unwrap_or(4096)
        }
    } else {
        args.ctx_size
    };
    tracing::info!(
        ctx_size,
        parallel = args.parallel,
        parallel_max = args.parallel_max,
        "context window per slot (parallel 0 = auto-size from memory)"
    );

    // Resolve the mmproj vision sidecar (unless --no-mmproj). The worker builds
    // the encoder from `mmproj_path`; the handler CPU-preprocesses with
    // `vision_config`. A sidecar that fails to parse degrades to text-only.
    let mmproj_path = if args.no_mmproj {
        None
    } else {
        crate::commands::download::find_sidecar_mmproj(&path)
    };
    // Parse the vision + audio projector configs from the one mmproj GGUF. The
    // handler uses these to assemble the image / audio blocks; the worker builds
    // the encoders from `mmproj_path`. An audio encoder is optional (qwen mmprojs
    // have none); a sidecar that yields neither degrades to text-only.
    let (vision_config, audio_config) = match mmproj_path.as_ref() {
        Some(p) => match GgufFile::open(p) {
            Ok(g) => {
                let vision = match crate::vision::parse_config(&g) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        tracing::warn!(path = ?p, error = %e, "mmproj vision config unparseable");
                        None
                    }
                };
                let audio = crate::audio::parse_config(&g).ok();
                (vision, audio)
            }
            Err(e) => {
                tracing::warn!(path = ?p, error = %e, "mmproj present but unreadable; serving text-only");
                (None, None)
            }
        },
        None => (None, None),
    };
    // If neither config parsed, don't hand the worker a path it can't use.
    let mmproj_path = if vision_config.is_some() || audio_config.is_some() {
        mmproj_path
    } else {
        None
    };

    // When the leading-prefix cache is on and a system prompt is set, render the
    // shared prefix here (handler-side tokenizer/template) and hand the worker
    // the tokens to prefill + PIN once at startup, so requests beginning with it
    // seed instead of re-prefilling it.
    let pin_prefix_tokens = if *crate::runtime_flags::PREFIX_CACHE {
        match resolve_system_prompt(args)? {
            Some(sys) => {
                let kwargs = args.chat_template_kwargs.clone().unwrap_or_default();
                let t = crate::server::convert::compute_pin_prefix(&bundle, &sys, &kwargs);
                if let Some(tk) = &t {
                    tracing::info!(
                        prefix_tokens = tk.len(),
                        "prefix cache: pinning system-prompt prefix"
                    );
                }
                t
            }
            None => None,
        }
    } else {
        None
    };

    // Optional MTP draft model (local path or HF repo) for single-stream
    // speculative decode. Resolved here (async); the worker attaches it.
    let spec_draft_path = crate::commands::download::resolve_spec_draft(
        args.spec.spec_draft_model.clone(),
        args.spec.spec_draft_hf.clone(),
        args.hf_token.clone(),
        args.offline,
    )
    .await?;

    let (handle, ready) = InferenceHandle::spawn(WorkerConfig {
        model_path: path.clone(),
        mmproj_path,
        n_ubatch: args.ubatch_size,
        // Embedding mode does one single-pass forward per input; match llama.cpp
        // and pin n_batch = n_ubatch (no chunked prefill across the pooled forward).
        n_batch: if args.embeddings {
            args.ubatch_size
        } else {
            args.batch_size
        },
        ctx_size,
        ctx_auto: args.ctx_size == 0,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        n_slots: args.parallel, // 0 = auto-size in the worker
        parallel_max: args.parallel_max,
        mem_fraction: args.mem_fraction,
        pin_prefix_tokens,
        spec_draft_path,
        spec_draft_n_max: args.spec.spec_draft_n_max,
        embeddings: args.embeddings,
        pooling: args.pooling,
        embd_normalize: args.embd_normalize,
        diffusion: is_diffusion.then(|| args.diffusion.to_config(args.max_tokens as usize)),
    });
    // The worker reports the *resolved* slot count (auto-sizing may differ from
    // the request) so `/slots` + `/props` report the real number.
    let n_slots = match ready.await {
        Ok(Ok(n)) => n,
        Ok(Err(e)) => return Err(format!("failed to load model: {e}").into()),
        Err(_) => return Err("inference worker exited during startup".into()),
    };

    let model_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    // GGUF-embedded sampling defaults (`general.sampling.*`) seed the server's
    // default sampler, overridden by explicit CLI flags then per-request fields.
    let gg_sampling = GgufSamplingDefaults::from_gguf(&gguf);
    if !gg_sampling.is_empty() {
        tracing::info!(?gg_sampling, "using GGUF general.sampling.* defaults");
    }

    Ok(AppState::new(AppStateInit {
        tokenizer: Arc::new(bundle),
        inference: handle,
        template_kwargs: {
            let mut kw = args.chat_template_kwargs.clone().unwrap_or_default();
            // diffusion-gemma is a thinking/harmony model: with `enable_thinking`
            // unset its template primes a closed empty `thought` channel that
            // derails the canvas. Match llama.cpp's diffusion-cli (thinking ON by
            // default); a user `--chat-template-kwargs` value still wins.
            if is_diffusion {
                kw.entry("enable_thinking".to_string())
                    .or_insert(serde_json::Value::Bool(true));
            }
            kw
        },
        default_sampler: args.sampler_config(&gg_sampling, ctx_size),
        default_max_tokens: args.max_tokens,
        default_ignore_eos: args.ignore_eos,
        default_system_prompt: resolve_system_prompt(args)?,
        ctx_size,
        n_slots,
        model_id,
        model_path: path.display().to_string(),
        vision_config,
        audio_config,
        embeddings: args.embeddings,
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
                !args.no_mmproj,
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
