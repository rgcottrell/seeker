//! GPU inference for `seeker serve`.
//!
//! The engine is synchronous, stateful, and holds Vulkan handles (`Engine` /
//! `KvCache` are not `Send`). axum handlers are async and multi-threaded. We
//! bridge them with **one dedicated worker thread** that *constructs and owns*
//! the engine — so the non-`Send` types never cross a thread boundary; only a
//! `PathBuf` + config move into the thread. Handlers render the chat template
//! and encode the prompt, then ship a [`GenJob`] (token ids + per-request
//! sampling) over a channel and receive a stream of [`GenEvent`]s back.
//!
//! Because this is a single-sequence engine, jobs are processed strictly one at
//! a time (matching the GPU's single command queue). The per-job reply channel
//! doubles as a cancellation signal: if the HTTP client disconnects, axum drops
//! the `Receiver`, the worker's next `blocking_send` returns `Err`, and the
//! decode loop bails — freeing the GPU for the next request.

use std::error::Error;
use std::path::{Path, PathBuf};

use tokio::sync::{mpsc, oneshot};

use crate::gguf::{GgmlType, GgufFile};
use crate::inference::Engine;
use crate::inference::kv_cache::{BatchKvCache, KvCacheConfig, PrefixSnapshot};
use crate::inference::sample::{Sampler, SamplerConfig};
use crate::tokenizer::{Tokenizer, build_tokenizer};
use crate::vision::encoder::{HostWeights, VisionEncoder};
use crate::vision::preprocess::PreprocessedImage;

/// Immutable per-process model/runtime config. All `Send` — built by the CLI
/// and moved into the worker thread, which constructs the `Engine` from it.
pub struct WorkerConfig {
    pub model_path: PathBuf,
    /// Path to the mmproj vision sidecar, if one was resolved (and not
    /// `--no-mmproj`). The worker builds the vision tower from it so chat
    /// requests can carry images. `None` → text-only serving.
    pub mmproj_path: Option<PathBuf>,
    pub n_ubatch: u32,
    pub n_batch: u32,
    pub ctx_size: u32,
    pub cache_type_k: GgmlType,
    pub cache_type_v: GgmlType,
    /// Number of independent KV-cache slots (llama.cpp `--parallel`). Each slot
    /// is a full `ctx_size` cache, so total KV(+SSM) memory is `n_slots ×`
    /// per-slot. `1` = single-cache behavior. Allocated eagerly at setup.
    /// `0` = **auto**: size from the device memory budget (see `parallel_max` /
    /// `mem_fraction`), so concurrent subagents get >1 warm slot by default.
    pub n_slots: u32,
    /// Upper bound for the `n_slots == 0` auto path (the per-slot full-context
    /// slab is large, so cap the count for the handful-of-subagents target).
    pub parallel_max: u32,
    /// Fraction of DEVICE_LOCAL memory the auto path may budget for KV slots
    /// (after weights + scratch). Leaves headroom for the OS / transient image
    /// scratch on the unified-memory APU. Ignored when `n_slots >= 1`.
    pub mem_fraction: f32,
    /// Pre-rendered shared leading-prefix tokens (the `--system-prompt` render)
    /// to prefill once and PIN in the leading-prefix cache at startup, so every
    /// request beginning with it seeds instead of re-prefilling it. `None` when
    /// no system prompt is set or `SEEKER_PREFIX_CACHE` is off. Rendered
    /// serve-side (the worker only ever sees tokens).
    pub pin_prefix_tokens: Option<Vec<u32>>,
}

/// Per-request generation parameters. The CLI sampling flags form the base
/// `SamplerConfig`; each API request overrides individual fields (see
/// `types::*::sampler_config`).
pub struct GenConfig {
    pub sampler: SamplerConfig,
    pub max_tokens: u32,
    /// Text stop sequences (OpenAI `stop` / Anthropic `stop_sequences` /
    /// llama `stop`). Generation halts when the decoded tail matches one.
    pub stop: Vec<String>,
    /// Never stop on an end-of-generation token (`--ignore-eos`).
    pub ignore_eos: bool,
    /// Pin to a specific cache slot (llama-native `id_slot`). `None` →
    /// auto-select by longest cached-prefix match / LRU. Out-of-range values
    /// fall back to auto-select.
    pub id_slot: Option<usize>,
}

/// One image attached to a chat request, carried to the worker — the only
/// thread with the GPU. The handler CPU-preprocesses it (it holds the
/// `VisionConfig`) and records where the vision block sits in `tokens`; the
/// worker encodes it through the vision tower and splices the embeddings during
/// that request's prefill. `PreprocessedImage` is `Send` (Vec<f32> + u32s).
pub struct ServeImage {
    pub pimg: PreprocessedImage,
    /// Local index of the first `<|image_pad|>` token in `GenJob::tokens`.
    pub image_start: usize,
    /// Merged-grid dims (`n_tok = nx*ny`).
    pub nx: usize,
    pub ny: usize,
}

/// An audio clip attached to a chat request. The handler decodes it to 16 kHz
/// mono on its thread (CPU) and records where the `<|audio|>` placeholders sit
/// in `tokens`; the worker encodes it through the gemma4ua projector and splices
/// the embeddings during that request's prefill. `samples` is `Send` (Vec<f32>).
pub struct ServeAudio {
    pub samples: Vec<f32>,
    /// Local index of the first `<|audio|>` token in `GenJob::tokens`.
    pub audio_start: usize,
    /// Number of 40 ms audio frames (placeholder tokens).
    pub n_tok: usize,
}

/// One unit of work for the worker. The handler has already rendered the chat
/// template and encoded it, so only `Send` data crosses the thread boundary.
pub struct GenJob {
    pub tokens: Vec<u32>,
    pub config: GenConfig,
    /// An attached image (chat requests with `image_url` content). `None` for
    /// the text path. Prefilled single-pass with the vision splice in the worker.
    pub image: Option<ServeImage>,
    /// An attached audio clip (chat requests with `input_audio` content). `None`
    /// otherwise. Prefilled with the gemma4ua audio splice in the worker.
    /// Mutually exclusive with `image`.
    pub audio: Option<ServeAudio>,
    /// Reply sink. Bounded so the worker back-pressures on a slow client; a
    /// *dropped* receiver (client disconnect) makes `blocking_send` return Err,
    /// which the decode loop treats as cancellation.
    pub reply: mpsc::Sender<GenEvent>,
}

/// The vision tower built once in the worker thread (when an mmproj was
/// resolved) and kept for the process. `vision` owns the uploaded mmproj weights
/// (the encoder's tensor views hold GPU buffer handles kept valid by it).
struct VisionCtx {
    vision: crate::vision::VisionModel,
    /// The transformer-tower encoder — `None` for the gemma4uv "no-tower"
    /// projector (image input via gemma4uv isn't wired in serve yet, but the
    /// mmproj still loads so its audio encoder can be used).
    encoder: Option<VisionEncoder>,
    /// CPU-side weights for the tower encoder's pos-embd resize — `None` for
    /// gemma4uv.
    host_weights: Option<HostWeights>,
    /// The gemma4ua audio config, when the mmproj carries an audio encoder.
    audio_cfg: Option<crate::audio::AudioConfig>,
}

/// Why generation stopped — maps to each API's finish/stop-reason field.
#[derive(Clone, Debug)]
pub enum StopReason {
    /// Hit an end-of-generation token (OpenAI `stop` / Anthropic `end_turn`).
    Eos,
    /// Matched a text stop sequence (OpenAI `stop` / Anthropic `stop_sequence`).
    StopSequence(String),
    /// Hit `max_tokens` (OpenAI `length` / Anthropic `max_tokens`).
    MaxTokens,
    /// Ran out of context headroom mid-reply (OpenAI `length`).
    ContextFull,
}

impl StopReason {
    /// OpenAI / llama-native `finish_reason`.
    pub fn openai_finish(&self) -> &'static str {
        match self {
            StopReason::Eos | StopReason::StopSequence(_) => "stop",
            StopReason::MaxTokens | StopReason::ContextFull => "length",
        }
    }

    /// Anthropic `stop_reason`.
    pub fn anthropic_reason(&self) -> &'static str {
        match self {
            StopReason::Eos => "end_turn",
            StopReason::StopSequence(_) => "stop_sequence",
            StopReason::MaxTokens | StopReason::ContextFull => "max_tokens",
        }
    }

    /// The matched stop string, if generation ended on a stop sequence.
    pub fn matched_sequence(&self) -> Option<&str> {
        match self {
            StopReason::StopSequence(s) => Some(s),
            _ => None,
        }
    }
}

/// Streamed output from the worker. The non-streaming handlers drain these and
/// concatenate; the streaming handlers map each to SSE frames.
#[derive(Clone, Debug)]
pub enum GenEvent {
    /// Emitted once, right after prefill — carries the real prompt token count
    /// (lets Anthropic `message_start` / OpenAI usage report `input_tokens`).
    Started { prompt_tokens: u32 },
    /// Newly-decoded text (complete UTF-8 chars, minus any tail held back for
    /// stop-sequence matching).
    Delta(String),
    /// Terminal success frame.
    Done {
        stop_reason: StopReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    /// Terminal failure frame (GPU error, prompt too long, …). The worker stays
    /// alive for the next job.
    Error(String),
}

/// Cheaply-cloneable handle stored in `AppState`. Wraps the job sender.
#[derive(Clone)]
pub struct InferenceHandle {
    jobs: mpsc::Sender<GenJob>,
}

impl InferenceHandle {
    /// Spawn the dedicated worker thread. Returns the handle plus a readiness
    /// channel: the worker runs the full model-load sequence and sends the
    /// *resolved* slot count `Ok(n_slots)` (or `Err(msg)`) *before* entering its
    /// job loop, so the caller can fail fast on a bad model / missing GPU and
    /// learn the auto-sized `n_slots` for `/slots` + `/props`.
    pub fn spawn(cfg: WorkerConfig) -> (InferenceHandle, oneshot::Receiver<Result<u32, String>>) {
        let (jobs_tx, jobs_rx) = mpsc::channel::<GenJob>(16);
        let (ready_tx, ready_rx) = oneshot::channel();
        std::thread::Builder::new()
            .name("seeker-inference".into())
            // The default 2 MiB thread stack overflows while loading a large
            // model (the GGUF metadata + per-tensor setup is stack-heavy — a
            // multi-GB checkpoint SIGSTKFLTs on the 2 MiB default, while the
            // 8 MiB main-thread stack of `seeker run` survives). Match a roomy
            // main-thread-sized budget.
            .stack_size(64 * 1024 * 1024)
            .spawn(move || worker_main(cfg, jobs_rx, ready_tx))
            .expect("spawn inference worker thread");
        (InferenceHandle { jobs: jobs_tx }, ready_rx)
    }

    /// Queue a job. Errors (returning the job) only if the worker has shut down.
    pub async fn submit(&self, job: GenJob) -> Result<(), GenJob> {
        self.jobs.send(job).await.map_err(|e| e.0)
    }

    /// Submit a job built from `tokens` + `config` and return the reply channel.
    /// The caller either drains it ([`collect`], non-streaming) or adapts it to
    /// SSE (`stream::*`).
    pub async fn start(
        &self,
        tokens: Vec<u32>,
        config: GenConfig,
    ) -> Result<mpsc::Receiver<GenEvent>, String> {
        self.start_mm(tokens, config, None, None).await
    }

    /// As [`Self::start`] but with an optional attached image or audio clip
    /// (multimodal chat). At most one of `image`/`audio` is `Some`.
    pub async fn start_mm(
        &self,
        tokens: Vec<u32>,
        config: GenConfig,
        image: Option<ServeImage>,
        audio: Option<ServeAudio>,
    ) -> Result<mpsc::Receiver<GenEvent>, String> {
        let (tx, rx) = mpsc::channel(32);
        self.submit(GenJob {
            tokens,
            config,
            image,
            audio,
            reply: tx,
        })
        .await
        .map_err(|_| "inference worker is unavailable".to_string())?;
        Ok(rx)
    }
}

/// The fully-collected result of a non-streaming generation.
pub struct GenOutput {
    pub text: String,
    pub stop_reason: StopReason,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Drain a reply channel to completion, concatenating the text deltas. Used by
/// the non-streaming handlers (the worker has one code path for both).
pub async fn collect(mut rx: mpsc::Receiver<GenEvent>) -> Result<GenOutput, String> {
    let mut out = GenOutput {
        text: String::new(),
        stop_reason: StopReason::MaxTokens,
        prompt_tokens: 0,
        completion_tokens: 0,
    };
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Started { prompt_tokens } => out.prompt_tokens = prompt_tokens,
            GenEvent::Delta(t) => out.text.push_str(&t),
            GenEvent::Done {
                stop_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                out.stop_reason = stop_reason;
                out.prompt_tokens = prompt_tokens;
                out.completion_tokens = completion_tokens;
            }
            GenEvent::Error(e) => return Err(e),
        }
    }
    Ok(out)
}

/// Per-slab persistent metadata for cross-request prefix reuse. `prior_tokens`
/// mirrors the slab's live cache when idle (`len == batch.positions[slab]`), so
/// a request that extends this prefix reuses it and prefills only the divergent
/// suffix. `active` marks the slab as owned by an in-flight [`ActiveSeq`] —
/// admission never reuses a busy slab.
struct SlotMeta {
    prior_tokens: Vec<u32>,
    /// Monotonic stamp of the last job that used this slab (LRU). `0` = cold.
    last_used: u64,
    active: bool,
}

/// One in-flight generation, occupying slab `slab`. Holds everything needed to
/// advance it one token per batched decode step and to stream / stop it
/// independently of the other sequences sharing the GPU forward.
struct ActiveSeq {
    slab: u32,
    sampler: Sampler,
    /// Rendered prompt — committed to the slab's `prior_tokens` on eviction.
    prompt: Vec<u32>,
    /// Tokens generated so far (each fed back as the next input, bar the last).
    generated: Vec<u32>,
    prompt_tokens: u32,
    stop: Vec<String>,
    ignore_eos: bool,
    max_tokens: u32,
    reply: mpsc::Sender<GenEvent>,
    // `tokenizers` DecodeStream state, driven via `step_decode_stream` so no
    // borrow of the tokenizer is stored across decode steps.
    stream_ids: Vec<u32>,
    stream_prefix: String,
    stream_prefix_index: usize,
    /// Decoded-but-held-back tail (stop-sequence matching).
    pending: String,
    /// Token to feed at the next batched decode step (legacy decode path).
    last_token: u32,
    /// This slab's context length (`max_seq_len`).
    ctx: u32,
    terminal: Option<StopReason>,
    disconnected: bool,
    /// Unified path: cache positions written so far for this slab (= the number
    /// of logical tokens `prompt ++ generated` already prefilled/decoded). The
    /// per-step contribution is `logical_len - num_computed`, chunked by the
    /// token budget. Equals `batch.positions[slab]`.
    num_computed: u32,
    /// Whether `GenEvent::Started` has been sent (deferred until the unified
    /// path finishes prefill and produces this sequence's first token).
    started: bool,
}

/// One cached leading prefix: the exact prefix token ids (length `p`, matched
/// via [`lcp`]), an LRU stamp, and a pin flag. Its KV+SSM bytes live in the
/// pool slot at the same index in [`PrefixCache::pool`].
struct PrefixEntry {
    tokens: Vec<u32>,
    p: u32,
    last_used: u64,
    pinned: bool,
}

/// In-process, GPU-resident leading-prefix snapshot cache for `seeker serve`.
/// A shared leading prefix (system prompt / few-shot block) is prefilled once
/// and snapshotted at sparse checkpoints; a later divergent request seeds a
/// fresh slab from the longest matching snapshot (GPU→GPU copy of `KV[0,P)` +
/// the SSM state at P) and prefills only its unique suffix. `pool[i]` holds the
/// bytes for `entries[i]` (parallel vecs; `entries[i] == None` ⇒ free slot).
/// Single-threaded with the Worker and every copy is a synchronous fenced
/// submit, so no in-flight reference counting is needed.
struct PrefixCache {
    pool: Vec<PrefixSnapshot>,
    entries: Vec<Option<PrefixEntry>>,
    /// Longest prefix (tokens) any pool slot can hold — caps the KV bytes.
    max_cached_len: u32,
    /// Snapshot at most once per this many prefill tokens (sparse: each is ~65 MiB).
    ckpt_stride: u32,
    /// Don't seed/snapshot a prefix shorter than this.
    p_min: u32,
    /// Monotonic LRU clock (shares nothing with the slot clock).
    clock: u64,
}

impl PrefixCache {
    fn new(
        batch: &BatchKvCache,
        device: &crate::inference::device::Device,
        capacity: u32,
        max_cached_len: u32,
        ckpt_stride: u32,
        p_min: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let mut pool = Vec::with_capacity(capacity as usize);
        for _ in 0..capacity {
            pool.push(batch.new_prefix_snapshot(device, max_cached_len)?);
        }
        let entries = (0..capacity as usize).map(|_| None).collect();
        Ok(Self {
            pool,
            entries,
            max_cached_len,
            ckpt_stride,
            p_min,
            clock: 0,
        })
    }

    /// Index of the longest cached entry whose tokens are a full leading prefix
    /// of `new_tokens` (the most reuse, least re-prefill).
    fn lookup(&self, new_tokens: &[u32]) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let Some(e) = e else { continue };
            let common = lcp(&e.tokens, new_tokens);
            if common == e.tokens.len() && best.is_none_or(|(_, bl)| e.tokens.len() > bl) {
                best = Some((i, e.tokens.len()));
            }
        }
        best.map(|(i, _)| i)
    }

    /// A pool index to capture into: a free slot, else the LRU unpinned entry.
    /// `None` only when every slot is occupied by a pinned entry.
    fn reserve_victim(&self) -> Option<usize> {
        if let Some(i) = self.entries.iter().position(|e| e.is_none()) {
            return Some(i);
        }
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().filter(|e| !e.pinned).map(|e| (i, e.last_used)))
            .min_by_key(|&(_, lu)| lu)
            .map(|(i, _)| i)
    }

    /// Whether a leading prefix of `tokens` at length ≥ `p` is already cached
    /// (so a redundant re-snapshot can be skipped).
    fn has_at_least(&self, tokens: &[u32], p: u32) -> bool {
        self.entries
            .iter()
            .flatten()
            .any(|e| e.p >= p && lcp(&e.tokens, tokens) >= p as usize)
    }

    /// Drop every cached entry (pool buffers stay allocated). The bench uses
    /// this to start each run with a cold cache for comparable measurements.
    fn clear(&mut self) {
        for e in &mut self.entries {
            *e = None;
        }
        self.clock = 0;
    }
}

/// GPU-resident state owned by the worker thread for the whole process. A
/// single [`BatchKvCache`] holds N slabs; idle slabs keep their conversation's
/// prefix (cross-request reuse), while the active subset advances together in
/// one batched forward (continuous batching). The batched forward gathers each
/// active sequence's (arbitrary, possibly non-contiguous) slab via its
/// per-sequence slot index, so reuse and batching coexist.
struct Worker {
    engine: Engine,
    model: Box<dyn crate::models::Model>,
    batch: BatchKvCache,
    slots: Vec<SlotMeta>,
    active: Vec<ActiveSeq>,
    eog_ids: Vec<u32>,
    /// Monotonic LRU clock (incremented per admission; avoids wall-clock).
    clock: u64,
    /// Per-step token budget for the unified scheduler (`max_num_batched_tokens`
    /// in vLLM terms) — `= --ubatch-size` (bounds one forward to the scratch
    /// reservation). Decode tokens are scheduled first,
    /// then prefill chunks fill the remainder, so a long prefill is chunked
    /// across steps instead of head-of-line-blocking active decodes.
    max_batch_tokens: u32,
    /// Whether the model implements the unified varlen forward. When true the
    /// scheduler uses the token-budget / chunked-prefill loop; otherwise it
    /// falls back to serial-prefill + batched-decode.
    unified: bool,
    /// The vision tower (when an mmproj was resolved) — lets chat requests carry
    /// images. `None` → text-only serving.
    vision: Option<VisionCtx>,
    /// Bytes the scratch region is sized for, so an image prefill (single-pass,
    /// plus the vision tower's working set) can grow it on demand.
    scratch_bytes: u64,
    /// Leading-prefix snapshot cache (`SEEKER_PREFIX_CACHE`). `None` when the
    /// feature is off — every seed/capture branch then short-circuits.
    prefix_cache: Option<PrefixCache>,
}

/// Length of the longest common prefix of two token sequences.
fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Pure slot-selection logic (testable without GPU state). `slots` is
/// `(prior_tokens, last_used)` per slot. An explicit, in-range `id_slot` pins
/// directly. Otherwise prefer the slot whose `prior_tokens` is a non-empty
/// prefix of `new_tokens` (pure extension → SSM-safe in-place reuse), choosing
/// the longest such prior (most reuse, least re-prefill); failing that, evict
/// the least-recently-used slot (cold slots have `last_used == 0`, so they go
/// first).
fn choose_slot(slots: &[(&[u32], u64)], new_tokens: &[u32], id_slot: Option<usize>) -> usize {
    if let Some(i) = id_slot
        && i < slots.len()
    {
        return i;
    }
    let mut best: Option<(usize, usize)> = None; // (slot, prior_len)
    for (i, (prior, _)) in slots.iter().enumerate() {
        if prior.is_empty() {
            continue;
        }
        let common = lcp(prior, new_tokens);
        if common == prior.len() && best.is_none_or(|(_, blen)| prior.len() > blen) {
            best = Some((i, prior.len()));
        }
    }
    if let Some((i, _)) = best {
        return i;
    }
    slots
        .iter()
        .enumerate()
        .min_by_key(|(_, (_, last_used))| *last_used)
        .map(|(i, _)| i)
        .expect("at least one slot")
}

/// Build the engine + model + N cache slots, mirroring `seeker chat`'s load
/// sequence. Slots are allocated eagerly (fail-fast if N×ctx doesn't fit).
fn setup(cfg: &WorkerConfig) -> Result<Worker, Box<dyn Error>> {
    let gguf = GgufFile::open(&cfg.model_path)?;
    let bundle = build_tokenizer(&gguf)?;

    let mut engine = Engine::new(cfg.n_ubatch, cfg.n_batch)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened for serve");
    let weights = engine.upload_weights(&gguf)?;
    // Capture the uploaded weight bytes before the handle moves into the model;
    // the auto-slot budget subtracts it from the device memory.
    let weights_bytes = weights.total_bytes;
    let model = crate::models::open(&gguf, weights, bundle, /*spec_enabled=*/ false)?;

    // Scratch is sized per-ubatch/ctx and shared across slots (one forward runs
    // at a time), so it does not scale with the slot count.
    let scratch_bytes = model.scratch_bytes_estimate(
        cfg.n_ubatch,
        cfg.ctx_size,
        cfg.cache_type_k,
        cfg.cache_type_v,
    );
    engine.allocate_scratch(scratch_bytes)?;

    let dims = model.cache_dims();
    let cache_config = KvCacheConfig {
        k_dtype: cfg.cache_type_k,
        v_dtype: cfg.cache_type_v,
        max_seq_len: cfg.ctx_size,
        n_head: dims.n_head,
    };
    let ssm_dims = model.ssm_state_dims();
    // Resolve the slot count: an explicit `--parallel N` is honored verbatim
    // (fail-fast below if N×ctx doesn't fit); `0` auto-sizes from the device
    // memory budget (capped at `parallel_max`).
    let n_slots = resolve_n_slots(
        &engine.device,
        &dims,
        cache_config,
        ssm_dims.as_ref(),
        weights_bytes,
        scratch_bytes,
        cfg,
    );
    // One BatchKvCache with N slabs: idle slabs keep their conversation's
    // prefix (reuse); the active subset batches in one forward. Allocated
    // eagerly (fail-fast if N×ctx doesn't fit).
    let mut batch = BatchKvCache::new(
        &engine.device,
        dims.n_layer,
        dims.head_dim,
        dims.n_head_kv,
        n_slots,
        cache_config,
    )?;
    if let Some(ssm) = &ssm_dims {
        batch.allocate_ssm_state(
            &engine.device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
    }

    let mib = 1u64 << 20;
    tracing::info!(
        slots = n_slots,
        auto = (cfg.n_slots == 0),
        ctx = cfg.ctx_size,
        total_mib = batch.total_bytes() / mib,
        "batched kv cache allocated",
    );

    let eog_ids = model.tokenizer().eog_ids.clone();
    let unified = model.supports_unified();
    // Per-step token budget = n_ubatch: the scratch region is sized for one
    // n_ubatch-token forward (`scratch_bytes_estimate(n_ubatch, ...)` above), so
    // a unified step must not pack more than that or it overflows scratch.
    // (n_batch is the higher-level logical batch; n_ubatch is the per-forward
    // micro-batch — the same split llama.cpp uses.)
    let max_batch_tokens = cfg.n_ubatch.max(1);
    tracing::info!(
        unified,
        max_batch_tokens,
        "scheduler mode (unified = token-budget chunked prefill + decode mixing)"
    );
    // Build the vision tower if an mmproj sidecar was resolved (chat image
    // input). A load failure degrades to text-only rather than failing serve.
    let vision = match &cfg.mmproj_path {
        Some(path) => match build_vision(&engine, path) {
            Ok(v) => {
                tracing::info!(path = ?path, "vision tower loaded for serve (image input enabled)");
                Some(v)
            }
            Err(e) => {
                tracing::warn!(path = ?path, error = %e, "failed to load mmproj; serving text-only");
                None
            }
        },
        None => None,
    };

    let slots = (0..n_slots)
        .map(|_| SlotMeta {
            prior_tokens: Vec::new(),
            last_used: 0,
            active: false,
        })
        .collect();

    // Optional leading-prefix snapshot cache (default-off). Its pool memory was
    // already reserved out of the auto `--parallel` budget in `resolve_n_slots`.
    let prefix_cache = if *crate::runtime_flags::PREFIX_CACHE {
        let cap = crate::runtime_flags::PREFIX_CACHE_SLOTS.unwrap_or(2).max(1);
        let max_cached_len = crate::runtime_flags::PREFIX_CACHE_MAXLEN
            .unwrap_or_else(|| cfg.ctx_size.min(4096))
            .clamp(1, cfg.ctx_size.max(1));
        let ckpt = crate::runtime_flags::PREFIX_CACHE_CKPT
            .unwrap_or(512)
            .max(1);
        let p_min = crate::runtime_flags::PREFIX_CACHE_PMIN.unwrap_or(64).max(1);
        match PrefixCache::new(&batch, &engine.device, cap, max_cached_len, ckpt, p_min) {
            Ok(pc) => {
                tracing::info!(
                    slots = cap,
                    max_cached_len,
                    ckpt_stride = ckpt,
                    p_min,
                    pool_mib = pc.pool.iter().map(|s| s.total_bytes()).sum::<u64>() / mib,
                    "prefix cache enabled (leading-prefix seed/snapshot)"
                );
                Some(pc)
            }
            Err(e) => {
                tracing::warn!(error = %e, "prefix cache alloc failed; serving without it");
                None
            }
        }
    } else {
        None
    };

    Ok(Worker {
        engine,
        model,
        batch,
        slots,
        active: Vec::new(),
        eog_ids,
        clock: 0,
        max_batch_tokens,
        unified,
        vision,
        scratch_bytes,
        prefix_cache,
    })
}

/// Resolve the KV-slot count. An explicit `cfg.n_slots >= 1` is returned as-is
/// (the eager `BatchKvCache::new` fail-fasts if it doesn't fit). `0` auto-sizes:
/// fit as many full-context slots as `mem_fraction` of DEVICE_LOCAL memory
/// allows after weights + scratch, clamped to `[1, parallel_max]`. Rounds down
/// and fail-soft to 1 — on unified DDR5 (DEVICE_LOCAL == system RAM) an
/// over-commit would OOM the box, not just a discrete GPU.
fn resolve_n_slots(
    device: &crate::inference::device::Device,
    dims: &crate::models::CacheDims,
    cache_config: KvCacheConfig,
    ssm_dims: Option<&crate::models::SsmStateDims>,
    weights_bytes: u64,
    scratch_bytes: u64,
    cfg: &WorkerConfig,
) -> u32 {
    if cfg.n_slots >= 1 {
        return cfg.n_slots;
    }
    let parallel_max = cfg.parallel_max.max(1);
    // Per-slot bytes via a throwaway 1-slot cache: exact (no estimator that
    // could drift from the allocator), and dropped before the real N-slot
    // allocation so it adds no memory peak.
    let per_slot = match probe_per_slot_bytes(device, dims, cache_config, ssm_dims) {
        Ok(b) if b > 0 => b,
        _ => {
            tracing::warn!("auto --parallel: could not size a KV slot; falling back to 1");
            return 1;
        }
    };
    let heap = device_local_bytes(device);
    let frac = cfg.mem_fraction.clamp(0.1, 1.0) as f64;
    let budget = (heap as f64 * frac) as u64;
    // Reserve the leading-prefix snapshot pool out of the budget so N×ctx + pool
    // still fits (each pool entry ≈ one sequence's SSM + `max_cached_len` of KV).
    let pool_bytes = if *crate::runtime_flags::PREFIX_CACHE {
        let slots = crate::runtime_flags::PREFIX_CACHE_SLOTS.unwrap_or(2).max(1) as u64;
        let max_cached_len = crate::runtime_flags::PREFIX_CACHE_MAXLEN
            .unwrap_or_else(|| cache_config.max_seq_len.min(4096));
        probe_prefix_snapshot_bytes(device, dims, cache_config, ssm_dims, max_cached_len)
            .unwrap_or(0)
            * slots
    } else {
        0
    };
    let avail = budget.saturating_sub(
        weights_bytes
            .saturating_add(scratch_bytes)
            .saturating_add(pool_bytes),
    );
    let n = ((avail / per_slot) as u32).clamp(1, parallel_max);
    if n < parallel_max {
        tracing::warn!(
            resolved = n,
            parallel_max,
            per_slot_mib = per_slot / (1 << 20),
            heap_mib = heap / (1 << 20),
            weights_mib = weights_bytes / (1 << 20),
            "auto --parallel: memory-bound below parallel-max; lower --ctx-size or KV quant for more slots"
        );
    } else {
        tracing::info!(resolved = n, "auto --parallel resolved");
    }
    n
}

/// Size of one full-context KV(+SSM) slot, measured by allocating and
/// immediately dropping a 1-slot cache. Exact, and the alloc is freed before
/// the real N-slot allocation so it adds no peak.
fn probe_per_slot_bytes(
    device: &crate::inference::device::Device,
    dims: &crate::models::CacheDims,
    cache_config: KvCacheConfig,
    ssm_dims: Option<&crate::models::SsmStateDims>,
) -> Result<u64, Box<dyn Error>> {
    let mut probe = BatchKvCache::new(
        device,
        dims.n_layer,
        dims.head_dim,
        dims.n_head_kv,
        1,
        cache_config,
    )?;
    if let Some(ssm) = ssm_dims {
        probe.allocate_ssm_state(
            device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
    }
    Ok(probe.total_bytes())
}

/// Bytes one prefix-cache snapshot occupies (KV mini-slabs for `max_cached_len`
/// tokens + one sequence's SSM state), measured exactly via a throwaway 1-slot
/// cache + snapshot that is dropped before the real allocation.
fn probe_prefix_snapshot_bytes(
    device: &crate::inference::device::Device,
    dims: &crate::models::CacheDims,
    cache_config: KvCacheConfig,
    ssm_dims: Option<&crate::models::SsmStateDims>,
    max_cached_len: u32,
) -> Result<u64, Box<dyn Error>> {
    let mut probe = BatchKvCache::new(
        device,
        dims.n_layer,
        dims.head_dim,
        dims.n_head_kv,
        1,
        cache_config,
    )?;
    if let Some(ssm) = ssm_dims {
        probe.allocate_ssm_state(
            device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
    }
    Ok(probe
        .new_prefix_snapshot(device, max_cached_len)?
        .total_bytes())
}

/// Total bytes across DEVICE_LOCAL memory heaps (≈ system RAM on the unified APU).
fn device_local_bytes(device: &crate::inference::device::Device) -> u64 {
    let mp = &device.mem_props;
    let mut total = 0u64;
    for i in 0..mp.memory_heap_count as usize {
        let heap = mp.memory_heaps[i];
        if heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
            total = total.saturating_add(heap.size);
        }
    }
    total
}

/// Build the vision tower from an mmproj GGUF (upload weights, build encoder +
/// host-side patch-embed copies). Mirrors `seeker chat`'s lazy `/image` build.
fn build_vision(engine: &Engine, mmproj_path: &Path) -> Result<VisionCtx, Box<dyn Error>> {
    let gguf = GgufFile::open(mmproj_path)?;
    let weights = engine.upload_weights(&gguf)?;
    let cfg = crate::vision::parse_config(&gguf)?;
    // gemma4uv is a "no-tower" projector with no transformer blocks; skip the
    // tower encoder (serve image input for gemma4uv isn't wired yet) but still
    // load the mmproj so its gemma4ua audio encoder is available.
    let (encoder, host_weights) = if cfg.projector_type == crate::vision::ProjectorType::Gemma4Uv {
        (None, None)
    } else {
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
    let audio_cfg = crate::audio::parse_config(&gguf).ok();
    let vision = crate::vision::VisionModel {
        config: cfg,
        weights,
    };
    Ok(VisionCtx {
        vision,
        encoder,
        host_weights,
        audio_cfg,
    })
}

/// Worker entry point. Loads the model, signals readiness, then runs the
/// continuous-batching scheduler until every `InferenceHandle` is dropped.
fn worker_main(
    mut cfg: WorkerConfig,
    mut jobs: mpsc::Receiver<GenJob>,
    ready: oneshot::Sender<Result<u32, String>>,
) {
    let mut worker = match setup(&cfg) {
        Ok(w) => w,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    // Report the resolved slot count (auto-sizing may differ from the request).
    if ready.send(Ok(worker.slots.len() as u32)).is_err() {
        // `serve::run` gave up waiting (process exiting). Nothing to serve.
        return;
    }

    // Pin the configured system-prompt prefix: prefill it once and keep it in
    // the cache un-evictably, so requests beginning with it seed instead of
    // re-prefilling it. After readiness (doesn't delay startup) and before the
    // first job is served (single-threaded loop, so no race).
    if let Some(tokens) = cfg.pin_prefix_tokens.take() {
        worker.pin_prefix(tokens);
    }

    // Scheduler loop: admit + prefill queued jobs onto free slabs, advance the
    // whole active set one token in a single batched forward, stream + reap
    // finishers. `blocking_recv` is safe here (plain OS thread, not in tokio);
    // `None` ⇒ all senders dropped ⇒ shut down (Engine drops on return).
    let unified = worker.unified;
    loop {
        if worker.active.is_empty() {
            // Idle: block for the next job (or shutdown).
            match jobs.blocking_recv() {
                // Media jobs prefill single-pass (the splice can't be chunked),
                // regardless of the global unified flag.
                Some(job) if job.image.is_some() => worker.admit_image(job),
                Some(job) if job.audio.is_some() => worker.admit_audio(job),
                Some(job) if unified => worker.admit_unified(job),
                Some(job) => worker.admit(job),
                None => return,
            }
        }
        // Drain queued jobs into any free slabs without blocking the step.
        while worker.free_slabs() > 0 {
            match jobs.try_recv() {
                Ok(job) if job.image.is_some() => worker.admit_image(job),
                Ok(job) if job.audio.is_some() => worker.admit_audio(job),
                Ok(job) if unified => worker.admit_unified(job),
                Ok(job) => worker.admit(job),
                Err(_) => break, // empty or all-senders-dropped
            }
        }
        if !worker.active.is_empty() {
            if unified {
                worker.schedule_step();
            } else {
                worker.decode_step();
            }
            worker.evict_finished();
        }
    }
}

impl Worker {
    /// Number of slabs not currently owned by an active sequence.
    fn free_slabs(&self) -> usize {
        self.slots.iter().filter(|s| !s.active).count()
    }

    /// Pick a *free* slab to serve `new_tokens` — prefix-reuse, else LRU (see
    /// [`choose_slot`]) — or `None` if every slab is busy. An in-range, free
    /// `id_slot` pins directly.
    fn select_free_slab(&self, new_tokens: &[u32], id_slot: Option<usize>) -> Option<usize> {
        if let Some(i) = id_slot
            && i < self.slots.len()
            && !self.slots[i].active
        {
            return Some(i);
        }
        let free: Vec<(usize, &[u32], u64)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.active)
            .map(|(i, s)| (i, s.prior_tokens.as_slice(), s.last_used))
            .collect();
        if free.is_empty() {
            return None;
        }
        let view: Vec<(&[u32], u64)> = free.iter().map(|&(_, p, l)| (p, l)).collect();
        let pick = choose_slot(&view, new_tokens, None);
        Some(free[pick].0)
    }

    /// Admit one job: select a free slab, prefill it (with prefix-reuse), and
    /// push it to the active set with its first sampled token. Any failure
    /// (no slab / oversized prompt / GPU error) sends a terminal frame and
    /// releases the slab; the worker stays alive for the next job.
    fn admit(&mut self, job: GenJob) {
        let GenJob {
            tokens: new_tokens,
            config,
            image: _, // text path: media jobs are routed to admit_image/admit_audio
            audio: _,
            reply,
        } = job;
        let GenConfig {
            sampler: cfg,
            max_tokens,
            stop,
            ignore_eos,
            id_slot,
        } = config;

        let prompt_tokens = new_tokens.len() as u32;
        if new_tokens.is_empty() {
            let _ =
                reply.blocking_send(GenEvent::Error("empty prompt — nothing to generate".into()));
            return;
        }
        let Some(idx) = self.select_free_slab(&new_tokens, id_slot) else {
            // Caller guards on free_slabs(), so this is defensive only.
            let _ = reply.blocking_send(GenEvent::Error("no free cache slot available".into()));
            return;
        };
        let ctx = self.batch.config.max_seq_len;
        if new_tokens.len() as u32 >= ctx {
            let _ = reply.blocking_send(GenEvent::Error(format!(
                "prompt is {} tokens but --ctx-size is {ctx} (no room to generate) — \
                 raise --ctx-size or shorten the prompt",
                new_tokens.len()
            )));
            return;
        }
        self.clock += 1;
        self.slots[idx].last_used = self.clock;

        // ── Safe prefix-reuse (per slab) ───────────────────────────────────
        // Reuse the cached prefix ONLY when this prompt strictly extends what
        // the slab holds (and the slab position agrees). Any divergence falls
        // back to a full reset + prefill — the only SSM/GDN-safe rewind.
        let common = lcp(&self.slots[idx].prior_tokens, &new_tokens);
        let cache_pos = self.batch.positions[idx];
        let pure_extension = common > 0
            && common == self.slots[idx].prior_tokens.len()
            && common == cache_pos as usize;
        let (start_pos, delta): (u32, Vec<u32>) = if pure_extension {
            if common < new_tokens.len() {
                (common as u32, new_tokens[common..].to_vec())
            } else {
                // Whole prompt already cached: rewind one and re-feed the last
                // token so there are logits to sample from.
                (common as u32 - 1, vec![*new_tokens.last().unwrap()])
            }
        } else {
            // Divergent → fresh prefill: zero just this slab's SSM state.
            self.batch.reset_slot(idx as u32);
            self.batch.positions[idx] = 0;
            (0, new_tokens.clone())
        };
        tracing::debug!(
            slab = idx,
            prompt = new_tokens.len(),
            reused = start_pos,
            prefill = delta.len(),
            "serve: admit + prefix reuse",
        );

        // ── Prefill via the borrowed single-slab cache ─────────────────────
        // Invalidate the engine's single-seq decode-replay cmdbuf — it binds a
        // specific slab's buffers, and the next batched forward drops it too.
        self.engine.decode_cache = None;
        let mut sampler = Sampler::new(cfg);
        let first = {
            let mut sc = self.batch.slot_kvcache(idx as u32);
            sc.position = start_pos;
            match self.engine.forward_sampled(
                &*self.model,
                &mut sc,
                &delta,
                start_pos,
                &mut sampler,
            ) {
                Ok(t) => {
                    self.batch.positions[idx] = sc.position;
                    t
                }
                Err(e) => {
                    // Slab/prior may be inconsistent after a GPU error — reset.
                    self.batch.reset_slot(idx as u32);
                    self.batch.positions[idx] = 0;
                    self.slots[idx].prior_tokens.clear();
                    let _ = reply.blocking_send(GenEvent::Error(e.to_string()));
                    return;
                }
            }
        };
        self.slots[idx].active = true;

        if reply
            .blocking_send(GenEvent::Started { prompt_tokens })
            .is_err()
        {
            // Client gone before we streamed anything — release the slab but
            // keep its cache for a future prefix-reuse.
            self.slots[idx].active = false;
            self.slots[idx].prior_tokens = new_tokens;
            return;
        }

        let mut seq = ActiveSeq {
            slab: idx as u32,
            sampler,
            prompt: new_tokens,
            generated: Vec::new(),
            prompt_tokens,
            stop,
            ignore_eos,
            max_tokens,
            reply,
            stream_ids: Vec::new(),
            stream_prefix: String::new(),
            stream_prefix_index: 0,
            pending: String::new(),
            last_token: first,
            ctx,
            terminal: None,
            disconnected: false,
            // Legacy path: admit() already prefilled + sent Started, so the
            // cache holds the whole prompt and num_computed mirrors it. (Unused
            // by the legacy decode_step; kept consistent for evict bookkeeping.)
            num_computed: self.batch.positions[idx],
            started: true,
        };
        // Stream the first (prefill) token now; it's fed back at the first
        // batched step. The rest advance in `decode_step`.
        process_token(
            &self.model.tokenizer().tokenizer,
            &self.eog_ids,
            &mut seq,
            first,
        );
        self.active.push(seq);
    }

    /// Unified-path admission: pick a free slab and apply safe prefix-reuse to
    /// set the starting cache position (`num_computed`), then push the request
    /// to the active set WITHOUT prefilling. The prefill runs (chunked) in
    /// [`Self::schedule_step`] alongside other requests' decode, so a long
    /// prompt never blocks the worker. Failure → terminal frame + slab released.
    fn admit_unified(&mut self, job: GenJob) {
        let GenJob {
            tokens: new_tokens,
            config,
            image: _,
            audio: _,
            reply,
        } = job;
        let GenConfig {
            sampler: cfg,
            max_tokens,
            stop,
            ignore_eos,
            id_slot,
        } = config;
        let prompt_tokens = new_tokens.len() as u32;
        if new_tokens.is_empty() {
            let _ =
                reply.blocking_send(GenEvent::Error("empty prompt — nothing to generate".into()));
            return;
        }
        let Some(idx) = self.select_free_slab(&new_tokens, id_slot) else {
            let _ = reply.blocking_send(GenEvent::Error("no free cache slot available".into()));
            return;
        };
        let ctx = self.batch.config.max_seq_len;
        if prompt_tokens >= ctx {
            let _ = reply.blocking_send(GenEvent::Error(format!(
                "prompt is {prompt_tokens} tokens but --ctx-size is {ctx} (no room to generate) — \
                 raise --ctx-size or shorten the prompt"
            )));
            return;
        }
        self.clock += 1;
        self.slots[idx].last_used = self.clock;

        // Safe prefix-reuse: reuse the cached prefix only on a pure extension
        // (same SSM/GDN-safe rule as the single-seq path), else reset.
        let common = lcp(&self.slots[idx].prior_tokens, &new_tokens);
        let cache_pos = self.batch.positions[idx];
        let pure_extension = common > 0
            && common == self.slots[idx].prior_tokens.len()
            && common == cache_pos as usize;
        let num_computed = if pure_extension {
            // Reuse [0, common); prefill [common, prompt). If the whole prompt
            // is cached, rewind one so there's a last token to sample from.
            if common < new_tokens.len() {
                common as u32
            } else {
                common as u32 - 1
            }
        } else if let Some(p_seed) = self.try_seed_prefix(idx, &new_tokens, prompt_tokens) {
            // No live-slab pure extension, but the leading-prefix cache holds a
            // shared prefix: seed KV[0,P)+SSM-at-P GPU→GPU and prefill only the
            // divergent suffix [p_seed, prompt). (seed_slab already overwrote the
            // slab's SSM state, so no reset_slot is needed.)
            p_seed
        } else {
            self.batch.reset_slot(idx as u32);
            0
        };
        self.batch.positions[idx] = num_computed;
        self.slots[idx].active = true;
        tracing::debug!(
            slab = idx,
            prompt = prompt_tokens,
            reused = num_computed,
            "serve: admit (unified)"
        );

        self.active.push(ActiveSeq {
            slab: idx as u32,
            sampler: Sampler::new(cfg),
            prompt: new_tokens,
            generated: Vec::new(),
            prompt_tokens,
            stop,
            ignore_eos,
            max_tokens,
            reply,
            stream_ids: Vec::new(),
            stream_prefix: String::new(),
            stream_prefix_index: 0,
            pending: String::new(),
            last_token: 0,
            ctx,
            terminal: None,
            disconnected: false,
            num_computed,
            started: false,
        });
    }

    /// On a cross-request leading-prefix cache hit, seed `slab` from the cached
    /// snapshot (GPU→GPU copy of `KV[0, P)` + the SSM state at P) and return the
    /// seeded position `p_seed`, so the caller prefills only `[p_seed, prompt)`.
    /// `None` ⇒ miss / below `p_min` / cache disabled, so the caller does a
    /// fresh prefill. Text-only: the seed forces `rope_lag = 0`.
    fn try_seed_prefix(
        &mut self,
        slab: usize,
        new_tokens: &[u32],
        prompt_tokens: u32,
    ) -> Option<u32> {
        // Decide (immutable phase); pull out scalars so no borrow is held across
        // the `&mut self.engine` seed call below.
        let (pool_idx, p_seed) = {
            let pc = self.prefix_cache.as_ref()?;
            let ei = pc.lookup(new_tokens)?;
            let p = pc.entries[ei].as_ref()?.p;
            // Leave ≥1 suffix token so the first sampled logit comes from a real
            // forward (`p ≤ prompt_tokens`, since the entry is a prefix of it).
            let p_seed = p.min(prompt_tokens.saturating_sub(1));
            if p_seed < pc.p_min {
                return None;
            }
            (ei, p_seed)
        };
        // GPU→GPU seed (disjoint field borrows: &mut engine, &batch, &pool slot).
        {
            let snap = &self.prefix_cache.as_ref().unwrap().pool[pool_idx];
            if let Err(e) = self
                .engine
                .seed_slab(&self.batch, snap, slab as u32, p_seed)
            {
                tracing::warn!(error = %e, "prefix-cache seed failed; full prefill");
                return None;
            }
        }
        self.batch.rope_lag[slab] = 0;
        let pc = self.prefix_cache.as_mut().unwrap();
        pc.clock += 1;
        if let Some(e) = pc.entries[pool_idx].as_mut() {
            e.last_used = pc.clock;
        }
        tracing::debug!(slab, p_seed, "serve: prefix-cache seed");
        Some(p_seed)
    }

    /// Snapshot the live prefix `[0, p)` of `slab` into the cache (a sparse
    /// checkpoint during prefill), so a later request sharing this leading
    /// prefix can seed from it. Requires `positions[slab] == p` (a chunk
    /// boundary, where the slab's live SSM state equals state-at-P) and a
    /// text-only slab. Best-effort; failures are logged, not fatal.
    fn capture_checkpoint(&mut self, slab: u32, p: u32, active_idx: usize, pinned: bool) {
        if self.batch.rope_lag[slab as usize] != 0 {
            return;
        }
        // Decide whether + where to capture (immutable phase).
        let pool_idx = {
            let Some(pc) = self.prefix_cache.as_ref() else {
                return;
            };
            if p < pc.p_min || p > pc.max_cached_len {
                return;
            }
            let prompt = &self.active[active_idx].prompt;
            if (p as usize) > prompt.len() {
                return;
            }
            if pc.has_at_least(&prompt[..p as usize], p) {
                return;
            }
            match pc.reserve_victim() {
                Some(i) => i,
                None => return,
            }
        };
        // GPU→GPU capture (disjoint field borrows: &mut engine, &batch, &pool slot).
        {
            let snap = &self.prefix_cache.as_ref().unwrap().pool[pool_idx];
            if let Err(e) = self.engine.capture_prefix(&self.batch, snap, slab, p) {
                tracing::warn!(error = %e, "prefix-cache capture failed");
                return;
            }
        }
        let tokens = self.active[active_idx].prompt[..p as usize].to_vec();
        let pc = self.prefix_cache.as_mut().unwrap();
        pc.clock += 1;
        pc.entries[pool_idx] = Some(PrefixEntry {
            tokens,
            p,
            last_used: pc.clock,
            pinned,
        });
        tracing::debug!(slab, p, "serve: prefix-cache capture");
    }

    /// Encode an image through the vision tower (GPU) → `[proj_dim, n_tok]` host
    /// f32 (column = merged token). Grows the scratch for the tower's working
    /// set first. Errors if no vision tower is loaded.
    fn encode_image(&mut self, image: &ServeImage) -> Result<Vec<f32>, Box<dyn Error>> {
        if self.vision.is_none() {
            return Err("no vision model loaded (mmproj)".into());
        }
        let n_pos = (image.pimg.grid_w as u64) * (image.pimg.grid_h as u64);
        // Mirror `vision_scratch_estimate` (commands::run / chat): `encode_image`
        // reclaims each stage's scratch between layers, so the working set is
        // O(n_pos) — ~28k floats/token for the largest stage, budget 40k (margin
        // + long-KV flash-attn split-K partials ~3k floats/token).
        let need = (40_000u64 * n_pos * 4).max(64 << 20);
        if need > self.scratch_bytes {
            self.engine.allocate_scratch(need)?;
            self.scratch_bytes = need;
        }
        let vc = self
            .vision
            .as_ref()
            .expect("vision present (checked above)");
        let weights = &vc.vision.weights;
        let encoder = vc.encoder.as_ref().ok_or(
            "this mmproj has no vision tower (gemma4uv image input is not supported in serve)",
        )?;
        let host_weights = vc
            .host_weights
            .as_ref()
            .ok_or("vision tower host weights missing")?;
        let pimg = &image.pimg;
        crate::vision::encoder::encode_image_chunked(
            &mut self.engine,
            weights,
            encoder,
            pimg,
            host_weights,
        )
    }

    /// Encode an audio clip through the gemma4ua projector (GPU) → `[proj_dim,
    /// n_tok]` host f32. Grows the scratch first. Errors if the mmproj has no
    /// audio encoder.
    fn encode_audio(&mut self, audio: &ServeAudio) -> Result<Vec<f32>, Box<dyn Error>> {
        let acfg = self
            .vision
            .as_ref()
            .and_then(|vc| vc.audio_cfg.clone())
            .ok_or("no audio model loaded (mmproj has no audio encoder)")?;
        let need = (40_000u64 * audio.n_tok as u64 * 4).max(64 << 20);
        if need > self.scratch_bytes {
            self.engine.allocate_scratch(need)?;
            self.scratch_bytes = need;
        }
        let weights = &self
            .vision
            .as_ref()
            .expect("vision present (checked above)")
            .vision
            .weights;
        let (embeddings, _n_tok) = crate::audio::encoder::encode_audio_gemma4(
            &mut self.engine,
            weights,
            &acfg,
            &audio.samples,
        )?;
        Ok(embeddings)
    }

    /// Admit an image chat request: encode the image, prefill the whole prompt
    /// single-pass with the embeddings spliced (`forward_image_sampled`), record
    /// the slab's M-RoPE lag, sample the first token, and push it to the active
    /// set. Decode then proceeds in the normal batched/unified step — the
    /// per-slab `rope_lag` keeps its positions continuous past the image. No
    /// prefix-reuse (always a fresh prefill) for image turns in this first cut.
    fn admit_image(&mut self, job: GenJob) {
        let GenJob {
            tokens: new_tokens,
            config,
            image,
            audio: _,
            reply,
        } = job;
        let GenConfig {
            sampler: cfg,
            max_tokens,
            stop,
            ignore_eos,
            id_slot,
        } = config;
        let Some(image) = image else { return }; // only routed here when Some
        if self.vision.is_none() {
            let _ = reply.blocking_send(GenEvent::Error(
                "this server has no vision model (mmproj); image requests are unsupported".into(),
            ));
            return;
        }
        let prompt_tokens = new_tokens.len() as u32;
        if new_tokens.is_empty() {
            let _ =
                reply.blocking_send(GenEvent::Error("empty prompt — nothing to generate".into()));
            return;
        }
        let ctx = self.batch.config.max_seq_len;
        if prompt_tokens >= ctx {
            let _ = reply.blocking_send(GenEvent::Error(format!(
                "prompt is {prompt_tokens} tokens but --ctx-size is {ctx} (no room to generate) — \
                 raise --ctx-size or shorten the prompt"
            )));
            return;
        }
        if image.image_start + image.nx * image.ny > new_tokens.len() {
            let _ = reply.blocking_send(GenEvent::Error("image span overruns the prompt".into()));
            return;
        }
        let Some(idx) = self.select_free_slab(&new_tokens, id_slot) else {
            let _ = reply.blocking_send(GenEvent::Error("no free cache slot available".into()));
            return;
        };
        self.clock += 1;
        self.slots[idx].last_used = self.clock;

        // Encode the image (grows scratch for the tower), then ensure scratch
        // fits the single-pass decoder prefill over the whole prompt.
        let embeddings = match self.encode_image(&image) {
            Ok(e) => e,
            Err(e) => {
                let _ = reply.blocking_send(GenEvent::Error(e.to_string()));
                return;
            }
        };
        let need = self.model.scratch_bytes_estimate(
            /*n_ubatch=*/ 0,
            prompt_tokens,
            self.batch.config.k_dtype,
            self.batch.config.v_dtype,
        );
        if need > self.scratch_bytes {
            if let Err(e) = self.engine.allocate_scratch(need) {
                let _ = reply.blocking_send(GenEvent::Error(e.to_string()));
                return;
            }
            self.scratch_bytes = need;
        }

        // Fresh full prefill (no prefix reuse for images).
        self.batch.reset_slot(idx as u32);
        self.batch.positions[idx] = 0;
        self.batch.rope_lag[idx] = 0;
        self.engine.decode_cache = None;

        let mut sampler = Sampler::new(cfg);
        let first = {
            let mut sc = self.batch.slot_kvcache(idx as u32);
            sc.position = 0;
            match self.engine.forward_image_sampled(
                &*self.model,
                &mut sc,
                &new_tokens,
                &embeddings,
                image.image_start,
                image.nx,
                image.ny,
                &mut sampler,
            ) {
                Ok(t) => {
                    self.batch.positions[idx] = sc.position;
                    self.batch.rope_lag[idx] = sc.rope_position_lag;
                    t
                }
                Err(e) => {
                    self.batch.reset_slot(idx as u32);
                    self.batch.positions[idx] = 0;
                    self.batch.rope_lag[idx] = 0;
                    self.slots[idx].prior_tokens.clear();
                    let _ = reply.blocking_send(GenEvent::Error(e.to_string()));
                    return;
                }
            }
        };
        self.slots[idx].active = true;

        if reply
            .blocking_send(GenEvent::Started { prompt_tokens })
            .is_err()
        {
            self.slots[idx].active = false;
            self.slots[idx].prior_tokens = new_tokens;
            return;
        }

        let mut seq = ActiveSeq {
            slab: idx as u32,
            sampler,
            prompt: new_tokens,
            generated: Vec::new(),
            prompt_tokens,
            stop,
            ignore_eos,
            max_tokens,
            reply,
            stream_ids: Vec::new(),
            stream_prefix: String::new(),
            stream_prefix_index: 0,
            pending: String::new(),
            last_token: first,
            ctx,
            terminal: None,
            disconnected: false,
            num_computed: self.batch.positions[idx],
            started: true,
        };
        process_token(
            &self.model.tokenizer().tokenizer,
            &self.eog_ids,
            &mut seq,
            first,
        );
        self.active.push(seq);
    }

    /// Admit an audio chat request — the gemma4ua analog of [`Self::admit_image`].
    /// Encodes the clip, prefills the whole prompt with the embeddings spliced
    /// over the `<|audio|>` rows (`forward_audio_sampled`, plain 1D positions →
    /// no rope lag), samples the first token, and pushes it to the active set.
    fn admit_audio(&mut self, job: GenJob) {
        let GenJob {
            tokens: new_tokens,
            config,
            image: _,
            audio,
            reply,
        } = job;
        let GenConfig {
            sampler: cfg,
            max_tokens,
            stop,
            ignore_eos,
            id_slot,
        } = config;
        let Some(audio) = audio else { return }; // only routed here when Some
        if self
            .vision
            .as_ref()
            .and_then(|vc| vc.audio_cfg.as_ref())
            .is_none()
        {
            let _ = reply.blocking_send(GenEvent::Error(
                "this server has no audio model (mmproj audio encoder); audio requests are \
                 unsupported"
                    .into(),
            ));
            return;
        }
        let prompt_tokens = new_tokens.len() as u32;
        if new_tokens.is_empty() {
            let _ =
                reply.blocking_send(GenEvent::Error("empty prompt — nothing to generate".into()));
            return;
        }
        let ctx = self.batch.config.max_seq_len;
        if prompt_tokens >= ctx {
            let _ = reply.blocking_send(GenEvent::Error(format!(
                "prompt is {prompt_tokens} tokens but --ctx-size is {ctx} (no room to generate) — \
                 raise --ctx-size or shorten the prompt"
            )));
            return;
        }
        if audio.audio_start + audio.n_tok > new_tokens.len() {
            let _ = reply.blocking_send(GenEvent::Error("audio span overruns the prompt".into()));
            return;
        }
        let Some(idx) = self.select_free_slab(&new_tokens, id_slot) else {
            let _ = reply.blocking_send(GenEvent::Error("no free cache slot available".into()));
            return;
        };
        self.clock += 1;
        self.slots[idx].last_used = self.clock;

        // Encode the audio (grows scratch for the encoder), then ensure scratch
        // fits the single-pass decoder prefill over the whole prompt.
        let embeddings = match self.encode_audio(&audio) {
            Ok(e) => e,
            Err(e) => {
                let _ = reply.blocking_send(GenEvent::Error(e.to_string()));
                return;
            }
        };
        let need = self.model.scratch_bytes_estimate(
            /*n_ubatch=*/ 0,
            prompt_tokens,
            self.batch.config.k_dtype,
            self.batch.config.v_dtype,
        );
        if need > self.scratch_bytes {
            if let Err(e) = self.engine.allocate_scratch(need) {
                let _ = reply.blocking_send(GenEvent::Error(e.to_string()));
                return;
            }
            self.scratch_bytes = need;
        }

        // Fresh full prefill (no prefix reuse for audio).
        self.batch.reset_slot(idx as u32);
        self.batch.positions[idx] = 0;
        self.batch.rope_lag[idx] = 0;
        self.engine.decode_cache = None;

        let mut sampler = Sampler::new(cfg);
        let first = {
            let mut sc = self.batch.slot_kvcache(idx as u32);
            sc.position = 0;
            match self.engine.forward_audio_sampled(
                &*self.model,
                &mut sc,
                &new_tokens,
                &embeddings,
                audio.audio_start,
                audio.n_tok,
                &mut sampler,
            ) {
                Ok(t) => {
                    self.batch.positions[idx] = sc.position;
                    self.batch.rope_lag[idx] = sc.rope_position_lag;
                    t
                }
                Err(e) => {
                    self.batch.reset_slot(idx as u32);
                    self.batch.positions[idx] = 0;
                    self.batch.rope_lag[idx] = 0;
                    self.slots[idx].prior_tokens.clear();
                    let _ = reply.blocking_send(GenEvent::Error(e.to_string()));
                    return;
                }
            }
        };
        self.slots[idx].active = true;

        if reply
            .blocking_send(GenEvent::Started { prompt_tokens })
            .is_err()
        {
            self.slots[idx].active = false;
            self.slots[idx].prior_tokens = new_tokens;
            return;
        }

        let mut seq = ActiveSeq {
            slab: idx as u32,
            sampler,
            prompt: new_tokens,
            generated: Vec::new(),
            prompt_tokens,
            stop,
            ignore_eos,
            max_tokens,
            reply,
            stream_ids: Vec::new(),
            stream_prefix: String::new(),
            stream_prefix_index: 0,
            pending: String::new(),
            last_token: first,
            ctx,
            terminal: None,
            disconnected: false,
            num_computed: self.batch.positions[idx],
            started: true,
        };
        process_token(
            &self.model.tokenizer().tokenizer,
            &self.eog_ids,
            &mut seq,
            first,
        );
        self.active.push(seq);
    }

    /// One unified scheduler step: assign each active sequence a slice of the
    /// per-step token budget — **decode tokens first** (so they never starve),
    /// then **prefill chunks fill the remainder** — and run a single
    /// `forward_unified`. A long prefill is chunked across steps, mixing with
    /// other requests' decode in each forward (continuous batching, vLLM-style).
    /// After the forward, sequences that caught up to their logical end sample +
    /// stream their next token; mid-prefill chunks discard the (unused) column.
    fn schedule_step(&mut self) {
        let ctx = self.batch.config.max_seq_len;
        let logical_len = |s: &ActiveSeq| s.prompt.len() + s.generated.len();

        // Mark ctx-full sequences terminal (no room to write the next token).
        for seq in self.active.iter_mut() {
            if seq.terminal.is_none() && !seq.disconnected && seq.num_computed + 1 >= ctx {
                seq.terminal = Some(StopReason::ContextFull);
            }
        }

        // Token budget: decodes / final-prefill-tokens (remaining == 1) first,
        // then prefill chunks (remaining > 1) fill whatever's left.
        let mut budget = self.max_batch_tokens as usize;
        let mut parts: Vec<(usize, u32)> = Vec::new(); // (active idx, num_new)
        for (i, seq) in self.active.iter().enumerate() {
            if seq.terminal.is_some() || seq.disconnected || budget == 0 {
                continue;
            }
            if logical_len(seq) - seq.num_computed as usize == 1 {
                parts.push((i, 1));
                budget -= 1;
            }
        }
        for (i, seq) in self.active.iter().enumerate() {
            if seq.terminal.is_some() || seq.disconnected || budget == 0 {
                continue;
            }
            let rem = logical_len(seq) - seq.num_computed as usize;
            if rem > 1 {
                let nn = (rem.min(budget)) as u32;
                parts.push((i, nn));
                budget -= nn as usize;
            }
        }
        if parts.is_empty() {
            return;
        }
        parts.sort_by_key(|&(i, _)| i); // ascending → aligns with sampler gather

        // Build the flat packed batch (tokens / positions per token; seq_lens /
        // slots per sequence).
        let mut tokens: Vec<u32> = Vec::new();
        let mut positions: Vec<u32> = Vec::new();
        let mut seq_lens: Vec<u32> = Vec::new();
        let mut slots: Vec<u32> = Vec::new();
        for &(i, nn) in &parts {
            let seq = &self.active[i];
            for off in 0..nn {
                let li = (seq.num_computed + off) as usize;
                let tok = if li < seq.prompt.len() {
                    seq.prompt[li]
                } else {
                    seq.generated[li - seq.prompt.len()]
                };
                tokens.push(tok);
                // The `positions` arg is the M-RoPE rope base; the forward takes
                // KV write offset + kv_len from `batch.positions[slot]` instead.
                // For an image slot the rope cursor trails the KV-slot count by
                // `rope_lag`; text slots have lag 0 (unchanged).
                positions.push(
                    (seq.num_computed + off).saturating_sub(self.batch.rope_lag[seq.slab as usize]),
                );
            }
            seq_lens.push(nn);
            slots.push(seq.slab);
        }

        let part_idxs: std::collections::HashSet<usize> = parts.iter().map(|&(i, _)| i).collect();
        let mut samplers: Vec<&mut Sampler> = self
            .active
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| part_idxs.contains(i))
            .map(|(_, s)| &mut s.sampler)
            .collect();

        let toks = match self.engine.forward_unified(
            &*self.model,
            &mut self.batch,
            &tokens,
            &positions,
            &seq_lens,
            &slots,
            &mut samplers,
        ) {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                for &(i, _) in &parts {
                    let _ = self.active[i]
                        .reply
                        .blocking_send(GenEvent::Error(msg.clone()));
                    self.active[i].disconnected = true;
                    self.active[i].terminal = Some(StopReason::ContextFull);
                }
                return;
            }
        };
        drop(samplers);

        // Advance each participant; a sequence that reached its logical end this
        // step (caught up) samples + streams `toks[k]`. Mid-prefill chunks just
        // advanced `num_computed` (their column is unused).
        for (k, &(i, nn)) in parts.iter().enumerate() {
            self.active[i].num_computed += nn;
            let caught_up = self.active[i].num_computed as usize == logical_len(&self.active[i]);
            if !caught_up {
                // Still prefilling. The forward synced `batch.positions[slab]` to
                // `num_computed`, so the slab's live SSM state is exactly
                // state-at-P here — a valid checkpoint. Snapshot the shared
                // leading prefix once per `ckpt_stride` tokens (each snapshot is
                // ~65 MiB, so keep it sparse) for later requests to seed from.
                if let Some(stride) = self.prefix_cache.as_ref().map(|pc| pc.ckpt_stride) {
                    let p = self.active[i].num_computed;
                    let prev = p - nn;
                    if p / stride > prev / stride {
                        let slab = self.active[i].slab;
                        self.capture_checkpoint(slab, p, i, /*pinned=*/ false);
                    }
                }
                continue;
            }
            if !self.active[i].started {
                self.active[i].started = true;
                let pt = self.active[i].prompt_tokens;
                if self.active[i]
                    .reply
                    .blocking_send(GenEvent::Started { prompt_tokens: pt })
                    .is_err()
                {
                    self.active[i].disconnected = true;
                    continue;
                }
            }
            let token = toks[k];
            process_token(
                &self.model.tokenizer().tokenizer,
                &self.eog_ids,
                &mut self.active[i],
                token,
            );
        }
    }

    /// One batched decode step: gather the active sequences that still have
    /// context room, run a single forward, then stream + stop-check each.
    fn decode_step(&mut self) {
        let ctx = self.batch.config.max_seq_len;
        // Participants (ascending `active` index). Mark ctx-full ones terminal.
        let mut parts: Vec<usize> = Vec::with_capacity(self.active.len());
        for (i, seq) in self.active.iter_mut().enumerate() {
            if seq.terminal.is_some() || seq.disconnected {
                continue;
            }
            if self.batch.positions[seq.slab as usize] + 1 >= ctx {
                seq.terminal = Some(StopReason::ContextFull);
                continue;
            }
            parts.push(i);
        }
        if parts.is_empty() {
            return;
        }

        // Advance every active sequence in one batched forward — even a lone
        // sequence (B=1). Routing B=1 through the single-sequence path (decode
        // replay + split-K flash) would be faster per token, but the single-seq
        // flash's split-K reduction order differs from the batched `k_num=1`
        // single-pass, so a sequence alternating paths as load changes could see
        // its greedy output drift. Keeping one path makes output independent of
        // concurrency. (The plan accepts disabling replay on the batched path as
        // the throughput-for-latency trade — see M4 notes.)
        let tokens: Vec<u32> = parts.iter().map(|&i| self.active[i].last_token).collect();
        let positions: Vec<u32> = parts
            .iter()
            .map(|&i| self.batch.positions[self.active[i].slab as usize])
            .collect();
        let slots: Vec<u32> = parts.iter().map(|&i| self.active[i].slab).collect();
        // Gather participant samplers (iter_mut + same ascending order as parts).
        let part_set: std::collections::HashSet<usize> = parts.iter().copied().collect();
        let mut samplers: Vec<&mut Sampler> = self
            .active
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| part_set.contains(i))
            .map(|(_, seq)| &mut seq.sampler)
            .collect();

        let toks = match self.engine.forward_batch_decode(
            &*self.model,
            &mut self.batch,
            &tokens,
            &positions,
            &slots,
            &mut samplers,
        ) {
            Ok(t) => t,
            Err(e) => {
                // A batched GPU error is fatal to every participant — fail them
                // all and let eviction reset their slabs.
                let msg = e.to_string();
                for &i in &parts {
                    let _ = self.active[i]
                        .reply
                        .blocking_send(GenEvent::Error(msg.clone()));
                    self.active[i].disconnected = true; // suppress a duplicate Done
                    self.active[i].terminal = Some(StopReason::ContextFull);
                }
                return;
            }
        };
        drop(samplers);

        for (k, &i) in parts.iter().enumerate() {
            let token = toks[k];
            process_token(
                &self.model.tokenizer().tokenizer,
                &self.eog_ids,
                &mut self.active[i],
                token,
            );
            self.active[i].last_token = token;
        }
    }

    /// Reap finished / cancelled sequences: flush any held-back text, send the
    /// terminal `Done`, commit the slab's `prior_tokens`, and free the slab.
    fn evict_finished(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].terminal.is_none() && !self.active[i].disconnected {
                i += 1;
                continue;
            }
            let mut seq = self.active.remove(i);
            let slab = seq.slab as usize;

            if !seq.disconnected && !seq.pending.is_empty() {
                let tail = std::mem::take(&mut seq.pending);
                if seq.reply.blocking_send(GenEvent::Delta(tail)).is_err() {
                    seq.disconnected = true;
                }
            }
            if !seq.disconnected {
                let _ = seq.reply.blocking_send(GenEvent::Done {
                    stop_reason: seq.terminal.clone().unwrap_or(StopReason::MaxTokens),
                    prompt_tokens: seq.prompt_tokens,
                    completion_tokens: seq.generated.len() as u32,
                });
            }

            // The slab's cache holds the prompt plus every generated token we
            // fed back (all but the last — an output, never an input), so
            // `prior_tokens.len() == batch.positions[slab]`.
            let mut prior = std::mem::take(&mut seq.prompt);
            if seq.generated.len() > 1 {
                prior.extend_from_slice(&seq.generated[..seq.generated.len() - 1]);
            }
            self.slots[slab].prior_tokens = prior;
            self.slots[slab].active = false;
        }
    }
}

/// Advance one sequence by `token`: EOS check, append, streaming-decode, stop
/// matching, and the max-token bound — setting `seq.terminal` / `seq.pending` /
/// `seq.disconnected`. Mirrors the single-sequence decode loop body, but for one
/// member of a batch. Streaming uses `step_decode_stream` over the sequence's
/// own `DecodeStream` state, so no tokenizer borrow is held across steps.
fn process_token(tokenizer: &Tokenizer, eog_ids: &[u32], seq: &mut ActiveSeq, token: u32) {
    if !seq.ignore_eos && eog_ids.contains(&token) {
        // EOS is now in the cache (K/V written) but never emitted.
        seq.generated.push(token);
        seq.terminal = Some(StopReason::Eos);
        return;
    }
    seq.generated.push(token);

    if let Ok(Some(piece)) = tokenizers::step_decode_stream(
        tokenizer,
        vec![token],
        /*skip_special_tokens=*/ true,
        &mut seq.stream_ids,
        &mut seq.stream_prefix,
        &mut seq.stream_prefix_index,
    ) {
        seq.pending.push_str(&piece);
        match scan_stop(&seq.pending, &seq.stop) {
            StopScan::Hit { upto, seq: matched } => {
                if upto > 0
                    && seq
                        .reply
                        .blocking_send(GenEvent::Delta(seq.pending[..upto].to_string()))
                        .is_err()
                {
                    seq.disconnected = true;
                } else {
                    seq.terminal = Some(StopReason::StopSequence(matched));
                }
                return;
            }
            StopScan::Emit(n) => {
                if n > 0 {
                    let chunk: String = seq.pending.drain(..n).collect();
                    if seq.reply.blocking_send(GenEvent::Delta(chunk)).is_err() {
                        seq.disconnected = true;
                        return;
                    }
                }
            }
        }
    }

    if seq.generated.len() as u32 >= seq.max_tokens {
        seq.terminal = Some(StopReason::MaxTokens);
    }
}

/// Result of scanning the decoded tail for stop sequences.
enum StopScan {
    /// A stop sequence is fully present; emit `pending[..upto]` then stop.
    Hit { upto: usize, seq: String },
    /// No full match; `n` leading bytes of `pending` are safe to emit now (the
    /// rest is a partial-stop suffix that must stay buffered).
    Emit(usize),
}

/// Detect text stop sequences that may span multiple decoded pieces. Returns a
/// full `Hit` (earliest match across all stops) or the number of leading bytes
/// safe to emit while holding back the longest suffix that could still grow
/// into a stop sequence.
fn scan_stop(pending: &str, stops: &[String]) -> StopScan {
    if stops.is_empty() {
        return StopScan::Emit(pending.len());
    }
    // Earliest full match wins.
    let mut hit: Option<(usize, &str)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(idx) = pending.find(s.as_str()) {
            match hit {
                Some((bidx, _)) if bidx <= idx => {}
                _ => hit = Some((idx, s)),
            }
        }
    }
    if let Some((idx, s)) = hit {
        return StopScan::Hit {
            upto: idx,
            seq: s.to_string(),
        };
    }
    // No full match — hold back the longest suffix of `pending` that is a
    // proper prefix of some stop sequence.
    let mut hold = 0usize;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        let maxk = pending.len().min(s.len());
        let mut k = maxk;
        while k > 0 {
            if s.is_char_boundary(k)
                && pending.is_char_boundary(pending.len() - k)
                && pending[pending.len() - k..] == s[..k]
            {
                hold = hold.max(k);
                break;
            }
            k -= 1;
        }
    }
    StopScan::Emit(pending.len() - hold)
}

// ─── Concurrent-throughput benchmark (the measure-first gate) ───────────────
//
// Drives the REAL scheduler (`admit_unified` + `schedule_step` +
// `evict_finished`) with B synthetic sequences, so each later phase's win shows
// up in the exact serve path it changes — no HTTP/template/SSE noise, fully
// deterministic (greedy, fixed seed, `ignore_eos`). Runs on a dedicated 64 MiB
// OS thread (not tokio) so the worker's `blocking_send` is valid, mirroring
// [`InferenceHandle::spawn`]. One `Worker` is built at `max(batches)` slots and
// reused across the sweep; slots are reset between runs for clean prefills.

/// Synthetic workload shape for the concurrent bench.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BenchWorkload {
    /// Each sequence's prompt diverges in its leading tokens (short `lcp`) —
    /// forces B independent full prefills (the no-shared-prompt subagent case).
    Distinct,
    /// All sequences share `shared_len` leading tokens then diverge — the
    /// subagent system-prompt case (re-prefilled per slab today; Phase 2 target).
    Shared,
}

impl BenchWorkload {
    fn name(self) -> &'static str {
        match self {
            BenchWorkload::Distinct => "distinct",
            BenchWorkload::Shared => "shared",
        }
    }
}

/// Parameters for [`run_concurrent_bench`]. All `Send` (moved into the thread).
pub struct BenchPlan {
    pub batches: Vec<u32>,
    pub prompt_len: u32,
    pub gen_len: u32,
    pub shared_len: u32,
    pub warmup: u32,
    pub prompt: String,
    pub check: bool,
    /// Warm the leading-prefix cache before each timed Shared run (prefill the
    /// shared prefix once) so the timed sequences seed from it — the measure
    /// gate for `SEEKER_PREFIX_CACHE`. No effect without the shared workload.
    pub prewarm: bool,
}

/// One result row (a single `(B, workload)` run).
struct BenchRow {
    prefill_tps: f64,
    decode_tps: f64,
    per_seq_tps: f64,
    ms_per_step: f64,
    ttft_p50: f64,
    ttft_p95: f64,
    /// seq-0's generated token ids (for the `--check` byte-identical gate).
    gen0: Vec<u32>,
}

/// Entry point for `seeker bench --concurrent`. Spawns the worker on a 64 MiB OS
/// thread (model load is stack-heavy; `blocking_send` needs a non-tokio thread)
/// and blocks until the sweep finishes, returning any error as a `String`.
pub fn run_concurrent_bench(cfg: WorkerConfig, plan: BenchPlan) -> Result<(), String> {
    std::thread::Builder::new()
        .name("seeker-bench".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || bench_thread(cfg, plan))
        .map_err(|e| e.to_string())?
        .join()
        .map_err(|_| "bench thread panicked".to_string())?
}

fn bench_thread(mut cfg: WorkerConfig, plan: BenchPlan) -> Result<(), String> {
    let max_b = plan.batches.iter().copied().max().unwrap_or(1).max(1);
    cfg.n_slots = max_b;
    let mut worker = setup(&cfg).map_err(|e| e.to_string())?;
    if !worker.unified {
        return Err(
            "bench --concurrent requires a model with the unified forward (e.g. qwen35moe)".into(),
        );
    }
    let base = bench_encode_base(&worker, &plan.prompt)?;
    let ctx = worker.batch.config.max_seq_len;
    eprintln!(
        "# concurrent bench: prompt_len={} gen_len={} shared_len={} warmup={} ctx={} ubatch={} \
         vocab_base_tokens={}",
        plan.prompt_len,
        plan.gen_len,
        plan.shared_len,
        plan.warmup,
        ctx,
        cfg.n_ubatch,
        base.len(),
    );

    let mut workloads = vec![BenchWorkload::Distinct];
    if plan.shared_len > 0 {
        workloads.push(BenchWorkload::Shared);
    }
    for workload in workloads {
        println!(
            "\nworkload={} prompt_len={} gen_len={} shared_len={}",
            workload.name(),
            plan.prompt_len,
            plan.gen_len,
            plan.shared_len,
        );
        println!(
            "  B    prefill_tps   decode_tps  per_seq_tps   ms/step  TTFT_p50_ms  TTFT_p95_ms  speedup"
        );
        let mut base_decode_tps = 0.0f64;
        let mut check0: Option<Vec<u32>> = None;
        for (bi, &b) in plan.batches.iter().enumerate() {
            worker.bench_reset_slots(/*clear_cache=*/ true);
            let row = drive_workload(&mut worker, &base, b, workload, &plan)?;
            if bi == 0 {
                base_decode_tps = row.decode_tps;
            }
            if plan.check {
                match &check0 {
                    Some(prev) => {
                        let n = prev.len().min(row.gen0.len());
                        if prev[..n] != row.gen0[..n] {
                            return Err(format!(
                                "byte-identical check FAILED: workload={} seq-0 tokens differ at B={b}",
                                workload.name()
                            ));
                        }
                    }
                    None => check0 = Some(row.gen0.clone()),
                }
            }
            let speedup = row.decode_tps / base_decode_tps.max(1e-9);
            println!(
                "  {b:<3}  {:>11.1}  {:>11.1}  {:>11.1}  {:>8.2}  {:>11.2}  {:>11.2}  {:>6.2}x",
                row.prefill_tps,
                row.decode_tps,
                row.per_seq_tps,
                row.ms_per_step,
                row.ttft_p50,
                row.ttft_p95,
                speedup,
            );
        }
    }
    if plan.check {
        println!("\nbyte-identical check: PASS (seq-0 greedy stream stable across B)");
    }
    Ok(())
}

/// Tokenize the bench passage into a base token vector (tiled/perturbed per
/// sequence by [`bench_synth_prompt`]).
fn bench_encode_base(worker: &Worker, prompt: &str) -> Result<Vec<u32>, String> {
    let enc = worker
        .model
        .tokenizer()
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("tokenize base prompt: {e}"))?;
    let ids = enc.get_ids().to_vec();
    if ids.is_empty() {
        return Err("bench base prompt tokenized to zero tokens".into());
    }
    Ok(ids)
}

/// Build sequence `seq`'s prompt of `prompt_len` tokens by tiling `base`, then
/// perturbing per the workload so `lcp` across sequences is what we intend.
fn bench_synth_prompt(
    base: &[u32],
    prompt_len: usize,
    seq: usize,
    workload: BenchWorkload,
    shared_len: usize,
) -> Vec<u32> {
    let mut v: Vec<u32> = (0..prompt_len).map(|i| base[i % base.len()]).collect();
    match workload {
        // Perturb the leading tokens so any two sequences diverge early → short
        // lcp → B independent full prefills (no accidental cross-slab reuse).
        BenchWorkload::Distinct => {
            for i in 0..prompt_len.min(4) {
                v[i] = base[(i + seq * 7 + 1) % base.len()];
            }
        }
        // Keep the first `shared_len` identical across sequences (the shared
        // system prompt), diverge in the suffix per sequence.
        BenchWorkload::Shared => {
            let start = shared_len.min(prompt_len);
            for (i, slot) in v.iter_mut().enumerate().skip(start) {
                *slot = base[(i + seq * 13 + 1) % base.len()];
            }
        }
    }
    v
}

/// Greedy sampler config — deterministic, sampling-behavior-free (matches the
/// single-seq `seeker bench`).
fn bench_greedy_cfg() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_penalty: 1.0,
        penalty_last_n: 0,
        seed: 0,
        logit_bias: Vec::new(),
    }
}

/// Drain every reply channel without blocking; stamp each sequence's first-token
/// time (TTFT) the first time its `Started` event appears.
fn bench_drain(
    rxs: &mut [mpsc::Receiver<GenEvent>],
    started_at: &mut [Option<f64>],
    t_submit: std::time::Instant,
) {
    for (s, rx) in rxs.iter_mut().enumerate() {
        while let Ok(ev) = rx.try_recv() {
            if matches!(ev, GenEvent::Started { .. }) && started_at[s].is_none() {
                started_at[s] = Some(t_submit.elapsed().as_secs_f64());
            }
        }
    }
}

/// Discard any queued reply events (keeps the bounded reply channels from
/// back-pressuring the worker during the decode window).
fn bench_drain_discard(rxs: &mut [mpsc::Receiver<GenEvent>]) {
    for rx in rxs.iter_mut() {
        while rx.try_recv().is_ok() {}
    }
}

fn bench_percentile(sorted_ms: &[f64], q: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * q).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

/// Run one `(B, workload)` measurement against an already-reset worker.
fn drive_workload(
    worker: &mut Worker,
    base: &[u32],
    b: u32,
    workload: BenchWorkload,
    plan: &BenchPlan,
) -> Result<BenchRow, String> {
    let b = b as usize;
    let prompt_len = plan.prompt_len.max(1) as usize;
    let gen_len = plan.gen_len.max(1) as usize;
    let warmup = plan.warmup as usize;
    let shared_len = plan.shared_len as usize;
    // High max_tokens (+ ignore_eos) so no sequence terminates inside the timed
    // window — the driver controls the step count exactly.
    let max_tokens = (warmup + gen_len + 32) as u32;

    // Optionally warm the leading-prefix cache (prefill the shared prefix once)
    // so the timed Shared burst seeds from it. The preceding bench_reset_slots
    // cleared the cache; bench_prewarm repopulates exactly [0, shared_len) and
    // leaves the slabs cold so admission takes the seed path, not pure-extension.
    if plan.prewarm && workload == BenchWorkload::Shared {
        worker.bench_prewarm(base, shared_len);
    }

    let mut rxs: Vec<mpsc::Receiver<GenEvent>> = Vec::with_capacity(b);
    for s in 0..b {
        let tokens = bench_synth_prompt(base, prompt_len, s, workload, shared_len);
        let (tx, rx) = mpsc::channel::<GenEvent>(gen_len + warmup + 64);
        rxs.push(rx);
        worker.admit_unified(GenJob {
            tokens,
            config: GenConfig {
                sampler: bench_greedy_cfg(),
                max_tokens,
                stop: Vec::new(),
                ignore_eos: true,
                id_slot: None,
            },
            image: None,
            audio: None,
            reply: tx,
        });
    }

    // ── Prefill window: step until every sequence has emitted Started. ──
    let mut started_at: Vec<Option<f64>> = vec![None; b];
    let t_submit = std::time::Instant::now();
    let mut guard = 0usize;
    while started_at.iter().any(|x| x.is_none()) {
        worker.schedule_step();
        worker.evict_finished();
        bench_drain(&mut rxs, &mut started_at, t_submit);
        guard += 1;
        if guard > 1_000_000 {
            return Err("bench prefill window did not converge".into());
        }
    }
    let prefill_secs = t_submit.elapsed().as_secs_f64();

    // ── Warm-up decode (untimed). ──
    for _ in 0..warmup {
        worker.schedule_step();
        worker.evict_finished();
        bench_drain_discard(&mut rxs);
    }

    // ── Timed decode window. ──
    let t_dec = std::time::Instant::now();
    for _ in 0..gen_len {
        worker.schedule_step();
        worker.evict_finished();
        bench_drain_discard(&mut rxs);
    }
    let decode_secs = t_dec.elapsed().as_secs_f64();

    // Sequences never terminated (high max_tokens), so `active` is still in
    // admit order — active[0] is seq-0.
    let gen0 = worker
        .active
        .first()
        .map(|s| s.generated.clone())
        .unwrap_or_default();

    let prefill_tps = (b * prompt_len) as f64 / prefill_secs.max(1e-9);
    let decode_tps = (b * gen_len) as f64 / decode_secs.max(1e-9);
    let per_seq_tps = gen_len as f64 / decode_secs.max(1e-9);
    let ms_per_step = decode_secs * 1000.0 / gen_len as f64;
    let mut ttfts: Vec<f64> = started_at
        .iter()
        .map(|x| x.unwrap_or(0.0) * 1000.0)
        .collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Ok(BenchRow {
        prefill_tps,
        decode_tps,
        per_seq_tps,
        ms_per_step,
        ttft_p50: bench_percentile(&ttfts, 0.50),
        ttft_p95: bench_percentile(&ttfts, 0.95),
        gen0,
    })
}

impl Worker {
    /// Clear every slot's cached prefix + recurrent state for a fresh, comparable
    /// bench run (no cross-run prefix reuse, no stale SSM state). Bench-only.
    /// `clear_cache` also wipes the leading-prefix snapshot cache — pass `true`
    /// between runs for comparability, `false` after a prewarm (which fills the
    /// cache and then cold-resets the slabs so the timed burst seeds from it).
    fn bench_reset_slots(&mut self, clear_cache: bool) {
        self.active.clear();
        self.clock = 0;
        self.engine.decode_cache = None;
        for i in 0..self.slots.len() {
            self.slots[i].prior_tokens.clear();
            self.slots[i].last_used = 0;
            self.slots[i].active = false;
            self.batch.reset_slot(i as u32);
            self.batch.positions[i] = 0;
            self.batch.rope_lag[i] = 0;
        }
        if clear_cache && let Some(pc) = self.prefix_cache.as_mut() {
            pc.clear();
        }
    }

    /// Prefill `tokens` as a standalone request and capture the prefix at its
    /// boundary (`positions[slab] == tokens.len()`, so the slab's live SSM state
    /// is exactly state-at-P), then cold-reset the slabs KEEPING the cache entry.
    /// `pinned` marks the entry un-evictable (the production system-prompt pin);
    /// the bench prewarm passes `false`. Returns whether a capture happened. No-op
    /// (with a warning) without the cache or when `tokens.len()` is outside
    /// `[p_min, max_cached_len]` (where `capture_checkpoint` would silently skip).
    fn prefill_and_capture(&mut self, tokens: Vec<u32>, pinned: bool) -> bool {
        let (p_min, max_cached_len) = match self.prefix_cache.as_ref() {
            Some(pc) => (pc.p_min, pc.max_cached_len),
            None => return false,
        };
        let n = tokens.len() as u32;
        if n == 0 {
            return false;
        }
        if n < p_min || n > max_cached_len {
            tracing::warn!(
                len = n,
                p_min,
                max_cached_len,
                "prefix cache: prefix length outside [p_min, max_cached_len]; not cached \
                 (adjust SEEKER_PREFIX_CACHE_PMIN/_MAXLEN)"
            );
            return false;
        }
        let (tx, mut rx) = mpsc::channel::<GenEvent>(64);
        self.admit_unified(GenJob {
            tokens,
            config: GenConfig {
                sampler: bench_greedy_cfg(),
                max_tokens: 4,
                stop: Vec::new(),
                ignore_eos: true,
                id_slot: None,
            },
            image: None,
            audio: None,
            reply: tx,
        });
        // admit_unified only pushes to `active` on success; an empty `active`
        // here means it rejected the request (e.g. no free slab).
        if self.active.is_empty() {
            tracing::warn!(
                "prefix cache: prefill request not admitted (no free slab?); not cached"
            );
            return false;
        }
        // Drive prefill to completion (the sequence emits `Started` once caught
        // up); no eviction in between so the slab is still live to capture.
        let mut guard = 0usize;
        loop {
            self.schedule_step();
            let mut started = false;
            while let Ok(ev) = rx.try_recv() {
                if matches!(ev, GenEvent::Started { .. }) {
                    started = true;
                }
            }
            if started {
                break;
            }
            if self.active.is_empty() {
                tracing::warn!("prefix cache: prefill ended before completing; not cached");
                self.bench_reset_slots(false);
                return false;
            }
            guard += 1;
            if guard > 100_000 {
                tracing::warn!("prefix cache: prefill did not converge; not cached");
                self.bench_reset_slots(false);
                return false;
            }
        }
        // Capture the prefix at its boundary (positions[slab] == tokens.len()).
        let captured = if let Some(slab) = self.active.first().map(|s| s.slab) {
            let p = self.batch.positions[slab as usize];
            self.capture_checkpoint(slab, p, 0, pinned);
            true
        } else {
            false
        };
        // Cold-reset the slabs but KEEP the cache entry.
        self.bench_reset_slots(false);
        captured
    }

    /// Warm the leading-prefix cache before a timed Shared bench run: prefill the
    /// `shared_len`-token shared prefix (which every Shared sequence starts with)
    /// and capture it (unpinned). No-op without the cache.
    fn bench_prewarm(&mut self, base: &[u32], shared_len: usize) {
        if self.prefix_cache.is_none() || shared_len == 0 || base.is_empty() {
            return;
        }
        // Identical to the first `shared_len` tokens of every Shared-workload
        // sequence (which tile `base` and only diverge in the suffix).
        let prefix: Vec<u32> = (0..shared_len).map(|i| base[i % base.len()]).collect();
        self.prefill_and_capture(prefix, /*pinned=*/ false);
    }

    /// Pin a shared leading prefix (the rendered system prompt) at startup so
    /// every request beginning with it seeds instead of re-prefilling it. The
    /// production analogue of `bench_prewarm`: prefill once, capture PINNED
    /// (never evicted). No-op without the cache.
    fn pin_prefix(&mut self, tokens: Vec<u32>) {
        let n = tokens.len();
        if self.prefill_and_capture(tokens, /*pinned=*/ true) {
            tracing::info!(
                prefix_tokens = n,
                "prefix cache: pinned the system-prompt prefix"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emit(pending: &str, stops: &[&str]) -> usize {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        match scan_stop(pending, &stops) {
            StopScan::Emit(n) => n,
            StopScan::Hit { .. } => panic!("unexpected hit"),
        }
    }

    fn hit(pending: &str, stops: &[&str]) -> (usize, String) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        match scan_stop(pending, &stops) {
            StopScan::Hit { upto, seq } => (upto, seq),
            StopScan::Emit(_) => panic!("expected hit"),
        }
    }

    #[test]
    fn no_stops_emits_everything() {
        assert_eq!(emit("hello world", &[]), 11);
    }

    #[test]
    fn holds_partial_stop_suffix() {
        // "abc</th" with stop "</think>" must hold back "</th" (4 bytes).
        assert_eq!(emit("abc</th", &["</think>"]), 3);
        // A bare tail that isn't a stop prefix is fully emitted.
        assert_eq!(emit("abcdef", &["</think>"]), 6);
    }

    #[test]
    fn detects_full_stop_across_pieces() {
        let (upto, seq) = hit("answer</think>", &["</think>"]);
        assert_eq!(upto, 6);
        assert_eq!(seq, "</think>");
    }

    #[test]
    fn earliest_match_wins() {
        let (upto, seq) = hit("aXbYc", &["Y", "X"]);
        assert_eq!(upto, 1); // "X" at index 1 is earlier than "Y" at 3
        assert_eq!(seq, "X");
    }

    #[test]
    fn multibyte_tail_is_not_split() {
        // A held-back boundary must stay on a char boundary (é = 2 bytes).
        // "caf" + "é" with stop "éxit": hold the "é" prefix.
        let n = emit("café", &["éxit"]);
        assert!("café".is_char_boundary(n), "cut at {n} is a char boundary");
    }

    #[test]
    fn choose_slot_prefers_longest_pure_extension() {
        // slot 0 holds [1,2] (a prefix of the request), slot 1 holds [1,2,3,4]
        // (a longer prefix). The request [1,2,3,4,5] extends both; pick slot 1
        // (most reuse). last_used is irrelevant when a match exists.
        let s0: &[u32] = &[1, 2];
        let s1: &[u32] = &[1, 2, 3, 4];
        let slots = [(s0, 9), (s1, 1)];
        assert_eq!(choose_slot(&slots, &[1, 2, 3, 4, 5], None), 1);
    }

    #[test]
    fn choose_slot_skips_non_prefix_match() {
        // slot 0 = [1,2,9] is NOT a prefix of [1,2,3] (diverges at index 2), so
        // it's not reusable; fall through to the LRU (cold slot 1, last_used 0).
        let s0: &[u32] = &[1, 2, 9];
        let s1: &[u32] = &[];
        let slots = [(s0, 5), (s1, 0)];
        assert_eq!(choose_slot(&slots, &[1, 2, 3], None), 1);
    }

    #[test]
    fn choose_slot_evicts_lru_when_no_match() {
        // No slot is a prefix of the request → evict the least-recently-used
        // (slot 1, last_used 2 < slot 0's 7).
        let a: &[u32] = &[5, 5];
        let b: &[u32] = &[6, 6];
        let slots = [(a, 7), (b, 2)];
        assert_eq!(choose_slot(&slots, &[9, 9], None), 1);
    }

    #[test]
    fn choose_slot_cold_slot_first() {
        // A cold slot (last_used 0) is the oldest, so a brand-new conversation
        // claims it instead of evicting a warm slot.
        let warm: &[u32] = &[1, 2, 3];
        let cold: &[u32] = &[];
        let slots = [(warm, 4), (cold, 0)];
        assert_eq!(choose_slot(&slots, &[7, 8], None), 1);
    }

    #[test]
    fn choose_slot_honors_valid_id_slot() {
        let a: &[u32] = &[1, 2];
        let b: &[u32] = &[];
        let slots = [(a, 1), (b, 0)];
        // In-range pin wins over prefix matching.
        assert_eq!(choose_slot(&slots, &[1, 2, 3], Some(1)), 1);
        // Out-of-range pin falls back to auto-select (slot 0 is a pure ext).
        assert_eq!(choose_slot(&slots, &[1, 2, 3], Some(9)), 0);
    }

    // ─── PrefixCache logic (no GPU; pool is empty, only `entries` is read) ────

    fn entry(tokens: &[u32], last_used: u64, pinned: bool) -> Option<PrefixEntry> {
        Some(PrefixEntry {
            tokens: tokens.to_vec(),
            p: tokens.len() as u32,
            last_used,
            pinned,
        })
    }

    fn pc(entries: Vec<Option<PrefixEntry>>, p_min: u32) -> PrefixCache {
        PrefixCache {
            pool: Vec::new(),
            entries,
            max_cached_len: 4096,
            ckpt_stride: 512,
            p_min,
            clock: 0,
        }
    }

    #[test]
    fn prefix_lookup_picks_longest_leading_prefix() {
        let c = pc(
            vec![entry(&[1, 2], 0, false), entry(&[1, 2, 3, 4], 0, false)],
            1,
        );
        // [1,2,3,4,5] extends both cached prefixes; pick the longer one (slot 1).
        assert_eq!(c.lookup(&[1, 2, 3, 4, 5]), Some(1));
    }

    #[test]
    fn prefix_lookup_requires_full_leading_prefix() {
        // A cached entry that is NOT a full leading prefix of the request must
        // not match (byte-exact, like choose_slot's pure-extension rule).
        let c = pc(vec![entry(&[1, 2, 9], 0, false)], 1);
        assert_eq!(c.lookup(&[1, 2, 3]), None);
        // And an entry LONGER than the request can't be a prefix of it.
        let c = pc(vec![entry(&[1, 2, 3], 0, false)], 1);
        assert_eq!(c.lookup(&[1, 2]), None);
    }

    #[test]
    fn prefix_reserve_victim_prefers_free_then_lru_unpinned() {
        // A free (None) slot is taken first.
        let c = pc(vec![entry(&[1], 7, false), None], 1);
        assert_eq!(c.reserve_victim(), Some(1));
        // No free slot: evict the least-recently-used UNPINNED entry. Slot 0 is
        // older (1) but pinned, so slot 2 (used 3) is the victim over slot 1 (5).
        let c = pc(
            vec![
                entry(&[1], 1, true),
                entry(&[2], 5, false),
                entry(&[3], 3, false),
            ],
            1,
        );
        assert_eq!(c.reserve_victim(), Some(2));
        // Every slot pinned ⇒ nothing to evict.
        let c = pc(vec![entry(&[1], 1, true), entry(&[2], 2, true)], 1);
        assert_eq!(c.reserve_victim(), None);
    }

    #[test]
    fn prefix_has_at_least_checks_length_and_match() {
        let c = pc(vec![entry(&[1, 2, 3, 4], 0, false)], 1);
        assert!(c.has_at_least(&[1, 2, 3, 4, 5], 3)); // cached p=4 covers ≥3, prefix matches
        assert!(c.has_at_least(&[1, 2, 3, 4, 5], 4));
        assert!(!c.has_at_least(&[1, 2, 3, 4, 5], 5)); // cached p=4 < 5
        assert!(!c.has_at_least(&[1, 2, 9], 3)); // diverges at index 2 (lcp=2 < 3)
    }
}
