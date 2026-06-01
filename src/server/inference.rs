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
use crate::inference::kv_cache::{BatchKvCache, KvCacheConfig};
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
    pub n_slots: u32,
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

/// One unit of work for the worker. The handler has already rendered the chat
/// template and encoded it, so only `Send` data crosses the thread boundary.
pub struct GenJob {
    pub tokens: Vec<u32>,
    pub config: GenConfig,
    /// An attached image (chat requests with `image_url` content). `None` for
    /// the text path. Prefilled single-pass with the vision splice in the worker.
    pub image: Option<ServeImage>,
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
    encoder: VisionEncoder,
    host_weights: HostWeights,
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
    /// channel: the worker runs the full model-load sequence and sends
    /// `Ok(())` (or `Err(msg)`) *before* entering its job loop, so the caller
    /// can fail fast on a bad model / missing GPU exactly like `seeker chat`.
    pub fn spawn(cfg: WorkerConfig) -> (InferenceHandle, oneshot::Receiver<Result<(), String>>) {
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
        self.start_mm(tokens, config, None).await
    }

    /// As [`Self::start`] but with an optional attached image (multimodal chat).
    pub async fn start_mm(
        &self,
        tokens: Vec<u32>,
        config: GenConfig,
        image: Option<ServeImage>,
    ) -> Result<mpsc::Receiver<GenEvent>, String> {
        let (tx, rx) = mpsc::channel(32);
        self.submit(GenJob {
            tokens,
            config,
            image,
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

    let cache_config = KvCacheConfig {
        k_dtype: cfg.cache_type_k,
        v_dtype: cfg.cache_type_v,
        max_seq_len: cfg.ctx_size,
    };
    let dims = model.cache_dims();
    let n_slots = cfg.n_slots.max(1);
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
    if let Some(ssm) = model.ssm_state_dims() {
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
    })
}

/// Build the vision tower from an mmproj GGUF (upload weights, build encoder +
/// host-side patch-embed copies). Mirrors `seeker chat`'s lazy `/image` build.
fn build_vision(engine: &Engine, mmproj_path: &Path) -> Result<VisionCtx, Box<dyn Error>> {
    let gguf = GgufFile::open(mmproj_path)?;
    let weights = engine.upload_weights(&gguf)?;
    let cfg = crate::vision::parse_config(&gguf)?;
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
    let vision = crate::vision::VisionModel {
        config: cfg,
        weights,
    };
    Ok(VisionCtx {
        vision,
        encoder,
        host_weights,
    })
}

/// Worker entry point. Loads the model, signals readiness, then runs the
/// continuous-batching scheduler until every `InferenceHandle` is dropped.
fn worker_main(
    cfg: WorkerConfig,
    mut jobs: mpsc::Receiver<GenJob>,
    ready: oneshot::Sender<Result<(), String>>,
) {
    let mut worker = match setup(&cfg) {
        Ok(w) => w,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        // `serve::run` gave up waiting (process exiting). Nothing to serve.
        return;
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
                // Image jobs prefill single-pass (the vision splice can't be
                // chunked), regardless of the global unified flag.
                Some(job) if job.image.is_some() => worker.admit_image(job),
                Some(job) if unified => worker.admit_unified(job),
                Some(job) => worker.admit(job),
                None => return,
            }
        }
        // Drain queued jobs into any free slabs without blocking the step.
        while worker.free_slabs() > 0 {
            match jobs.try_recv() {
                Ok(job) if job.image.is_some() => worker.admit_image(job),
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
            image: _, // text path: image jobs are routed to admit_image
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
        let (encoder, host_weights, weights) = (&vc.encoder, &vc.host_weights, &vc.vision.weights);
        let pimg = &image.pimg;
        crate::vision::encoder::encode_image_chunked(
            &mut self.engine,
            weights,
            encoder,
            pimg,
            host_weights,
        )
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
}
