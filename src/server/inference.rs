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
use std::path::PathBuf;

use tokio::sync::{mpsc, oneshot};

use crate::gguf::{GgmlType, GgufFile};
use crate::inference::kv_cache::{KvCache, KvCacheConfig};
use crate::inference::sample::{Sampler, SamplerConfig};
use crate::inference::Engine;
use crate::tokenizer::build_tokenizer;

/// Immutable per-process model/runtime config. All `Send` — built by the CLI
/// and moved into the worker thread, which constructs the `Engine` from it.
pub struct WorkerConfig {
    pub model_path: PathBuf,
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

/// One unit of work for the worker. The handler has already rendered the chat
/// template and encoded it, so only `Send` data crosses the thread boundary.
pub struct GenJob {
    pub tokens: Vec<u32>,
    pub config: GenConfig,
    /// Reply sink. Bounded so the worker back-pressures on a slow client; a
    /// *dropped* receiver (client disconnect) makes `blocking_send` return Err,
    /// which the decode loop treats as cancellation.
    pub reply: mpsc::Sender<GenEvent>,
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
        let (tx, rx) = mpsc::channel(32);
        self.submit(GenJob {
            tokens,
            config,
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

/// One independent KV-cache slot. `prior_tokens` mirrors `cache.position` (the
/// tokens whose K/V is live), so a request that extends this slot's prefix
/// reuses it and prefills only the divergent suffix.
struct Slot {
    cache: KvCache,
    prior_tokens: Vec<u32>,
    /// Monotonic stamp of the last job that used this slot (LRU). `0` = cold.
    last_used: u64,
}

/// GPU-resident state owned by the worker thread for the whole process. Holds a
/// pool of N independent cache slots; the single-sequence engine processes one
/// job at a time, so slots provide cross-request cache reuse, not parallelism.
struct Worker {
    engine: Engine,
    model: Box<dyn crate::models::Model>,
    slots: Vec<Slot>,
    eog_ids: Vec<u32>,
    /// Monotonic LRU clock (incremented per job; avoids wall-clock).
    clock: u64,
    /// Slot the previous job used — a switch must invalidate the engine's
    /// decode-replay cmdbuf (it binds the prior slot's K/V buffers).
    last_slot: Option<usize>,
}

impl Worker {
    /// Pick the slot to serve `new_tokens` from (see [`choose_slot`]).
    fn select_slot(&self, new_tokens: &[u32], id_slot: Option<usize>) -> usize {
        let view: Vec<(&[u32], u64)> = self
            .slots
            .iter()
            .map(|s| (s.prior_tokens.as_slice(), s.last_used))
            .collect();
        choose_slot(&view, new_tokens, id_slot)
    }
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
    engine.allocate_scratch(model.scratch_bytes_estimate(
        cfg.n_ubatch,
        cfg.ctx_size,
        cfg.cache_type_k,
        cfg.cache_type_v,
    ))?;

    let cache_config = KvCacheConfig {
        k_dtype: cfg.cache_type_k,
        v_dtype: cfg.cache_type_v,
        max_seq_len: cfg.ctx_size,
    };
    let dims = model.cache_dims();
    let ssm = model.ssm_state_dims();
    let n_slots = cfg.n_slots.max(1);
    let mut slots = Vec::with_capacity(n_slots as usize);
    for _ in 0..n_slots {
        let mut cache =
            engine.allocate_kv_cache(dims.n_layer, dims.head_dim, dims.n_head_kv, cache_config)?;
        if let Some(ssm) = &ssm {
            cache.allocate_ssm_state(
                &engine.device,
                ssm.n_ssm_layers,
                ssm.conv_state_floats,
                ssm.gdn_state_floats,
            )?;
        }
        slots.push(Slot {
            cache,
            prior_tokens: Vec::new(),
            last_used: 0,
        });
    }

    let per_slot = slots[0].cache.region.size
        + slots[0]
            .cache
            .ssm_region
            .as_ref()
            .map_or(0, |r| r.size);
    let mib = 1u64 << 20;
    tracing::info!(
        slots = n_slots,
        ctx = cfg.ctx_size,
        per_slot_mib = per_slot / mib,
        total_mib = (per_slot * n_slots as u64) / mib,
        "kv cache slots allocated",
    );

    let eog_ids = model.tokenizer().eog_ids.clone();
    Ok(Worker {
        engine,
        model,
        slots,
        eog_ids,
        clock: 0,
        last_slot: None,
    })
}

/// Worker entry point. Loads the model, signals readiness, then serves jobs
/// until every `InferenceHandle` is dropped (server shutdown).
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

    // `blocking_recv` is safe here: this is a plain OS thread, not inside the
    // tokio runtime. `None` ⇒ all senders dropped ⇒ shut down (Engine drops).
    while let Some(job) = jobs.blocking_recv() {
        run_job(&mut worker, job);
    }
}

/// Run one generation job to completion, streaming `GenEvent`s. Never panics on
/// inference error (converts to `GenEvent::Error`) and never propagates out, so
/// the worker survives a bad job and serves the next one.
fn run_job(w: &mut Worker, job: GenJob) {
    let GenJob {
        tokens: new_tokens,
        config,
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
        let _ = reply.blocking_send(GenEvent::Error("empty prompt — nothing to generate".into()));
        return;
    }

    // ── Slot selection ──────────────────────────────────────────────────
    let idx = w.select_slot(&new_tokens, id_slot);
    // The decode-replay cmdbuf is keyed only on sampler/grid, not on which
    // cache recorded it — so a slot switch must invalidate it, or an L==1 first
    // forward could replay against the previous slot's K/V buffers.
    if w.last_slot != Some(idx) {
        w.engine.decode_cache = None;
        w.last_slot = Some(idx);
    }
    w.clock += 1;
    w.slots[idx].last_used = w.clock;

    let ctx = w.slots[idx].cache.config.max_seq_len;
    // Need room for the whole prompt plus at least one generated token.
    if new_tokens.len() as u32 >= ctx {
        let _ = reply.blocking_send(GenEvent::Error(format!(
            "prompt is {} tokens but --ctx-size is {ctx} (no room to generate) — \
             raise --ctx-size or shorten the prompt",
            new_tokens.len()
        )));
        return;
    }

    let slot = &mut w.slots[idx];

    // ── Safe prefix-reuse (per slot) ────────────────────────────────────
    // Reuse the cached prefix ONLY when this prompt strictly extends what the
    // slot holds (and `cache.position` agrees). Any divergence falls back to a
    // full reset + prefill, the only SSM/GDN-safe rewind (recurrent state has no
    // per-position undo — see `KvCache::reset`).
    let common = slot
        .prior_tokens
        .iter()
        .zip(new_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let pure_extension =
        common > 0 && common == slot.prior_tokens.len() && common == slot.cache.position as usize;
    let (start_pos, delta): (u32, Vec<u32>) = if pure_extension {
        if common < new_tokens.len() {
            (common as u32, new_tokens[common..].to_vec())
        } else {
            // Whole prompt already cached (identical re-request): rewind one
            // and re-feed the last token so there are logits to sample from.
            (common as u32 - 1, vec![*new_tokens.last().unwrap()])
        }
    } else {
        slot.cache.reset();
        (0, new_tokens.clone())
    };
    slot.cache.position = start_pos;
    tracing::debug!(
        slot = idx,
        prompt = new_tokens.len(),
        reused = start_pos,
        prefill = delta.len(),
        "serve: prefix reuse",
    );

    // ── Prefill + decode ────────────────────────────────────────────────
    let mut sampler = Sampler::new(cfg);
    let mut generated: Vec<u32> = Vec::new();
    let mut step: Vec<u32> = delta;
    let mut stream = w.model.tokenizer().tokenizer.decode_stream(/*skip_special=*/ true);
    let mut pending = String::new(); // decoded-but-held-back tail (stop matching)
    let mut forwards = 0usize;
    let mut disconnected = false;
    let mut terminal: Option<StopReason> = None;

    loop {
        // Context headroom for the next forward (chat parity).
        if slot.cache.position as usize + step.len() + 1 > ctx as usize {
            terminal = Some(StopReason::ContextFull);
            break;
        }
        let position = slot.cache.position;
        let token = match w
            .engine
            .forward_sampled(&*w.model, &mut slot.cache, &step, position, &mut sampler)
        {
            Ok(t) => t,
            Err(e) => {
                // Cache/prior may be inconsistent after a GPU error — reset so
                // the next job full-prefills. Force the next job to re-record
                // the decode cmdbuf (last_slot cleared). Keep the worker alive.
                slot.cache.reset();
                slot.prior_tokens.clear();
                w.last_slot = None;
                let _ = reply.blocking_send(GenEvent::Error(e.to_string()));
                return;
            }
        };

        if forwards == 0 && reply.blocking_send(GenEvent::Started { prompt_tokens }).is_err() {
            disconnected = true;
            break;
        }
        forwards += 1;

        if !ignore_eos && w.eog_ids.contains(&token) {
            // EOS is now in the cache (K/V written) but not emitted.
            generated.push(token);
            terminal = Some(StopReason::Eos);
            break;
        }
        generated.push(token);

        if let Ok(Some(piece)) = stream.step(token) {
            pending.push_str(&piece);
            match scan_stop(&pending, &stop) {
                StopScan::Hit { upto, seq } => {
                    if upto > 0 && reply.blocking_send(GenEvent::Delta(pending[..upto].to_string())).is_err() {
                        disconnected = true;
                        break;
                    }
                    terminal = Some(StopReason::StopSequence(seq));
                    break;
                }
                StopScan::Emit(n) => {
                    if n > 0 {
                        let chunk: String = pending.drain(..n).collect();
                        if reply.blocking_send(GenEvent::Delta(chunk)).is_err() {
                            disconnected = true;
                            break;
                        }
                    }
                }
            }
        }

        if generated.len() as u32 >= max_tokens {
            terminal = Some(StopReason::MaxTokens);
            break;
        }
        step = vec![token];
    }

    // Flush any text still buffered for stop matching (no stop matched).
    if !disconnected
        && !pending.is_empty()
        && reply
            .blocking_send(GenEvent::Delta(std::mem::take(&mut pending)))
            .is_err()
    {
        disconnected = true;
    }

    // ── Commit prior_tokens to mirror the slot's cache ──────────────────
    // The cache holds the full rendered prompt plus every generated token we
    // fed back (all but the last — the last was an output, never an input).
    // This invariant (`prior_tokens.len() == cache.position`) holds on every
    // exit because the prefill forward always ran (guarded above).
    let mut prior = new_tokens;
    if generated.len() > 1 {
        prior.extend_from_slice(&generated[..generated.len() - 1]);
    }
    slot.prior_tokens = prior;

    if !disconnected {
        let _ = reply.blocking_send(GenEvent::Done {
            stop_reason: terminal.unwrap_or(StopReason::MaxTokens),
            prompt_tokens,
            completion_tokens: generated.len() as u32,
        });
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
