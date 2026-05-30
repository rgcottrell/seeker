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

/// GPU-resident state owned by the worker thread for the whole process.
struct Worker {
    engine: Engine,
    model: Box<dyn crate::models::Model>,
    cache: KvCache,
    eog_ids: Vec<u32>,
    /// Token ids currently in the KV cache, mirroring `cache.position`. Lets us
    /// reuse the cached prefix across requests that strictly extend it.
    prior_tokens: Vec<u32>,
}

/// Build the engine + model + cache, mirroring `seeker chat`'s load sequence.
fn setup(cfg: &WorkerConfig) -> Result<Worker, Box<dyn Error>> {
    let gguf = GgufFile::open(&cfg.model_path)?;
    let bundle = build_tokenizer(&gguf)?;

    let mut engine = Engine::new(cfg.n_ubatch, cfg.n_batch)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened for serve");
    let weights = engine.upload_weights(&gguf)?;
    let model = crate::models::open(&gguf, weights, bundle, /*spec_enabled=*/ false)?;

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
    let mut cache =
        engine.allocate_kv_cache(dims.n_layer, dims.head_dim, dims.n_head_kv, cache_config)?;
    if let Some(ssm) = model.ssm_state_dims() {
        cache.allocate_ssm_state(
            &engine.device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
    }

    let eog_ids = model.tokenizer().eog_ids.clone();
    Ok(Worker {
        engine,
        model,
        cache,
        eog_ids,
        prior_tokens: Vec::new(),
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
    } = config;

    let ctx = w.cache.config.max_seq_len;
    let prompt_tokens = new_tokens.len() as u32;

    if new_tokens.is_empty() {
        let _ = reply.blocking_send(GenEvent::Error("empty prompt — nothing to generate".into()));
        return;
    }
    // Need room for the whole prompt plus at least one generated token.
    if new_tokens.len() as u32 >= ctx {
        let _ = reply.blocking_send(GenEvent::Error(format!(
            "prompt is {} tokens but --ctx-size is {ctx} (no room to generate) — \
             raise --ctx-size or shorten the prompt",
            new_tokens.len()
        )));
        return;
    }

    // ── Safe prefix-reuse ───────────────────────────────────────────────
    // Reuse the cached prefix ONLY when this prompt strictly extends what is
    // already in the cache (and `cache.position` agrees). Any divergence falls
    // back to a full reset + prefill, the only SSM/GDN-safe rewind (recurrent
    // state has no per-position undo — see `KvCache::reset`).
    let common = w
        .prior_tokens
        .iter()
        .zip(new_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let pure_extension =
        common > 0 && common == w.prior_tokens.len() && common == w.cache.position as usize;
    let (start_pos, delta): (u32, Vec<u32>) = if pure_extension {
        if common < new_tokens.len() {
            (common as u32, new_tokens[common..].to_vec())
        } else {
            // Whole prompt already cached (identical re-request): rewind one
            // and re-feed the last token so there are logits to sample from.
            (common as u32 - 1, vec![*new_tokens.last().unwrap()])
        }
    } else {
        w.cache.reset();
        (0, new_tokens.clone())
    };
    w.cache.position = start_pos;
    tracing::debug!(
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
        if w.cache.position as usize + step.len() + 1 > ctx as usize {
            terminal = Some(StopReason::ContextFull);
            break;
        }
        let position = w.cache.position;
        let token = match w
            .engine
            .forward_sampled(&*w.model, &mut w.cache, &step, position, &mut sampler)
        {
            Ok(t) => t,
            Err(e) => {
                // Cache/prior may be inconsistent after a GPU error — reset so
                // the next job full-prefills. Keep the worker alive.
                w.cache.reset();
                w.prior_tokens.clear();
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

    // ── Commit prior_tokens to mirror the cache ─────────────────────────
    // The cache holds the full rendered prompt plus every generated token we
    // fed back (all but the last — the last was an output, never an input).
    // This invariant (`prior_tokens.len() == cache.position`) holds on every
    // exit because the prefill forward always ran (guarded above).
    let mut prior = new_tokens;
    if generated.len() > 1 {
        prior.extend_from_slice(&generated[..generated.len() - 1]);
    }
    w.prior_tokens = prior;

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
}
