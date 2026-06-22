//! GPU-resident sampler. Records a chain of compute dispatches that read the
//! model's logits (already on device after the forward pass) and produce a
//! single u32 token id. The host pulls back 4 bytes per decode step instead
//! of the full vocab.
//!
//! Chain order, after `llama.cpp`'s common-sampler convention:
//!
//! ```text
//! penalties → top_k → top_p → min_p → temp → dist
//! ```
//!
//! Penalties run *first* because the repetition-penalty multiply/divide is
//! sign-conditional on the raw logit. Temperature runs *last* (right before
//! the categorical) because `top_p` does its own internal softmax — applying
//! `1/T` ahead of that would shift the cumulative cutoff.
//!
//! Greedy short-circuit: when `temperature == 0` we skip
//! top_k/top_p/min_p/dist but still apply penalties before `argmax`.

use std::collections::VecDeque;
use std::error::Error;
use std::hash::{Hash, Hasher};

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::buffer::BufferRange;
use super::context::DispatchContext;
use super::decode_dyn;
use super::weights::TensorView;

/// Built-in fallback sampler defaults — llama.cpp's `common_params_sampling`
/// values, used when neither a CLI flag nor a GGUF `general.sampling.*` key
/// supplies a knob. The resolution precedence is **CLI flag → GGUF default →
/// these built-ins** (see [`GgufSamplingDefaults`]).
pub const DEFAULT_TEMPERATURE: f32 = 0.8;
pub const DEFAULT_TOP_K: u32 = 40;
pub const DEFAULT_TOP_P: f32 = 0.95;
pub const DEFAULT_MIN_P: f32 = 0.05;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.0;
pub const DEFAULT_PENALTY_LAST_N: i32 = 64;

/// Sampler defaults a model author embedded in the GGUF under `general.sampling.*`
/// (llama.cpp reads these too — see `common/common.cpp`). Each field is `None`
/// when the corresponding key is absent. A CLI flag overrides any of these; an
/// absent field falls back to the `DEFAULT_*` built-ins above.
#[derive(Debug, Clone, Copy, Default)]
pub struct GgufSamplingDefaults {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub penalty_last_n: Option<i32>,
}

impl GgufSamplingDefaults {
    /// Read the `general.sampling.*` keys from a GGUF (the set llama.cpp also
    /// honors). Missing keys stay `None`.
    pub fn from_gguf(gguf: &crate::gguf::GgufFile) -> Self {
        Self {
            temperature: gguf.meta_f32("general.sampling.temp"),
            top_k: gguf.meta_u32("general.sampling.top_k"),
            top_p: gguf.meta_f32("general.sampling.top_p"),
            min_p: gguf.meta_f32("general.sampling.min_p"),
            repeat_penalty: gguf.meta_f32("general.sampling.penalty_repeat"),
            penalty_last_n: gguf.meta_i32("general.sampling.penalty_last_n"),
        }
    }

    /// True when the GGUF supplied at least one sampling override (for logging).
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.repeat_penalty.is_none()
            && self.penalty_last_n.is_none()
    }
}

/// User-facing sampler knobs. Mirrors the relevant subset of llama.cpp's
/// `common_params_sampling`.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0.0 → greedy (skip stochastic chain entirely).
    pub temperature: f32,
    /// 0 → top_k disabled (operate on full vocab — slow path for large vocabs).
    pub top_k: u32,
    /// 1.0 → top_p disabled.
    pub top_p: f32,
    /// 0.0 → min_p disabled.
    pub min_p: f32,
    /// 0.0 → no presence penalty.
    pub presence_penalty: f32,
    /// 0.0 → no frequency penalty.
    pub frequency_penalty: f32,
    /// 1.0 → no repetition penalty.
    pub repeat_penalty: f32,
    /// Number of trailing generated tokens that count toward the penalties.
    pub penalty_last_n: usize,
    /// RNG seed for stochastic sampling.
    pub seed: u64,
    /// Static `(token_id, bias)` adjustments added to the raw logits before
    /// any other stage (`--logit-bias`). Empty → no bias dispatch. A `-inf`
    /// bias hard-bans a token; `+inf` forces it. Constant for the session.
    pub logit_bias: Vec<(u32, f32)>,
}

impl Default for SamplerConfig {
    /// llama.cpp's `common_params_sampling` defaults — model-agnostic and
    /// sensible across families (Llama, Qwen, Mistral, …).
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_k: DEFAULT_TOP_K,
            top_p: DEFAULT_TOP_P,
            min_p: DEFAULT_MIN_P,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: DEFAULT_REPEAT_PENALTY,
            penalty_last_n: DEFAULT_PENALTY_LAST_N as usize,
            seed: 0,
            logit_bias: Vec::new(),
        }
    }
}

impl SamplerConfig {
    /// Build a config from raw CLI fields, resolving `penalty_last_n < 0` to the
    /// whole context window (`ctx_size`) — llama.cpp's `--repeat-last-n -1`.
    /// Shared by `seeker chat` and `seeker serve` so both map flags identically.
    #[allow(clippy::too_many_arguments)]
    pub fn from_cli(
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        repeat_penalty: f32,
        penalty_last_n: i32,
        ctx_size: u32,
        seed: u64,
        logit_bias: Vec<(u32, f32)>,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p,
            presence_penalty,
            frequency_penalty,
            repeat_penalty,
            // -1 → whole context window; otherwise the literal count (0 = off).
            penalty_last_n: if penalty_last_n < 0 {
                ctx_size as usize
            } else {
                penalty_last_n as usize
            },
            seed,
            logit_bias,
        }
    }

    /// True if greedy short-circuit applies. Penalties still run.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// True if at least one penalty term is non-identity.
    pub fn any_penalty(&self) -> bool {
        self.repeat_penalty != 1.0 || self.frequency_penalty != 0.0 || self.presence_penalty != 0.0
    }

    /// Hash of every config field that affects the recorded GPU graph
    /// shape or spec constants. Used by the persistent-decode-cmdbuf
    /// path to invalidate the cached recording when the caller changes
    /// sampler knobs mid-session. RNG seed and the recent-token window
    /// are deliberately excluded — they affect only the values fed
    /// into the dispatched graph, not the graph itself.
    pub fn graph_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        // Cast floats to bit patterns so NaN compares structurally and
        // we don't have to bound `Hash` on `SamplerConfig`.
        self.temperature.to_bits().hash(&mut h);
        self.top_k.hash(&mut h);
        self.top_p.to_bits().hash(&mut h);
        self.min_p.to_bits().hash(&mut h);
        self.repeat_penalty.to_bits().hash(&mut h);
        self.frequency_penalty.to_bits().hash(&mut h);
        self.presence_penalty.to_bits().hash(&mut h);
        self.penalty_last_n.hash(&mut h);
        // logit_bias is uploaded to scratch at record time, so any change
        // (count *or* value) must force a re-record to re-upload it.
        for &(tid, b) in &self.logit_bias {
            tid.hash(&mut h);
            b.to_bits().hash(&mut h);
        }
        h.finish()
    }
}

/// Owns the per-step RNG and the recent-token bookkeeping. Recorded into the
/// command buffer by [`Sampler::record_chain`]; accepts each sampled token
/// via [`Sampler::accept`] so the next step sees the updated penalty window.
pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,
    recent: VecDeque<u32>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            rng,
            recent: VecDeque::new(),
        }
    }

    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Clear the recent-token ring (used when a `/clear` ends one logical
    /// conversation but the same sampler instance keeps generating). The
    /// RNG state is left alone so deterministic seeds still reproduce.
    pub fn reset_recent(&mut self) {
        self.recent.clear();
    }

    /// Update the recent-token window after a token has been sampled. The
    /// next `record_chain` call will use this for penalties.
    pub fn accept(&mut self, token: u32) {
        let n = self.config.penalty_last_n;
        if n == 0 {
            return;
        }
        if self.recent.len() == n {
            self.recent.pop_front();
        }
        self.recent.push_back(token);
    }

    /// Build the `(token_id, count)` list for the penalty shader from the
    /// current recent-token window. Returns at most `penalty_last_n` entries.
    pub fn penalty_pairs(&self) -> Vec<(u32, u32)> {
        if self.recent.is_empty() {
            return Vec::new();
        }
        let mut counts: std::collections::HashMap<u32, u32> = Default::default();
        for &t in &self.recent {
            *counts.entry(t).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// Draw a uniform `[0, 1)` float for the categorical step. Done on host
    /// because we need a tiny upload anyway.
    pub fn draw_uniform(&mut self) -> f32 {
        use rand::Rng;
        self.rng.r#gen::<f32>()
    }

    /// Host-side refresh of the sampler-owned scratch slots between
    /// submits of a cached decode command buffer. Pairs with
    /// [`Model::refresh_replay_inputs`] (the model handles its own
    /// token/positions buffers); this writes the RNG uniform and the
    /// `(token_id, count)` penalty pairs + count into the slots whose
    /// offsets were captured during the recording pass.
    ///
    /// Mutates the sampler RNG state — call exactly once per replay.
    // `host_ptr` is a caller-mapped GPU scratch pointer (deref is `unsafe`-scoped).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn refresh_replay_inputs(
        &mut self,
        host_ptr: *mut u8,
        plan: &decode_dyn::ReplayPlan,
    ) -> Result<(), Box<dyn Error>> {
        // Uniform draw goes into the DecodeDyn slot. Greedy paths use 0
        // (sample_categorical isn't recorded in that case anyway).
        let uniform = if self.config.is_greedy() {
            0.0
        } else {
            self.draw_uniform()
        };
        decode_dyn::write_field(
            host_ptr,
            plan.decode_dyn_offset,
            decode_dyn::OFFSET_UNIFORM_RNG,
            uniform,
        );

        // Penalty pairs (when the chain recorded apply_penalties).
        if let Some((pairs_off, max_pairs)) = plan.penalty_pairs {
            let pairs = self.penalty_pairs();
            if pairs.len() as u32 > max_pairs {
                return Err(format!(
                    "penalty_pairs {} exceeds recorded max {max_pairs}",
                    pairs.len()
                )
                .into());
            }
            unsafe {
                let dst = host_ptr.add(pairs_off as usize) as *mut u32;
                for (i, &(tid, count)) in pairs.iter().enumerate() {
                    std::ptr::write(dst.add(2 * i), tid);
                    std::ptr::write(dst.add(2 * i + 1), count);
                }
                for i in pairs.len()..max_pairs as usize {
                    std::ptr::write(dst.add(2 * i), 0u32);
                    std::ptr::write(dst.add(2 * i + 1), 0u32);
                }
            }
            decode_dyn::write_field(
                host_ptr,
                plan.decode_dyn_offset,
                decode_dyn::OFFSET_PENALTY_COUNT,
                pairs.len() as u32,
            );
        }
        Ok(())
    }

    /// Host-side speculative **sample-and-compare** over a batch of
    /// verify-position logits. This is the acceptance step for MTP
    /// speculative decoding and is *lossless at any temperature*: at each
    /// position we draw a genuine sample from the main-model distribution
    /// (the same sampler chain the GPU path runs, applied on the host) and
    /// accept the draft only if our sample equals it. Every emitted token
    /// is therefore a faithful sample of the target — the draft only
    /// decides when we stop.
    ///
    /// `logits[i]` is the main model's vocab-wide logits at verify
    /// position `i` (length `n_draft + 1`). `drafts[i]` is the MTP head's
    /// proposal for position `i` (length `n_draft`). Returns the emitted
    /// tokens `s_0..s_accept_len` (length `accept_len + 1`, always ≥ 1):
    /// the matched prefix plus one bonus/corrective token.
    ///
    /// Mutates the sampler RNG + recent-token window for **every** emitted
    /// token (we never sample past the stop point, so there is nothing to
    /// roll back). Penalties at position `i` see the window updated by the
    /// tokens accepted at `0..i` — matching llama.cpp's per-token
    /// `common_sampler_accept` within a speculative batch.
    pub fn sample_and_compare(&mut self, logits: &[Vec<f32>], drafts: &[u32]) -> Vec<u32> {
        let n = drafts.len();
        debug_assert_eq!(logits.len(), n + 1, "need n_draft+1 logit rows");
        let mut emitted = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let tok = self.sample_one(&logits[i]);
            self.accept(tok);
            emitted.push(tok);
            if i < n && tok == drafts[i] {
                continue; // accepted — verify the next draft
            }
            break; // mismatch (or the final position) — tok is the bonus token
        }
        emitted
    }

    /// Draw one token from a single row of vocab-wide logits on the host,
    /// mirroring the GPU chain order
    /// (penalties → top_k → top_p → min_p → temp → categorical). Greedy
    /// (`temperature == 0`) short-circuits to argmax after penalties.
    /// Used by [`sample_and_compare`]; consumes one RNG draw on the
    /// stochastic path.
    pub fn sample_one(&mut self, logits_in: &[f32]) -> u32 {
        let cfg = &self.config;
        let mut logits: Vec<f32> = logits_in.to_vec();

        // 0. Static logit bias first (matches the GPU chain / llama.cpp): a
        //    `-inf` bias bans a token, `+inf` forces it.
        for &(tid, b) in &cfg.logit_bias {
            if let Some(l) = logits.get_mut(tid as usize) {
                *l += b;
            }
        }

        // 1. Penalties on the raw logits (sign-conditional repeat penalty,
        //    then frequency/presence), matching `apply_penalties_f32`.
        if cfg.any_penalty() {
            for (tid, count) in self.penalty_pairs() {
                let l = &mut logits[tid as usize];
                if cfg.repeat_penalty != 1.0 {
                    *l = if *l > 0.0 {
                        *l / cfg.repeat_penalty
                    } else {
                        *l * cfg.repeat_penalty
                    };
                }
                *l -= count as f32 * cfg.frequency_penalty;
                if count > 0 {
                    *l -= cfg.presence_penalty;
                }
            }
        }

        // 2. Greedy short-circuit — argmax (lowest index on ties).
        if cfg.is_greedy() {
            return argmax(&logits);
        }

        // 3. top_k: candidate ids sorted DESC by logit, truncated to k.
        let vocab = logits.len();
        let mut cand: Vec<u32> = (0..vocab as u32).collect();
        let k = if cfg.top_k > 0 && (cfg.top_k as usize) < vocab {
            cfg.top_k as usize
        } else {
            vocab
        };
        // Sort by logit DESC, breaking ties by ascending id (deterministic).
        cand.sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        cand.truncate(k);
        let mut kept: Vec<f32> = cand.iter().map(|&i| logits[i as usize]).collect();

        // 4. top_p: keep tokens while the *exclusive* cumulative prob ≤ p
        //    (so the crossing token is kept, matching llama.cpp / the GPU
        //    `record_top_p`). Operates on softmax of the kept logits.
        if cfg.top_p < 1.0 && cfg.top_p > 0.0 {
            let probs = softmax(&kept);
            let mut cum = 0.0f32; // exclusive cumulative
            for j in 0..kept.len() {
                if cum > cfg.top_p {
                    kept[j] = f32::NEG_INFINITY;
                }
                cum += probs[j];
            }
        }

        // 5. min_p: log-space cutoff = max_logit + ln(min_p) (kept[0] is the
        //    max since `cand` is sorted DESC), matching the GPU `record_min_p`.
        if cfg.min_p > 0.0 {
            let max_logit = kept[0];
            let cutoff = max_logit + cfg.min_p.ln();
            for v in kept.iter_mut() {
                if *v < cutoff {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        // 6. Temperature scale (last, before the categorical softmax).
        if (cfg.temperature - 1.0).abs() > 1e-9 {
            let inv_t = 1.0 / cfg.temperature.max(1e-9);
            for v in kept.iter_mut() {
                if v.is_finite() {
                    *v *= inv_t;
                }
            }
        }

        // 7. Categorical inverse-CDF sample with the per-step uniform draw.
        //    Mirrors the GPU `record_categorical`: pick the first index whose
        //    cumulative probability ≥ u.
        let probs = softmax(&kept);
        let u = self.draw_uniform();
        let mut cum = 0.0f32;
        let mut chosen = kept.len() - 1; // clamp to last on FP shortfall
        for (j, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= u {
                chosen = j;
                break;
            }
        }
        cand[chosen]
    }

    /// Record the sampler chain into the command buffer. Returns a
    /// 4-byte `BufferRange` holding the sampled token id (u32). The engine
    /// reads back exactly those 4 bytes.
    ///
    /// `logits` is the F32 vocab-wide tensor produced by the model's forward
    /// pass — shape `[n_vocab, 1, 1, 1]`.
    pub fn record_chain(
        &mut self,
        ctx: &mut DispatchContext,
        logits: TensorView,
    ) -> Result<BufferRange, Box<dyn Error>> {
        // Boundary marker — everything emitted from here until the
        // next `mark(…)` (or end of forward) is attributed to
        // BlockClass::Sampler. The final `forward_sampled` mark closes
        // it implicitly via the pair-walk on host readback.
        ctx.mark(super::profile::BlockClass::Sampler);
        let pairs = self.penalty_pairs();
        let uniform = if self.config.is_greedy() {
            0.0
        } else {
            self.draw_uniform()
        };
        super::ops::sampler::record_chain(ctx, &self.config, logits, &pairs, uniform)
    }
}

/// Index of the largest element, ties broken by lowest index (matching the
/// GPU argmax reduce). Empty input returns 0.
fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Numerically-stable softmax over a slice. `-inf` entries (filtered by
/// top_p / min_p masking) map to probability 0.
fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All masked / empty — uniform fallback avoids NaN.
        let n = xs.len().max(1) as f32;
        return vec![1.0 / n; xs.len()];
    }
    let mut out: Vec<f32> = xs.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out
}
