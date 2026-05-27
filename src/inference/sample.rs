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
}

impl Default for SamplerConfig {
    /// llama.cpp's `common_params_sampling` defaults — model-agnostic and
    /// sensible across families (Llama, Qwen, Mistral, …).
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            penalty_last_n: 64,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    /// True if greedy short-circuit applies. Penalties still run.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// True if at least one penalty term is non-identity.
    pub fn any_penalty(&self) -> bool {
        self.repeat_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
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
