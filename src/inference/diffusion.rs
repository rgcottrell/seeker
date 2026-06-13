//! Entropy-bound diffusion denoiser for `diffusion-gemma` (non-autoregressive
//! block text-diffusion). Ports llama.cpp's `diffusion_generate_entropy_bound`:
//! a fixed-length "canvas" is initialized with random tokens and iteratively
//! refined over a temperature schedule. Each step runs one bidirectional
//! forward (`[prompt | canvas]` → per-canvas-position logits), then on the host:
//! argmax / partition-function / entropy / multinomial-sample per position,
//! accept the lowest-entropy positions within a cumulative-entropy bound,
//! renoise the rest, and stop early once the argmax canvas stabilizes and its
//! mean entropy drops below a threshold. Longer-than-canvas replies chain
//! blocks: a committed block becomes context (prompt) for the next.
//!
//! The per-position host reduction reads the full `[vocab, C]` logits each step
//! (a big readback); phase 4 moves it (and self-conditioning) onto the GPU.

use std::error::Error;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Entropy-bound denoiser knobs. Defaults match llama.cpp's `diffusion_eb_params`.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Maximum denoising steps per block (`S`).
    pub steps: u32,
    /// Temperature at the last step (most confident).
    pub t_min: f32,
    /// Temperature at the first step (most noisy).
    pub t_max: f32,
    /// Cumulative-entropy (mutual-information) acceptance bound, in nats.
    pub entropy_bound: f32,
    /// Argmax canvas must hold stable this many steps before stopping.
    pub stability_threshold: u32,
    /// Mean canvas entropy must fall below this (nats) to stop.
    pub confidence_threshold: f32,
    /// RNG seed (canvas init, multinomial draws, renoise).
    pub seed: u64,
    /// Cap on total generated tokens across blocks (multi-block chaining).
    pub max_tokens: usize,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            steps: 48,
            t_min: 0.4,
            t_max: 0.8,
            entropy_bound: 0.1,
            stability_threshold: 1,
            confidence_threshold: 0.005,
            seed: 0,
            max_tokens: 256,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct PosOut {
    argmax: u32,
    entropy: f32,
    sampled: u32,
}

/// Per-canvas-position reduction over one logits row (`[vocab]`): temperature-
/// scaled argmax, partition function `Z`, Shannon entropy `H`, and an
/// inverse-CDF multinomial sample using the pre-drawn uniform `u`. `exps` is a
/// reusable `[vocab]` scratch (one per worker thread).
fn reduce_position(row: &[f32], temp_inv: f32, u: f32, exps: &mut [f32]) -> PosOut {
    let n = row.len();
    let mut m = f32::NEG_INFINITY;
    let mut amax = 0u32;
    for (v, &lg) in row.iter().enumerate() {
        let z = lg * temp_inv;
        if z > m {
            m = z;
            amax = v as u32;
        }
    }
    let mut z_sum = 0f32;
    for v in 0..n {
        let e = (row[v] * temp_inv - m).exp();
        exps[v] = e;
        z_sum += e;
    }
    let target = u * z_sum;
    let mut cum = 0f32;
    let mut h = 0f32;
    let mut sampled = (n - 1) as u32;
    let mut picked = false;
    for (v, &e) in exps.iter().enumerate().take(n) {
        let p = e / z_sum;
        if p > 0.0 {
            h -= p * p.ln();
        }
        cum += e;
        if !picked && cum >= target {
            sampled = v as u32;
            picked = true;
        }
    }
    PosOut {
        argmax: amax,
        entropy: h,
        sampled,
    }
}

/// Reduce all `c` canvas positions of `logits` (`[vocab, c]`, column-major) in
/// parallel across worker threads.
fn reduce_canvas(
    logits: &[f32],
    c: usize,
    n_vocab: usize,
    temp_inv: f32,
    u: &[f32],
) -> Vec<PosOut> {
    let mut outs = vec![PosOut::default(); c];
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(c.max(1));
    let chunk = c.div_ceil(n_threads.max(1));
    std::thread::scope(|scope| {
        for (ci, out_chunk) in outs.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            scope.spawn(move || {
                let mut exps = vec![0f32; n_vocab];
                for (j, slot) in out_chunk.iter_mut().enumerate() {
                    let pos = base + j;
                    let row = &logits[pos * n_vocab..(pos + 1) * n_vocab];
                    *slot = reduce_position(row, temp_inv, u[pos], &mut exps);
                }
            });
        }
    });
    outs
}

/// Denoise one canvas block against `prompt_ext` (the prompt plus any
/// already-committed blocks). Returns the final argmax canvas (`canvas_len`
/// tokens). `forward(full, n_prompt)` runs the bidirectional forward over
/// `[prompt_ext | canvas]` and returns the canvas logits `[vocab, C]`.
fn denoise_block<F>(
    prompt_ext: &[u32],
    canvas_len: usize,
    n_vocab: usize,
    cfg: &DiffusionConfig,
    rng: &mut StdRng,
    forward: &mut F,
) -> Result<Vec<u32>, Box<dyn Error>>
where
    F: FnMut(&[u32], u32) -> Result<Vec<f32>, Box<dyn Error>>,
{
    let big_p = prompt_ext.len();
    let c = canvas_len;
    let s = cfg.steps.max(1);
    let vocab_u = n_vocab as u32;

    let mut canvas: Vec<u32> = (0..c).map(|_| rng.gen_range(0..vocab_u)).collect();
    let mut argmax_canvas = vec![0u32; c];
    let mut prev_argmax = vec![u32::MAX; c];
    let mut held = 0u32;
    let mut full: Vec<u32> = Vec::with_capacity(big_p + c);

    for cur_step in (1..=s).rev() {
        full.clear();
        full.extend_from_slice(prompt_ext);
        full.extend_from_slice(&canvas);

        let t = cfg.t_min + (cfg.t_max - cfg.t_min) * (cur_step as f32 / s as f32);
        let temp_inv = 1.0 / t;

        let logits = forward(&full, big_p as u32)?;
        if logits.len() != c * n_vocab {
            return Err(format!(
                "diffusion forward returned {} logits, expected {}",
                logits.len(),
                c * n_vocab
            )
            .into());
        }

        // Pre-draw randomness single-threaded so the seed is reproducible
        // regardless of the worker-thread split.
        let u: Vec<f32> = (0..c).map(|_| rng.r#gen::<f32>()).collect();
        let renoise: Vec<u32> = (0..c).map(|_| rng.gen_range(0..vocab_u)).collect();

        let outs = reduce_canvas(&logits, c, n_vocab, temp_inv, &u);

        // Accept the lowest-entropy positions whose cumulative entropy (before
        // this one) stays within the bound; renoise the rest.
        let mut order: Vec<usize> = (0..c).collect();
        order.sort_by(|&a, &b| {
            outs[a]
                .entropy
                .partial_cmp(&outs[b].entropy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut accepted = vec![false; c];
        let mut cum_e = 0f64;
        for &pos in &order {
            cum_e += outs[pos].entropy as f64;
            if cum_e - outs[pos].entropy as f64 <= cfg.entropy_bound as f64 {
                accepted[pos] = true;
            }
        }

        let mut entropy_sum = 0f32;
        for pos in 0..c {
            canvas[pos] = if accepted[pos] {
                outs[pos].sampled
            } else {
                renoise[pos]
            };
            argmax_canvas[pos] = outs[pos].argmax;
            entropy_sum += outs[pos].entropy;
        }

        // Adaptive stop: argmax stable for `stability_threshold` steps AND mean
        // entropy below `confidence_threshold`.
        held = if prev_argmax == argmax_canvas {
            held + 1
        } else {
            0
        };
        let confident = (entropy_sum / c as f32) < cfg.confidence_threshold;
        if held >= cfg.stability_threshold && confident {
            break;
        }
        prev_argmax.copy_from_slice(&argmax_canvas);
    }

    Ok(argmax_canvas)
}

/// Generate a reply by chaining diffusion blocks until a stop token or
/// `cfg.max_tokens`. `forward(full, n_prompt)` runs the canvas forward;
/// `on_block(tokens)` streams each committed block (trimmed at the first
/// end-of-generation token). Returns all generated tokens (EOG-trimmed).
pub fn generate<F>(
    prompt: &[u32],
    canvas_len: usize,
    n_vocab: usize,
    eog_ids: &[u32],
    cfg: &DiffusionConfig,
    mut forward: F,
    mut on_block: impl FnMut(&[u32]),
) -> Result<Vec<u32>, Box<dyn Error>>
where
    F: FnMut(&[u32], u32) -> Result<Vec<f32>, Box<dyn Error>>,
{
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut prompt_ext = prompt.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    while generated.len() < cfg.max_tokens {
        let canvas = denoise_block(
            &prompt_ext,
            canvas_len,
            n_vocab,
            cfg,
            &mut rng,
            &mut forward,
        )?;

        // Trim the block at the first end-of-generation token (exclusive).
        if let Some(e) = canvas.iter().position(|t| eog_ids.contains(t)) {
            let emit = &canvas[..e];
            generated.extend_from_slice(emit);
            on_block(emit);
            break;
        }
        generated.extend_from_slice(&canvas);
        on_block(&canvas);
        prompt_ext.extend_from_slice(&canvas);
    }

    generated.truncate(cfg.max_tokens);
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_position_argmax_and_low_entropy_sample() {
        // A near-one-hot distribution: vocab 4, logit 100 at index 2.
        let row = [0.0f32, 0.0, 100.0, 0.0];
        let mut exps = vec![0f32; 4];
        let out = reduce_position(&row, 1.0, 0.5, &mut exps);
        assert_eq!(out.argmax, 2);
        // Sampling concentrates on index 2; entropy ~ 0.
        assert_eq!(out.sampled, 2);
        assert!(out.entropy < 1e-3, "entropy {} should be ~0", out.entropy);
    }

    #[test]
    fn reduce_position_uniform_high_entropy() {
        let row = [0.0f32, 0.0, 0.0, 0.0];
        let mut exps = vec![0f32; 4];
        let out = reduce_position(&row, 1.0, 0.99, &mut exps);
        // Uniform over 4 → entropy ln(4).
        assert!((out.entropy - (4f32).ln()).abs() < 1e-4);
        // u=0.99 lands in the last bin.
        assert_eq!(out.sampled, 3);
    }

    #[test]
    fn generate_chains_blocks_and_stops_on_eog() {
        // forward always makes the canvas argmax to a fixed pattern ending in
        // EOG id 9, so generation stops mid-canvas on the first block.
        let n_vocab = 16;
        let canvas_len = 4;
        let eog = [9u32];
        let cfg = DiffusionConfig {
            steps: 1,
            confidence_threshold: f32::INFINITY, // force a single-step stop
            stability_threshold: 0,
            max_tokens: 100,
            ..Default::default()
        };
        // Logits: position p gets argmax token (p<2 ? p+1 : 9).
        let forward = |full: &[u32], n_prompt: u32| -> Result<Vec<f32>, Box<dyn Error>> {
            let c = full.len() - n_prompt as usize;
            let mut logits = vec![0f32; c * n_vocab];
            for pos in 0..c {
                let tok = if pos < 2 { pos + 1 } else { 9 };
                logits[pos * n_vocab + tok] = 50.0;
            }
            Ok(logits)
        };
        let mut emitted = Vec::new();
        let out = generate(
            &[100, 101],
            canvas_len,
            n_vocab,
            &eog,
            &cfg,
            forward,
            |blk| emitted.extend_from_slice(blk),
        )
        .unwrap();
        // Tokens 1,2 then EOG(9) at pos 2 → emit [1,2], stop.
        assert_eq!(out, vec![1, 2]);
        assert_eq!(emitted, vec![1, 2]);
    }
}
