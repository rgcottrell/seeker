//! Shared, GPU-free embedding post-processing: apply the model's final
//! `output_norm` to each token's hidden state, pool over positions, and
//! normalize. Used by both the `seeker embedding` CLI command and the
//! `seeker serve` embedding endpoints, so the two can never diverge.
//!
//! The transformer forward (`Engine::forward_full_readback`) returns the
//! per-position pre-`output_norm` residual `[n_embd, L]` (position-major); this
//! module turns that into the final embedding(s), matching llama.cpp's
//! *norm-all-then-pool* order.

use clap::ValueEnum;

/// Pooling over token hidden states (llama.cpp `--pooling`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Pooling {
    /// Hidden state of the last token (Qwen3-Embedding default).
    Last,
    /// Mean over all token positions.
    Mean,
    /// First token ([CLS]).
    Cls,
    /// No pooling — one (normalized) vector per token.
    None,
}

impl Pooling {
    /// Map a GGUF `*.pooling_type` value (0 none, 1 mean, 2 cls, 3 last, …) to a
    /// [`Pooling`]; anything else (incl. absent) defaults to `Last`.
    pub fn from_gguf(pooling_type: Option<u32>) -> Self {
        match pooling_type {
            Some(0) => Pooling::None,
            Some(1) => Pooling::Mean,
            Some(2) => Pooling::Cls,
            _ => Pooling::Last, // 3 (last) or unspecified
        }
    }
}

/// RMSNorm a single hidden-state column with the learned weight (matches the
/// `rms_norm.slang` kernel: `x / sqrt(mean(x²)+eps) * w`).
pub fn rmsnorm_col(col: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = col.len() as f32;
    let ms = col.iter().map(|x| x * x).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    col.iter().zip(weight).map(|(x, w)| x * inv * w).collect()
}

/// Pool the per-position (already output_norm'd) vectors into the embedding(s).
/// Returns one vector for `Last`/`Mean`/`Cls`, or `L` vectors for `None`.
pub fn pool(normed: &[Vec<f32>], pooling: Pooling) -> Vec<Vec<f32>> {
    match pooling {
        Pooling::Last => vec![normed.last().cloned().unwrap_or_default()],
        Pooling::Cls => vec![normed.first().cloned().unwrap_or_default()],
        Pooling::Mean => {
            let l = normed.len().max(1);
            let dim = normed.first().map(Vec::len).unwrap_or(0);
            let mut acc = vec![0.0f32; dim];
            for v in normed {
                for (a, x) in acc.iter_mut().zip(v) {
                    *a += *x;
                }
            }
            for a in &mut acc {
                *a /= l as f32;
            }
            vec![acc]
        }
        Pooling::None => normed.to_vec(),
    }
}

/// In-place embedding normalization, matching llama.cpp `common_embd_normalize`:
/// p<0 none, 0 max-abs, 1 L1/taxicab, 2 L2/euclidean, p>2 p-norm.
pub fn normalize(v: &mut [f32], p: i32) {
    let sum: f64 = match p {
        i32::MIN..=-1 => 1.0,
        0 => v.iter().fold(0.0f64, |m, x| m.max(x.abs() as f64)),
        1 => v.iter().map(|x| x.abs() as f64).sum(),
        2 => v
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt(),
        p => {
            let pf = p as f64;
            v.iter()
                .map(|x| (x.abs() as f64).powf(pf))
                .sum::<f64>()
                .powf(1.0 / pf)
        }
    };
    let norm = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for x in v.iter_mut() {
        *x = (*x as f64 * norm) as f32;
    }
}

/// End-to-end embedding extraction from a forward's residual: apply `output_norm`
/// to every position (`residual` is `[n_embd, L]` position-major), pool, then
/// normalize each pooled vector. The single source of truth for the CLI command
/// and the serve endpoints.
pub fn pool_and_normalize(
    residual: &[f32],
    n_embd: usize,
    output_norm: &[f32],
    eps: f32,
    pooling: Pooling,
    embd_normalize: i32,
) -> Vec<Vec<f32>> {
    let l = residual.len().checked_div(n_embd).unwrap_or(0);
    let normed: Vec<Vec<f32>> = (0..l)
        .map(|t| rmsnorm_col(&residual[t * n_embd..(t + 1) * n_embd], output_norm, eps))
        .collect();
    let mut pooled = pool(&normed, pooling);
    for v in &mut pooled {
        normalize(v, embd_normalize);
    }
    pooled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_col_matches_hand_compute() {
        // x=[1,2,3], w=[1,1,1], eps=0 → ms=14/3, inv=1/sqrt(14/3).
        let out = rmsnorm_col(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 0.0);
        let inv = 1.0f32 / (14.0f32 / 3.0).sqrt();
        for (o, x) in out.iter().zip([1.0, 2.0, 3.0]) {
            assert!((o - x * inv).abs() < 1e-6, "{o} vs {}", x * inv);
        }
    }

    #[test]
    fn rmsnorm_col_applies_weight() {
        let out = rmsnorm_col(&[1.0, 1.0], &[2.0, 4.0], 0.0);
        assert!((out[0] - 2.0).abs() < 1e-6 && (out[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn pool_selects_right_column() {
        let cols = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0]];
        assert_eq!(pool(&cols, Pooling::Last)[0], vec![3.0, 0.0]);
        assert_eq!(pool(&cols, Pooling::Cls)[0], vec![1.0, 0.0]);
        assert_eq!(pool(&cols, Pooling::Mean)[0], vec![2.0, 0.0]);
        assert_eq!(pool(&cols, Pooling::None).len(), 3);
    }

    #[test]
    fn normalize_l2_is_unit() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v, 2);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_l1_and_none() {
        let mut v = vec![1.0, 3.0];
        normalize(&mut v, 1);
        assert!((v[0] - 0.25).abs() < 1e-6 && (v[1] - 0.75).abs() < 1e-6);
        let mut w = vec![1.0, 3.0];
        normalize(&mut w, -1);
        assert_eq!(w, vec![1.0, 3.0]);
    }

    #[test]
    fn from_gguf_maps_pooling_type() {
        assert_eq!(Pooling::from_gguf(Some(3)), Pooling::Last);
        assert_eq!(Pooling::from_gguf(Some(1)), Pooling::Mean);
        assert_eq!(Pooling::from_gguf(Some(2)), Pooling::Cls);
        assert_eq!(Pooling::from_gguf(Some(0)), Pooling::None);
        assert_eq!(Pooling::from_gguf(None), Pooling::Last);
    }

    #[test]
    fn pool_and_normalize_last_token_l2() {
        // n_embd=2, L=2; output_norm=[1,1], eps=0. residual columns:
        // t0=[3,4], t1=[1,0]. Last pooling → norm(t1) with rms then L2.
        let residual = vec![3.0, 4.0, 1.0, 0.0];
        let out = pool_and_normalize(&residual, 2, &[1.0, 1.0], 0.0, Pooling::Last, 2);
        assert_eq!(out.len(), 1);
        // rmsnorm([1,0]) = [1,0]*1/sqrt(0.5) = [sqrt2,0]; L2 → [1,0].
        assert!((out[0][0] - 1.0).abs() < 1e-6 && out[0][1].abs() < 1e-6);
    }

    #[test]
    fn pool_and_normalize_none_returns_per_token() {
        let residual = vec![3.0, 4.0, 1.0, 0.0];
        let out = pool_and_normalize(&residual, 2, &[1.0, 1.0], 0.0, Pooling::None, 2);
        assert_eq!(out.len(), 2);
        for v in &out {
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-6);
        }
    }
}
