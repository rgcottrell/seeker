//! M3 — the attention core on the NPU: per-head GQA QKᵀ → causal mask → softmax → ·V,
//! for all 16 q-heads (8 kv-heads, ratio 2), producing attn_out[L][q_dim].
//!
//! To isolate the attention math, the inputs (q_roped/k_roped/v) are computed host-side
//! in f32 from real blk.0 weights and quantized to bf16; the scale 1/√128 is folded into
//! q. The on-NPU path is then:
//!   scores_h[L][1024] = q_h[L][128] · k_kv_padᵀ        (b_col_maj GEMM; keys padded
//!                                                        512→1024 so each row is a
//!                                                        softmax tile)
//!   scores += mask   (causal + pad columns → −1e4)      (batched over all heads)
//!   probs   = softmax(scores)                            (per-1024-row, batched)
//!   out_h[L][128] = probs_h[L][1024] · v_kv_pad[1024][256][:, :128]
//! Validated by cosine vs a host f32 attention reference.
//!
//! N=head_dim=128 violates the GEMM N%256 rule, so ·V pads V's feature dim to 256 and
//! slices the first 128 back out. Build (besides earlier kernels):
//!   kernels/gemm/build.sh 512 128 1024 bf16 1   (QKᵀ)
//!   kernels/gemm/build.sh 512 1024 256 bf16 0   (·V)
//!   kernels/norm/build.sh softmax 8388608
//!   kernels/eltwise/build.sh add bf16 8388608
//! then: cargo run -p seeker-npu --example attention
#[path = "common/mod.rs"]
mod common;
use common::*;
use seeker_core::gguf::GgufFile;

const L: usize = 512;
const HEAD_DIM: usize = 128;
const RMS_EPS: f32 = 1e-5;
const ROPE_BASE: f32 = 1e6;
const KEYS: usize = 1024; // L padded up to the softmax tile width
const VPAD: usize = 256; // head_dim padded up to the GEMM N%256 rule
const MASK_NEG: f32 = -1e4; // exp(−1e4 − max) → 0; bf16-representable

/// Host: per-head rmsnorm(128)·norm_w + NEOX rope on a token-major projection.
fn norm_rope(proj: &[f32], n_heads: usize, norm_w: &[f32]) -> Vec<f32> {
    let d = n_heads * HEAD_DIM;
    let mut out = vec![0.0f32; L * d];
    for t in 0..L {
        for h in 0..n_heads {
            let base = t * d + h * HEAD_DIM;
            let head = &proj[base..base + HEAD_DIM];
            let ms = head.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32;
            let inv = 1.0 / (ms + RMS_EPS).sqrt();
            let nd: Vec<f32> = (0..HEAD_DIM).map(|i| head[i] * inv * norm_w[i]).collect();
            for j in 0..HEAD_DIM / 2 {
                let theta = t as f32 * ROPE_BASE.powf(-2.0 * j as f32 / HEAD_DIM as f32);
                let (c, s) = (theta.cos(), theta.sin());
                out[base + j] = nd[j] * c - nd[j + HEAD_DIM / 2] * s;
                out[base + j + HEAD_DIM / 2] = nd[j] * s + nd[j + HEAD_DIM / 2] * c;
            }
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("NPU_QWEN3_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let gguf = GgufFile::open(&model)?;
    let te = gguf
        .tensor("token_embd.weight")
        .ok_or("missing token_embd")?;
    let (n_embd, vocab) = (te.dims[0] as usize, te.dims[1] as usize);
    let q_dim = gguf.tensor("blk.0.attn_q.weight").ok_or("missing wq")?.dims[1] as usize;
    let kv_dim = gguf.tensor("blk.0.attn_v.weight").ok_or("missing wv")?.dims[1] as usize;
    let (n_head, n_kv) = (q_dim / HEAD_DIM, kv_dim / HEAD_DIM);
    let gqa = n_head / n_kv;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    println!("n_head={n_head} n_kv={n_kv} gqa={gqa} q_dim={q_dim} kv_dim={kv_dim} L={L}");

    // ── host: build q_roped (scaled), k_roped, v from real weights ──
    let token_embd = f16_to_f32(&gguf, "token_embd.weight")?;
    let ids: Vec<usize> = (0..L).map(|t| (t * 37 + 1) % vocab).collect();
    let attn_norm = f32_vec(&gguf, "blk.0.attn_norm.weight", n_embd)?;
    let wq = f16_to_f32(&gguf, "blk.0.attn_q.weight")?;
    let wk = f16_to_f32(&gguf, "blk.0.attn_k.weight")?;
    let wv = f16_to_f32(&gguf, "blk.0.attn_v.weight")?;
    let q_norm = f32_vec(&gguf, "blk.0.attn_q_norm.weight", HEAD_DIM)?;
    let k_norm = f32_vec(&gguf, "blk.0.attn_k_norm.weight", HEAD_DIM)?;

    let (mut qp, mut kp, mut v) = (
        vec![0.0; L * q_dim],
        vec![0.0; L * kv_dim],
        vec![0.0; L * kv_dim],
    );
    for (t, &id) in ids.iter().enumerate() {
        let e = &token_embd[id * n_embd..(id + 1) * n_embd];
        let ms = e.iter().map(|x| x * x).sum::<f32>() / n_embd as f32;
        let inv = 1.0 / (ms + RMS_EPS).sqrt();
        let xnw: Vec<f32> = (0..n_embd).map(|i| e[i] * inv * attn_norm[i]).collect();
        qp[t * q_dim..(t + 1) * q_dim].copy_from_slice(&matvec(&wq, &xnw, q_dim, n_embd));
        kp[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&matvec(&wk, &xnw, kv_dim, n_embd));
        v[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&matvec(&wv, &xnw, kv_dim, n_embd));
    }
    let mut q_roped = norm_rope(&qp, n_head, &q_norm);
    let k_roped = norm_rope(&kp, n_kv, &k_norm);
    for x in &mut q_roped {
        *x *= scale; // fold the attention scale into q
    }

    // ── NPU: per-head QKᵀ (keys padded to 1024) stacked over heads ──
    let mut scores = vec![0u16; n_head * L * KEYS];
    for h in 0..n_head {
        let kv = h / gqa;
        let mut q_h = vec![0.0f32; L * HEAD_DIM];
        let mut k_pad = vec![0.0f32; KEYS * HEAD_DIM]; // [keys][128] b_col_maj B; rows≥L = 0
        for t in 0..L {
            q_h[t * HEAD_DIM..(t + 1) * HEAD_DIM].copy_from_slice(
                &q_roped[t * q_dim + h * HEAD_DIM..t * q_dim + (h + 1) * HEAD_DIM],
            );
            k_pad[t * HEAD_DIM..(t + 1) * HEAD_DIM].copy_from_slice(
                &k_roped[t * kv_dim + kv * HEAD_DIM..t * kv_dim + (kv + 1) * HEAD_DIM],
            );
        }
        let s = run_kernel(
            "gemm",
            "gemm_512x128x1024_bcm_bf16",
            &[&bits(&q_h), &bits(&k_pad)],
            L * KEYS,
        )?;
        scores[h * L * KEYS..(h + 1) * L * KEYS].copy_from_slice(&s);
    }

    // ── mask (causal + pad cols) then batched softmax over [n_head*L][1024] ──
    let mut mask = vec![0.0f32; n_head * L * KEYS];
    for h in 0..n_head {
        for tq in 0..L {
            let row = (h * L + tq) * KEYS;
            for tk in 0..KEYS {
                if tk > tq || tk >= L {
                    mask[row + tk] = MASK_NEG;
                }
            }
        }
    }
    let n_sc = n_head * L * KEYS;
    let scores = run_kernel(
        "eltwise",
        "eltwise_add_bf16_8388608",
        &[&scores, &bits(&mask)],
        n_sc,
    )?;
    let probs = run_kernel("norm", "softmax_8388608", &[&scores], n_sc)?;

    // ── ·V per head (V's feature dim padded 128→256), gather into attn_out ──
    let mut attn_out = vec![0.0f32; L * q_dim];
    for h in 0..n_head {
        let kv = h / gqa;
        let mut v_pad = vec![0.0f32; KEYS * VPAD]; // [keys][256] row-major B; cols≥128, rows≥L = 0
        for t in 0..L {
            v_pad[t * VPAD..t * VPAD + HEAD_DIM]
                .copy_from_slice(&v[t * kv_dim + kv * HEAD_DIM..t * kv_dim + (kv + 1) * HEAD_DIM]);
        }
        let probs_h = &probs[h * L * KEYS..(h + 1) * L * KEYS];
        let o = run_kernel(
            "gemm",
            "gemm_512x1024x256_bf16",
            &[probs_h, &bits(&v_pad)],
            L * VPAD,
        )?;
        let o = deq(&o);
        for t in 0..L {
            attn_out[t * q_dim + h * HEAD_DIM..t * q_dim + (h + 1) * HEAD_DIM]
                .copy_from_slice(&o[t * VPAD..t * VPAD + HEAD_DIM]);
        }
    }

    // ── host f32 attention reference ──
    let mut want = vec![0.0f32; L * q_dim];
    for h in 0..n_head {
        let kv = h / gqa;
        for tq in 0..L {
            let q_h = &q_roped[tq * q_dim + h * HEAD_DIM..tq * q_dim + (h + 1) * HEAD_DIM];
            let mut sc = vec![0.0f32; tq + 1];
            for (tk, s) in sc.iter_mut().enumerate() {
                let k_h = &k_roped[tk * kv_dim + kv * HEAD_DIM..tk * kv_dim + (kv + 1) * HEAD_DIM];
                *s = q_h.iter().zip(k_h).map(|(a, b)| a * b).sum();
            }
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0.0f32;
            for s in &mut sc {
                *s = (*s - m).exp();
                den += *s;
            }
            for (tk, &p) in sc.iter().enumerate() {
                let w = p / den;
                let v_h = &v[tk * kv_dim + kv * HEAD_DIM..tk * kv_dim + (kv + 1) * HEAD_DIM];
                for d in 0..HEAD_DIM {
                    want[tq * q_dim + h * HEAD_DIM + d] += w * v_h[d];
                }
            }
        }
    }

    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (g, w) in attn_out.iter().zip(&want) {
        dot += *g as f64 * *w as f64;
        na += *g as f64 * *g as f64;
        nb += *w as f64 * *w as f64;
    }
    let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
    println!("attention attn_out cosine={cos:.6} vs host f32 (GQA, causal, {n_head} heads)");
    if cos >= 0.99 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("attention core wrong: cosine {cos} < 0.99").into())
    }
}
