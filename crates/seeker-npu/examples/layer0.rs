//! M3 capstone — the **entire** Qwen3 layer-0 chained on the NPU, end to end, on real
//! `blk.0` weights and real token embeddings, all bf16 token-major:
//!
//!   x_norm  = rmsnorm(x)·attn_norm
//!   q,k,v   = x_norm·{Wq,Wk,Wv}ᵀ
//!   q,k     = rope(rmsnorm₁₂₈(·)·{q,k}_norm)         (scale 1/√128 folded into q)
//!   attn    = softmax(causal QKᵀ)·V                   (GQA, per head)
//!   x       = x + attn·Woᵀ                            (residual; no post-attn norm)
//!   x       = x + (silu(x2·ffn_gateᵀ)*(x2·ffn_upᵀ))·ffn_downᵀ,  x2 = rmsnorm(x)·ffn_norm
//!
//! This integrates every piece validated in the focused examples (the input block,
//! q/k norm-and-rope, the attention core, the ffn block) into one on-NPU forward, and
//! checks the layer output against a host f32 reference that mirrors seeker-vulkan's
//! qwen3 forward_inner exactly. The embedding cross-check vs `--device vulkan` is M4.
//!
//! Run (after building every kernel the focused examples list):
//!   cargo run -p seeker-npu --example layer0
#[path = "common/mod.rs"]
mod common;
use common::*;
use seeker_core::gguf::GgufFile;

const L: usize = 512;
const HEAD_DIM: usize = 128;
const RMS_EPS: f32 = 1e-5;
const ROPE_BASE: f32 = 1e6;
const KEYS: usize = 1024;
const VPAD: usize = 256;
const MASK_NEG: f32 = -1e4;

fn rope_tables(n_heads: usize, scale: f32) -> (Vec<f32>, Vec<f32>) {
    let d = n_heads * HEAD_DIM;
    let (mut cos, mut sin) = (vec![0.0f32; L * d], vec![0.0f32; L * d]);
    for t in 0..L {
        for i in 0..d {
            let j = (i % HEAD_DIM) % (HEAD_DIM / 2);
            let theta = t as f32 * ROPE_BASE.powf(-2.0 * j as f32 / HEAD_DIM as f32);
            cos[t * d + i] = theta.cos() * scale;
            sin[t * d + i] = theta.sin() * scale;
        }
    }
    (cos, sin)
}

fn rot_half(x: &[f32]) -> Vec<f32> {
    let mut r = vec![0.0f32; x.len()];
    for blk in 0..(x.len() / HEAD_DIM) {
        let b = blk * HEAD_DIM;
        for i in 0..HEAD_DIM / 2 {
            r[b + i] = -x[b + i + HEAD_DIM / 2];
            r[b + i + HEAD_DIM / 2] = x[b + i];
        }
    }
    r
}

/// NPU per-head RMSNorm(128)·norm_w + NEOX rope (cos/sin may carry the attn scale).
fn qk_norm_rope(
    proj: &[u16],
    n_heads: usize,
    norm_w: &[f32],
    cos: &[u16],
    sin: &[u16],
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let n = L * n_heads * HEAD_DIM;
    let (rms, mul, add) = (
        format!("rmsnorm_128_{n}"),
        format!("eltwise_mul_bf16_{n}"),
        format!("eltwise_add_bf16_{n}"),
    );
    let normed = run_kernel("norm", &rms, &[proj], n)?;
    let wt: Vec<f32> = (0..n).map(|i| norm_w[i % HEAD_DIM]).collect();
    let normed = run_kernel("eltwise", &mul, &[&normed, &bits(&wt)], n)?;
    let t1 = run_kernel("eltwise", &mul, &[&normed, cos], n)?;
    let t2 = run_kernel("eltwise", &mul, &[&bits(&rot_half(&deq(&normed))), sin], n)?;
    run_kernel("eltwise", &add, &[&t1, &t2], n)
}

/// NPU GQA attention → attn_out[L][q_dim] (bf16 bits). q is pre-scaled.
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    n_head: usize,
    n_kv: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let gqa = n_head / n_kv;
    let (qf, kf, vf) = (deq(q), deq(k), deq(v));
    let mut scores = vec![0u16; n_head * L * KEYS];
    for h in 0..n_head {
        let kv = h / gqa;
        let (mut q_h, mut k_pad) = (vec![0.0f32; L * HEAD_DIM], vec![0.0f32; KEYS * HEAD_DIM]);
        for t in 0..L {
            q_h[t * HEAD_DIM..(t + 1) * HEAD_DIM]
                .copy_from_slice(&qf[t * q_dim + h * HEAD_DIM..t * q_dim + (h + 1) * HEAD_DIM]);
            k_pad[t * HEAD_DIM..(t + 1) * HEAD_DIM]
                .copy_from_slice(&kf[t * kv_dim + kv * HEAD_DIM..t * kv_dim + (kv + 1) * HEAD_DIM]);
        }
        let s = run_kernel(
            "gemm",
            "gemm_512x128x1024_bcm_bf16",
            &[&bits(&q_h), &bits(&k_pad)],
            L * KEYS,
        )?;
        scores[h * L * KEYS..(h + 1) * L * KEYS].copy_from_slice(&s);
    }
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
    let mut attn_out = vec![0.0f32; L * q_dim];
    for h in 0..n_head {
        let kv = h / gqa;
        let mut v_pad = vec![0.0f32; KEYS * VPAD];
        for t in 0..L {
            v_pad[t * VPAD..t * VPAD + HEAD_DIM]
                .copy_from_slice(&vf[t * kv_dim + kv * HEAD_DIM..t * kv_dim + (kv + 1) * HEAD_DIM]);
        }
        let o = deq(&run_kernel(
            "gemm",
            "gemm_512x1024x256_bf16",
            &[&probs[h * L * KEYS..(h + 1) * L * KEYS], &bits(&v_pad)],
            L * VPAD,
        )?);
        for t in 0..L {
            attn_out[t * q_dim + h * HEAD_DIM..t * q_dim + (h + 1) * HEAD_DIM]
                .copy_from_slice(&o[t * VPAD..t * VPAD + HEAD_DIM]);
        }
    }
    Ok(bits(&attn_out))
}

// ── host f32 reference (mirrors qwen3.rs forward_inner) ──
fn host_norm_rope(proj: &[f32], n_heads: usize, norm_w: &[f32], scale: f32) -> Vec<f32> {
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
                let (c, s) = (theta.cos() * scale, theta.sin() * scale);
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
    let q_dim = gguf.tensor("blk.0.attn_q.weight").ok_or("wq")?.dims[1] as usize;
    let kv_dim = gguf.tensor("blk.0.attn_v.weight").ok_or("wv")?.dims[1] as usize;
    let n_ff = gguf.tensor("blk.0.ffn_gate.weight").ok_or("ffn_gate")?.dims[1] as usize;
    let (n_head, n_kv) = (q_dim / HEAD_DIM, kv_dim / HEAD_DIM);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    println!("layer-0 on NPU: n_embd={n_embd} q_dim={q_dim} kv_dim={kv_dim} n_ff={n_ff} L={L}");

    // weights
    let token_embd = f16_to_f32(&gguf, "token_embd.weight")?;
    let attn_norm = f32_vec(&gguf, "blk.0.attn_norm.weight", n_embd)?;
    let (wq, wk, wv) = (
        f16_to_f32(&gguf, "blk.0.attn_q.weight")?,
        f16_to_f32(&gguf, "blk.0.attn_k.weight")?,
        f16_to_f32(&gguf, "blk.0.attn_v.weight")?,
    );
    let q_norm = f32_vec(&gguf, "blk.0.attn_q_norm.weight", HEAD_DIM)?;
    let k_norm = f32_vec(&gguf, "blk.0.attn_k_norm.weight", HEAD_DIM)?;
    let wo = f16_to_f32(&gguf, "blk.0.attn_output.weight")?;
    let ffn_norm = f32_vec(&gguf, "blk.0.ffn_norm.weight", n_embd)?;
    let (w_gate, w_up, w_down) = (
        f16_to_f32(&gguf, "blk.0.ffn_gate.weight")?,
        f16_to_f32(&gguf, "blk.0.ffn_up.weight")?,
        f16_to_f32(&gguf, "blk.0.ffn_down.weight")?,
    );

    // get_rows (host)
    let ids: Vec<usize> = (0..L).map(|t| (t * 37 + 1) % vocab).collect();
    let mut x = vec![0.0f32; L * n_embd];
    for (t, &id) in ids.iter().enumerate() {
        x[t * n_embd..(t + 1) * n_embd]
            .copy_from_slice(&token_embd[id * n_embd..(id + 1) * n_embd]);
    }

    // ── NPU forward ──
    let attn_norm_t: Vec<f32> = (0..L * n_embd).map(|i| attn_norm[i % n_embd]).collect();
    let ffn_norm_t: Vec<f32> = (0..L * n_embd).map(|i| ffn_norm[i % n_embd]).collect();
    let x_bits = bits(&x);
    let xn = run_kernel("norm", "rmsnorm_1024_524288", &[&x_bits], L * n_embd)?;
    let xnw = run_kernel(
        "eltwise",
        "eltwise_mul_bf16_524288",
        &[&xn, &bits(&attn_norm_t)],
        L * n_embd,
    )?;
    let q = run_kernel(
        "gemm",
        "gemm_512x1024x2048_bcm_bf16",
        &[&xnw, &bits(&wq)],
        L * q_dim,
    )?;
    let k = run_kernel(
        "gemm",
        "gemm_512x1024x1024_bcm_bf16",
        &[&xnw, &bits(&wk)],
        L * kv_dim,
    )?;
    let v = run_kernel(
        "gemm",
        "gemm_512x1024x1024_bcm_bf16",
        &[&xnw, &bits(&wv)],
        L * kv_dim,
    )?;
    let (cos_q, sin_q) = rope_tables(n_head, scale);
    let (cos_k, sin_k) = rope_tables(n_kv, 1.0);
    let q_roped = qk_norm_rope(&q, n_head, &q_norm, &bits(&cos_q), &bits(&sin_q))?;
    let k_roped = qk_norm_rope(&k, n_kv, &k_norm, &bits(&cos_k), &bits(&sin_k))?;
    let attn = attention(&q_roped, &k_roped, &v, n_head, n_kv, q_dim, kv_dim)?;
    let proj = run_kernel(
        "gemm",
        "gemm_512x2048x1024_bcm_bf16",
        &[&attn, &bits(&wo)],
        L * n_embd,
    )?;
    let resid = run_kernel(
        "eltwise",
        "eltwise_add_bf16_524288",
        &[&x_bits, &proj],
        L * n_embd,
    )?;
    let xn2 = run_kernel("norm", "rmsnorm_1024_524288", &[&resid], L * n_embd)?;
    let xn2w = run_kernel(
        "eltwise",
        "eltwise_mul_bf16_524288",
        &[&xn2, &bits(&ffn_norm_t)],
        L * n_embd,
    )?;
    let gate = run_kernel(
        "gemm",
        "gemm_512x1024x3072_bcm_bf16",
        &[&xn2w, &bits(&w_gate)],
        L * n_ff,
    )?;
    let up = run_kernel(
        "gemm",
        "gemm_512x1024x3072_bcm_bf16",
        &[&xn2w, &bits(&w_up)],
        L * n_ff,
    )?;
    let gsilu = run_kernel("activation", "silu_1572864", &[&gate], L * n_ff)?;
    let hidden = run_kernel(
        "eltwise",
        "eltwise_mul_bf16_1572864",
        &[&gsilu, &up],
        L * n_ff,
    )?;
    let down = run_kernel(
        "gemm",
        "gemm_512x3072x1024_bcm_bf16",
        &[&hidden, &bits(&w_down)],
        L * n_embd,
    )?;
    let out = run_kernel(
        "eltwise",
        "eltwise_add_bf16_524288",
        &[&resid, &down],
        L * n_embd,
    )?;

    // ── host f32 reference ──
    let mut hx = x.clone();
    // attn pre-norm + qkv
    let mut xnw_h = vec![0.0f32; L * n_embd];
    for t in 0..L {
        let xt = &hx[t * n_embd..(t + 1) * n_embd];
        let ms = xt.iter().map(|a| a * a).sum::<f32>() / n_embd as f32;
        let inv = 1.0 / (ms + RMS_EPS).sqrt();
        for i in 0..n_embd {
            xnw_h[t * n_embd + i] = xt[i] * inv * attn_norm[i];
        }
    }
    let (mut qh, mut kh, mut vh) = (
        vec![0.0; L * q_dim],
        vec![0.0; L * kv_dim],
        vec![0.0; L * kv_dim],
    );
    for t in 0..L {
        let xt = &xnw_h[t * n_embd..(t + 1) * n_embd];
        qh[t * q_dim..(t + 1) * q_dim].copy_from_slice(&matvec(&wq, xt, q_dim, n_embd));
        kh[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&matvec(&wk, xt, kv_dim, n_embd));
        vh[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&matvec(&wv, xt, kv_dim, n_embd));
    }
    let qr = host_norm_rope(&qh, n_head, &q_norm, scale);
    let kr = host_norm_rope(&kh, n_kv, &k_norm, 1.0);
    let gqa = n_head / n_kv;
    let mut attn_h = vec![0.0f32; L * q_dim];
    for h in 0..n_head {
        let kv = h / gqa;
        for tq in 0..L {
            let q_h = &qr[tq * q_dim + h * HEAD_DIM..tq * q_dim + (h + 1) * HEAD_DIM];
            let mut sc = vec![0.0f32; tq + 1];
            for (tk, s) in sc.iter_mut().enumerate() {
                let k_h = &kr[tk * kv_dim + kv * HEAD_DIM..tk * kv_dim + (kv + 1) * HEAD_DIM];
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
                let v_h = &vh[tk * kv_dim + kv * HEAD_DIM..tk * kv_dim + (kv + 1) * HEAD_DIM];
                for d in 0..HEAD_DIM {
                    attn_h[tq * q_dim + h * HEAD_DIM + d] += w * v_h[d];
                }
            }
        }
    }
    // O-proj + residual
    for t in 0..L {
        let p = matvec(&wo, &attn_h[t * q_dim..(t + 1) * q_dim], n_embd, q_dim);
        for i in 0..n_embd {
            hx[t * n_embd + i] += p[i];
        }
    }
    // FFN
    let mut want = vec![0.0f32; L * n_embd];
    for t in 0..L {
        let r = &hx[t * n_embd..(t + 1) * n_embd];
        let ms = r.iter().map(|a| a * a).sum::<f32>() / n_embd as f32;
        let inv = 1.0 / (ms + RMS_EPS).sqrt();
        let x2: Vec<f32> = (0..n_embd).map(|i| r[i] * inv * ffn_norm[i]).collect();
        let g = matvec(&w_gate, &x2, n_ff, n_embd);
        let u = matvec(&w_up, &x2, n_ff, n_embd);
        let hd: Vec<f32> = (0..n_ff)
            .map(|o| (g[o] / (1.0 + (-g[o]).exp())) * u[o])
            .collect();
        let d = matvec(&w_down, &hd, n_embd, n_ff);
        for i in 0..n_embd {
            want[t * n_embd + i] = r[i] + d[i];
        }
    }

    let cos = cosine(&out, &want);
    println!("layer-0 output cosine={cos:.6} vs host f32 (full attn+ffn block on NPU)");
    if cos >= 0.99 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("layer-0 wrong: cosine {cos} < 0.99").into())
    }
}
