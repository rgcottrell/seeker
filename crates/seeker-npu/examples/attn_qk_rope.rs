//! M3 — per-head Q/K RMSNorm + NEOX RoPE on the NPU, chained on real `blk.0` weights.
//! Continues the input block: from `x = rmsnorm(embed)·attn_norm`, projects q/k, then
//!   q_normed = rmsnorm₁₂₈(q) · attn_q_norm   (per-head norm over head_dim=128)
//!   q_roped  = q_normed ⊙ cos + rot_half(q_normed) ⊙ sin   (NEOX, θ=pos·1e6^(−2j/128))
//! and likewise for k. V is raw (no norm/rope). Validated by cosine vs a host f32 ref.
//!
//! RoPE arithmetic (the two muls + add) runs on the NPU; the cos/sin tables and the
//! rotate-half shuffle (a within-128-block permute+negate, [−hi, lo]) are host data-prep
//! — same category as get_rows. In the resident-BO forward (M5) rot_half becomes an
//! on-NPU strided DMA. The conventions match seeker-vulkan's qwen3 rope_neox exactly:
//!   d[j]    = x[j]·cosⱼ − x[j+64]·sinⱼ ,  d[j+64] = x[j]·sinⱼ + x[j+64]·cosⱼ
//!
//! Build (besides the input-block kernels):
//!   kernels/norm/build.sh rmsnorm 1048576 128   (q per-head norm)
//!   kernels/norm/build.sh rmsnorm 524288 128    (k per-head norm)
//!   kernels/gemm/build.sh 512 1024 1024 bf16 1  (wk)
//!   kernels/eltwise/build.sh mul bf16 1048576 ; add bf16 1048576
//! then: cargo run -p seeker-npu --example attn_qk_rope
#[path = "common/mod.rs"]
mod common;
use common::*;
use seeker_core::gguf::GgufFile;

const L: usize = 512;
const HEAD_DIM: usize = 128;
const RMS_EPS: f32 = 1e-5; // the aie2p rms_norm.cc value (Qwen3 is 1e-6, negligible)
const ROPE_BASE: f32 = 1e6;

/// cos/sin broadcast to token-major `[L][n_heads·128]`: per 128-block, entry i uses
/// pair j = i%64 at the token's position; both halves of a block share cosⱼ/sinⱼ.
fn rope_tables(n_heads: usize) -> (Vec<f32>, Vec<f32>) {
    let d = n_heads * HEAD_DIM;
    let mut cos = vec![0.0f32; L * d];
    let mut sin = vec![0.0f32; L * d];
    for t in 0..L {
        for i in 0..d {
            let j = (i % HEAD_DIM) % (HEAD_DIM / 2); // pair index in [0,64)
            let theta = t as f32 * ROPE_BASE.powf(-2.0 * j as f32 / HEAD_DIM as f32);
            cos[t * d + i] = theta.cos();
            sin[t * d + i] = theta.sin();
        }
    }
    (cos, sin)
}

/// rot_half within each 128-block: out = [−hi, lo] (i<64 → −x[i+64]; i≥64 → x[i−64]).
/// Operates on every contiguous head_dim block, so it's head-count agnostic.
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

/// Per-head RMSNorm(128)·norm_w, then NEOX RoPE — on the NPU. Returns roped bf16 bits.
fn qk_norm_rope(
    proj: &[u16], // token-major [L][n_heads*128] bf16 bits (q or k post-projection)
    n_heads: usize,
    norm_w: &[f32], // attn_{q,k}_norm [128]
    cos: &[u16],
    sin: &[u16],
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let d = n_heads * HEAD_DIM;
    let n = L * d;
    let rms = format!("rmsnorm_128_{n}");
    let mul = format!("eltwise_mul_bf16_{n}");
    let add = format!("eltwise_add_bf16_{n}");
    // per-head RMSNorm over each 128-block, then ·norm_w (broadcast per block).
    let normed = run_kernel("norm", &rms, &[proj], n)?;
    let w_tiled: Vec<f32> = (0..n).map(|i| norm_w[i % HEAD_DIM]).collect();
    let normed = run_kernel("eltwise", &mul, &[&normed, &bits(&w_tiled)], n)?;
    // RoPE: normed⊙cos + rot_half(normed)⊙sin.
    let t1 = run_kernel("eltwise", &mul, &[&normed, cos], n)?;
    let rh = bits(&rot_half(&deq(&normed)));
    let t2 = run_kernel("eltwise", &mul, &[&rh, sin], n)?;
    run_kernel("eltwise", &add, &[&t1, &t2], n)
}

/// Host f32 reference: per-head rmsnorm(128)·norm_w then NEOX rope.
fn host_qk_norm_rope(proj: &[f32], n_heads: usize, norm_w: &[f32]) -> Vec<f32> {
    let d = n_heads * HEAD_DIM;
    let mut out = vec![0.0f32; L * d];
    for t in 0..L {
        for h in 0..n_heads {
            let base = t * d + h * HEAD_DIM;
            let head = &proj[base..base + HEAD_DIM];
            let ms = head.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32;
            let inv = 1.0 / (ms + RMS_EPS).sqrt();
            let normed: Vec<f32> = (0..HEAD_DIM).map(|i| head[i] * inv * norm_w[i]).collect();
            for j in 0..HEAD_DIM / 2 {
                let theta = t as f32 * ROPE_BASE.powf(-2.0 * j as f32 / HEAD_DIM as f32);
                let (c, s) = (theta.cos(), theta.sin());
                out[base + j] = normed[j] * c - normed[j + HEAD_DIM / 2] * s;
                out[base + j + HEAD_DIM / 2] = normed[j] * s + normed[j + HEAD_DIM / 2] * c;
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
    let kv_dim = gguf.tensor("blk.0.attn_k.weight").ok_or("missing wk")?.dims[1] as usize;
    let (n_head, n_kv) = (q_dim / HEAD_DIM, kv_dim / HEAD_DIM);
    println!("n_embd={n_embd} q_dim={q_dim} kv_dim={kv_dim} n_head={n_head} n_kv={n_kv} L={L}");

    // ── input block: get_rows → rmsnorm·attn_norm → wq/wk (token-major, bf16) ──
    let token_embd = f16_to_f32(&gguf, "token_embd.weight")?;
    let ids: Vec<usize> = (0..L).map(|t| (t * 37 + 1) % vocab).collect();
    let mut x = vec![0.0f32; L * n_embd];
    for (t, &id) in ids.iter().enumerate() {
        x[t * n_embd..(t + 1) * n_embd]
            .copy_from_slice(&token_embd[id * n_embd..(id + 1) * n_embd]);
    }
    let attn_norm = f32_vec(&gguf, "blk.0.attn_norm.weight", n_embd)?;
    let wq = f16_to_f32(&gguf, "blk.0.attn_q.weight")?;
    let wk = f16_to_f32(&gguf, "blk.0.attn_k.weight")?;
    let q_norm = f32_vec(&gguf, "blk.0.attn_q_norm.weight", HEAD_DIM)?;
    let k_norm = f32_vec(&gguf, "blk.0.attn_k_norm.weight", HEAD_DIM)?;

    let xn = run_kernel("norm", "rmsnorm_1024_524288", &[&bits(&x)], L * n_embd)?;
    let wtile: Vec<f32> = (0..L * n_embd).map(|i| attn_norm[i % n_embd]).collect();
    let xnw = run_kernel(
        "eltwise",
        "eltwise_mul_bf16_524288",
        &[&xn, &bits(&wtile)],
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

    // ── per-head norm + RoPE on NPU ──
    let (cos_q, sin_q) = rope_tables(n_head);
    let (cos_k, sin_k) = rope_tables(n_kv);
    let q_roped = qk_norm_rope(&q, n_head, &q_norm, &bits(&cos_q), &bits(&sin_q))?;
    let k_roped = qk_norm_rope(&k, n_kv, &k_norm, &bits(&cos_k), &bits(&sin_k))?;

    // ── host f32 reference of the whole chain ──
    let host_x: Vec<f32> = {
        // recompute the input block in f32 so the q/k fed to norm+rope are the f32 ideal
        let mut xnw_h = vec![0.0f32; L * n_embd];
        for t in 0..L {
            let xt = &x[t * n_embd..(t + 1) * n_embd];
            let ms = xt.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
            let inv = 1.0 / (ms + RMS_EPS).sqrt();
            for i in 0..n_embd {
                xnw_h[t * n_embd + i] = xt[i] * inv * attn_norm[i];
            }
        }
        xnw_h
    };
    let mut q_h = vec![0.0f32; L * q_dim];
    let mut k_h = vec![0.0f32; L * kv_dim];
    for t in 0..L {
        let xt = &host_x[t * n_embd..(t + 1) * n_embd];
        q_h[t * q_dim..(t + 1) * q_dim].copy_from_slice(&matvec(&wq, xt, q_dim, n_embd));
        k_h[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&matvec(&wk, xt, kv_dim, n_embd));
    }
    let q_ref = host_qk_norm_rope(&q_h, n_head, &q_norm);
    let k_ref = host_qk_norm_rope(&k_h, n_kv, &k_norm);

    let cq = cosine(&q_roped, &q_ref);
    let ck = cosine(&k_roped, &k_ref);
    println!("q_roped cosine={cq:.6}  k_roped cosine={ck:.6} vs host f32 (full chain on NPU)");
    if cq >= 0.99 && ck >= 0.99 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("per-head norm/rope wrong: q {cq} k {ck} (< 0.99)").into())
    }
}
