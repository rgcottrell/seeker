//! Diagnostic — a pure **host f32** Qwen3-Embedding forward using the SAME
//! conventions the NPU path mirrors (no NPU involved), then the shared
//! `pool_and_normalize`. Comparing this to `--device vulkan` isolates whether an
//! NPU↔Vulkan embedding gap is a convention mismatch (this won't match Vulkan
//! either) or bf16 accumulation (this matches Vulkan, the NPU doesn't).
//!
//!   cargo run -p seeker-npu --example host_forward -- "the prompt"
#[path = "common/mod.rs"]
mod common;
use common::*;
use half::bf16;
use seeker_core::embed::{Pooling, pool_and_normalize};
use seeker_core::gguf::GgufFile;
use seeker_core::tokenizer::build_tokenizer;

const HEAD_DIM: usize = 128;

fn b16(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

/// Mat-vec with optional NPU-precision simulation:
///   SEEKER_SIM_BF16_GEMM=1 → round weights+activations to bf16 (f32 accum)
///   SEEKER_SIM_BF16_OUT=1  → also round each output to bf16
/// to isolate how much the NPU's bf16 matmul (vs the LUT activation kernels) costs.
fn mm(w: &[f32], x: &[f32], out_dim: usize, k: usize) -> Vec<f32> {
    let sim = std::env::var("SEEKER_SIM_BF16_GEMM").is_ok();
    let sim_out = std::env::var("SEEKER_SIM_BF16_OUT").is_ok();
    if !sim {
        return matvec(w, x, out_dim, k);
    }
    let xb: Vec<f32> = x.iter().map(|&v| b16(v)).collect();
    (0..out_dim)
        .map(|o| {
            let acc: f32 = w[o * k..(o + 1) * k]
                .iter()
                .zip(&xb)
                .map(|(a, b)| b16(*a) * b)
                .sum();
            if sim_out { b16(acc) } else { acc }
        })
        .collect()
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

#[allow(clippy::too_many_arguments)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".to_string());
    let model = std::env::var("NPU_QWEN3_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let gguf = GgufFile::open(&model)?;
    let arch = gguf.architecture().unwrap_or("qwen3").to_string();
    let mu = |k: &str| gguf.meta_u32(&format!("{arch}.{k}"));
    let n_layers = mu("block_count").unwrap() as usize;
    let n_embd = mu("embedding_length").unwrap() as usize;
    let n_head = mu("attention.head_count").unwrap() as usize;
    let n_kv = mu("attention.head_count_kv").unwrap_or(n_head as u32) as usize;
    let n_ff = mu("feed_forward_length").unwrap() as usize;
    let q_dim = gguf.tensor("blk.0.attn_q.weight").unwrap().dims[1] as usize;
    let kv_dim = gguf.tensor("blk.0.attn_v.weight").unwrap().dims[1] as usize;
    let rope_base = gguf
        .meta_f32(&format!("{arch}.rope.freq_base"))
        .unwrap_or(1e6);
    let eps = gguf
        .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let gqa = n_head / n_kv;

    let tok = build_tokenizer(&gguf)?;
    let ids: Vec<usize> = tok
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| format!("{e}"))?
        .get_ids()
        .iter()
        .map(|&i| i as usize)
        .collect();
    let l = ids.len();
    eprintln!("tokens={l} n_layers={n_layers} eps={eps} rope_base={rope_base}");

    let token_embd = f16_to_f32(&gguf, "token_embd.weight")?;
    let mut resid = vec![0.0f32; l * n_embd];
    for (t, &id) in ids.iter().enumerate() {
        resid[t * n_embd..(t + 1) * n_embd]
            .copy_from_slice(&token_embd[id * n_embd..(id + 1) * n_embd]);
    }

    let norm_rope = |proj: &[f32], n_heads: usize, nw: &[f32], sc: f32| -> Vec<f32> {
        let d = n_heads * HEAD_DIM;
        let mut out = vec![0.0f32; l * d];
        for t in 0..l {
            for h in 0..n_heads {
                let base = t * d + h * HEAD_DIM;
                let nd = rmsnorm(&proj[base..base + HEAD_DIM], nw, eps);
                for j in 0..HEAD_DIM / 2 {
                    let theta = t as f32 * rope_base.powf(-2.0 * j as f32 / HEAD_DIM as f32);
                    let (c, s) = (theta.cos() * sc, theta.sin() * sc);
                    out[base + j] = nd[j] * c - nd[j + HEAD_DIM / 2] * s;
                    out[base + j + HEAD_DIM / 2] = nd[j] * s + nd[j + HEAD_DIM / 2] * c;
                }
            }
        }
        out
    };

    let max = std::env::var("SEEKER_NPU_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(n_layers);
    for i in 0..n_layers.min(max) {
        let p = format!("blk.{i}");
        let attn_norm = f32_vec(&gguf, &format!("{p}.attn_norm.weight"), n_embd)?;
        let wq = f16_to_f32(&gguf, &format!("{p}.attn_q.weight"))?;
        let wk = f16_to_f32(&gguf, &format!("{p}.attn_k.weight"))?;
        let wv = f16_to_f32(&gguf, &format!("{p}.attn_v.weight"))?;
        let q_norm = f32_vec(&gguf, &format!("{p}.attn_q_norm.weight"), HEAD_DIM)?;
        let k_norm = f32_vec(&gguf, &format!("{p}.attn_k_norm.weight"), HEAD_DIM)?;
        let wo = f16_to_f32(&gguf, &format!("{p}.attn_output.weight"))?;
        let ffn_norm = f32_vec(&gguf, &format!("{p}.ffn_norm.weight"), n_embd)?;
        let w_gate = f16_to_f32(&gguf, &format!("{p}.ffn_gate.weight"))?;
        let w_up = f16_to_f32(&gguf, &format!("{p}.ffn_up.weight"))?;
        let w_down = f16_to_f32(&gguf, &format!("{p}.ffn_down.weight"))?;

        // attn
        let (mut q, mut k, mut v) = (
            vec![0.0; l * q_dim],
            vec![0.0; l * kv_dim],
            vec![0.0; l * kv_dim],
        );
        for t in 0..l {
            let xn = rmsnorm(&resid[t * n_embd..(t + 1) * n_embd], &attn_norm, eps);
            q[t * q_dim..(t + 1) * q_dim].copy_from_slice(&mm(&wq, &xn, q_dim, n_embd));
            k[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&mm(&wk, &xn, kv_dim, n_embd));
            v[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&mm(&wv, &xn, kv_dim, n_embd));
        }
        let qr = norm_rope(&q, n_head, &q_norm, scale);
        let kr = norm_rope(&k, n_kv, &k_norm, 1.0);
        let mut attn = vec![0.0f32; l * q_dim];
        for h in 0..n_head {
            let kv = h / gqa;
            for tq in 0..l {
                let qh = &qr[tq * q_dim + h * HEAD_DIM..tq * q_dim + (h + 1) * HEAD_DIM];
                let mut sc = vec![0.0f32; tq + 1];
                for (tk, sv) in sc.iter_mut().enumerate() {
                    let kh = &kr[tk * kv_dim + kv * HEAD_DIM..tk * kv_dim + (kv + 1) * HEAD_DIM];
                    *sv = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
                }
                let m = sc.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0.0;
                for sv in &mut sc {
                    *sv = (*sv - m).exp();
                    den += *sv;
                }
                for (tk, &pw) in sc.iter().enumerate() {
                    let w = pw / den;
                    let vh = &v[tk * kv_dim + kv * HEAD_DIM..tk * kv_dim + (kv + 1) * HEAD_DIM];
                    for d in 0..HEAD_DIM {
                        attn[tq * q_dim + h * HEAD_DIM + d] += w * vh[d];
                    }
                }
            }
        }
        for t in 0..l {
            let pr = mm(&wo, &attn[t * q_dim..(t + 1) * q_dim], n_embd, q_dim);
            for i in 0..n_embd {
                resid[t * n_embd + i] += pr[i];
            }
        }
        // ffn
        for t in 0..l {
            let r = &resid[t * n_embd..(t + 1) * n_embd];
            let x2 = rmsnorm(r, &ffn_norm, eps);
            let g = mm(&w_gate, &x2, n_ff, n_embd);
            let u = mm(&w_up, &x2, n_ff, n_embd);
            let hd: Vec<f32> = (0..n_ff)
                .map(|o| (g[o] / (1.0 + (-g[o]).exp())) * u[o])
                .collect();
            let dn = mm(&w_down, &hd, n_embd, n_ff);
            for i in 0..n_embd {
                resid[t * n_embd + i] += dn[i];
            }
        }
    }

    let on = f32_vec(&gguf, "output_norm.weight", n_embd)?;
    let emb = pool_and_normalize(&resid, n_embd, &on, eps, Pooling::Last, 2);
    let v = &emb[0];
    print!("[[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{x}");
    }
    println!("]]");
    Ok(())
}
