//! M3 — the Qwen3 layer's input-projection block, chained on the NPU with real
//! `blk.0` weights and real token embeddings:
//!   get_rows(token_embd)  →  input RMSNorm  →  · attn_norm.weight  →  Wq projection
//! i.e. `q = (rmsnorm(x) · w_attn_norm) · Wqᵀ`, all token-major (no transposes).
//! Validates the multi-op resident chain by cosine vs a host f32 reference.
//!
//! get_rows runs on the host (a gather + dequant); the three compute ops run on the
//! NPU as separate xclbins, the activation flowing host-side between them (a memcpy;
//! keeping it resident across xclbins via import_host_ptr is an M5 perf concern).
//!
//! Build first:
//!   kernels/norm/build.sh rmsnorm 524288 1024
//!   kernels/eltwise/build.sh mul bf16 524288
//!   kernels/gemm/build.sh 512 1024 2048 bf16 1
//! then: cargo run -p seeker-npu --example input_block
use std::path::PathBuf;

use half::{bf16, f16};
use seeker_core::gguf::{GgmlType, GgufFile};
use seeker_npu::npu::{Buffer, Context};

const DEFAULT_GGUF: &str = "/models/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-f16.gguf";
const L: usize = 512; // token count (padded to the GEMM M%512 constraint)
const RMS_EPS: f32 = 1e-5; // the aie2p rms_norm.cc hardcodes 1e-5

fn artifact(subdir: &str, stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels")
        .join(subdir)
        .join("build")
        .join(format!("{stem}.{ext}"))
}

/// Run one NPU kernel: alloc a BO per input (bf16 bits) + one output BO, bind them
/// in order (inputs.., output), run, and return the output's bf16 bits.
fn run_kernel(
    subdir: &str,
    stem: &str,
    inputs: &[&[u16]],
    out_elems: usize,
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let ctx = Context::new(&artifact(subdir, stem, "xclbin"), "MLIR_AIE")?;
    let insts = std::fs::read(artifact(subdir, stem, "insts.bin"))?;
    let mut instr = ctx.alloc_instr(insts.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts);
    instr.sync_to_device()?;

    let mut bos: Vec<Buffer> = Vec::new();
    for inp in inputs {
        let mut b = ctx.alloc_data(inp.len() * 2)?;
        b.as_mut_slice::<u16>().copy_from_slice(inp);
        b.sync_to_device()?;
        bos.push(b);
    }
    let mut out = ctx.alloc_data(out_elems * 2)?;
    out.as_mut_bytes().fill(0);
    out.sync_to_device()?;

    let refs: Vec<&Buffer> = bos.iter().chain(std::iter::once(&out)).collect();
    ctx.run(&instr, insts.len() as u32, &refs)?;
    drop(refs);
    out.sync_from_device()?;
    Ok(out.as_slice::<u16>().to_vec())
}

fn f16_tensor_to_f32(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let info = gguf.tensor(name).ok_or(format!("missing {name}"))?;
    if info.ggml_type != GgmlType::F16 {
        return Err(format!("{name} expected F16, got {:?}", info.ggml_type).into());
    }
    let raw = gguf.tensor_data(name).ok_or("no data")?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("NPU_QWEN3_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let gguf = GgufFile::open(&model)?;

    // dims: token_embd [n_embd, vocab]; wq [n_embd(K), q_dim(N)].
    let te = gguf
        .tensor("token_embd.weight")
        .ok_or("missing token_embd")?;
    let (n_embd, vocab) = (te.dims[0] as usize, te.dims[1] as usize);
    let wq = gguf.tensor("blk.0.attn_q.weight").ok_or("missing wq")?;
    let (k, q_dim) = (wq.dims[0] as usize, wq.dims[1] as usize);
    assert_eq!(k, n_embd);
    println!("n_embd={n_embd} q_dim={q_dim} L={L}  (get_rows -> rmsnorm -> ·attn_norm -> Wq)");

    // ── host: get_rows for L token ids spread across the vocab ──
    let token_embd = f16_tensor_to_f32(&gguf, "token_embd.weight")?;
    let ids: Vec<usize> = (0..L).map(|t| (t * 37 + 1) % vocab).collect();
    let mut x = vec![0.0f32; L * n_embd]; // token-major [L][n_embd]
    for (t, &id) in ids.iter().enumerate() {
        x[t * n_embd..(t + 1) * n_embd]
            .copy_from_slice(&token_embd[id * n_embd..(id + 1) * n_embd]);
    }

    // weights
    let attn_norm = {
        let info = gguf
            .tensor("blk.0.attn_norm.weight")
            .ok_or("missing attn_norm")?;
        let raw = gguf
            .tensor_data("blk.0.attn_norm.weight")
            .ok_or("no data")?;
        if info.ggml_type != GgmlType::F32 || raw.len() != n_embd * 4 {
            return Err("attn_norm must be F32 [n_embd]".into());
        }
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<f32>>()
    };
    let wq_f32 = f16_tensor_to_f32(&gguf, "blk.0.attn_q.weight")?; // [q_dim][n_embd] = B (b_col_maj)
    // attn_norm broadcast to token-major [L][n_embd].
    let w_tiled: Vec<f32> = (0..L * n_embd).map(|i| attn_norm[i % n_embd]).collect();

    // ── NPU chain (bf16) ──
    let x_norm = run_kernel("norm", "rmsnorm_1024_524288", &[&bits(&x)], L * n_embd)?;
    let x_normw = run_kernel(
        "eltwise",
        "eltwise_mul_bf16_524288",
        &[&x_norm, &bits(&w_tiled)],
        L * n_embd,
    )?;
    let q = run_kernel(
        "gemm",
        "gemm_512x1024x2048_bcm_bf16",
        &[&x_normw, &bits(&wq_f32)],
        L * q_dim,
    )?;

    // ── host f32 reference of the same chain ──
    let mut want = vec![0.0f32; L * q_dim];
    for t in 0..L {
        let xt = &x[t * n_embd..(t + 1) * n_embd];
        let ms = xt.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
        let inv = 1.0 / (ms + RMS_EPS).sqrt();
        let xnw: Vec<f32> = (0..n_embd).map(|i| xt[i] * inv * attn_norm[i]).collect();
        for o in 0..q_dim {
            let wrow = &wq_f32[o * n_embd..(o + 1) * n_embd];
            want[t * q_dim + o] = xnw.iter().zip(wrow).map(|(a, b)| a * b).sum();
        }
    }

    // cosine over the whole q (the chain-correctness metric; bf16 across 3 ops is noise).
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (i, w) in want.iter().enumerate() {
        let g = bf16::from_bits(q[i]).to_f32();
        dot += g as f64 * *w as f64;
        na += g as f64 * g as f64;
        nb += *w as f64 * *w as f64;
    }
    let cosine = (dot / (na.sqrt() * nb.sqrt())) as f32;
    println!("input block on NPU: q cosine={cosine:.6} vs host f32 (L={L})");
    if cosine >= 0.99 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("input block chain wrong: cosine {cosine} < 0.99").into())
    }
}
