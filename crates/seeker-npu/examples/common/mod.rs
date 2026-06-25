//! Shared helpers for the in-tree NPU validators (`examples/*.rs`). Included via
//! `#[path = "common/mod.rs"] mod common;` — Cargo does not build a subdir without a
//! `main.rs` as its own example, so this stays a plain module compiled into each
//! example that pulls it in.
#![allow(dead_code)]
use std::path::PathBuf;

use half::{bf16, f16};
use seeker_core::gguf::{GgmlType, GgufFile};
use seeker_npu::npu::{Buffer, Context};

pub const DEFAULT_GGUF: &str = "/models/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-f16.gguf";

/// Path to a built kernel artifact: `kernels/<subdir>/build/<stem>.<ext>`.
pub fn artifact(subdir: &str, stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels")
        .join(subdir)
        .join("build")
        .join(format!("{stem}.{ext}"))
}

/// Run one NPU kernel: alloc a BO per input (bf16 bits) + one output BO, bind them in
/// order (inputs.., output), run, and return the output's bf16 bits. Each op is its own
/// xclbin/Context; activations flow host-side between ops (resident BOs across Contexts
/// are an M5 perf concern, not a correctness one).
pub fn run_kernel(
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

/// f32 -> bf16 bits.
pub fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}

/// bf16 bits -> f32.
pub fn deq(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
}

/// Dequantize an F16 GGUF tensor to f32 (row-major as stored).
pub fn f16_to_f32(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let info = gguf.tensor(name).ok_or(format!("missing {name}"))?;
    if info.ggml_type != GgmlType::F16 {
        return Err(format!("{name} expected F16, got {:?}", info.ggml_type).into());
    }
    Ok(gguf
        .tensor_data(name)
        .ok_or("no data")?
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

/// Read an F32 GGUF vector of an expected length.
pub fn f32_vec(
    gguf: &GgufFile,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let raw = gguf.tensor_data(name).ok_or(format!("missing {name}"))?;
    if raw.len() != len * 4 {
        return Err(format!("{name} expected F32[{len}]").into());
    }
    Ok(raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Cosine similarity of NPU bf16-bit output vs a host f32 reference (the chain-
/// correctness metric — bf16 rounding across many ops is high-frequency noise that
/// barely moves cosine but dominates element-wise error).
pub fn cosine(npu: &[u16], host: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (i, w) in host.iter().enumerate() {
        let g = bf16::from_bits(npu[i]).to_f32();
        dot += g as f64 * *w as f64;
        na += g as f64 * g as f64;
        nb += *w as f64 * *w as f64;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// `out[o] = Σ_k w[o*k + ki] · x[ki]` — a column-major-weight (b_col_maj) mat-vec,
/// i.e. weight row `o` is the contiguous `w[o*K..(o+1)*K]` slice (GGUF `[out][in]`).
pub fn matvec(w: &[f32], x: &[f32], out_dim: usize, k: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            w[o * k..(o + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}
