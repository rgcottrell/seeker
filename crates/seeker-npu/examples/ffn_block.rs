//! M3 — the Qwen3 layer's FFN (SwiGLU) block, chained on the NPU with real `blk.0`
//! weights, all bf16:
//!   x2 = rmsnorm(residual)·ffn_norm ; ffn_hidden = silu(x2·ffn_gateᵀ) * (x2·ffn_upᵀ)
//!   out = residual + ffn_hidden·ffn_downᵀ
//! Token-major throughout (b_col_maj weights). Validates the chain — and in
//! particular whether the down GEMM's bf16 output over K=n_ff=3072 holds up — by
//! cosine vs a host f32 reference. (The residual here is real token embeddings
//! standing in for a layer's running activation.)
//!
//! Build first:
//!   kernels/norm/build.sh rmsnorm 524288 1024
//!   kernels/eltwise/build.sh mul bf16 524288   (broadcast ffn_norm)
//!   kernels/gemm/build.sh 512 1024 3072 bf16 1 (gate/up)
//!   kernels/activation/build.sh 1572864        (silu)
//!   kernels/eltwise/build.sh mul bf16 1572864  (gate*up)
//!   GEMM_ALLOW_VERIFY_FAIL=1 kernels/gemm/build.sh 512 3072 1024 bf16 1 (down)
//!     ^ bf16 output over K=n_ff=3072 exceeds wa.py's self-check tolerance (expected;
//!       the f32-out build of the same shape passes) — see kernels/gemm/build.sh.
//!   kernels/eltwise/build.sh add bf16 524288   (residual add)
//! then: cargo run -p seeker-npu --example ffn_block
use std::path::PathBuf;

use half::{bf16, f16};
use seeker_core::gguf::{GgmlType, GgufFile};
use seeker_npu::npu::{Buffer, Context};

const DEFAULT_GGUF: &str = "/models/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-f16.gguf";
const L: usize = 512;
const RMS_EPS: f32 = 1e-5;

fn artifact(subdir: &str, stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels")
        .join(subdir)
        .join("build")
        .join(format!("{stem}.{ext}"))
}

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

fn f16_to_f32(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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

fn f32_vec(
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

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}

// q-major weight Wᵀ row o (the b_col_maj B row): w[o*K .. (o+1)*K].
fn matvec(w: &[f32], x: &[f32], out_dim: usize, k: usize) -> Vec<f32> {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("NPU_QWEN3_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let gguf = GgufFile::open(&model)?;

    let te = gguf
        .tensor("token_embd.weight")
        .ok_or("missing token_embd")?;
    let (n_embd, vocab) = (te.dims[0] as usize, te.dims[1] as usize);
    let n_ff = gguf
        .tensor("blk.0.ffn_gate.weight")
        .ok_or("missing ffn_gate")?
        .dims[1] as usize;
    println!(
        "n_embd={n_embd} n_ff={n_ff} L={L}  (rmsnorm·ffn_norm -> gate/up -> SiLU·up -> down -> +residual)"
    );

    // residual = real token embeddings (stand-in for a layer's running activation).
    let token_embd = f16_to_f32(&gguf, "token_embd.weight")?;
    let ids: Vec<usize> = (0..L).map(|t| (t * 37 + 1) % vocab).collect();
    let mut residual = vec![0.0f32; L * n_embd];
    for (t, &id) in ids.iter().enumerate() {
        residual[t * n_embd..(t + 1) * n_embd]
            .copy_from_slice(&token_embd[id * n_embd..(id + 1) * n_embd]);
    }

    let ffn_norm = f32_vec(&gguf, "blk.0.ffn_norm.weight", n_embd)?;
    let w_gate = f16_to_f32(&gguf, "blk.0.ffn_gate.weight")?; // [n_ff][n_embd] = B (bcm)
    let w_up = f16_to_f32(&gguf, "blk.0.ffn_up.weight")?;
    let w_down = f16_to_f32(&gguf, "blk.0.ffn_down.weight")?; // [n_embd][n_ff] = B (bcm)
    let wnorm_tiled: Vec<f32> = (0..L * n_embd).map(|i| ffn_norm[i % n_embd]).collect();

    // ── NPU chain (bf16) ──
    let x2 = run_kernel(
        "norm",
        "rmsnorm_1024_524288",
        &[&bits(&residual)],
        L * n_embd,
    )?;
    let x2w = run_kernel(
        "eltwise",
        "eltwise_mul_bf16_524288",
        &[&x2, &bits(&wnorm_tiled)],
        L * n_embd,
    )?;
    let gate = run_kernel(
        "gemm",
        "gemm_512x1024x3072_bcm_bf16",
        &[&x2w, &bits(&w_gate)],
        L * n_ff,
    )?;
    let up = run_kernel(
        "gemm",
        "gemm_512x1024x3072_bcm_bf16",
        &[&x2w, &bits(&w_up)],
        L * n_ff,
    )?;
    let gate_silu = run_kernel("activation", "silu_1572864", &[&gate], L * n_ff)?;
    let hidden = run_kernel(
        "eltwise",
        "eltwise_mul_bf16_1572864",
        &[&gate_silu, &up],
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
        &[&bits(&residual), &down],
        L * n_embd,
    )?;

    // ── host f32 reference (and isolate the down projection itself) ──
    let mut want = vec![0.0f32; L * n_embd];
    let mut down_host = vec![0.0f32; L * n_embd];
    for t in 0..L {
        let r = &residual[t * n_embd..(t + 1) * n_embd];
        let ms = r.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
        let inv = 1.0 / (ms + RMS_EPS).sqrt();
        let xnw: Vec<f32> = (0..n_embd).map(|i| r[i] * inv * ffn_norm[i]).collect();
        let g = matvec(&w_gate, &xnw, n_ff, n_embd);
        let u = matvec(&w_up, &xnw, n_ff, n_embd);
        let h: Vec<f32> = (0..n_ff)
            .map(|o| (g[o] / (1.0 + (-g[o]).exp())) * u[o])
            .collect();
        let d = matvec(&w_down, &h, n_embd, n_ff);
        for i in 0..n_embd {
            down_host[t * n_embd + i] = d[i];
            want[t * n_embd + i] = r[i] + d[i];
        }
    }

    let cosine = |npu: &[u16], host: &[f32]| -> f32 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (i, w) in host.iter().enumerate() {
            let g = bf16::from_bits(npu[i]).to_f32();
            dot += g as f64 * *w as f64;
            na += g as f64 * g as f64;
            nb += *w as f64 * *w as f64;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    };
    // The down GEMM alone (bf16 output over K=n_ff=3072) is the precision floor here;
    // the residual add then mixes it with the (exact) residual. Reporting both tells
    // M4 whether large-K GEMMs need f32 output to hold the 0.999 budget over 28 layers.
    let cos_down = cosine(&down, &down_host);
    let cos_out = cosine(&out, &want);
    println!(
        "FFN block on NPU: down(K={n_ff}) cosine={cos_down:.6}  full-block cosine={cos_out:.6} vs host f32"
    );
    if cos_out >= 0.99 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("FFN block chain wrong: cosine {cos_out} < 0.99").into())
    }
}
