//! M6 — validate the FUSED FFN gate kernel (`kernels/fused/ffn.py` with the SiLU
//! epilogue) on real `blk.0` weights: one NPU dispatch computes `SiLU(x2 · ffn_gateᵀ)`
//! (GEMM + on-chip SiLU), vs the host f32 reference. This is the proof that fused
//! multi-op AIE kernels work in our stack (a GEMM with an in-core activation epilogue).
//!
//! Build first:  kernels/fused/build.sh 512 1024 3072 1 1
//! then:  cargo run -p seeker-npu --example ffn_fused
#[path = "common/mod.rs"]
mod common;
use common::*;
use seeker_core::gguf::GgufFile;

const L: usize = 512;
const RMS_EPS: f32 = 1e-5;

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
    // The fused xclbin is fixed-shape (Qwen3-Embedding-0.6B); reject a model whose dims
    // don't match, or the BOs wouldn't line up with what the kernel reads/writes.
    if (n_embd, n_ff) != (1024, 3072) {
        return Err(format!(
            "ffn_silu_512x1024x3072_bcm is built for n_embd=1024 n_ff=3072; \
             this model has n_embd={n_embd} n_ff={n_ff}"
        )
        .into());
    }
    println!(
        "fused FFN gate: SiLU(x2 · ffn_gateᵀ) in one NPU dispatch — n_embd={n_embd} n_ff={n_ff} L={L}"
    );

    // x2 = rmsnorm(residual)·ffn_norm, residual = real token embeddings (host, as in ffn_block).
    let token_embd = f16_to_f32(&gguf, "token_embd.weight")?;
    let ffn_norm = f32_vec(&gguf, "blk.0.ffn_norm.weight", n_embd)?;
    let w_gate = f16_to_f32(&gguf, "blk.0.ffn_gate.weight")?; // [n_ff][n_embd] = B (bcm)
    let ids: Vec<usize> = (0..L).map(|t| (t * 37 + 1) % vocab).collect();
    let mut x2 = vec![0.0f32; L * n_embd];
    for (t, &id) in ids.iter().enumerate() {
        let e = &token_embd[id * n_embd..(id + 1) * n_embd];
        let ms = e.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
        let inv = 1.0 / (ms + RMS_EPS).sqrt();
        for i in 0..n_embd {
            x2[t * n_embd + i] = e[i] * inv * ffn_norm[i];
        }
    }

    // ── one fused NPU dispatch: GEMM + SiLU epilogue ──
    let gate_silu = run_kernel(
        "fused",
        "ffn_silu_512x1024x3072_bcm",
        &[&bits(&x2), &bits(&w_gate)],
        L * n_ff,
    )?;

    // ── host f32 reference: SiLU(x2 · ffn_gateᵀ) ──
    let mut want = vec![0.0f32; L * n_ff];
    for t in 0..L {
        let xt = &x2[t * n_embd..(t + 1) * n_embd];
        let g = matvec(&w_gate, xt, n_ff, n_embd);
        for (o, gv) in g.iter().enumerate() {
            want[t * n_ff + o] = gv / (1.0 + (-gv).exp());
        }
    }

    let cos = cosine(&gate_silu, &want);
    println!(
        "fused gate+SiLU cosine={cos:.6} vs host f32 (one dispatch; bf16 K-accum + bf16 SiLU LUT)"
    );
    if cos >= 0.99 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("fused FFN gate wrong: cosine {cos} < 0.99").into())
    }
}
