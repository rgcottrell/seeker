//! M3 step 1 — real-weight GEMM on the NPU. Loads the actual Qwen3-Embedding-0.6B
//! `blk.0.attn_q.weight` (F16) from the GGUF, dequantizes it to bf16, and runs the
//! Q-projection GEMM `Y = Wq · X` on the NPU (Wq is [q_dim=2048, n_embd=1024], X a
//! synthetic [n_embd, N=256] activation), validating against a host f32 reference
//! over the same bf16-rounded operands. This proves the GGUF -> bf16 -> NPU-GEMM
//! pipeline with real model weights before the full layer is wired.
//!
//! Build the GEMM first (`kernels/gemm/build.sh 2048 1024 256`), then:
//!   NPU_QWEN3_GGUF=/path/to/Qwen3-Embedding-0.6B-f16.gguf \
//!     cargo run -p seeker-npu --example layer0_wq
use std::path::PathBuf;

use half::{bf16, f16};
use seeker_core::gguf::{GgmlType, GgufFile};
use seeker_npu::npu::Context;

const DEFAULT_GGUF: &str = "/models/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-f16.gguf";
const N: usize = 256; // padded token count (L bucket)

fn artifact(stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/gemm/build")
        .join(format!("{stem}.{ext}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("NPU_QWEN3_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let gguf = GgufFile::open(&model)?;

    let name = "blk.0.attn_q.weight";
    let info = gguf.tensor(name).ok_or("missing blk.0.attn_q.weight")?;
    if info.ggml_type != GgmlType::F16 {
        return Err(format!(
            "expected F16 wq, got {:?} (use the f16 GGUF)",
            info.ggml_type
        )
        .into());
    }
    // GGUF dims are [ne0, ne1] = [in, out] = [n_embd(K), q_dim(M)]; row-major data
    // is [M][K], exactly the whole_array GEMM's A[M,K].
    let (k, m) = (info.dims[0] as usize, info.dims[1] as usize);
    println!("wq: {name} F16 dims=[{k},{m}] (K=n_embd, M=q_dim)");

    // Dequantize the real weights F16 -> bf16 into A[M,K].
    let raw = gguf.tensor_data(name).ok_or("no wq data")?;
    let a_bf: Vec<bf16> = raw
        .chunks_exact(2)
        .map(|b| bf16::from_f32(f16::from_le_bytes([b[0], b[1]]).to_f32()))
        .collect();
    assert_eq!(a_bf.len(), m * k, "wq element count");

    // Synthetic activation X[K,N] (a real X comes from embed+RMSNorm in M3b).
    let b_bf: Vec<bf16> = (0..k * N)
        .map(|i| bf16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
        .collect();

    // Host f32 reference over the bf16-rounded operands (matches the NPU f32 accum).
    let mut want = vec![0.0f32; m * N];
    for row in 0..m {
        for kk in 0..k {
            let a = a_bf[row * k + kk].to_f32();
            if a == 0.0 {
                continue;
            }
            let (arow, brow) = (row * N, kk * N);
            for col in 0..N {
                want[arow + col] += a * b_bf[brow + col].to_f32();
            }
        }
    }

    // NPU GEMM (bf16 -> f32) for the Qwen3 wq shape.
    let stem = format!("gemm_{m}x{k}x{N}");
    let ctx = Context::new(&artifact(&stem, "xclbin"), "MLIR_AIE")?;
    let insts_bytes = std::fs::read(artifact(&stem, "insts.bin"))?;
    let mut instr = ctx.alloc_instr(insts_bytes.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts_bytes);
    instr.sync_to_device()?;

    let mut a_bo = ctx.alloc_data(m * k * 2)?;
    let mut b_bo = ctx.alloc_data(k * N * 2)?;
    let mut c_bo = ctx.alloc_data(m * N * 4)?;
    a_bo.as_mut_slice::<u16>()
        .copy_from_slice(&a_bf.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    b_bo.as_mut_slice::<u16>()
        .copy_from_slice(&b_bf.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    c_bo.as_mut_slice::<f32>().fill(0.0);
    a_bo.sync_to_device()?;
    b_bo.sync_to_device()?;
    c_bo.sync_to_device()?;

    ctx.run(&instr, insts_bytes.len() as u32, &[&a_bo, &b_bo, &c_bo])?;
    c_bo.sync_from_device()?;

    let got = c_bo.as_slice::<f32>();
    let mut max_rel = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        max_rel = max_rel.max((g - w).abs() / w.abs().max(1e-2));
    }
    println!(
        "wq GEMM {m}x{k}x{N} on NPU: max_rel_err={max_rel:.5}  c[0]={} (want {:.5})",
        got[0], want[0]
    );
    if max_rel < 3e-2 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("wq GEMM mismatch: max_rel_err {max_rel}").into())
    }
}
