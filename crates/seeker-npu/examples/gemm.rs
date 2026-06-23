//! NPU GEMM validation: run a fixed-shape bf16->f32 `C = A @ B` on the Strix Halo
//! NPU through the in-tree `seeker_npu::npu` wrapper and compare against a host f32
//! reference (computed over the same bf16-rounded inputs the NPU sees).
//!
//! Build the xclbin first (see `kernels/gemm/build.sh`), then:
//!   cargo run -p seeker-npu --example gemm
//! Override the shape/artifacts via NPU_GEMM_{M,K,N,XCLBIN,INSTS}.
use std::path::PathBuf;

use half::bf16;
use seeker_npu::npu::Context;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn artifact(env_key: &str, default_name: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_key) {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/gemm/build")
        .join(default_name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default to the committed-shape Qwen3 `wq` GEMM (q_dim x n_embd x L_bucket).
    let m = env_usize("NPU_GEMM_M", 2048);
    let k = env_usize("NPU_GEMM_K", 1024);
    let n = env_usize("NPU_GEMM_N", 256);
    let stem = format!("gemm_{m}x{k}x{n}");
    let xclbin = artifact("NPU_GEMM_XCLBIN", &format!("{stem}.xclbin"));
    let insts = artifact("NPU_GEMM_INSTS", &format!("{stem}.insts.bin"));
    println!("GEMM {m}x{k}x{n} bf16->f32  xclbin={}", xclbin.display());

    // Deterministic small inputs, rounded to bf16 (exactly what the NPU consumes).
    let a_bf: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
        .collect();
    let b_bf: Vec<bf16> = (0..k * n)
        .map(|i| bf16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
        .collect();

    // Host f32 reference over the bf16-rounded inputs (matches the NPU's f32 accum).
    let mut want = vec![0.0f32; m * n];
    for row in 0..m {
        for kk in 0..k {
            let a = a_bf[row * k + kk].to_f32();
            if a == 0.0 {
                continue;
            }
            let arow = row * n;
            let brow = kk * n;
            for col in 0..n {
                want[arow + col] += a * b_bf[brow + col].to_f32();
            }
        }
    }

    // ── NPU run ──
    let ctx = Context::new(&xclbin, "MLIR_AIE")?;
    let insts_bytes = std::fs::read(&insts)?;
    let mut instr = ctx.alloc_instr(insts_bytes.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts_bytes);
    instr.sync_to_device()?;

    let mut a_bo = ctx.alloc_data(m * k * 2)?; // bf16 = 2 bytes
    let mut b_bo = ctx.alloc_data(k * n * 2)?;
    let mut c_bo = ctx.alloc_data(m * n * 4)?; // f32 = 4 bytes
    // bf16 is bit-identical to u16; copy the raw bits into the device buffers.
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

    // ── Compare ──
    let got = c_bo.as_slice::<f32>();
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        let abs = (g - w).abs();
        max_abs = max_abs.max(abs);
        let denom = w.abs().max(1e-3);
        max_rel = max_rel.max(abs / denom);
    }
    println!(
        "max_abs_err={max_abs:.5}  max_rel_err={max_rel:.5}  c[0]={} (want {:.5})",
        got[0], want[0]
    );
    // bf16 inputs are applied identically host+NPU; the residual is f32 MAC-order /
    // rounding noise over K accumulations — a small relative tolerance covers it.
    if max_rel < 2e-2 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("GEMM mismatch: max_rel_err {max_rel} >= 2e-2").into())
    }
}
