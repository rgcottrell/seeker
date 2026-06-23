//! NPU normalization validation: run bf16 `rmsnorm` and `softmax` (per-1024-tile =
//! per-token, since Qwen3 n_embd == 1024) on the Strix Halo NPU through the in-tree
//! `seeker_npu::npu` wrapper and compare against host f32 references.
//!
//! rmsnorm here is gamma=1 / eps=1e-5 (the learned weight is applied separately via
//! the eltwise `mul`). Build first: `kernels/norm/build.sh rmsnorm 8192` and
//! `softmax 8192`, then: `cargo run -p seeker-npu --example norm`.
use std::path::PathBuf;

use half::bf16;
use seeker_npu::npu::Context;

const TILE: usize = 1024;

fn artifact(stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/norm/build")
        .join(format!("{stem}.{ext}"))
}

fn run(op: &str, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let stem = format!("{op}_{n}");
    let ctx = Context::new(&artifact(&stem, "xclbin"), "MLIR_AIE")?;
    let insts_bytes = std::fs::read(artifact(&stem, "insts.bin"))?;
    let mut instr = ctx.alloc_instr(insts_bytes.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts_bytes);
    instr.sync_to_device()?;

    // Deterministic bf16 input, varied per row so each tile is a distinct test.
    let xb: Vec<bf16> = (0..n)
        .map(|i| bf16::from_f32((((i * 31 + 7) % 97) as f32 - 48.0) * 0.05))
        .collect();
    let mut x = ctx.alloc_data(n * 2)?;
    let mut out = ctx.alloc_data(n * 2)?;
    x.as_mut_slice::<u16>()
        .copy_from_slice(&xb.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    out.as_mut_slice::<u16>().fill(0);
    x.sync_to_device()?;
    out.sync_to_device()?;

    ctx.run(&instr, insts_bytes.len() as u32, &[&x, &out])?;
    out.sync_from_device()?;

    let ob = out.as_slice::<u16>();
    let (atol, rtol) = (2e-2f32, 5e-2f32);
    let mut max_abs = 0.0f32;
    let mut worst = 0.0f32;
    for row in 0..n / TILE {
        let xr: Vec<f32> = (0..TILE).map(|j| xb[row * TILE + j].to_f32()).collect();
        let want: Vec<f32> = if op == "rmsnorm" {
            let ms = xr.iter().map(|v| v * v).sum::<f32>() / TILE as f32;
            let inv = 1.0 / (ms + 1e-5).sqrt();
            xr.iter().map(|v| v * inv).collect()
        } else {
            let m = xr.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = xr.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = e.iter().sum();
            e.iter().map(|v| v / s).collect()
        };
        for (j, w) in want.iter().enumerate() {
            let got = bf16::from_bits(ob[row * TILE + j]).to_f32();
            let err = (got - w).abs();
            max_abs = max_abs.max(err);
            worst = worst.max(err - (atol + rtol * w.abs()));
        }
    }
    println!("{op} n={n}: max_abs_err={max_abs:.5} (allclose atol={atol} rtol={rtol})");
    if worst <= 0.0 {
        Ok(())
    } else {
        Err(format!("{op} mismatch: exceeds allclose tol by {worst}").into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::var("NPU_NORM_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    run("rmsnorm", n)?;
    run("softmax", n)?;
    println!("PASS");
    Ok(())
}
