//! NPU SiLU validation: run the bf16 SiLU activation (the SwiGLU gate function)
//! on the Strix Halo NPU through the in-tree `seeker_npu::npu` wrapper and compare
//! against a host f32 reference `x * sigmoid(x)`.
//!
//! Build the xclbin first (`kernels/activation/build.sh 8192`), then:
//!   cargo run -p seeker-npu --example silu
use std::path::PathBuf;

use half::bf16;
use seeker_npu::npu::Context;

fn artifact(stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/activation/build")
        .join(format!("{stem}.{ext}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::var("NPU_SILU_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let stem = format!("silu_{n}");
    let ctx = Context::new(&artifact(&stem, "xclbin"), "MLIR_AIE")?;
    let insts_bytes = std::fs::read(artifact(&stem, "insts.bin"))?;
    let mut instr = ctx.alloc_instr(insts_bytes.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts_bytes);
    instr.sync_to_device()?;

    // Deterministic bf16 inputs spanning the SiLU's interesting range.
    let xb: Vec<bf16> = (0..n)
        .map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.3))
        .collect();
    let mut x = ctx.alloc_data(n * 2)?; // bf16 = 2 bytes
    let mut out = ctx.alloc_data(n * 2)?;
    x.as_mut_slice::<u16>()
        .copy_from_slice(&xb.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    out.as_mut_slice::<u16>().fill(0);
    x.sync_to_device()?;
    out.sync_to_device()?;

    ctx.run(&instr, insts_bytes.len() as u32, &[&x, &out])?;
    out.sync_from_device()?;

    // numpy.allclose criterion (the shipped bf16 LUT kernel's own tolerance):
    // |got - want| <= atol + rtol*|want|, with atol=2e-2, rtol=3e-2.
    const ATOL: f32 = 2e-2;
    const RTOL: f32 = 3e-2;
    let ob = out.as_slice::<u16>();
    let mut max_abs = 0.0f32;
    let mut worst = 0.0f32; // (err - tol), <= 0 means within tolerance
    for (i, xv) in xb.iter().enumerate() {
        let xf = xv.to_f32();
        let want = xf / (1.0 + (-xf).exp());
        let got = bf16::from_bits(ob[i]).to_f32();
        let err = (got - want).abs();
        max_abs = max_abs.max(err);
        worst = worst.max(err - (ATOL + RTOL * want.abs()));
    }
    println!("silu n={n}: max_abs_err={max_abs:.5} (allclose atol={ATOL} rtol={RTOL})");
    if worst <= 0.0 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("silu mismatch: exceeds allclose tol by {worst}").into())
    }
}
