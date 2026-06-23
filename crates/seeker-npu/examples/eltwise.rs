//! NPU element-wise validation: run the `add` (residual) and `mul` (SwiGLU
//! gate*up) kernels — in both f32 and bf16 — on the Strix Halo NPU through the
//! in-tree `seeker_npu::npu` wrapper, comparing against a host f32 reference. The
//! bf16 variants are the ones the bf16-activation forward uses.
//!
//! Build the xclbins first, e.g. `kernels/eltwise/build.sh add bf16 4096`, then:
//!   cargo run -p seeker-npu --example eltwise
use std::path::PathBuf;

use half::bf16;
use seeker_npu::npu::Context;

fn artifact(stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/eltwise/build")
        .join(format!("{stem}.{ext}"))
}

/// Run one (op, dtype) eltwise kernel and validate vs a host f32 reference.
/// `elem_bytes` is 4 for f32, 2 for bf16. `to_f32`/`from_f32` convert a buffer
/// element's raw bits.
fn run(op: &str, dtype: &str, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let stem = format!("eltwise_{op}_{dtype}_{n}");
    let ctx = Context::new(&artifact(&stem, "xclbin"), "MLIR_AIE")?;
    let insts_bytes = std::fs::read(artifact(&stem, "insts.bin"))?;
    let mut instr = ctx.alloc_instr(insts_bytes.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts_bytes);
    instr.sync_to_device()?;

    let bf = dtype == "bf16";
    let esz = if bf { 2 } else { 4 };
    // Host inputs as f32, then encoded into the device buffers at the right width.
    let af: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let bvf: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
    let enc = |v: &[f32], buf: &mut seeker_npu::npu::Buffer| {
        if bf {
            buf.as_mut_slice::<u16>().copy_from_slice(
                &v.iter()
                    .map(|x| bf16::from_f32(*x).to_bits())
                    .collect::<Vec<_>>(),
            );
        } else {
            buf.as_mut_slice::<f32>().copy_from_slice(v);
        }
    };

    let mut a = ctx.alloc_data(n * esz)?;
    let mut b = ctx.alloc_data(n * esz)?;
    let mut c = ctx.alloc_data(n * esz)?;
    enc(&af, &mut a);
    enc(&bvf, &mut b);
    c.as_mut_bytes().fill(0);
    a.sync_to_device()?;
    b.sync_to_device()?;
    c.sync_to_device()?;

    ctx.run(&instr, insts_bytes.len() as u32, &[&a, &b, &c])?;
    c.sync_from_device()?;

    let dec = |buf: &seeker_npu::npu::Buffer, i: usize| -> f32 {
        if bf {
            bf16::from_bits(buf.as_slice::<u16>()[i]).to_f32()
        } else {
            buf.as_slice::<f32>()[i]
        }
    };
    // bf16 inputs are rounded; compare the kernel output against the reference over
    // the SAME rounded inputs, with a dtype-appropriate tolerance.
    let (atol, rtol) = if bf { (2e-2f32, 3e-2f32) } else { (1e-5, 1e-4) };
    let mut worst = 0.0f32;
    for i in 0..n {
        let (ai, bi) = (dec(&a, i), dec(&b, i));
        let want = if op == "add" { ai + bi } else { ai * bi };
        let err = (dec(&c, i) - want).abs();
        worst = worst.max(err - (atol + rtol * want.abs()));
    }
    println!(
        "eltwise {op} {dtype} n={n}: {}",
        if worst <= 0.0 { "ok" } else { "OUT OF TOL" }
    );
    if worst <= 0.0 {
        Ok(())
    } else {
        Err(format!("eltwise {op} {dtype} exceeds tol by {worst}").into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::var("NPU_ELTWISE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    for op in ["add", "mul"] {
        for dtype in ["f32", "bf16"] {
            run(op, dtype, n)?;
        }
    }
    println!("PASS");
    Ok(())
}
