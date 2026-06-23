//! NPU element-wise validation: run the f32 `add` (residual) and `mul` (SwiGLU
//! gate*up) kernels on the Strix Halo NPU through the in-tree `seeker_npu::npu`
//! wrapper and compare against a host f32 reference.
//!
//! Build the xclbins first (`kernels/eltwise/build.sh add 4096` and `mul 4096`),
//! then: `cargo run -p seeker-npu --example eltwise`.
use std::path::PathBuf;

use seeker_npu::npu::Context;

fn artifact(stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/eltwise/build")
        .join(format!("{stem}.{ext}"))
}

fn run_op(op: &str, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let stem = format!("eltwise_{op}_{n}");
    let ctx = Context::new(&artifact(&stem, "xclbin"), "MLIR_AIE")?;
    let insts_bytes = std::fs::read(artifact(&stem, "insts.bin"))?;
    let mut instr = ctx.alloc_instr(insts_bytes.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts_bytes);
    instr.sync_to_device()?;

    let mut a = ctx.alloc_data(n * 4)?; // f32
    let mut b = ctx.alloc_data(n * 4)?;
    let mut c = ctx.alloc_data(n * 4)?;
    {
        let av = a.as_mut_slice::<f32>();
        let bv = b.as_mut_slice::<f32>();
        for i in 0..n {
            av[i] = ((i % 13) as f32 - 6.0) * 0.1;
            bv[i] = ((i % 7) as f32 - 3.0) * 0.2;
        }
    }
    c.as_mut_slice::<f32>().fill(0.0);
    a.sync_to_device()?;
    b.sync_to_device()?;
    c.sync_to_device()?;

    ctx.run(&instr, insts_bytes.len() as u32, &[&a, &b, &c])?;
    c.sync_from_device()?;

    let (av, bv, cv) = (
        a.as_slice::<f32>(),
        b.as_slice::<f32>(),
        c.as_slice::<f32>(),
    );
    let mut max_err = 0.0f32;
    for i in 0..n {
        let want = if op == "add" {
            av[i] + bv[i]
        } else {
            av[i] * bv[i]
        };
        max_err = max_err.max((cv[i] - want).abs());
    }
    println!("eltwise {op} n={n}: max_abs_err={max_err:.7}");
    if max_err < 1e-4 {
        Ok(())
    } else {
        Err(format!("eltwise {op} mismatch: max_abs_err {max_err}").into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::var("NPU_ELTWISE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    run_op("add", n)?;
    run_op("mul", n)?;
    println!("PASS");
    Ok(())
}
