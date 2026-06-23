//! NPU bring-up smoke test: run the IRON-generated `MLIR_AIE` vector-add kernel
//! (c = a + b over 64 i32 elements) through the in-tree `seeker_npu::npu` wrapper.
//!
//! Requires a Strix Halo NPU + XRT, and a prebuilt vadd xclbin + instruction blob
//! (e.g. from `~/workspace/gpu-npu-demo/kernels/aie/vadd/build`). Point at them via:
//!   NPU_XCLBIN=.../vadd.xclbin NPU_INSTS=.../insts.bin \
//!     cargo run -p seeker-npu --example vadd
use std::path::PathBuf;

use seeker_npu::npu::Context;

const N: usize = 64;

fn artifact(env_key: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    std::env::var(env_key)
        .map(PathBuf::from)
        .map_err(|_| format!("set {env_key} to the vadd artifact path").into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xclbin = artifact("NPU_XCLBIN")?;
    let insts = artifact("NPU_INSTS")?;

    let ctx = Context::new(&xclbin, "MLIR_AIE")?;

    let insts_bytes = std::fs::read(&insts)?;
    let mut instr = ctx.alloc_instr(insts_bytes.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts_bytes);
    instr.sync_to_device()?;

    let mut a = ctx.alloc_data(N * 4)?;
    let mut b = ctx.alloc_data(N * 4)?;
    let mut c = ctx.alloc_data(N * 4)?;
    {
        let av = a.as_mut_slice::<i32>();
        let bv = b.as_mut_slice::<i32>();
        let cv = c.as_mut_slice::<i32>();
        for i in 0..N {
            av[i] = i as i32 + 1;
            bv[i] = 10 * (i as i32 + 1);
            cv[i] = -1;
        }
    }
    // Every buffer (including the output) must be resident on the device before the run.
    a.sync_to_device()?;
    b.sync_to_device()?;
    c.sync_to_device()?;

    ctx.run(&instr, insts_bytes.len() as u32, &[&a, &b, &c])?;

    c.sync_from_device()?;
    let cv = c.as_slice::<i32>();
    let mut errors = 0;
    for (i, &got) in cv.iter().enumerate() {
        let want = (i as i32 + 1) + 10 * (i as i32 + 1);
        if got != want {
            if errors < 8 {
                eprintln!("  mismatch[{i}]: got {got} want {want}");
            }
            errors += 1;
        }
    }
    println!("N={N} c[0]={} (want 11) errors={errors}", cv[0]);
    if errors == 0 {
        println!("PASS");
        Ok(())
    } else {
        Err("vadd mismatch".into())
    }
}
