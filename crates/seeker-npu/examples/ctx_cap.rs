//! M5 probe — how many NPU hardware contexts (loaded xclbins) can be held resident
//! at once before `CREATE_HWCTX` fails? This sizes the forward's Context cache.
//!   cargo run -p seeker-npu --example ctx_cap
use std::path::PathBuf;

use seeker_npu::npu::Context;

fn gemm(stem: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/gemm/build")
        .join(format!("{stem}.xclbin"))
}

fn main() {
    // The 7 distinct f32-output GEMM xclbins the hybrid forward dispatches.
    let stems = [
        "gemm_512x1024x2048_bcm",
        "gemm_512x1024x1024_bcm",
        "gemm_512x2048x1024_bcm",
        "gemm_512x1024x3072_bcm",
        "gemm_512x3072x1024_bcm",
        "gemm_512x128x1024_bcm",
        "gemm_512x1024x256",
    ];
    // Hold distinct xclbins resident.
    let mut held: Vec<Context> = Vec::new();
    println!("-- distinct xclbins held simultaneously --");
    for (i, s) in stems.iter().enumerate() {
        match Context::new(&gemm(s), "MLIR_AIE") {
            Ok(c) => {
                held.push(c);
                println!("  {} contexts OK ({s})", i + 1);
            }
            Err(e) => {
                println!("  FAILED at {} ({s}): {e}", i + 1);
                break;
            }
        }
    }
    let distinct = held.len();
    held.clear();

    // Hold N copies of the SAME xclbin resident (does the cap count handles or columns?).
    println!("-- copies of one xclbin held simultaneously --");
    let mut n = 0;
    loop {
        match Context::new(&gemm(stems[0]), "MLIR_AIE") {
            Ok(c) => {
                held.push(c);
                n += 1;
                if n >= 32 {
                    println!("  reached {n} (stopping)");
                    break;
                }
            }
            Err(e) => {
                println!("  FAILED at {}: {e}", n + 1);
                break;
            }
        }
    }
    println!("\ncap: {distinct} distinct, {n} same-xclbin held concurrently");
}
