//! M3 — validate the Qwen3 layer's *projection layout* on real weights. The layer
//! keeps activations **token-major** (`x[L][n_embd]`) and feeds each weight as the
//! **column-major B operand** (`b_col_maj`, i.e. exactly GGUF's `[out][in]`
//! storage), so `q[L][q_dim] = x[L][n_embd] @ Wqᵀ` is computed transpose-free and
//! the result stays token-major (each token's q_dim contiguous → its per-head
//! 128-chunks are contiguous for the q-norm). This is the layout the whole layer
//! uses to avoid transpose kernels between ops.
//!
//! Build first: `kernels/gemm/build.sh 512 1024 2048 bf16 1` (M=L padded to 512,
//! K=n_embd, N=q_dim, bf16, b_col_maj). Then:
//!   NPU_QWEN3_GGUF=/path/to/Qwen3-Embedding-0.6B-f16.gguf \
//!     cargo run -p seeker-npu --example layer_proj
use std::path::PathBuf;

use half::{bf16, f16};
use seeker_core::gguf::{GgmlType, GgufFile};
use seeker_npu::npu::Context;

const DEFAULT_GGUF: &str = "/models/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-f16.gguf";
const L: usize = 512; // token dim (padded to the b_col_maj transfer-block multiple)

fn artifact(stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels/gemm/build")
        .join(format!("{stem}.{ext}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("NPU_QWEN3_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let gguf = GgufFile::open(&model)?;

    let info = gguf.tensor("blk.0.attn_q.weight").ok_or("missing wq")?;
    if info.ggml_type != GgmlType::F16 {
        return Err(format!("expected F16 wq, got {:?}", info.ggml_type).into());
    }
    // GGUF dims [ne0,ne1] = [in,out] = [n_embd(K), q_dim(N)]; stored row-major as
    // [out][in] = [N][K], which is exactly the b_col_maj B operand.
    if info.dims.len() != 2 {
        return Err(format!("wq must be 2-D, got dims {:?}", info.dims).into());
    }
    let (k, n) = (info.dims[0] as usize, info.dims[1] as usize); // K=n_embd, N=q_dim
    let m = L; // token dim
    println!(
        "layer wq projection: A=x[{m},{k}] (token-major) · B=Wq[{n},{k}] (b_col_maj) -> q[{m},{n}]"
    );

    // B = real Wq, F16 -> bf16, [N=q_dim][K=n_embd] (GGUF order, unchanged).
    let raw = gguf
        .tensor_data("blk.0.attn_q.weight")
        .ok_or("no wq data")?;
    let b_bf: Vec<bf16> = raw
        .chunks_exact(2)
        .map(|b| bf16::from_f32(f16::from_le_bytes([b[0], b[1]]).to_f32()))
        .collect();
    if b_bf.len() != n * k {
        return Err(format!(
            "wq has {} elements, expected {} ({n}×{k})",
            b_bf.len(),
            n * k
        )
        .into());
    }

    // A = synthetic token-major activation x[L][n_embd] (a real X comes from
    // get_rows + RMSNorm in the next step).
    let a_bf: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
        .collect();

    // Host f32 reference: q[t][o] = sum_k x[t][k] * Wq[o][k]  (b_col_maj => Bᵀ).
    let mut want = vec![0.0f32; m * n];
    for t in 0..m {
        for kk in 0..k {
            let a = a_bf[t * k + kk].to_f32();
            if a == 0.0 {
                continue;
            }
            for o in 0..n {
                want[t * n + o] += a * b_bf[o * k + kk].to_f32();
            }
        }
    }

    // NPU GEMM: b_col_maj, bf16 output (the resident-activation forward path).
    let stem = format!("gemm_{m}x{k}x{n}_bcm_bf16");
    let ctx = Context::new(&artifact(&stem, "xclbin"), "MLIR_AIE")?;
    let insts_bytes = std::fs::read(artifact(&stem, "insts.bin"))?;
    let mut instr = ctx.alloc_instr(insts_bytes.len())?;
    instr.as_mut_bytes().copy_from_slice(&insts_bytes);
    instr.sync_to_device()?;

    let mut a_bo = ctx.alloc_data(m * k * 2)?;
    let mut b_bo = ctx.alloc_data(n * k * 2)?;
    let mut c_bo = ctx.alloc_data(m * n * 2)?; // bf16 output
    a_bo.as_mut_slice::<u16>()
        .copy_from_slice(&a_bf.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    b_bo.as_mut_slice::<u16>()
        .copy_from_slice(&b_bf.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    c_bo.as_mut_bytes().fill(0);
    a_bo.sync_to_device()?;
    b_bo.sync_to_device()?;
    c_bo.sync_to_device()?;

    ctx.run(&instr, insts_bytes.len() as u32, &[&a_bo, &b_bo, &c_bo])?;
    c_bo.sync_from_device()?;

    // The layout is what's under test; bf16 *output* rounding over K=1024 is real
    // but noise. Cosine similarity vs the host f32 reference is the layout-correctness
    // metric (and the quantity the embedding ultimately cares about).
    let got = c_bo.as_slice::<u16>();
    let (mut dot, mut na, mut nb, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (i, w) in want.iter().enumerate() {
        let g = bf16::from_bits(got[i]).to_f32();
        dot += g as f64 * *w as f64;
        na += g as f64 * g as f64;
        nb += *w as f64 * *w as f64;
        max_abs = max_abs.max((g - w).abs());
    }
    let cosine = (dot / (na.sqrt() * nb.sqrt())) as f32;
    println!(
        "q on NPU: cosine={cosine:.6}  max_abs_err={max_abs:.4}  q[0]={} (want {:.5})",
        bf16::from_bits(got[0]).to_f32(),
        want[0]
    );
    if cosine >= 0.999 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("layer wq projection layout wrong: cosine {cosine} < 0.999").into())
    }
}
