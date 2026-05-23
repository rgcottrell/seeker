//! `seeker run` — single-shot forward pass: feed the prompt through the
//! model, take logits at the last position, argmax → print the predicted
//! next token. Exits.

use std::error::Error;
use std::path::PathBuf;

use clap::Args;

use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::GgufFile;
use crate::inference::sample::argmax;
use crate::inference::Engine;
use crate::tokenizer::build_tokenizer;

const SCRATCH_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Args)]
pub struct RunArgs {
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]". (short: -hf, -hfr)
    #[arg(long = "hf-repo", required_unless_present = "model", conflicts_with = "model")]
    hf_repo: Option<String>,

    /// Specific file within the repo. (short: -hff)
    #[arg(long = "hf-file", requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,

    /// HF auth token (defaults to HF_TOKEN env / ~/.cache/huggingface/token). (short: -hft)
    #[arg(long = "hf-token", requires = "hf_repo", conflicts_with = "model")]
    hf_token: Option<String>,

    /// Resolve files from the local cache only; never hit the network.
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    offline: bool,

    /// Path to a local .gguf model file.
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,

    /// Prompt to feed through one forward pass.
    #[arg(long, default_value = "Once upon a time")]
    prompt: String,
}

pub async fn run(args: RunArgs) -> Result<(), Box<dyn Error>> {
    let path = resolve_model_path(&args).await?;
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;

    let add_special = bundle.add_bos_default || bundle.add_eos_default;
    let encoding = bundle
        .tokenizer
        .encode(args.prompt.as_str(), add_special)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let tokens: Vec<u32> = encoding.get_ids().to_vec();

    let mut engine = Engine::new(SCRATCH_BYTES)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened");

    let weights = engine.upload_weights(&gguf)?;
    tracing::info!(
        tensors = weights.views.len(),
        bytes = weights.region.cursor,
        "weights uploaded to GPU",
    );

    let model = crate::models::open(&gguf, weights, bundle)?;

    // DEBUG: isolated matmul test with hardcoded 2x2 input.
    if std::env::var("SEEKER_MATMUL_TEST").is_ok() {
        matmul_smoke_test(&mut engine, model.weights())?;
        return Ok(());
    }
    if std::env::var("SEEKER_RMSNORM_TEST").is_ok() {
        rms_norm_smoke_test(&mut engine, model.weights())?;
        return Ok(());
    }
    if std::env::var("SEEKER_GETROWS_TEST").is_ok() {
        get_rows_smoke_test(&mut engine, model.weights())?;
        return Ok(());
    }

    let logits = engine.forward(model.weights(), |ctx| model.record_forward(ctx, &tokens))?;
    let next_id = argmax(&logits) as u32;
    let piece = model
        .tokenizer()
        .tokenizer
        .decode(&[next_id], false)
        .unwrap_or_else(|_| format!("<id={next_id}>"));

    println!("prompt: {}", args.prompt);
    println!("tokens: {tokens:?}");
    println!("next:   {next_id} -> {piece:?}");
    Ok(())
}

fn matmul_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::gguf::GgmlType;
    use crate::inference::ops::matmul;

    // Production size: M=49152 (vocab), K=576 (hidden), N=4 (tokens)
    let m: u64 = 49152;
    let k: u64 = 576;
    let n: u64 = 4;
    let logits = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        let a = ctx.alloc_tensor([k, m, 1, 1], GgmlType::F16)?;
        let b = ctx.alloc_tensor([k, n, 1, 1], GgmlType::F32)?;
        let d = ctx.alloc_tensor([m, n, 1, 1], GgmlType::F32)?;
        let host_ptr = ctx.scratch.host_ptr.unwrap();
        unsafe {
            // A = all 1.0 (F16)
            let a_ptr = host_ptr.add(a.byte_offset as usize) as *mut u16;
            for i in 0..(k * m) as usize {
                *a_ptr.add(i) = f32_to_f16_bits(1.0);
            }
            // B = column index per row: B[k, n] = n+1
            let b_ptr = host_ptr.add(b.byte_offset as usize) as *mut f32;
            for col in 0..n as usize {
                for row in 0..k as usize {
                    *b_ptr.add(col * k as usize + row) = (col + 1) as f32;
                }
            }
        }
        matmul::record(ctx, a, b, d)?;
        Ok(d.range())
    })?;

    // Expected: D[m, n] = sum_k A[k,m] * B[k,n] = sum_k 1.0 * (n+1) = K * (n+1)
    // So col 0 (n=0): all values = K*1 = 576. col 1: 1152. col 2: 1728. col 3: 2304.
    let nonzero = logits.iter().filter(|x| **x != 0.0).count();
    println!("matmul production-size test: M={m}, K={k}, N={n}");
    println!("  total: {}", logits.len());
    println!("  non-zero: {nonzero} (expected {})", m * n);
    for col in 0..n as usize {
        let col_start = col * m as usize;
        let col_slice = &logits[col_start..col_start + m as usize];
        let col_nonzero = col_slice.iter().filter(|x| **x != 0.0).count();
        let col_max = col_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let col_first = col_slice[0];
        let col_last = col_slice[col_slice.len() - 1];
        println!("  col{col}: nonzero={col_nonzero}/{m}, max={col_max}, first={col_first}, last={col_last} (expected {})", k * (col + 1) as u64);
    }
    Ok(())
}

fn get_rows_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::gguf::GgmlType;
    use crate::inference::ops::elementwise;
    // Embedding table: 4 rows × hidden=576 F32. Row i is value (i*100 + j).
    let hidden: u64 = 576;
    let n_rows: u64 = 49152;  // production size
    let n_indices: u64 = 4;
    let logits = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        // F16 table — matches production token_embd.
        let table = ctx.alloc_tensor([hidden, n_rows, 1, 1], GgmlType::F16)?;
        let indices_buf = ctx.alloc_scratch(n_indices * 4)?;
        let dst = ctx.alloc_tensor([hidden, n_indices, 1, 1], GgmlType::F32)?;
        // Fill F16 table: position(j, i) = i*100 + j as F16
        let host_ptr = ctx.scratch.host_ptr.unwrap();
        unsafe {
            let table_ptr = host_ptr.add(table.byte_offset as usize) as *mut u16;
            for i in 0..n_rows {
                for j in 0..hidden {
                    let v = ((i * 100 + j) % 50000) as f32;
                    *table_ptr.add((i * hidden + j) as usize) = f32_to_f16_bits(v);
                }
            }
            let indices_data: [u32; 4] = [0, 1, 2, 3];
            std::ptr::copy_nonoverlapping(indices_data.as_ptr(), host_ptr.add(indices_buf.offset as usize) as *mut u32, indices_data.len());
        }
        elementwise::record_get_rows(ctx, table, indices_buf, n_indices as u32, dst)?;
        Ok(dst.range())
    })?;
    let nonzero = logits.iter().filter(|x| **x != 0.0).count();
    println!("get_rows test: hidden={hidden}, n_rows={n_rows}, n_indices={n_indices}");
    println!("  total values: {}", logits.len());
    println!("  non-zero:     {nonzero} (expected {})", hidden * n_indices);
    for t in 0..n_indices {
        let row = &logits[(t * hidden) as usize..((t+1) * hidden) as usize];
        println!("  row{t} first 4: {:?}", &row[..4]);
        println!("  row{t} last 4:  {:?}", &row[row.len()-4..]);
    }
    Ok(())
}

fn rms_norm_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::gguf::GgmlType;
    use crate::inference::ops::rms_norm;
    // [hidden=4, L=4] F32 input, weight=[1,1,1,1] → rmsnorm should produce
    // x / sqrt(mean(x^2)+eps) for each row.
    // Test with production-size hidden=576, L=4. All values = 1.0 so
    // mean(x²)=1, rsqrt(1+eps)≈1, output should be all 1.0.
    let hidden: u64 = 576;
    let l: u64 = 4;
    let logits = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        let src = ctx.alloc_tensor([hidden, l, 1, 1], GgmlType::F32)?;
        let weight = ctx.alloc_tensor([hidden, 1, 1, 1], GgmlType::F32)?;
        let dst = ctx.alloc_tensor([hidden, l, 1, 1], GgmlType::F32)?;
        let src_data: Vec<f32> = vec![1.0; (hidden * l) as usize];
        let weight_data: Vec<f32> = vec![1.0; hidden as usize];
        let host_ptr = ctx.scratch.host_ptr.unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(src_data.as_ptr(), host_ptr.add(src.byte_offset as usize) as *mut f32, src_data.len());
            std::ptr::copy_nonoverlapping(weight_data.as_ptr(), host_ptr.add(weight.byte_offset as usize) as *mut f32, weight_data.len());
        }
        rms_norm::record(ctx, src, weight, dst, 1e-5)?;
        Ok(dst.range())
    })?;
    // Expected (per row):
    // row0: x=[1,2,3,4], mean(x^2) = 30/4 = 7.5, rsqrt = 0.3651 → [0.365, 0.730, 1.095, 1.461]
    // row1: x=[2,4,6,8], mean=120/4=30, rsqrt = 0.1826 → [0.365, 0.730, 1.095, 1.461]
    // row2: x=[5,5,5,5], mean=25, rsqrt=0.2 → [1.0, 1.0, 1.0, 1.0]
    // row3: x=[10,20,30,40], mean=3000/4=750, rsqrt=0.0365 → [0.365, 0.730, 1.095, 1.461]
    let nonzero = logits.iter().filter(|x| **x != 0.0).count();
    let close_to_one = logits.iter().filter(|x| (**x - 1.0).abs() < 0.01).count();
    println!("rms_norm production-size test: hidden={hidden}, L={l}");
    println!("  total values: {}", logits.len());
    println!("  non-zero:     {nonzero}");
    println!("  ~1.0 values:  {close_to_one}");
    println!("  first 4: {:?}", &logits[..4]);
    println!("  last 4:  {:?}", &logits[logits.len()-4..]);
    Ok(())
}

fn f32_to_f16_bits(v: f32) -> u16 {
    // IEEE 754 conversion. Simple version — assumes finite values, no
    // denormals, no NaN handling beyond the spec.
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mant32 = bits & 0x7FFFFF;
    if exp32 == 0 {
        return sign << 15;
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 <= 0 {
        return sign << 15;
    }
    if exp16 >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    let mant16 = (mant32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mant16
}

async fn resolve_model_path(args: &RunArgs) -> Result<PathBuf, Box<dyn Error>> {
    match (args.hf_repo.clone(), args.model.clone()) {
        (Some(repo), None) => Ok(resolve_hf(
            &HfResolveArgs {
                repo,
                file: args.hf_file.clone(),
                token: args.hf_token.clone(),
                offline: args.offline,
            },
            false,
        )
        .await?
        .main),
        (None, Some(model)) => Ok(model),
        _ => unreachable!("clap group invariant"),
    }
}
