//! `seeker run` — single-shot forward pass: feed the prompt through the
//! model, sample from the last-position logits via the GPU sampler chain,
//! print the generated tokens. Exits.

use std::error::Error;
use std::path::PathBuf;

use clap::Args;

use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::{GgmlType, GgufFile};
use crate::inference::kv_cache::{parse_dtype, KvCacheConfig};
use crate::inference::sample::{Sampler, SamplerConfig};
use crate::inference::Engine;
use crate::tokenizer::build_tokenizer;

const SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

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

    /// Number of tokens to generate (>= 1). 1 = single prefill, exit.
    #[arg(long, default_value_t = 1)]
    max_tokens: u32,

    /// KV cache K dtype. One of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1.
    #[arg(long = "cache-type-k", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_k: GgmlType,

    /// KV cache V dtype. Same legal values as --cache-type-k.
    #[arg(long = "cache-type-v", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_v: GgmlType,

    // ─── Sampling ───────────────────────────────────────────────────────
    /// Sampling temperature. 0 → greedy argmax.
    #[arg(long = "temp", alias = "temperature", default_value_t = 0.0)]
    temperature: f32,

    /// Top-K filter (0 = disabled, full vocab).
    #[arg(long = "top-k", default_value_t = 20)]
    top_k: u32,

    /// Top-P (nucleus) filter (1.0 = disabled).
    #[arg(long = "top-p", default_value_t = 0.95)]
    top_p: f32,

    /// Min-P filter (0.0 = disabled).
    #[arg(long = "min-p", default_value_t = 0.0)]
    min_p: f32,

    /// Presence penalty (subtract from any repeated-token logit; 0.0 = off).
    #[arg(long = "presence-penalty", default_value_t = 0.0)]
    presence_penalty: f32,

    /// Frequency penalty (subtract count×p from repeated-token logits; 0.0 = off).
    #[arg(long = "frequency-penalty", default_value_t = 0.0)]
    frequency_penalty: f32,

    /// Repetition penalty (multiply/divide repeated logits; 1.0 = off).
    #[arg(long = "repeat-penalty", alias = "repetition-penalty", default_value_t = 1.0)]
    repeat_penalty: f32,

    /// How many trailing tokens contribute to penalties (≤ scratch budget).
    #[arg(long = "penalty-last-n", default_value_t = 64)]
    penalty_last_n: usize,

    /// RNG seed for stochastic sampling.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

impl RunArgs {
    fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_penalty: self.repeat_penalty,
            penalty_last_n: self.penalty_last_n,
            seed: self.seed,
        }
    }
}

fn parse_dtype_arg(s: &str) -> Result<GgmlType, String> {
    parse_dtype(s)
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
    if let Ok(dt_name) = std::env::var("SEEKER_QUANT_TEST") {
        quant_roundtrip_test(&mut engine, model.weights(), &dt_name)?;
        return Ok(());
    }
    if let Ok(tname) = std::env::var("SEEKER_KQUANT_MATMUL_TEST") {
        kquant_matmul_test(&mut engine, model.weights(), &tname)?;
        return Ok(());
    }

    let max_seq_len = args
        .max_tokens
        .saturating_add(tokens.len() as u32)
        .max(tokens.len() as u32);
    let cache_config = KvCacheConfig {
        k_dtype: args.cache_type_k,
        v_dtype: args.cache_type_v,
        max_seq_len,
    };
    let dims = model.cache_dims();
    let mut cache = engine.allocate_kv_cache(
        dims.n_layer,
        dims.head_dim,
        dims.n_head_kv,
        cache_config,
    )?;
    tracing::info!(
        n_layer = dims.n_layer,
        max_seq_len,
        k_dtype = ?args.cache_type_k,
        v_dtype = ?args.cache_type_v,
        bytes = cache.region.cursor,
        "kv cache allocated",
    );

    let mut sampler = Sampler::new(args.sampler_config());
    let mut step_tokens: Vec<u32> = tokens.clone();
    let mut generated: Vec<u32> = Vec::with_capacity(args.max_tokens as usize);

    // Prefill = first forward pass (N = prompt length). Decode = every
    // subsequent N=1 step. Reporting them separately matches llama.cpp's
    // `prompt eval` / `eval` timings and is what we want when comparing.
    let mut prefill_secs: f64 = 0.0;
    let prefill_tokens = step_tokens.len();
    let mut decode_secs: f64 = 0.0;

    for step in 0..args.max_tokens {
        let position_offset = cache.position;
        let t0 = std::time::Instant::now();
        let next_id = engine.forward_sampled(model.weights(), &mut sampler, |ctx| {
            model.record_forward(ctx, &mut cache, &step_tokens, position_offset)
        })?;
        let elapsed = t0.elapsed().as_secs_f64();
        if step == 0 {
            prefill_secs = elapsed;
        } else {
            decode_secs += elapsed;
        }
        generated.push(next_id);
        step_tokens = vec![next_id];
    }

    let generated_text = model
        .tokenizer()
        .tokenizer
        .decode(&generated, false)
        .unwrap_or_else(|_| {
            generated
                .iter()
                .map(|id| format!("<id={id}>"))
                .collect::<Vec<_>>()
                .join("")
        });

    println!("prompt:    {}", args.prompt);
    println!("tokens:    {tokens:?}");
    println!("generated: {generated_text}");
    println!("ids:       {generated:?}");
    let prefill_tps = (prefill_tokens as f64) / prefill_secs.max(1e-9);
    let decode_steps = args.max_tokens.saturating_sub(1) as f64;
    let decode_tps = if decode_steps > 0.0 {
        decode_steps / decode_secs.max(1e-9)
    } else {
        0.0
    };
    println!(
        "timing:    prefill {prefill_tokens} tok in {:.3}s ({prefill_tps:.1} tok/s), decode {} tok in {:.3}s ({decode_tps:.1} tok/s)",
        prefill_secs, decode_steps as u32, decode_secs,
    );
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

fn quant_roundtrip_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
    dt_name: &str,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::kv_cache::parse_dtype;
    use crate::inference::ops::cast::record_cast;
    let cache_dtype = parse_dtype(dt_name)?;
    // 64 hidden, 3 heads, 4 tokens = 768 elements
    let hd: u64 = 64;
    let nhkv: u64 = 3;
    let l: u64 = 4;
    let nel = (hd * nhkv * l) as usize;
    let logits = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        let f32_src = ctx.alloc_tensor([hd, nhkv, l, 1], GgmlType::F32)?;
        let cache_buf = ctx.alloc_tensor([hd, nhkv, l, 1], cache_dtype)?;
        let f32_dst = ctx.alloc_tensor([hd, nhkv, l, 1], GgmlType::F32)?;
        // fill src: value = (i % 100) / 100.0 (range [0, 1))
        let src_data: Vec<f32> = (0..nel).map(|i| (i as f32 % 100.0) / 100.0).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_data.as_ptr(),
                ctx.scratch.host_ptr.unwrap().add(f32_src.byte_offset as usize) as *mut f32,
                nel,
            );
        }
        record_cast(ctx, f32_src, cache_buf)?;
        crate::inference::command::record_global_barrier(ctx.device, ctx.cmd);
        record_cast(ctx, cache_buf, f32_dst)?;
        Ok(f32_dst.range())
    })?;
    let expected: Vec<f32> = (0..nel).map(|i| (i as f32 % 100.0) / 100.0).collect();
    let mut max_err = 0.0f32;
    for (a, b) in logits.iter().zip(expected.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    println!("quant roundtrip via {dt_name}:");
    println!("  src first 4: {:?}", &expected[..4]);
    println!("  dst first 4: {:?}", &logits[..4]);
    println!("  src last 4:  {:?}", &expected[nel-4..]);
    println!("  dst last 4:  {:?}", &logits[nel-4..]);
    println!("  max abs err: {max_err}");
    Ok(())
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0f32 } else { 1.0f32 };
    let exp = (h >> 10) & 0x1F;
    let mant = (h & 0x3FF) as f32;
    let val = if exp == 0 {
        // subnormal (or zero): mant × 2^-24
        mant * (2.0f32).powi(-24)
    } else if exp == 0x1F {
        if mant == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        // normal: (1 + mant/1024) × 2^(exp-15)
        (1.0 + mant / 1024.0) * (2.0f32).powi(exp as i32 - 15)
    };
    sign * val
}

/// CPU reference matmul-vec for a K-quant weight, compared against the GPU
/// `mul_mat_vec_q{4,6}_k` kernel. Picks `blk.0.<name>.weight`, builds a
/// deterministic input vector, and reports max abs / rel error.
fn kquant_matmul_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
    tensor_name: &str,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::ops::matmul;

    let full_name = format!("blk.0.{tensor_name}.weight");
    let a = *weights
        .views
        .get(&full_name)
        .ok_or_else(|| format!("tensor {full_name} not found"))?;
    let k = a.dims[0]; // contracting dim
    let m = a.dims[1]; // output rows
    println!("kquant matmul test: {full_name} dtype={:?} K={k} M={m}", a.dtype);

    // Raw quant bytes live in the (host-visible) weights region.
    let host_ptr = weights
        .region
        .host_ptr
        .ok_or("weights region not host-visible")?;
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(
            host_ptr.add(a.byte_offset as usize),
            a.byte_size as usize,
        )
    };

    // Deterministic input vector b[i] = sin(i * 0.01) — varied, bounded.
    let b_host: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();

    // GPU matmul.
    let gpu = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        let b = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
        let d = ctx.alloc_tensor([m, 1, 1, 1], GgmlType::F32)?;
        unsafe {
            let bp = ctx.scratch.host_ptr.unwrap().add(b.byte_offset as usize) as *mut f32;
            std::ptr::copy_nonoverlapping(b_host.as_ptr(), bp, b_host.len());
        }
        matmul::record(ctx, a, b, d)?;
        Ok(d.range())
    })?;

    // CPU reference: dequant each row of A, dot with b.
    let cpu: Vec<f32> = (0..m)
        .map(|row| {
            let mut row_vals = vec![0f32; k as usize];
            dequant_kquant_row(a.dtype, raw, row, k as usize, &mut row_vals);
            row_vals.iter().zip(b_host.iter()).map(|(w, x)| w * x).sum()
        })
        .collect();

    let mut max_abs = 0f32;
    let mut worst_row = 0;
    for (row, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let abs = (g - c).abs();
        if abs > max_abs {
            max_abs = abs;
            worst_row = row;
        }
    }
    let bad = (0..m as usize).filter(|&r| (gpu[r] - cpu[r]).abs() > 0.01).count();
    println!("  gpu[0..4]    = {:?}", &gpu[..4.min(gpu.len())]);
    println!("  cpu[0..4]    = {:?}", &cpu[..4.min(cpu.len())]);
    println!("  max abs err  = {max_abs} (worst row {worst_row})");
    println!("  bad rows     = {bad}/{m} (threshold 0.01)");
    Ok(())
}

/// Dequantize one row (`k` elements) of a K-quant tensor stored in `raw`,
/// into `out`. Mirrors llama.cpp's `dequantize_row_q{4,6}_K`.
fn dequant_kquant_row(dtype: GgmlType, raw: &[u8], row: u64, k: usize, out: &mut [f32]) {
    match dtype {
        GgmlType::Q4_K => {
            const BLK: usize = 144; // bytes per Q4_K superblock (256 elements)
            let blocks_per_row = k / 256;
            for blk in 0..blocks_per_row {
                let base = (row as usize * blocks_per_row + blk) * BLK;
                let d = f16_bits_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([raw[base + 2], raw[base + 3]]));
                let scales = &raw[base + 4..base + 16];
                let qs = &raw[base + 16..base + 144];
                let get_sc_min = |j: usize| -> (u8, u8) {
                    if j < 4 {
                        (scales[j] & 63, scales[j + 4] & 63)
                    } else {
                        (
                            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
                            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
                        )
                    }
                };
                let mut y = blk * 256;
                let mut is = 0usize;
                for j in (0..256).step_by(64) {
                    let (sc1, m1) = get_sc_min(is);
                    let (sc2, m2) = get_sc_min(is + 1);
                    let d1 = d * sc1 as f32;
                    let mm1 = dmin * m1 as f32;
                    let d2 = d * sc2 as f32;
                    let mm2 = dmin * m2 as f32;
                    let q = &qs[j / 2..];
                    for l in 0..32 {
                        out[y] = d1 * (q[l] & 0xF) as f32 - mm1;
                        y += 1;
                    }
                    for l in 0..32 {
                        out[y] = d2 * (q[l] >> 4) as f32 - mm2;
                        y += 1;
                    }
                    is += 2;
                }
            }
        }
        GgmlType::Q6_K => {
            const BLK: usize = 210;
            let blocks_per_row = k / 256;
            for blk in 0..blocks_per_row {
                let base = (row as usize * blocks_per_row + blk) * BLK;
                let ql = &raw[base..base + 128];
                let qh = &raw[base + 128..base + 192];
                let sc = &raw[base + 192..base + 208]; // i8
                let d = f16_bits_to_f32(u16::from_le_bytes([raw[base + 208], raw[base + 209]]));
                for n in 0..2 {
                    let ql_b = n * 64;
                    let qh_b = n * 32;
                    let sc_b = n * 8;
                    let y_b = blk * 256 + n * 128;
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = (((ql[ql_b + l] & 0xF) | (((qh[qh_b + l] >> 0) & 3) << 4)) as i8) as i32 - 32;
                        let q2 = (((ql[ql_b + l + 32] & 0xF) | (((qh[qh_b + l] >> 2) & 3) << 4)) as i8) as i32 - 32;
                        let q3 = (((ql[ql_b + l] >> 4) | (((qh[qh_b + l] >> 4) & 3) << 4)) as i8) as i32 - 32;
                        let q4 = (((ql[ql_b + l + 32] >> 4) | (((qh[qh_b + l] >> 6) & 3) << 4)) as i8) as i32 - 32;
                        out[y_b + l] = d * (sc[sc_b + is] as i8 as f32) * q1 as f32;
                        out[y_b + l + 32] = d * (sc[sc_b + is + 2] as i8 as f32) * q2 as f32;
                        out[y_b + l + 64] = d * (sc[sc_b + is + 4] as i8 as f32) * q3 as f32;
                        out[y_b + l + 96] = d * (sc[sc_b + is + 6] as i8 as f32) * q4 as f32;
                    }
                }
            }
        }
        GgmlType::Q8_0 => {
            const BLK: usize = 34; // 2 (d) + 32 (qs i8)
            let blocks_per_row = k / 32;
            for blk in 0..blocks_per_row {
                let base = (row as usize * blocks_per_row + blk) * BLK;
                let d = f16_bits_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
                for i in 0..32 {
                    let q = raw[base + 2 + i] as i8 as f32;
                    out[blk * 32 + i] = d * q;
                }
            }
        }
        GgmlType::BF16 => {
            let base = row as usize * k * 2;
            for i in 0..k {
                let bits = u16::from_le_bytes([raw[base + i * 2], raw[base + i * 2 + 1]]);
                // bf16 → f32: high 16 bits of the f32.
                out[i] = f32::from_bits((bits as u32) << 16);
            }
        }
        GgmlType::F16 => {
            let base = row as usize * k * 2;
            for i in 0..k {
                let bits = u16::from_le_bytes([raw[base + i * 2], raw[base + i * 2 + 1]]);
                out[i] = f16_bits_to_f32(bits);
            }
        }
        other => panic!("dequant_kquant_row: unsupported dtype {other:?}"),
    }
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
