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
    /// Sampling temperature. Defaults to 0 (greedy) — `run` is a single-shot
    /// inspection tool, so deterministic argmax is the useful default. Pass
    /// `--temp 0.8` for llama.cpp-style stochastic sampling.
    #[arg(long = "temp", alias = "temperature", default_value_t = 0.0)]
    temperature: f32,

    /// Top-K filter (0 = disabled, full vocab). (llama.cpp default: 40)
    #[arg(long = "top-k", default_value_t = 40)]
    top_k: u32,

    /// Top-P (nucleus) filter (1.0 = disabled).
    #[arg(long = "top-p", default_value_t = 0.95)]
    top_p: f32,

    /// Min-P filter (0.0 = disabled). (llama.cpp default: 0.05)
    #[arg(long = "min-p", default_value_t = 0.05)]
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

    // ─── Batch limits (llama.cpp parity) ────────────────────────────────
    /// Logical batch size (max tokens per submit). Validation-only in this
    /// single-sequence engine; `--ubatch-size` is the memory-relevant knob.
    #[arg(short = 'b', long = "batch-size", default_value_t = 2048)]
    batch_size: u32,

    /// Physical micro-batch size: prefill is split into ≤ this many tokens
    /// per GPU pass so scratch memory stays bounded on long prompts.
    /// 0 = unbounded (single pass). (short: -ub)
    #[arg(long = "ubatch-size", default_value_t = 512)]
    ubatch_size: u32,

    /// Max MTP draft tokens per step for speculative decoding (llama.cpp's
    /// `--spec-draft-n-max`). 0 (default) disables it and uses the normal
    /// non-speculative decode loop. Only the qwen35moe model with NextN
    /// weights supports it; other models silently ignore the flag.
    #[arg(long = "spec-draft-n-max", default_value_t = 0)]
    spec_draft_n_max: u32,
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
            logit_bias: Vec::new(),
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

    let mut engine = Engine::new(args.ubatch_size, args.batch_size)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened");

    let weights = engine.upload_weights(&gguf)?;
    tracing::info!(
        tensors = weights.views.len(),
        bytes = weights.total_bytes,
        "weights uploaded to GPU",
    );

    let model = crate::models::open(&gguf, weights, bundle, args.spec_draft_n_max > 0)?;

    // Size the scratch (compute buffer) for this model + n_ubatch before any
    // forward (including the debug-dump paths below), replacing the placeholder
    // from Engine::new. With the within-chunk mask this is context-independent
    // for a homogeneous KV cache.
    let max_seq_len = args
        .max_tokens
        .saturating_add(tokens.len() as u32)
        .max(tokens.len() as u32);
    engine.allocate_scratch(model.scratch_bytes_estimate(
        args.ubatch_size,
        max_seq_len,
        args.cache_type_k,
        args.cache_type_v,
    ))?;

    // Op smoke-test harnesses — model bring-up scaffolding, gated behind
    // the `gpu_debug` feature. In production builds this whole dispatch
    // block (and the test functions below) compiles out, so `seeker run`
    // does no `SEEKER_*_TEST` env probing at startup and the binary
    // carries none of the test code.
    #[cfg(feature = "gpu_debug")]
    {
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
        if std::env::var("SEEKER_INPLACE_BINARY_TEST").is_ok() {
            inplace_binary_smoke_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_MM_CM_SMOKE").is_ok() {
            mm_cm_smoke_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_MOE_TEST").is_ok() {
            moe_smoke_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if let Ok(stage) = std::env::var("SEEKER_MOE_DIAG") {
            moe_diag_test(&mut engine, model.weights(), &stage)?;
            return Ok(());
        }
        if std::env::var("SEEKER_SSM_CONV_TEST").is_ok() {
            ssm_conv_smoke_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_L2_NORM_TEST").is_ok() {
            l2_norm_smoke_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_GDN_TEST").is_ok() {
            gdn_smoke_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_FA_BATCH_TEST").is_ok() {
            fa_batch_smoke_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_GDN_BATCH_TEST").is_ok() {
            gdn_batch_smoke_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_BATCH_DECODE_TEST").is_ok() {
            batch_decode_smoke_test(&mut engine, model.as_ref())?;
            return Ok(());
        }
        if std::env::var("SEEKER_WQ_DUMP").is_ok() {
            wq_dump_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_NORM_WEIGHTS_DUMP").is_ok() {
            norm_weights_dump(model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_EMBED_NORM_DUMP").is_ok() {
            embed_norm_dump(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_TAP_SANITY").is_ok() {
            tap_sanity_test(&mut engine, model.weights())?;
            return Ok(());
        }
        if std::env::var("SEEKER_RMS_TEST").is_ok() {
            rms_norm_qwen_smoke(&mut engine, model.weights())?;
            return Ok(());
        }
    }

    // Throughput benchmark — outside the `gpu_debug` gate so it runs in a
    // release build (no validation-layer overhead skewing the numbers).
    if std::env::var("SEEKER_BATCH_BENCH").is_ok() {
        batch_decode_bench(&mut engine, model.as_ref())?;
        return Ok(());
    }

    // `max_seq_len` was computed above (used to size the scratch region).
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
    // Hybrid models (qwen35moe etc.) also need persistent SSM/GDN
    // recurrent state. Allocate it on the cache; the model reads/writes
    // it across forwards.
    if let Some(ssm) = model.ssm_state_dims() {
        cache.allocate_ssm_state(
            &engine.device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
        tracing::info!(
            n_ssm_layers = ssm.n_ssm_layers,
            conv_state_floats = ssm.conv_state_floats,
            gdn_state_floats = ssm.gdn_state_floats,
            "ssm state allocated",
        );
        // Per-position SSM checkpoint buffers for spec decode — sized for
        // L = (clamped n_draft)+1 snapshots. Lets partial acceptance roll
        // the recurrent state back without a re-run.
        if args.spec_draft_n_max > 0 && model.supports_mtp_spec() {
            let max_snapshots = args.spec_draft_n_max.clamp(1, 8) + 1;
            cache.allocate_ssm_snapshots(&engine.device, &ssm, max_snapshots)?;
        }
    }
    tracing::info!(
        n_layer = dims.n_layer,
        max_seq_len,
        k_dtype = ?args.cache_type_k,
        v_dtype = ?args.cache_type_v,
        bytes = cache.region.cursor,
        "kv cache allocated",
    );

    let mut sampler = Sampler::new(args.sampler_config());
    let mut generated: Vec<u32> = Vec::with_capacity(args.max_tokens as usize);

    // Prefill = first forward pass (N = prompt length). Decode = every
    // subsequent N=1 step. Reporting them separately matches llama.cpp's
    // `prompt eval` / `eval` timings and is what we want when comparing.
    let mut prefill_secs: f64 = 0.0;
    let prefill_tokens = tokens.len();
    let mut decode_secs: f64 = 0.0;

    // Speculative decode is active only when --spec-draft-n-max > 0 AND the
    // model loaded an MTP head. Otherwise fall through to the unchanged
    // single-token loop (byte-for-byte identical to before).
    let spec = args.spec_draft_n_max > 0 && model.supports_mtp_spec();
    if spec {
        let n_max = args.spec_draft_n_max;
        // Prefill via the hidden-exposing forward; sample the first token on
        // the host (the spec path samples on host throughout).
        let t0 = std::time::Instant::now();
        let (logits, residual) =
            engine.forward_full_readback(&*model, &mut cache, &tokens, 0, /*full_logits=*/ false)?;
        let prompt_len = tokens.len();
        let hsz = residual.len() / prompt_len;
        let mut h_last = residual[(prompt_len - 1) * hsz..prompt_len * hsz].to_vec();
        // Seed the MTP draft head's KV from the prompt's main hidden states so
        // the first draft attends to real prior context (raises acceptance).
        // MTP-KV[i] (i in [0, prompt_len-1)) uses h_i + embed(t_{i+1}).
        if prompt_len >= 2 {
            let seed_hiddens = &residual[0..(prompt_len - 1) * hsz];
            let seed_tokens = &tokens[1..prompt_len];
            engine.run_mtp_seed(&*model, &mut cache, seed_hiddens, seed_tokens, 0)?;
        }
        prefill_secs = t0.elapsed().as_secs_f64();
        let first = sampler.sample_one(&logits);
        sampler.accept(first);
        generated.push(first);
        let mut last_token = first;

        let mut total_accepted: u64 = 0;
        let mut spec_steps: u64 = 0;
        let t_dec = std::time::Instant::now();
        while (generated.len() as u32) < args.max_tokens {
            let position = cache.position;
            let out = engine.decode_speculative(
                &*model, &mut cache, last_token, &h_last, position, &mut sampler, n_max,
            )?;
            total_accepted += out.accepted as u64;
            spec_steps += 1;
            for &tk in &out.emitted {
                if (generated.len() as u32) >= args.max_tokens {
                    break;
                }
                generated.push(tk);
            }
            last_token = out.last_token;
            h_last = out.h_last;
        }
        decode_secs = t_dec.elapsed().as_secs_f64();
        let mean_acc = if spec_steps > 0 {
            total_accepted as f64 / spec_steps as f64
        } else {
            0.0
        };
        eprintln!(
            "SPEC: {spec_steps} steps, mean accepted {mean_acc:.2}/{n_max} drafts ({} tokens)",
            generated.len(),
        );
    } else {
        let mut step_tokens: Vec<u32> = tokens.clone();
        for step in 0..args.max_tokens {
            let position_offset = cache.position;
            let t0 = std::time::Instant::now();
            let next_id = engine.forward_sampled(&*model, &mut cache, &step_tokens, position_offset, &mut sampler)?;
            let elapsed = t0.elapsed().as_secs_f64();
            if step == 0 {
                prefill_secs = elapsed;
            } else {
                decode_secs += elapsed;
            }
            generated.push(next_id);
            step_tokens = vec![next_id];
        }
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
    // Decode tokens = everything after the first (prefill-sampled) token.
    // For spec decode this counts the multiple tokens emitted per step.
    let decode_steps = generated.len().saturating_sub(1) as f64;
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

/// End-to-end batched-decode correctness test: decode B distinct prompts both
/// serially (each in its own KvCache) and as a batch (one BatchKvCache, one
/// `forward_batch_decode` per step), with greedy sampling, and assert the token
/// streams are **byte-identical** per sequence. This is M1's gate — proof that
/// batching N sequences into one forward changes nothing about each sequence's
/// output. Run with `SEEKER_BATCH_DECODE_TEST=1 seeker run --features gpu_debug`.
#[cfg(feature = "gpu_debug")]
fn batch_decode_smoke_test(
    engine: &mut Engine,
    model: &dyn crate::models::Model,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::kv_cache::{BatchKvCache, KvCacheConfig};
    use crate::inference::sample::{Sampler, SamplerConfig};

    let prompts: Vec<&str> = if std::env::var("SEEKER_BATCH_B1").is_ok() {
        vec!["The capital of France is"]
    } else if std::env::var("SEEKER_BATCH_SAME").is_ok() {
        vec![
            "The capital of France is",
            "The capital of France is",
            "The capital of France is",
        ]
    } else {
        vec![
            "The capital of France is",
            "Once upon a time, in a land",
            "Two plus two equals",
        ]
    };
    let b = prompts.len();
    let n_gen = 16usize; // tokens to generate per sequence (incl. the first)

    let greedy = SamplerConfig {
        temperature: 0.0,
        ..SamplerConfig::default()
    };

    // Tokenize each prompt.
    let bundle = model.tokenizer();
    let add_special = bundle.add_bos_default || bundle.add_eos_default;
    let prompts_tokens: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            bundle
                .tokenizer
                .encode(*p, add_special)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| format!("tokenize failed: {e}"))
        })
        .collect::<Result<_, _>>()?;

    let max_prompt = prompts_tokens.iter().map(|t| t.len()).max().unwrap();
    let max_seq_len = (max_prompt + n_gen + 4) as u32;
    let config = KvCacheConfig {
        k_dtype: GgmlType::F16,
        v_dtype: GgmlType::F16,
        max_seq_len,
    };
    let dims = model.cache_dims();

    println!(
        "Batch-decode test: B={b}, gen={n_gen}, prompts={:?}",
        prompts_tokens.iter().map(|t| t.len()).collect::<Vec<_>>()
    );

    // ---- Serial reference: each prompt in its own cache ----
    let mut serial: Vec<Vec<u32>> = Vec::with_capacity(b);
    for tokens in &prompts_tokens {
        let mut cache =
            engine.allocate_kv_cache(dims.n_layer, dims.head_dim, dims.n_head_kv, config)?;
        if let Some(ssm) = model.ssm_state_dims() {
            cache.allocate_ssm_state(
                &engine.device,
                ssm.n_ssm_layers,
                ssm.conv_state_floats,
                ssm.gdn_state_floats,
            )?;
        }
        let mut sampler = Sampler::new(greedy.clone());
        // Prefill (samples the first token), then greedy decode the rest.
        let first = engine.forward_sampled(model, &mut cache, tokens, 0, &mut sampler)?;
        let mut stream = vec![first];
        let mut last = first;
        for _ in 1..n_gen {
            let pos = cache.position;
            last = engine.forward_sampled(model, &mut cache, &[last], pos, &mut sampler)?;
            stream.push(last);
        }
        serial.push(stream);
    }
    eprintln!("[batch test] serial reference done ({} streams)", serial.len());

    // ---- Batched: all prompts in one BatchKvCache ----
    // `SEEKER_BATCH_PERM` parks sequence `s` in a *non-contiguous* slab
    // (`2*s+1`, leaving even slabs empty) to exercise the gathered, slot-indexed
    // batched decode (continuous-batching with prefix-reuse parking). The
    // per-sequence result must be independent of which slab it lives in.
    let perm: Vec<u32> = if std::env::var("SEEKER_BATCH_PERM").is_ok() {
        (0..b as u32).map(|s| 2 * s + 1).collect()
    } else {
        (0..b as u32).collect()
    };
    let n_slots = perm.iter().copied().max().unwrap_or(0) + 1;
    println!("Batched slab assignment (slot per seq): {perm:?}, n_slots={n_slots}");
    let mut batch =
        BatchKvCache::new(&engine.device, dims.n_layer, dims.head_dim, dims.n_head_kv, n_slots, config)?;
    if let Some(ssm) = model.ssm_state_dims() {
        batch.allocate_ssm_state(
            &engine.device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
    }
    let mut samplers: Vec<Sampler> = (0..b).map(|_| Sampler::new(greedy.clone())).collect();
    let mut last: Vec<u32> = vec![0; b];
    let mut batched: Vec<Vec<u32>> = Vec::with_capacity(b);
    // Prefill each sequence into its (possibly non-contiguous) slab.
    for (s, tokens) in prompts_tokens.iter().enumerate() {
        let mut sc = batch.slot_kvcache(perm[s]);
        let first = engine.forward_sampled(model, &mut sc, tokens, 0, &mut samplers[s])?;
        batch.positions[perm[s] as usize] = sc.position;
        last[s] = first;
        batched.push(vec![first]);
        eprintln!("[batch test] prefill seq {s} done (slab {}, first token {first})", perm[s]);
    }
    // Batched decode steps.
    for step in 1..n_gen {
        let positions: Vec<u32> = (0..b).map(|s| batch.positions[perm[s] as usize]).collect();
        let mut srefs: Vec<&mut Sampler> = samplers.iter_mut().collect();
        let toks = engine
            .forward_batch_decode(model, &mut batch, &last, &positions, &perm, &mut srefs)?;
        let matches: Vec<bool> = (0..b)
            .map(|s| serial[s].get(step).map(|&t| t == toks[s]).unwrap_or(false))
            .collect();
        eprintln!("[batch step {step}] toks={toks:?} match={matches:?}");
        for s in 0..b {
            batched[s].push(toks[s]);
            last[s] = toks[s];
        }
    }

    // ---- Compare ----
    let mut all_ok = true;
    for s in 0..b {
        let ok = serial[s] == batched[s];
        all_ok &= ok;
        let first_diff = serial[s]
            .iter()
            .zip(batched[s].iter())
            .position(|(a, b)| a != b);
        println!(
            "  seq {s}: {} (serial={:?} batched={:?}{})",
            if ok { "MATCH" } else { "DIVERGE" },
            &serial[s][..serial[s].len().min(8)],
            &batched[s][..batched[s].len().min(8)],
            first_diff
                .map(|i| format!(", first diff at token {i}"))
                .unwrap_or_default(),
        );
    }
    println!(
        "  RESULT: {}",
        if all_ok {
            "PASS (batched decode is byte-identical to serial)"
        } else {
            "FAIL (batched decode diverged from serial)"
        }
    );
    Ok(())
}

/// Aggregate decode-throughput benchmark for batched decode. Decode is
/// bandwidth-bound (each token re-reads the whole weight set), so batching B
/// sequences into one forward amortizes that read across B tokens — aggregate
/// tok/s should scale up with B until the path turns compute-bound. Prefills B
/// contiguous slabs with a fixed prompt, times `n_decode` batched steps per
/// batch size, and prints aggregate / per-sequence tok/s. The B=1 row is the
/// single-stream baseline. Run with `SEEKER_BATCH_BENCH=1 seeker run`. Override
/// with `SEEKER_BENCH_BATCHES=1,2,4,8`, `SEEKER_BENCH_PROMPT=128`,
/// `SEEKER_BENCH_DECODE=128`.
fn batch_decode_bench(
    engine: &mut Engine,
    model: &dyn crate::models::Model,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::kv_cache::{BatchKvCache, KvCacheConfig};
    use crate::inference::sample::{Sampler, SamplerConfig};
    use std::time::Instant;

    let env_usize = |k: &str, d: usize| {
        std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
    };
    let prompt_len = env_usize("SEEKER_BENCH_PROMPT", 128);
    let n_decode = env_usize("SEEKER_BENCH_DECODE", 128);
    let warmup = 4usize;
    let batches: Vec<usize> = std::env::var("SEEKER_BENCH_BATCHES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);

    let greedy = SamplerConfig {
        temperature: 0.0,
        ..SamplerConfig::default()
    };
    // A real prompt tokenized then padded (by repetition) to `prompt_len`, so
    // every sequence carries the same realistic-length cache.
    let bundle = model.tokenizer();
    let seed = bundle
        .tokenizer
        .encode("The history of computing began long before the invention of", true)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let prompt: Vec<u32> = (0..prompt_len).map(|i| seed[i % seed.len()]).collect();

    let dims = model.cache_dims();
    let max_seq = (prompt_len + n_decode + warmup + 8) as u32;
    let config = KvCacheConfig {
        k_dtype: GgmlType::F16,
        v_dtype: GgmlType::F16,
        max_seq_len: max_seq,
    };

    println!(
        "Batch-decode throughput: prompt_len={prompt_len}, decode_steps={n_decode}, model dims n_layer={}",
        dims.n_layer
    );
    println!("  B   aggregate tok/s   per-seq tok/s   ms/step   speedup");
    let mut baseline = 0f64;
    for &b in &batches {
        let mut batch = BatchKvCache::new(
            &engine.device,
            dims.n_layer,
            dims.head_dim,
            dims.n_head_kv,
            b as u32,
            config,
        )?;
        if let Some(ssm) = model.ssm_state_dims() {
            batch.allocate_ssm_state(
                &engine.device,
                ssm.n_ssm_layers,
                ssm.conv_state_floats,
                ssm.gdn_state_floats,
            )?;
        }
        let mut samplers: Vec<Sampler> = (0..b).map(|_| Sampler::new(greedy.clone())).collect();
        let mut last: Vec<u32> = vec![0; b];
        for s in 0..b {
            let mut sc = batch.slot_kvcache(s as u32);
            last[s] = engine.forward_sampled(model, &mut sc, &prompt, 0, &mut samplers[s])?;
            batch.positions[s] = sc.position;
        }
        let slot_ids: Vec<u32> = (0..b as u32).collect();
        let step = |engine: &mut Engine, batch: &mut BatchKvCache, last: &mut Vec<u32>, samplers: &mut [Sampler]| -> Result<(), Box<dyn Error>> {
            let positions: Vec<u32> = (0..b).map(|s| batch.positions[s]).collect();
            let mut srefs: Vec<&mut Sampler> = samplers.iter_mut().collect();
            *last = engine
                .forward_batch_decode(model, batch, last, &positions, &slot_ids, &mut srefs)?;
            Ok(())
        };
        for _ in 0..warmup {
            step(engine, &mut batch, &mut last, &mut samplers)?;
        }
        let t0 = Instant::now();
        for _ in 0..n_decode {
            step(engine, &mut batch, &mut last, &mut samplers)?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let agg = (b as f64 * n_decode as f64) / elapsed;
        let per_seq = n_decode as f64 / elapsed;
        let ms_step = elapsed * 1000.0 / n_decode as f64;
        if b == batches[0] {
            baseline = agg;
        }
        println!(
            "  {b:<3} {agg:>14.1}   {per_seq:>13.1}   {ms_step:>7.2}   {:>5.2}x",
            agg / baseline.max(1e-9)
        );
    }
    Ok(())
}

/// Batched-decode flash-attention smoke test: B sequences, one query token
/// each, each attending to its own KV slab over its own `kv_len`. Runs the
/// batched dispatch (`flash_attn::record_batched`) on synthetic F32 Q/K/V and
/// compares against a CPU reference (per-(seq, head) softmax attention). This
/// proves the per-sequence `kv_len` keystone works for `n_seqs > 1` — the
/// linchpin of continuous batching — independent of the slab-KV / forward
/// plumbing. Run with `SEEKER_FA_BATCH_TEST=1 seeker run --features gpu_debug`.
#[cfg(feature = "gpu_debug")]
fn fa_batch_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::buffer::BufferRange;
    use crate::inference::ops::flash_attn;

    let head_dim: usize = std::env::var("SEEKER_FA_HEAD_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64); // multiple of 32 (HSV_PER_THREAD = HSV/32)
    let n_head: usize = std::env::var("SEEKER_FA_N_HEAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n_head_kv: usize = std::env::var("SEEKER_FA_N_HEAD_KV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2); // gqa_ratio = n_head / n_head_kv
    let gqa = (n_head / n_head_kv) as u32;
    let hidden = head_dim * n_head;
    // `SEEKER_FA_KV_LENS=200,137,256` overrides the per-sequence cache lengths
    // (default short). Long lengths (> a few Bc=32 blocks) let the split-K
    // heuristic — or a pinned `SEEKER_FA_SPLIT_KNUM` — fire, so this test also
    // validates the batched split-K dispatch + combine against the CPU ref.
    let kv_lens: Vec<u32> = std::env::var("SEEKER_FA_KV_LENS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<u32>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![5, 8, 3]);
    let b = kv_lens.len();
    let max_kv = *kv_lens.iter().max().unwrap() as usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    println!(
        "FA batch smoke: B={b}, head_dim={head_dim}, n_head={n_head}, n_head_kv={n_head_kv}, kv_lens={kv_lens:?}"
    );

    // Deterministic synthetic inputs (host-side, also drive the CPU reference).
    let q_val = |s: usize, h: usize, d: usize| ((s * 7 + h * 3 + d) as f32 * 0.013).sin();
    let k_val =
        |s: usize, kh: usize, j: usize, d: usize| ((s * 5 + kh * 4 + j * 2 + d) as f32 * 0.021).cos();
    let v_val = |s: usize, kh: usize, j: usize, d: usize| {
        ((s * 9 + kh * 2 + j * 3 + d) as f32 * 0.017).sin() * 0.5
    };

    // CPU reference: out[s][h][d].
    let mut reference = vec![0f32; b * hidden];
    for s in 0..b {
        for h in 0..n_head {
            let kh = h / (gqa as usize);
            let klen = kv_lens[s] as usize;
            let mut scores = vec![0f32; klen];
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..klen {
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += q_val(s, h, d) * k_val(s, kh, j, d);
                }
                scores[j] = dot * scale;
                maxs = maxs.max(scores[j]);
            }
            let mut denom = 0f32;
            for sj in scores.iter_mut() {
                *sj = (*sj - maxs).exp();
                denom += *sj;
            }
            for d in 0..head_dim {
                let mut acc = 0f32;
                for j in 0..klen {
                    acc += scores[j] * v_val(s, kh, j, d);
                }
                reference[s * hidden + h * head_dim + d] = acc / denom;
            }
        }
    }

    // GPU batched dispatch on synthetic F32 Q/K/V laid out as the shader expects:
    //   Q [head_dim, 1, n_head, B]   (contiguous: [B][n_head][head_dim])
    //   K/V [head_dim, max_kv, n_head_kv, B]  ([B][n_head_kv][max_kv][head_dim])
    let gpu_out = engine.forward(weights, |ctx| -> Result<BufferRange, Box<dyn Error>> {
        let host = ctx.scratch.host_ptr.ok_or("scratch not host-visible")?;
        let q = ctx.alloc_tensor([head_dim as u64, 1, n_head as u64, b as u64], GgmlType::F32)?;
        let k = ctx.alloc_tensor(
            [head_dim as u64, max_kv as u64, n_head_kv as u64, b as u64],
            GgmlType::F32,
        )?;
        let v = ctx.alloc_tensor(
            [head_dim as u64, max_kv as u64, n_head_kv as u64, b as u64],
            GgmlType::F32,
        )?;
        unsafe {
            let qp = host.add(q.byte_offset as usize) as *mut f32;
            for s in 0..b {
                for h in 0..n_head {
                    for d in 0..head_dim {
                        *qp.add(s * hidden + h * head_dim + d) = q_val(s, h, d);
                    }
                }
            }
            let kp = host.add(k.byte_offset as usize) as *mut f32;
            let vp = host.add(v.byte_offset as usize) as *mut f32;
            let slab = n_head_kv * max_kv * head_dim;
            for s in 0..b {
                for kh in 0..n_head_kv {
                    for j in 0..max_kv {
                        for d in 0..head_dim {
                            let idx = s * slab + kh * (max_kv * head_dim) + j * head_dim + d;
                            *kp.add(idx) = k_val(s, kh, j, d);
                            *vp.add(idx) = v_val(s, kh, j, d);
                        }
                    }
                }
            }
        }
        let out = ctx.alloc_tensor([hidden as u64, b as u64, 1, 1], GgmlType::F32)?;
        let params = flash_attn::FlashAttnParams {
            head_dim_k: head_dim as u32,
            head_dim_v: head_dim as u32,
            gqa_ratio: gqa,
            scale,
        };
        let fa_dyn = crate::inference::decode_dyn::alloc_array(ctx, b as u32)?;
        flash_attn::record_batched(ctx, q, k, v, out, params, &kv_lens, fa_dyn, None)?;
        Ok(out.range())
    })?;

    let mut max_err = 0f32;
    let mut worst = (0usize, 0usize, 0usize);
    for s in 0..b {
        for h in 0..n_head {
            for d in 0..head_dim {
                let idx = s * hidden + h * head_dim + d;
                let e = (gpu_out[idx] - reference[idx]).abs();
                if e > max_err {
                    max_err = e;
                    worst = (s, h, d);
                }
            }
        }
    }
    let (ws, wh, wd) = worst;
    let widx = ws * hidden + wh * head_dim + wd;
    println!(
        "  max_err = {max_err:.3e} at (seq={ws}, head={wh}, d={wd}); gpu={:.5}, ref={:.5}",
        gpu_out[widx], reference[widx]
    );
    if max_err < 1e-4 {
        println!("  RESULT: PASS (batched attention matches per-sequence CPU reference)");
    } else {
        println!("  RESULT: FAIL — batched attention diverges from the CPU reference");
    }
    Ok(())
}

/// Regression test for in-place elementwise add/mul: `record_add(d, b, d)`
/// must apply `b` exactly once even though the kernel tiles the buffer in
/// 512-wide blocks. A previous version dispatched overlapping workgroups and
/// relied on redundant writes being idempotent — true for a distinct `dst`,
/// but in-place it double-applied the op nondeterministically (the root cause
/// of run-to-run nondeterminism in the residual stream). We size N to span
/// several blocks and deliberately straddle the 512 boundary.
#[cfg(feature = "gpu_debug")]
fn inplace_binary_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::buffer::BufferRange;
    use crate::inference::ops::elementwise;

    let n: u64 = 512 * 7 + 137; // straddles block boundaries; not a multiple of 512
    let a_val = |i: usize| (i as f32) * 0.5 - 3.0;
    let b_val = |i: usize| ((i % 13) as f32) - 6.0;

    let out = engine.forward(weights, |ctx| -> Result<BufferRange, Box<dyn Error>> {
        let d = ctx.alloc_tensor([n, 1, 1, 1], GgmlType::F32)?;
        let b = ctx.alloc_tensor([n, 1, 1, 1], GgmlType::F32)?;
        let host = ctx.scratch.host_ptr.ok_or("scratch not host-visible")?;
        unsafe {
            let dp = host.add(d.byte_offset as usize) as *mut f32;
            let bp = host.add(b.byte_offset as usize) as *mut f32;
            for i in 0..n as usize {
                std::ptr::write(dp.add(i), a_val(i));
                std::ptr::write(bp.add(i), b_val(i));
            }
        }
        // In-place: dst aliases src0.
        elementwise::record_add(ctx, d, b, d)?;
        Ok(d.range())
    })?;

    let mut wrong = 0usize;
    let mut max_err = 0f32;
    for i in 0..n as usize {
        let expected = a_val(i) + b_val(i);
        let err = (out[i] - expected).abs();
        if err > 1e-4 {
            wrong += 1;
            max_err = max_err.max(err);
        }
    }
    println!("in-place add: N={n}, {wrong} wrong elements, max_err={max_err}");
    if wrong == 0 {
        println!("  RESULT: PASS (each element written exactly once)");
    } else {
        println!("  RESULT: FAIL (in-place op applied a non-unit number of times)");
    }
    Ok(())
}

/// SSM conv1d smoke test: dispatches `record_ssm_conv` on a synthetic
/// input (qkv = consecutive integers, zero conv prefix) using blk.0's
/// real `ssm_conv1d` kernel; reads the output back and compares against
/// a CPU reference computed inline. This isolates whether the conv
/// dispatch and the transpose-via-cast preprocessing both work
/// correctly on real model weights.
///
/// On success this proves the conv step is bug-free; failures show
/// which (channel, token) position diverges and by how much, pointing
/// at either a stride bug (transpose cast) or a shader bug (conv).
#[cfg(feature = "gpu_debug")]
fn ssm_conv_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::ops::{cast, ssm};

    let kernel = weights
        .view("blk.0.ssm_conv1d.weight")
        .map_err(|_| "missing blk.0.ssm_conv1d.weight")?;
    let conv_channels = kernel.dims[1]; // 8192
    let conv_kernel = kernel.dims[0]; // 4
    let l: u64 = 5;
    let n_padded = (conv_kernel - 1) + l;

    println!(
        "SSM conv smoke: channels={conv_channels}, kernel={conv_kernel}, L={l}, n_padded={n_padded}"
    );

    // Read kernel weights to host once so we can build the CPU reference.
    let kernel_host: Vec<f32> = {
        let base = weights.debug_host_base(&kernel).ok_or(
            "host-side weight reads unsupported (weights are device-local)",
        )?;
        let total = (conv_kernel * conv_channels) as usize;
        let mut out = vec![0f32; total];
        unsafe {
            let src = base.add(kernel.byte_offset as usize) as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), total);
        }
        out
    };

    let gpu_out = engine.forward(
        weights,
        |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
            // Build qkv tensor (synthetic): channel-innermost like a matmul
            // output. Values: qkv[c, t] = (c % 16) * 0.1 + t * 0.01.
            let qkv = ctx.alloc_tensor([conv_channels, l, 1, 1], GgmlType::F32)?;
            let host_ptr = ctx.scratch.host_ptr.ok_or("scratch not host-visible")?;
            unsafe {
                let p = host_ptr.add(qkv.byte_offset as usize) as *mut f32;
                for t in 0..l as usize {
                    for c in 0..conv_channels as usize {
                        *p.add(t * conv_channels as usize + c) =
                            (c % 16) as f32 * 0.1 + t as f32 * 0.01;
                    }
                }
            }

            // Conv input with zero prefix; transpose qkv into the tail.
            let conv_input = ctx.alloc_tensor([n_padded, conv_channels, 1, 1], GgmlType::F32)?;
            unsafe {
                std::ptr::write_bytes(
                    host_ptr.add(conv_input.byte_offset as usize) as *mut u8,
                    0,
                    conv_input.byte_size as usize,
                );
            }
            let elem = 4u64;
            let qkv_as_token_inner = crate::inference::weights::TensorView {
                buffer: qkv.buffer,
                byte_offset: qkv.byte_offset,
                byte_size: qkv.byte_size,
                dims: [l, conv_channels, 1, 1],
                byte_stride: [
                    conv_channels * elem,
                    elem,
                    conv_channels * l * elem,
                    conv_channels * l * elem,
                ],
                element_stride: [conv_channels, 1, conv_channels * l, conv_channels * l],
                dtype: qkv.dtype,
            };
            let conv_input_tail = crate::inference::weights::TensorView {
                buffer: conv_input.buffer,
                byte_offset: conv_input.byte_offset + (conv_kernel - 1) * elem,
                byte_size: conv_input.byte_size - (conv_kernel - 1) * elem,
                dims: [l, conv_channels, 1, 1],
                byte_stride: [
                    elem,
                    n_padded * elem,
                    l * conv_channels * elem,
                    l * conv_channels * elem,
                ],
                element_stride: [1, n_padded, l * conv_channels, l * conv_channels],
                dtype: conv_input.dtype,
            };
            cast::record_cast(ctx, qkv_as_token_inner, conv_input_tail)?;

            let conv_out = ctx.alloc_tensor([conv_channels, l, 1, 1], GgmlType::F32)?;
            ssm::record_ssm_conv(
                ctx,
                conv_input,
                kernel,
                conv_out,
                conv_channels as u32,
                n_padded as u32,
                l as u32,
                1,
                conv_kernel as u32,
                /* fuse_silu = */ false,
            )?;
            Ok(conv_out.range())
        },
    )?;

    // CPU reference: for each (c, t), sum kernel[c][k] * input[c][t + k] for k=0..3,
    // with input prefix being zeros and input[c][kernel-1 + t'] = qkv[c, t'].
    let mut max_err = 0f32;
    let mut worst_c = 0usize;
    let mut worst_t = 0usize;
    for t in 0..l as usize {
        for c in 0..conv_channels as usize {
            let mut sum = 0f32;
            for k in 0..conv_kernel as usize {
                let pos = (conv_kernel as usize - 1 + t).wrapping_sub(conv_kernel as usize - 1 - k);
                // pos = t + k - (conv_kernel - 1) when k < conv_kernel - 1 (gives a prefix idx = 0)
                // Simpler: the conv at output position t reads input positions [t, t+1, ..., t+conv_kernel-1].
                // After our zero-prefix + qkv tail layout:
                //   input[c][p] = 0 if p < conv_kernel-1, else qkv[c, p - (conv_kernel-1)]
                let p = t + k;
                let input_val = if p < (conv_kernel as usize - 1) {
                    0.0
                } else {
                    let qkv_t = p - (conv_kernel as usize - 1);
                    (c % 16) as f32 * 0.1 + qkv_t as f32 * 0.01
                };
                // kernel layout: ne[0]=4 inner, ne[1]=8192. memory[c*4 + k] = kernel[k, c].
                let k_val = kernel_host[c * conv_kernel as usize + k];
                sum += k_val * input_val;
                let _ = pos; // unused
            }
            // GPU output: ne0=channels inner, ne1=tokens. memory[t * channels + c] = out[c, t].
            let gpu_val = gpu_out[t * conv_channels as usize + c];
            let err = (gpu_val - sum).abs();
            if err > max_err {
                max_err = err;
                worst_c = c;
                worst_t = t;
            }
        }
    }
    println!(
        "  max_err = {max_err:.6e} at (c={worst_c}, t={worst_t}); gpu[c,t] = {}, ref = computed",
        gpu_out[worst_t * conv_channels as usize + worst_c]
    );
    if max_err < 1e-3 {
        println!("  RESULT: PASS (conv1d math matches CPU reference)");
    } else {
        println!("  RESULT: FAIL — conv1d output diverges from CPU reference");
    }
    Ok(())
}

/// L2-norm + strided-slice smoke test. Builds a synthetic conv-output-
/// shaped tensor `[conv_channels=8192, L=5]` with KNOWN values, then
/// slices the first `key_dim=4096` channels and reshapes as
/// `[s_v=256, num_k=16, L, 1]` exactly as the SSM block does. Dispatches
/// `record_l2_norm` and compares against a CPU reference (per-(head,
/// token) L2 normalization). This isolates whether our strided slice
/// view + dispatch shape correctly hit every (head, token) row.
#[cfg(feature = "gpu_debug")]
fn l2_norm_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::ops::elementwise;

    let conv_channels: u64 = 8192;
    let l: u64 = 5;
    let s_v: u64 = 256;
    let num_k: u64 = 16;
    let _key_dim: u64 = s_v * num_k; // 4096

    println!("L2-norm smoke: conv_channels={conv_channels}, L={l}, s_v={s_v}, num_k={num_k}");

    // Build CPU reference: synthetic conv_out[c, t] = sin(c*0.01 + t*0.13).
    // Then take Q slice = conv_out[0..key_dim, t]. Per-head L2 norm.
    let mut conv_out_host = vec![0f32; (conv_channels * l) as usize];
    for t in 0..l as usize {
        for c in 0..conv_channels as usize {
            conv_out_host[t * conv_channels as usize + c] =
                (c as f32 * 0.01 + t as f32 * 0.13).sin();
        }
    }
    let mut ref_q_normed = vec![0f32; (s_v * num_k * l) as usize];
    let eps = 1e-6f32;
    for t in 0..l as usize {
        for h in 0..num_k as usize {
            // Source: conv_out[t * conv_channels + h * s_v + i0] for i0 in [0, s_v).
            let mut sum = 0f32;
            for i0 in 0..s_v as usize {
                let v = conv_out_host[t * conv_channels as usize + h * s_v as usize + i0];
                sum += v * v;
            }
            let scale = 1.0 / sum.sqrt().max(eps);
            for i0 in 0..s_v as usize {
                let v = conv_out_host[t * conv_channels as usize + h * s_v as usize + i0];
                // Output: q_normed[s_v=inner, num_k, L, 1] contiguous.
                ref_q_normed[t * (s_v * num_k) as usize + h * s_v as usize + i0] = v * scale;
            }
        }
    }

    let gpu_out = engine.forward(
        weights,
        |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
            let conv_out = ctx.alloc_tensor([conv_channels, l, 1, 1], GgmlType::F32)?;
            let host_ptr = ctx.scratch.host_ptr.ok_or("scratch not host-visible")?;
            unsafe {
                let p = host_ptr.add(conv_out.byte_offset as usize) as *mut f32;
                std::ptr::copy_nonoverlapping(
                    conv_out_host.as_ptr(),
                    p,
                    conv_out_host.len(),
                );
            }
            let elem = 4u64;
            // Q slice: same view the SSM block builds.
            let q_view = crate::inference::weights::TensorView {
                buffer: conv_out.buffer,
                byte_offset: conv_out.byte_offset,
                byte_size: conv_out.byte_size,
                dims: [s_v, num_k, l, 1],
                byte_stride: [
                    elem,
                    s_v * elem,
                    conv_channels * elem,
                    conv_channels * l * elem,
                ],
                element_stride: [1, s_v, conv_channels, conv_channels * l],
                dtype: conv_out.dtype,
            };
            let q_normed = ctx.alloc_tensor([s_v, num_k, l, 1], GgmlType::F32)?;
            elementwise::record_l2_norm(ctx, q_view, q_normed, eps)?;
            Ok(q_normed.range())
        },
    )?;

    let mut max_err = 0f32;
    let mut worst = (0usize, 0usize, 0usize);
    for t in 0..l as usize {
        for h in 0..num_k as usize {
            for i0 in 0..s_v as usize {
                let idx = t * (s_v * num_k) as usize + h * s_v as usize + i0;
                let err = (gpu_out[idx] - ref_q_normed[idx]).abs();
                if err > max_err {
                    max_err = err;
                    worst = (t, h, i0);
                }
            }
        }
    }
    let (wt, wh, wi) = worst;
    let idx = wt * (s_v * num_k) as usize + wh * s_v as usize + wi;
    println!(
        "  max_err = {max_err:.6e} at (t={wt}, h={wh}, i={wi}); gpu = {}, ref = {}",
        gpu_out[idx], ref_q_normed[idx]
    );
    if max_err < 1e-4 {
        println!("  RESULT: PASS (per-head L2 norm + strided slice both correct)");
    } else {
        println!("  RESULT: FAIL — L2 norm output diverges from CPU reference");
    }
    Ok(())
}

/// Run rms_norm on a known input vs CPU reference. If the GPU output
/// doesn't match the CPU computation of `x[i] / sqrt(mean(x^2)+eps) * w[i]`,
/// the rms_norm shader has a numeric bug.
#[cfg(feature = "gpu_debug")]
fn rms_norm_qwen_smoke(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    let n: u64 = 2048;
    let eps: f32 = 1e-6;
    // Build a known input (embedding of token 9419 = "Hello") + known weight
    // (blk.0.attn_norm.weight). Compute CPU reference.
    let embed_view = weights.view("token_embd.weight")?;
    let weight_view = weights.view("blk.0.attn_norm.weight")?;
    // Read embed row for token 9419 (Q8_0 — need to dequantize)
    // Simpler: use the GPU get_rows to materialize. Or use F32 weight directly.
    // Just use a synthetic input + read weight.
    // Get the actual "Hello" (token 9419) embedding via the GPU's get_rows,
    // then read it back for the CPU reference computation.
    let embed_host: Vec<f32> = engine.forward(
        weights,
        |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
            let token_buf = ctx.alloc_scratch(4)?;
            let host_ptr = ctx.scratch.host_ptr.ok_or("no host ptr")?;
            unsafe { *(host_ptr.add(token_buf.offset as usize) as *mut u32) = 9419; }
            let out = ctx.alloc_tensor([n, 1, 1, 1], GgmlType::F32)?;
            crate::inference::ops::elementwise::record_get_rows(ctx, embed_view, token_buf, 1, out)?;
            Ok(out.range())
        },
    )?;
    let input = embed_host.clone();
    println!("input head={:?}, max_abs={:.4}", &input[..5],
        input.iter().map(|v| v.abs()).fold(0.0, f32::max));
    let mut w_host = vec![0f32; n as usize];
    let weight_base = weights.debug_host_base(&weight_view).ok_or(
        "host-side weight reads unsupported (weights are device-local)",
    )?;
    unsafe {
        let p = weight_base.add(weight_view.byte_offset as usize) as *const f32;
        std::ptr::copy_nonoverlapping(p, w_host.as_mut_ptr(), n as usize);
    }
    // CPU reference
    let sum_sq: f32 = input.iter().map(|x| x*x).sum();
    let mean = sum_sq / n as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    let mut ref_out = vec![0f32; n as usize];
    for i in 0..n as usize {
        ref_out[i] = scale * input[i] * w_host[i];
    }
    let ref_sum: f32 = ref_out.iter().sum();
    let ref_max: f32 = ref_out.iter().map(|v| v.abs()).fold(0.0, f32::max);
    println!("CPU: sum_sq={sum_sq:.6}, mean={mean:.9}, scale={scale:.4}");
    println!("CPU ref: sum={ref_sum:.4}, max_abs={ref_max:.4}, head={:?}",
        &ref_out[..5]);
    // GPU — replicate the qwen forward's prologue exactly:
    // 1. token_buf + positions_buf + mask alloc
    // 2. record_get_rows to populate residual
    // 3. write_causal_mask (host write)
    // 4. then rms_norm
    let gpu_out = engine.forward(
        weights,
        |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
            // Mirror the qwen prologue
            let token_buf = ctx.alloc_scratch(4)?;
            let positions_buf = ctx.alloc_scratch(4 * 4)?;
            let host_ptr = ctx.scratch.host_ptr.ok_or("no host ptr")?;
            unsafe {
                *(host_ptr.add(token_buf.offset as usize) as *mut u32) = 9419;
                for axis in 0..4 {
                    *(host_ptr.add(positions_buf.offset as usize + axis*4) as *mut u32) = 0;
                }
            }
            let mask = ctx.alloc_tensor([1, 1, 1, 1], GgmlType::F32)?;
            unsafe {
                *(host_ptr.add(mask.byte_offset as usize) as *mut f32) = 0.0;
            }
            let residual = ctx.alloc_tensor([n, 1, 1, 1], GgmlType::F32)?;
            crate::inference::ops::elementwise::record_get_rows(ctx, embed_view, token_buf, 1, residual)?;
            let dst = ctx.alloc_tensor([n, 1, 1, 1], GgmlType::F32)?;
            crate::inference::ops::rms_norm::record(ctx, residual, weight_view, dst, eps)?;
            let _ = positions_buf;
            Ok(dst.range())
        },
    )?;
    let gpu_sum: f32 = gpu_out.iter().sum();
    let gpu_max: f32 = gpu_out.iter().map(|v| v.abs()).fold(0.0, f32::max);
    println!("GPU: sum={gpu_sum:.4}, max_abs={gpu_max:.4}, head={:?}", &gpu_out[..5]);
    let mut max_diff = 0f32;
    for i in 0..n as usize {
        let d = (gpu_out[i] - ref_out[i]).abs();
        max_diff = max_diff.max(d);
    }
    println!("Max diff: {max_diff:.6e}");
    let _ = embed_view;
    if max_diff < 1e-3 {
        println!("RESULT: PASS");
    } else {
        println!("RESULT: FAIL (>{max_diff:.4} diff)");
    }
    Ok(())
}

/// Sanity-test the diff-dump tap mechanism: write 1.0 to every element
/// of a scratch tensor, then tap it 4 times consecutively. All four
/// taps must report the same sum (= number of elements). If they
/// diverge, the tap path is broken.
#[cfg(feature = "gpu_debug")]
fn tap_sanity_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    unsafe { std::env::set_var("SEEKER_QWEN_DIFF_DUMP", "1"); }
    let _ = engine.forward(
        weights,
        |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
            let buf = ctx.alloc_tensor([2048, 1, 1, 1], GgmlType::F32)?;
            // Write 1.0 to every element via host pointer.
            let host_ptr = ctx.scratch.host_ptr.ok_or("no host ptr")?;
            unsafe {
                let p = host_ptr.add(buf.byte_offset as usize) as *mut f32;
                for i in 0..2048 {
                    *p.add(i) = 1.0;
                }
            }
            ctx.tap("ones-1", buf)?;
            ctx.tap("ones-2", buf)?;
            ctx.tap("ones-3", buf)?;
            ctx.tap("ones-4", buf)?;
            Ok(buf.range())
        },
    )?;
    Ok(())
}

/// Compute the embedding norm for tokens 12656 ("iel") and 198649
/// ("diserta") — the tokens that attention path collapses to — vs
/// average norm. If these have unusually large magnitudes, they may
/// be naturally favored argmax winners when residual is small/noisy.
#[cfg(feature = "gpu_debug")]
fn embed_norm_dump(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    let token_embd = weights.view("token_embd.weight")?;
    let n_embd = token_embd.dims[0] as usize;
    let n_vocab = token_embd.dims[1] as usize;
    let dtype = token_embd.dtype;
    println!("token_embd: n_embd={n_embd}, n_vocab={n_vocab}, dtype={dtype:?}");

    // Sample a bunch of token indices including 12656 and 198649.
    let sample_tokens: Vec<u32> =
        vec![0, 100, 1000, 9707, 11, 12656, 100000, 198649, 220, 248044, 248046];

    for &tok in &sample_tokens {
        if tok as usize >= n_vocab { continue; }
        let out = engine.forward(
            weights,
            |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
                let token_buf = ctx.alloc_scratch(4)?;
                let host_ptr = ctx.scratch.host_ptr.ok_or("no host ptr")?;
                unsafe { *(host_ptr.add(token_buf.offset as usize) as *mut u32) = tok; }
                let residual = ctx.alloc_tensor([n_embd as u64, 1, 1, 1], GgmlType::F32)?;
                crate::inference::ops::elementwise::record_get_rows(ctx, token_embd, token_buf, 1, residual)?;
                Ok(residual.range())
            },
        )?;
        let norm_sq: f32 = out.iter().map(|x| x*x).sum();
        let mean: f32 = out.iter().sum::<f32>() / n_embd as f32;
        let max_abs = out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        println!(
            "  token {tok}: norm={:.4}, mean={mean:+.4}, max_abs={max_abs:.4}",
            norm_sq.sqrt()
        );
    }
    Ok(())
}

/// Dump the magnitudes of the per-head Q/K norm weights for each
/// attention layer. If these are large (>2 typical), they amplify Q/K
/// values before flash_attn, potentially saturating softmax.
#[cfg(feature = "gpu_debug")]
fn norm_weights_dump(
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    for layer in [3u32, 7, 11, 15, 19, 23, 27, 31, 35, 39] {
        for kind in ["attn_q_norm", "attn_k_norm"] {
            let name = format!("blk.{layer}.{kind}.weight");
            let view = weights.view(&name)?;
            let n = view.dims[0] as usize;
            let base = weights.debug_host_base(&view).ok_or(
                "host-side weight reads unsupported (weights are device-local)",
            )?;
            let mut sum_sq = 0f32;
            let mut max_abs = 0f32;
            let mut mean = 0f32;
            unsafe {
                let p = base.add(view.byte_offset as usize) as *const f32;
                for i in 0..n {
                    let v = *p.add(i);
                    mean += v;
                    sum_sq += v * v;
                    max_abs = max_abs.max(v.abs());
                }
            }
            mean /= n as f32;
            let rms = (sum_sq / n as f32).sqrt();
            println!("  {name}: mean={mean:.3}, rms={rms:.3}, max_abs={max_abs:.3}");
        }
    }
    Ok(())
}

/// Dump `wq @ x_norm` output for `blk.3.attn_q.weight` on a synthetic
/// x_norm input. Reads back the full 8192-wide output and computes
/// per-head statistics for the Q-half (positions 0..256 of each head)
/// vs the Gate-half (positions 256..512). If these have wildly
/// different magnitudes, the wq matmul or the GGUF tensor layout is
/// the suspect.
#[cfg(feature = "gpu_debug")]
fn wq_dump_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    let wq = weights
        .view("blk.3.attn_q.weight")
        .map_err(|_| "missing blk.3.attn_q.weight")?;
    let n_embd = wq.dims[0]; // 2048
    let wq_out = wq.dims[1]; // 8192

    println!("wq dump: K={n_embd}, M={wq_out}");

    let out = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        // Use the actual model embedding for token 9707 (= "Hello") + attn_norm
        // to feed a realistic x_norm. Embedding lookup → rms_norm → wq.
        let token_buf = ctx.alloc_scratch(4)?;
        let host_ptr = ctx.scratch.host_ptr.ok_or("no host ptr")?;
        unsafe {
            *(host_ptr.add(token_buf.offset as usize) as *mut u32) = 9707u32;
        }
        let residual = ctx.alloc_tensor([n_embd, 1, 1, 1], GgmlType::F32)?;
        let token_embd = weights.view("token_embd.weight")?;
        crate::inference::ops::elementwise::record_get_rows(ctx, token_embd, token_buf, 1, residual)?;
        let attn_norm = weights.view("blk.3.attn_norm.weight")?;
        let x_norm = ctx.alloc_tensor([n_embd, 1, 1, 1], GgmlType::F32)?;
        crate::inference::ops::rms_norm::record(ctx, residual, attn_norm, x_norm, 1e-6)?;
        let q_full = ctx.alloc_tensor([wq_out, 1, 1, 1], GgmlType::F32)?;
        crate::inference::ops::matmul::record(ctx, wq, x_norm, q_full)?;
        Ok(q_full.range())
    })?;

    let head_dim = 256;
    let n_head = 16;
    assert_eq!(out.len(), (wq_out) as usize);
    // For each head h, q-half = positions [h*512, h*512+256), gate-half = [h*512+256, h*512+512).
    println!("Per-head magnitude stats (mean abs, max abs):");
    println!("  head  q_mean   q_max    gate_mean gate_max");
    let mut q_mean_global = 0f32;
    let mut g_mean_global = 0f32;
    let mut q_max_global = 0f32;
    let mut g_max_global = 0f32;
    for h in 0..n_head {
        let base = h * 512;
        let mut q_mean = 0f32; let mut q_max = 0f32;
        let mut g_mean = 0f32; let mut g_max = 0f32;
        for d in 0..head_dim {
            let qv = out[base + d].abs();
            let gv = out[base + head_dim + d].abs();
            q_mean += qv; g_mean += gv;
            q_max = q_max.max(qv);
            g_max = g_max.max(gv);
        }
        q_mean /= head_dim as f32; g_mean /= head_dim as f32;
        println!("  {h:3}  {q_mean:7.3}  {q_max:7.3}  {g_mean:7.3}    {g_max:7.3}");
        q_mean_global += q_mean; g_mean_global += g_mean;
        q_max_global = q_max_global.max(q_max);
        g_max_global = g_max_global.max(g_max);
    }
    q_mean_global /= n_head as f32; g_mean_global /= n_head as f32;
    println!("  ALL  q_mean={q_mean_global:.3}, q_max={q_max_global:.3}");
    println!("       g_mean={g_mean_global:.3}, g_max={g_max_global:.3}");
    // Sigmoid distribution for gate values
    let mut bins = [0usize; 5]; // [0,0.1), [0.1, 0.3), [0.3, 0.7), [0.7, 0.9), [0.9, 1.0]
    for h in 0..n_head {
        let base = h * 512;
        for d in 0..head_dim {
            let g = out[base + head_dim + d];
            let s = 1.0 / (1.0 + (-g).exp());
            let bin = if s < 0.1 { 0 } else if s < 0.3 { 1 } else if s < 0.7 { 2 } else if s < 0.9 { 3 } else { 4 };
            bins[bin] += 1;
        }
    }
    println!("Sigmoid(gate) distribution:");
    println!("  [0, 0.1) : {} ({:.1}%)", bins[0], 100.0 * bins[0] as f32 / (n_head * head_dim) as f32);
    println!("  [0.1, 0.3) : {} ({:.1}%)", bins[1], 100.0 * bins[1] as f32 / (n_head * head_dim) as f32);
    println!("  [0.3, 0.7) : {} ({:.1}%)", bins[2], 100.0 * bins[2] as f32 / (n_head * head_dim) as f32);
    println!("  [0.7, 0.9) : {} ({:.1}%)", bins[3], 100.0 * bins[3] as f32 / (n_head * head_dim) as f32);
    println!("  [0.9, 1.0] : {} ({:.1}%)", bins[4], 100.0 * bins[4] as f32 / (n_head * head_dim) as f32);
    Ok(())
}

/// Batched gated-delta-net smoke test: B sequences, each with its own
/// recurrent state, run through GDN at `n_seqs = B` in one dispatch, vs a
/// per-sequence CPU reference. Proves the recurrent state batches correctly
/// (`seq_id` indexing) — the linchpin of batched decode for the qwen35moe
/// SSM hybrid (M2). Run with `SEEKER_GDN_BATCH_TEST=1 ... --features gpu_debug`.
#[cfg(feature = "gpu_debug")]
fn gdn_batch_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::ops::ssm::{record_gated_delta_net, GdnStrides};

    let s_v: u64 = 128;
    let num_v: u64 = 2; // heads
    let l: u64 = 3; // tokens per sequence
    let b: u64 = 3; // sequences (n_seqs)
    let scale: f32 = 1.0;
    let _ = weights;
    let per_seq = (s_v * num_v * l) as usize; // q/k/v per sequence
    let gb_per_seq = (num_v * l) as usize; // g/beta per sequence

    println!("GDN batch smoke: B={b}, s_v={s_v}, num_v={num_v}, L={l}");

    // Per-sequence synthetic inputs (vary by sequence `s`).
    let mut q_host = vec![0f32; b as usize * per_seq];
    let mut k_host = vec![0f32; b as usize * per_seq];
    let mut v_host = vec![0f32; b as usize * per_seq];
    let mut g_host = vec![0f32; b as usize * gb_per_seq];
    let mut beta_host = vec![0f32; b as usize * gb_per_seq];
    for s in 0..b as usize {
        for t in 0..l as usize {
            for h in 0..num_v as usize {
                for i in 0..s_v as usize {
                    let off = s * per_seq + t * (s_v * num_v) as usize + h * s_v as usize + i;
                    q_host[off] = ((s * 13 + t * 7 + h * 3 + i) as f32 * 0.13).sin();
                    k_host[off] = ((s * 17 + t * 11 + h * 5 + i) as f32 * 0.07).cos();
                    v_host[off] = ((s * 19 + t * 5 + h * 2 + i) as f32 * 0.19).sin() * 0.5;
                }
                let gh = s * gb_per_seq + t * num_v as usize + h;
                g_host[gh] = -0.1 * (s as f32 + 1.0) * (h as f32 + 1.0);
                beta_host[gh] = 0.5 + 0.1 * t as f32 + 0.05 * s as f32;
            }
        }
    }

    // CPU reference: per-sequence recurrence (state resets per sequence).
    let attn_per_seq = (l * num_v * s_v) as usize;
    let mut ref_out = vec![0f32; b as usize * attn_per_seq];
    for s in 0..b as usize {
        for h in 0..num_v as usize {
            let mut state = vec![0f32; (s_v * s_v) as usize]; // col-major [S_V*S_V]
            for t in 0..l as usize {
                let base = s * per_seq + t * (s_v * num_v) as usize + h * s_v as usize;
                let q = &q_host[base..base + s_v as usize];
                let k = &k_host[base..base + s_v as usize];
                let v = &v_host[base..base + s_v as usize];
                let g_val = g_host[s * gb_per_seq + t * num_v as usize + h].exp();
                let beta = beta_host[s * gb_per_seq + t * num_v as usize + h];
                let mut kv = vec![0f32; s_v as usize];
                for c in 0..s_v as usize {
                    let mut acc = 0f32;
                    for r in 0..s_v as usize {
                        acc += g_val * state[c * s_v as usize + r] * k[r];
                    }
                    kv[c] = acc;
                }
                for c in 0..s_v as usize {
                    let delta = (v[c] - kv[c]) * beta;
                    for r in 0..s_v as usize {
                        state[c * s_v as usize + r] =
                            g_val * state[c * s_v as usize + r] + k[r] * delta;
                    }
                }
                for c in 0..s_v as usize {
                    let mut acc = 0f32;
                    for r in 0..s_v as usize {
                        acc += state[c * s_v as usize + r] * q[r];
                    }
                    let out_idx =
                        s * attn_per_seq + t * (num_v * s_v) as usize + h * s_v as usize + c;
                    ref_out[out_idx] = acc * scale;
                }
            }
        }
    }

    let gpu_out = engine.forward(
        weights,
        |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
            let elem = 4u64;
            // q/k/v: [s_v, num_v, l, B]; g/beta: [num_v, l, B] (seq outermost).
            let q = ctx.alloc_tensor([s_v, num_v, l, b], GgmlType::F32)?;
            let k = ctx.alloc_tensor([s_v, num_v, l, b], GgmlType::F32)?;
            let v = ctx.alloc_tensor([s_v, num_v, l, b], GgmlType::F32)?;
            let g = ctx.alloc_tensor([num_v, l, b, 1], GgmlType::F32)?;
            let beta = ctx.alloc_tensor([num_v, l, b, 1], GgmlType::F32)?;
            let host_ptr = ctx.scratch.host_ptr.ok_or("scratch not host-visible")?;
            let cp = |dst: &crate::inference::weights::TensorView, src: &[f32]| unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    host_ptr.add(dst.byte_offset as usize) as *mut f32,
                    src.len(),
                );
            };
            cp(&q, &q_host);
            cp(&k, &k_host);
            cp(&v, &v_host);
            cp(&g, &g_host);
            cp(&beta, &beta_host);

            let attn_floats = b * l * num_v * s_v;
            let state_floats = b * num_v * s_v * s_v;
            let gdn_total = attn_floats + state_floats;
            let gdn_dst = ctx.alloc_scratch(gdn_total * elem)?;
            let state_in = ctx.alloc_scratch(state_floats * elem)?;
            unsafe {
                std::ptr::write_bytes(
                    host_ptr.add(state_in.offset as usize) as *mut u8,
                    0,
                    state_in.size as usize,
                );
            }

            record_gated_delta_net(
                ctx,
                q,
                k,
                v,
                g,
                beta,
                state_in,
                gdn_dst,
                num_v as u32,
                num_v as u32,
                l as u32,
                b as u32, // n_seqs = B
                attn_floats as u32, // s_off: state region starts after all B attn outputs
                scale,
                GdnStrides { s1: s_v as u32, s2: (s_v * num_v) as u32, s3: (s_v * num_v * l) as u32 },
                GdnStrides { s1: s_v as u32, s2: (s_v * num_v) as u32, s3: (s_v * num_v * l) as u32 },
                GdnStrides { s1: 1, s2: num_v as u32, s3: (num_v * l) as u32 },
                s_v as u32,
                1, // k_snapshots = 1
            )?;
            Ok(crate::inference::buffer::BufferRange {
                buffer: gdn_dst.buffer,
                offset: gdn_dst.offset,
                size: attn_floats * elem,
            })
        },
    )?;

    // Relative error: the recurrent sums reach large magnitudes (~1e3), so an
    // absolute threshold is meaningless — compare against |ref| (floored at 1).
    let mut max_rel = 0f32;
    let mut worst = (0usize, 0usize, 0usize, 0usize);
    for s in 0..b as usize {
        for t in 0..l as usize {
            for h in 0..num_v as usize {
                for c in 0..s_v as usize {
                    let idx = s * attn_per_seq + t * (num_v * s_v) as usize + h * s_v as usize + c;
                    let rel = (gpu_out[idx] - ref_out[idx]).abs() / ref_out[idx].abs().max(1.0);
                    if rel > max_rel {
                        max_rel = rel;
                        worst = (s, t, h, c);
                    }
                }
            }
        }
    }
    let (ws, wt, wh, wc) = worst;
    let widx = ws * attn_per_seq + wt * (num_v * s_v) as usize + wh * s_v as usize + wc;
    println!(
        "  max_rel_err = {max_rel:.3e} at (seq={ws}, t={wt}, head={wh}, c={wc}); gpu={:.5}, ref={:.5}",
        gpu_out[widx], ref_out[widx]
    );
    if max_rel < 1e-4 {
        println!("  RESULT: PASS (batched GDN recurrence matches per-sequence reference)");
    } else {
        println!("  RESULT: FAIL — batched GDN diverges from the per-sequence reference");
    }
    Ok(())
}

/// Gated-delta-net smoke test. Feeds known Q/K/V/g/β to GDN with
/// zero initial state, reads back output, compares against a CPU
/// reference of the recurrent update. Verifies the most complex op
/// in the SSM block.
#[cfg(feature = "gpu_debug")]
fn gdn_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::ops::ssm::{record_gated_delta_net, GdnStrides};

    // Production S_V (smaller values give ROWS_PER_LANE = S_V/32 = 0 and
    // the shader silently returns zeros). num_v small for fast reference.
    let s_v: u64 = 128;
    let num_v: u64 = 2; // heads
    let l: u64 = 3; // tokens
    let n_seqs: u64 = 1;
    let scale: f32 = 1.0;
    let _ = weights;

    println!("GDN smoke: s_v={s_v}, num_v={num_v}, L={l}");

    // CPU reference inputs (deterministic).
    let mut q_host = vec![0f32; (s_v * num_v * l) as usize];
    let mut k_host = vec![0f32; (s_v * num_v * l) as usize];
    let mut v_host = vec![0f32; (s_v * num_v * l) as usize];
    let mut g_host = vec![0f32; (num_v * l) as usize];
    let mut beta_host = vec![0f32; (num_v * l) as usize];
    for t in 0..l as usize {
        for h in 0..num_v as usize {
            for i in 0..s_v as usize {
                let off = t * (s_v * num_v) as usize + h * s_v as usize + i;
                q_host[off] = ((t * 7 + h * 3 + i) as f32 * 0.13).sin();
                k_host[off] = ((t * 11 + h * 5 + i) as f32 * 0.07).cos();
                v_host[off] = ((t * 5 + h * 2 + i) as f32 * 0.19).sin() * 0.5;
            }
            let gh = t * num_v as usize + h;
            g_host[gh] = -0.1 * (h as f32 + 1.0); // gate exponent (negative → decay)
            beta_host[gh] = 0.5 + 0.1 * t as f32;
        }
    }

    // CPU reference: replicate the shader's per-(head, seq) update exactly.
    // Per token t:
    //   g_exp = exp(g[t,h])                  # scalar gate
    //   kv[c]    = sum_r g_exp * S[r][c] * k[r]
    //   delta[c] = (v[c] - kv[c]) * beta
    //   S[r][c]  = g_exp * S[r][c] + k[r] * delta[c]
    //   attn[c]  = sum_r S[r][c] * q[r]
    //   out[t,h,c] = attn[c] * scale
    // Storage: S is column-major (col stride = S_V, row stride = 1) to
    // match the shader's `data_state[col * S_V + r]` indexing.
    let mut ref_out = vec![0f32; (l * num_v * s_v) as usize];
    for h in 0..num_v as usize {
        let mut state = vec![0f32; (s_v * s_v) as usize]; // [S_V * S_V], col-major
        for t in 0..l as usize {
            // Read inputs for this (t, h).
            let q_base = t * (s_v * num_v) as usize + h * s_v as usize;
            let q = &q_host[q_base..q_base + s_v as usize];
            let k = &k_host[q_base..q_base + s_v as usize];
            let v = &v_host[q_base..q_base + s_v as usize];
            let g_val = g_host[t * num_v as usize + h].exp();
            let beta = beta_host[t * num_v as usize + h];
            // kv[c]
            let mut kv = vec![0f32; s_v as usize];
            for c in 0..s_v as usize {
                let mut acc = 0f32;
                for r in 0..s_v as usize {
                    acc += g_val * state[c * s_v as usize + r] * k[r];
                }
                kv[c] = acc;
            }
            // delta and state update
            for c in 0..s_v as usize {
                let delta = (v[c] - kv[c]) * beta;
                for r in 0..s_v as usize {
                    state[c * s_v as usize + r] =
                        g_val * state[c * s_v as usize + r] + k[r] * delta;
                }
            }
            // attn out
            for c in 0..s_v as usize {
                let mut acc = 0f32;
                for r in 0..s_v as usize {
                    acc += state[c * s_v as usize + r] * q[r];
                }
                // Shader output layout: dst[seq * n_tokens * H + t * H + h, col]
                // i.e. for our flat readback: ref_out[t*num_v*s_v + h*s_v + c]
                let out_idx = t * (num_v * s_v) as usize + h * s_v as usize + c;
                ref_out[out_idx] = acc * scale;
            }
        }
    }
    let gpu_out = engine.forward(
        weights,
        |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
            let elem = 4u64;
            let q = ctx.alloc_tensor([s_v, num_v, l, 1], GgmlType::F32)?;
            let k = ctx.alloc_tensor([s_v, num_v, l, 1], GgmlType::F32)?;
            let v = ctx.alloc_tensor([s_v, num_v, l, 1], GgmlType::F32)?;
            let g = ctx.alloc_tensor([num_v, l, 1, 1], GgmlType::F32)?;
            let beta = ctx.alloc_tensor([num_v, l, 1, 1], GgmlType::F32)?;
            let host_ptr = ctx.scratch.host_ptr.ok_or("scratch not host-visible")?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    q_host.as_ptr(),
                    host_ptr.add(q.byte_offset as usize) as *mut f32,
                    q_host.len(),
                );
                std::ptr::copy_nonoverlapping(
                    k_host.as_ptr(),
                    host_ptr.add(k.byte_offset as usize) as *mut f32,
                    k_host.len(),
                );
                std::ptr::copy_nonoverlapping(
                    v_host.as_ptr(),
                    host_ptr.add(v.byte_offset as usize) as *mut f32,
                    v_host.len(),
                );
                std::ptr::copy_nonoverlapping(
                    g_host.as_ptr(),
                    host_ptr.add(g.byte_offset as usize) as *mut f32,
                    g_host.len(),
                );
                std::ptr::copy_nonoverlapping(
                    beta_host.as_ptr(),
                    host_ptr.add(beta.byte_offset as usize) as *mut f32,
                    beta_host.len(),
                );
            }

            // Allocate GDN dst (attn output + state-out, packed).
            let attn_floats = l * num_v * s_v;
            let state_floats = num_v * s_v * s_v;
            let gdn_total = attn_floats + state_floats;
            let gdn_dst = ctx.alloc_scratch(gdn_total * elem)?;
            let gdn_state_in = ctx.alloc_scratch(state_floats * elem)?;
            unsafe {
                std::ptr::write_bytes(
                    host_ptr.add(gdn_state_in.offset as usize) as *mut u8,
                    0,
                    gdn_state_in.size as usize,
                );
            }

            record_gated_delta_net(
                ctx,
                q,
                k,
                v,
                g,
                beta,
                gdn_state_in,
                gdn_dst,
                num_v as u32,
                num_v as u32, // head_count_k = head_count_v (no GQA in this toy)
                l as u32,
                n_seqs as u32,
                attn_floats as u32,
                scale,
                GdnStrides {
                    s1: s_v as u32,
                    s2: (s_v * num_v) as u32,
                    s3: (s_v * num_v * l) as u32,
                },
                GdnStrides {
                    s1: s_v as u32,
                    s2: (s_v * num_v) as u32,
                    s3: (s_v * num_v * l) as u32,
                },
                GdnStrides {
                    s1: 1,
                    s2: num_v as u32,
                    s3: (num_v * l) as u32,
                },
                s_v as u32,
                1, // k_snapshots = 1 (normal mode: final state only, no checkpoint)
            )?;
            // Return only the attn portion (first attn_floats of gdn_dst).
            Ok(crate::inference::buffer::BufferRange {
                buffer: gdn_dst.buffer,
                offset: gdn_dst.offset,
                size: attn_floats * elem,
            })
        },
    )?;

    // Compare GPU output against CPU reference.
    let mut max_err = 0f32;
    let mut worst = (0usize, 0usize, 0usize);
    for t in 0..l as usize {
        for h in 0..num_v as usize {
            for c in 0..s_v as usize {
                let idx = t * (num_v * s_v) as usize + h * s_v as usize + c;
                let err = (gpu_out[idx] - ref_out[idx]).abs();
                if err > max_err {
                    max_err = err;
                    worst = (t, h, c);
                }
            }
        }
    }
    let (wt, wh, wc) = worst;
    let widx = wt * (num_v * s_v) as usize + wh * s_v as usize + wc;
    println!(
        "  max_err = {:.6e} at (t={}, h={}, c={}); gpu = {}, ref = {}",
        max_err, wt, wh, wc, gpu_out[widx], ref_out[widx]
    );
    let max_ref = ref_out.iter().fold(0f32, |a, &b| a.max(b.abs()));
    println!("  max_abs_ref = {:.4}, relative_err = {:.4e}", max_ref, max_err / max_ref.max(1e-12));
    if max_err / max_ref.max(1e-12) < 1e-3 {
        println!("  RESULT: PASS (GDN math matches CPU reference)");
    } else {
        println!("  RESULT: FAIL — GDN output diverges from CPU reference");
    }
    Ok(())
}

/// MoE FFN diagnostic test: same chain as the smoke test but with
/// REAL embedding+norm input (not synthetic), returning intermediate
/// tensors by stage so we can pinpoint where NaN first appears.
///
/// Stages (set via SEEKER_MOE_DIAG=<stage>):
///   xnorm        — output of rms_norm(embedding, post_attn_norm)
///   logits       — gate_logits = ffn_gate_inp @ x_norm
///   weights      — topk_moe routing weights [n_expert_used]
///   ids          — topk_moe expert ids [n_experts] (first n_expert_used valid)
///   gate         — matvec_q4k_id(ffn_gate_exps, x_norm, ids) [ff, n_used]
///   up           — matvec_q4k_id(ffn_up_exps,   x_norm, ids)
///   ffnh         — silu(gate) * up
///   routed       — moe_down_q5k(ffn_down_exps, ffn_h, ids, weights) [n_embd, 1]
#[cfg(feature = "gpu_debug")]
fn moe_diag_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
    stage: &str,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::ops::{elementwise, matmul, moe, rms_norm};

    let view = |n: &str| weights.view(n).map_err(|_| format!("missing tensor: {n}"));
    let token_embd = view("token_embd.weight")?;
    let post_attn_norm = view("blk.0.post_attention_norm.weight")?;
    let ffn_gate_inp = view("blk.0.ffn_gate_inp.weight")?;
    let ffn_gate_exps = view("blk.0.ffn_gate_exps.weight")?;
    let ffn_up_exps = view("blk.0.ffn_up_exps.weight")?;
    let ffn_down_exps = view("blk.0.ffn_down_exps.weight")?;

    let n_embd = ffn_gate_exps.dims[0];
    let ff = ffn_gate_exps.dims[1];
    let n_experts = ffn_gate_exps.dims[2] as u32;
    let n_expert_used = 8u32;
    let token: u32 = 9419; // "Hello"

    println!("MoE diag — stage={stage}, token={token}, n_embd={n_embd}, ff={ff}");

    let stage_clone = stage.to_string();
    let out = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        // 1. Embedding lookup.
        let token_buf = ctx.alloc_scratch(4)?;
        let host = ctx.scratch.host_ptr.unwrap();
        unsafe { *(host.add(token_buf.offset as usize) as *mut u32) = token; }
        let residual = ctx.alloc_tensor([n_embd, 1, 1, 1], GgmlType::F32)?;
        elementwise::record_get_rows(ctx, token_embd, token_buf, 1, residual)?;

        // 2. post_attn_norm
        let x_norm = ctx.alloc_tensor([n_embd, 1, 1, 1], GgmlType::F32)?;
        rms_norm::record(ctx, residual, post_attn_norm, x_norm, 1e-6)?;
        if stage_clone == "xnorm" { return Ok(x_norm.range()); }

        // 3. Router logits
        let gate_logits = ctx.alloc_tensor([n_experts as u64, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, ffn_gate_inp, x_norm, gate_logits)?;
        if stage_clone == "logits" { return Ok(gate_logits.range()); }

        // 4. topk
        let ids = ctx.alloc_scratch((n_experts as u64) * 4)?;
        let weights_buf = ctx.alloc_scratch((n_expert_used as u64) * 4)?;
        moe::record_topk_moe(
            ctx, gate_logits, weights_buf, ids,
            moe::TopkMoeParams {
                n_experts, n_expert_used,
                gating_func: moe::GATING_SOFTMAX,
                with_norm: false,
            },
        )?;
        if stage_clone == "weights" { return Ok(weights_buf); }
        if stage_clone == "ids" { return Ok(ids); }

        // 5. gate matvec_id
        let gate = ctx.alloc_tensor([ff, n_expert_used as u64, 1, 1], GgmlType::F32)?;
        moe::record_matvec_q4k_id(ctx, ffn_gate_exps, x_norm, ids, gate, n_expert_used)?;
        if stage_clone == "gate" { return Ok(gate.range()); }

        // 6. up matvec_id
        let up = ctx.alloc_tensor([ff, n_expert_used as u64, 1, 1], GgmlType::F32)?;
        moe::record_matvec_q4k_id(ctx, ffn_up_exps, x_norm, ids, up, n_expert_used)?;
        if stage_clone == "up" { return Ok(up.range()); }

        // 7. SwiGLU
        let gate_silu = ctx.alloc_tensor([ff, n_expert_used as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_silu(ctx, gate, gate_silu)?;
        let ffn_h = ctx.alloc_tensor([ff, n_expert_used as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_mul(ctx, gate_silu, up, ffn_h)?;
        if stage_clone == "ffnh" { return Ok(ffn_h.range()); }

        // 8. Fused down
        let routed = ctx.alloc_tensor([n_embd, 1, 1, 1], GgmlType::F32)?;
        moe::record_moe_down_q5k(ctx, ffn_down_exps, ffn_h, ids, weights_buf, routed, n_expert_used)?;
        Ok(routed.range())
    })?;

    let n = out.len();
    let finite = out.iter().filter(|x| x.is_finite()).count();
    let nan = out.iter().filter(|x| x.is_nan()).count();
    let inf = out.iter().filter(|x| x.is_infinite()).count();
    let max_abs = out.iter().filter(|x| x.is_finite()).fold(0f32, |a, &b| a.max(b.abs()));
    let mean_abs = if finite > 0 {
        out.iter().filter(|x| x.is_finite()).map(|x| x.abs()).sum::<f32>() / finite as f32
    } else { 0.0 };
    println!("  total: {n}, finite: {finite}, NaN: {nan}, Inf: {inf}");
    println!("  max |x|  : {max_abs:.6}");
    println!("  mean |x| : {mean_abs:.6}");
    println!("  first 8 : {:?}", &out[..8.min(n)]);
    println!("  last 8  : {:?}", &out[n - 8.min(n)..]);
    if stage == "ids" {
        // Re-interpret as u32 — ids are stored as u32
        let bytes = unsafe {
            std::slice::from_raw_parts(out.as_ptr() as *const u32, out.len())
        };
        println!("  first 8 ids (u32 reinterpret): {:?}", &bytes[..8]);
    }
    Ok(())
}

/// MoE FFN smoke test: exercises the full GPU-only MoE expert dispatch
/// (`topk_moe` → `mat_vec_q4_k_id` × 2 → swiglu → `moe_down_q5_k`) on
/// real Qwen35MoE block-0 weights. Doesn't verify against a CPU
/// reference — just confirms the dispatch chain runs to completion and
/// produces non-NaN finite values for a deterministic input. A later
/// pass should diff against llama.cpp's per-layer output.
#[cfg(feature = "gpu_debug")]
fn moe_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::ops::{elementwise, matmul, moe};

    // Block-0 MoE weights from the qwen35moe checkpoint. The first SSM
    // block has the same MoE tensors as any attention block — we just
    // need any block's FFN slab.
    let view = |name: &str| -> Result<crate::inference::weights::TensorView, Box<dyn Error>> {
        weights
            .view(name)
            .map_err(|_| format!("missing tensor: {name}").into())
    };
    let ffn_gate_inp = view("blk.0.ffn_gate_inp.weight")?;
    let ffn_gate_exps = view("blk.0.ffn_gate_exps.weight")?;
    let ffn_up_exps = view("blk.0.ffn_up_exps.weight")?;
    let ffn_down_exps = view("blk.0.ffn_down_exps.weight")?;

    // Shape sanity — bail loudly if the loaded checkpoint doesn't match
    // the Qwen3.5-A3B layout we coded against.
    if ffn_gate_exps.dims[2] == 0 || ffn_gate_exps.dims[2] != ffn_up_exps.dims[2] {
        return Err(format!(
            "MoE smoke test expects 3-D expert slabs; got gate_exps dims {:?}",
            ffn_gate_exps.dims
        )
        .into());
    }
    let n_embd = ffn_gate_exps.dims[0];
    let ff = ffn_gate_exps.dims[1];
    let n_experts = ffn_gate_exps.dims[2] as u32;
    let n_expert_used = 8u32;

    println!(
        "MoE smoke test setup: n_embd={n_embd}, ff={ff}, n_experts={n_experts}, n_expert_used={n_expert_used}"
    );

    let output = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        // 1. Synthetic x_norm — deterministic, varied.
        let x_norm = ctx.alloc_tensor([n_embd, 1, 1, 1], GgmlType::F32)?;
        let host_ptr = ctx.scratch.host_ptr.ok_or("scratch not host-visible")?;
        unsafe {
            let p = host_ptr.add(x_norm.byte_offset as usize) as *mut f32;
            for i in 0..n_embd as usize {
                *p.add(i) = (i as f32 * 0.01).sin();
            }
        }

        // 2. Router logits = ffn_gate_inp @ x_norm. F32 weights, F32 input;
        //    matmul::record dispatches mul_mat_vec_f32 (N=1).
        let gate_logits = ctx.alloc_tensor([n_experts as u64, 1, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, ffn_gate_inp, x_norm, gate_logits)?;

        // 3. topk_moe → ids[n_experts] (n_expert_used valid prefix) + weights[n_expert_used].
        //    The ids buffer is sized n_experts per token per the shader's ids_offset math.
        let ids = ctx.alloc_scratch((n_experts as u64) * 4)?;
        let weights_buf = ctx.alloc_scratch((n_expert_used as u64) * 4)?;
        moe::record_topk_moe(
            ctx,
            gate_logits,
            weights_buf,
            ids,
            moe::TopkMoeParams {
                n_experts,
                n_expert_used,
                gating_func: moe::GATING_SOFTMAX,
                with_norm: false,
            },
        )?;

        // 4 + 5. Per-expert gate and up matvecs (Q4_K, expert-indirect).
        let gate = ctx.alloc_tensor([ff, n_expert_used as u64, 1, 1], GgmlType::F32)?;
        moe::record_matvec_q4k_id(ctx, ffn_gate_exps, x_norm, ids, gate, n_expert_used)?;
        let up = ctx.alloc_tensor([ff, n_expert_used as u64, 1, 1], GgmlType::F32)?;
        moe::record_matvec_q4k_id(ctx, ffn_up_exps, x_norm, ids, up, n_expert_used)?;

        // 6 + 7. SwiGLU: silu(gate) * up → ffn_h.
        let gate_silu = ctx.alloc_tensor([ff, n_expert_used as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_silu(ctx, gate, gate_silu)?;
        let ffn_h = ctx.alloc_tensor([ff, n_expert_used as u64, 1, 1], GgmlType::F32)?;
        elementwise::record_mul(ctx, gate_silu, up, ffn_h)?;

        // 8. Fused down: routing-weighted sum across experts in one dispatch.
        let output = ctx.alloc_tensor([n_embd, 1, 1, 1], GgmlType::F32)?;
        moe::record_moe_down_q5k(
            ctx,
            ffn_down_exps,
            ffn_h,
            ids,
            weights_buf,
            output,
            n_expert_used,
        )?;

        Ok(output.range())
    })?;

    let nonzero = output.iter().filter(|x| x.abs() > 1e-9).count();
    let non_finite = output.iter().filter(|x| !x.is_finite()).count();
    let max_abs = output.iter().fold(0f32, |a, &b| a.max(b.abs()));
    let sum: f32 = output.iter().sum();
    println!("  output[0..8]                = {:?}", &output[..8.min(output.len())]);
    println!(
        "  output[{}..{}] = {:?}",
        output.len() - 8.min(output.len()),
        output.len(),
        &output[output.len() - 8.min(output.len())..],
    );
    println!("  nonzero  : {nonzero}/{}", output.len());
    println!("  non-finite: {non_finite}");
    println!("  max |x|  : {max_abs:.6}");
    println!("  sum      : {sum:.6}");
    if non_finite > 0 {
        println!("  RESULT: FAIL — NaN/Inf in output");
    } else if nonzero == 0 {
        println!("  RESULT: FAIL — output is all zeros");
    } else if max_abs < 1e-6 || max_abs > 1e6 {
        println!("  RESULT: SUSPECT — magnitudes outside expected range");
    } else {
        println!("  RESULT: pass (non-NaN, non-zero, reasonable magnitude)");
    }
    Ok(())
}

/// Cooperative-matrix matmul smoke test: builds a small F16 weight A in
/// scratch, an F32 input B, dispatches `matmul::record` with
/// `SEEKER_MM_CM=1` set internally, and compares against the obvious CPU
/// dot-product. Uses M=N=K=32 (smallest size that exercises CM) so any
/// output-store layout error is loud.
#[cfg(feature = "gpu_debug")]
fn mm_cm_smoke_test(
    engine: &mut Engine,
    weights: &crate::inference::weights::WeightsHandle,
) -> Result<(), Box<dyn Error>> {
    use crate::inference::ops::matmul;

    let m: u64 = 32;
    let k: u64 = 32;
    let n: u64 = 32;

    // Force CM dispatch path inside `matmul::record`.
    // SAFETY: single-threaded test; std::env mutation is fine in this context.
    unsafe { std::env::set_var("SEEKER_MM_CM", "1") };

    let logits = engine.forward(weights, |ctx| -> Result<crate::inference::buffer::BufferRange, Box<dyn Error>> {
        let a = ctx.alloc_tensor([k, m, 1, 1], GgmlType::F16)?;
        let b = ctx.alloc_tensor([k, n, 1, 1], GgmlType::F32)?;
        let d = ctx.alloc_tensor([m, n, 1, 1], GgmlType::F32)?;
        let host_ptr = ctx.scratch.host_ptr.unwrap();
        unsafe {
            // A[m, k] = (m + 0.1 * k) as F16. Stored A[k, m] in ggml.
            // memory[m*k_idx + k_idx... wait, A is laid out K as ne[0], M as ne[1]
            //   → A_mem[m*K + k] = A_mathematical[k, m] = (m + 0.1 * k) ?
            // Let's just use: A_mem[i] = i * 0.01 (any deterministic pattern).
            let a_ptr = host_ptr.add(a.byte_offset as usize) as *mut u16;
            for i in 0..(k * m) as usize {
                *a_ptr.add(i) = f32_to_f16_bits(i as f32 * 0.01);
            }
            // B[n, k] memory: B_mem[n * K + k] = (n + 1) * (k + 1) * 0.1
            let b_ptr = host_ptr.add(b.byte_offset as usize) as *mut f32;
            for n_i in 0..n as usize {
                for k_i in 0..k as usize {
                    *b_ptr.add(n_i * k as usize + k_i) =
                        (n_i as f32 + 1.0) * (k_i as f32 + 1.0) * 0.1;
                }
            }
        }
        matmul::record(ctx, a, b, d)?;
        Ok(d.range())
    })?;

    // CPU reference: D[m, n] = sum_k A_mem[m*K + k] * B_mem[n*K + k]
    // (column-major output: D_mem[n * M + m]).
    let mut cpu = vec![0f32; (m * n) as usize];
    for n_i in 0..n as usize {
        for m_i in 0..m as usize {
            let mut s = 0f32;
            for k_i in 0..k as usize {
                let a_val = f16_bits_to_f32(f32_to_f16_bits((m_i * k as usize + k_i) as f32 * 0.01));
                let b_val = (n_i as f32 + 1.0) * (k_i as f32 + 1.0) * 0.1;
                s += a_val * b_val;
            }
            cpu[n_i * m as usize + m_i] = s;
        }
    }

    let mut max_abs = 0f32;
    let mut nonzero_gpu = 0usize;
    let mut worst_idx = 0usize;
    for i in 0..(m * n) as usize {
        if logits[i].abs() > 1e-6 {
            nonzero_gpu += 1;
        }
        let e = (logits[i] - cpu[i]).abs();
        if e > max_abs {
            max_abs = e;
            worst_idx = i;
        }
    }
    println!("mm_cm smoke (F16 A, M=N=K=32):");
    println!("  gpu[0..4] = {:?}", &logits[..4]);
    println!("  cpu[0..4] = {:?}", &cpu[..4]);
    println!("  gpu[end-4..] = {:?}", &logits[(m * n) as usize - 4..]);
    println!("  cpu[end-4..] = {:?}", &cpu[(m * n) as usize - 4..]);
    println!("  nonzero gpu elements: {nonzero_gpu}/{}", m * n);
    println!(
        "  max abs err = {max_abs} (idx {worst_idx}: gpu={} cpu={})",
        logits[worst_idx], cpu[worst_idx],
    );
    Ok(())
}

#[cfg(feature = "gpu_debug")]
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

#[cfg(feature = "gpu_debug")]
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

#[cfg(feature = "gpu_debug")]
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

#[cfg(feature = "gpu_debug")]
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

#[cfg(feature = "gpu_debug")]
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
#[cfg(feature = "gpu_debug")]
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

    // Raw quant bytes — weights are per-tensor device-local now, so go through
    // the debug host-base accessor (same as the other gpu_debug harnesses).
    let base = weights
        .debug_host_base(&a)
        .ok_or("host-side weight reads unsupported (weights are device-local)")?;
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(
            base.add(a.byte_offset as usize),
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
#[cfg(feature = "gpu_debug")]
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

#[cfg(feature = "gpu_debug")]
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
