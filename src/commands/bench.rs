//! `seeker bench` — load a model, run a prefill + N decode steps, report
//! prefill/decode tok/s and per-token latency stats. Used as the baseline
//! signal for the Strix Halo optimization work; every change should leave
//! `--dump-logits` output bit-identical (or within a stated tolerance) and
//! show a positive tok/s delta on the bench numbers.
//!
//! No new dep — just `std::time::Instant`.
//!
//! The dump-logits path uses `Engine::forward` (full logits readback) for
//! the prefill, writes them to stderr as `LOGIT <i> <f32>` lines, then
//! greedy-samples on CPU and continues the decode loop via
//! `forward_sampled` as usual. The prefill therefore costs an extra
//! `vocab_size * 4`-byte host readback when dumping, which is fine —
//! `--dump-logits` is a correctness mode, not a timing mode.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::{GgmlType, GgufFile};
use crate::inference::kv_cache::{parse_dtype, KvCacheConfig};
use crate::inference::sample::{Sampler, SamplerConfig};
use crate::inference::Engine;
use crate::tokenizer::build_tokenizer;

#[derive(Args)]
pub struct BenchArgs {
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]".
    #[arg(long = "hf-repo", required_unless_present = "model", conflicts_with = "model")]
    hf_repo: Option<String>,

    /// Specific file within the repo.
    #[arg(long = "hf-file", requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,

    /// HF auth token.
    #[arg(long = "hf-token", requires = "hf_repo", conflicts_with = "model")]
    hf_token: Option<String>,

    /// Resolve files from the local cache only; never hit the network.
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    offline: bool,

    /// Path to a local .gguf model file.
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,

    /// Prompt text to feed through the prefill pass. Defaults to a stable
    /// 500-ish-token English passage so successive bench runs measure the
    /// same prefill workload.
    #[arg(long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Number of decode tokens to time after the prefill.
    #[arg(long, default_value_t = 64)]
    decode_tokens: u32,

    /// Number of warm-up decode tokens to run before the timed loop.
    /// Discards JIT/page-fault costs on the first few iterations.
    #[arg(long, default_value_t = 4)]
    warmup: u32,

    /// Dump first-token logits to stderr (one `LOGIT <i> <f32>` line per
    /// value). Use for golden-output comparison across optimization steps:
    /// capture once on `main`, re-run after each change, diff.
    #[arg(long = "dump-logits")]
    dump_logits: bool,

    /// If set, truncate the tokenized prompt to this many tokens. Use to
    /// force a prefill length aligned to BM/BN multiples for cooperative-
    /// matrix testing.
    #[arg(long = "prompt-tokens")]
    prompt_tokens: Option<u32>,

    /// KV cache K dtype.
    #[arg(long = "cache-type-k", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_k: GgmlType,

    /// KV cache V dtype.
    #[arg(long = "cache-type-v", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_v: GgmlType,

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
}

fn parse_dtype_arg(s: &str) -> Result<GgmlType, String> {
    parse_dtype(s)
}

/// A ~500-token English passage. Deterministic content; pick `--prompt` to
/// substitute. Length is intentionally large so the prefill pass exercises
/// the batched matmul (N>=32) path that step 2 of the optimization plan
/// will target.
const DEFAULT_PROMPT: &str = "The history of computing hardware covers the developments from early simple devices to aid calculation to modern day computers. The first aids to computation were purely mechanical devices which required the operator to set up the initial values of an elementary arithmetic operation, then manipulate the device to obtain the result. Later, computers represented numbers in a continuous form, for instance distance along a scale, rotation of a shaft, or a voltage. Numbers could also be represented in the form of digits, automatically manipulated by a mechanism. Although this approach generally required more complex mechanisms, it greatly increased the precision of results. The development of transistor technology and then the integrated circuit chip led to a series of breakthroughs, starting with transistor computers and then integrated circuit computers, causing digital computers to largely replace analog computers. Metal-oxide-semiconductor large-scale integration then enabled semiconductor memory and the microprocessor, leading to another key breakthrough, the miniaturized personal computer, in the 1970s. The cost of computers gradually became so low that personal computers by the 1990s, and then mobile computers in the 2000s, became ubiquitous. The earliest known tool for use in computation is the abacus, developed in the period between 2700 to 2300 BCE in Sumer. The Sumerian abacus consisted of a table of successive columns which delimited the successive orders of magnitude of their sexagesimal number system. Its original style of usage was by lines drawn in sand with pebbles. Abaci of a more modern design are still used as calculation tools today, such as the Chinese abacus.";

pub async fn run(args: BenchArgs) -> Result<(), Box<dyn Error>> {
    let path = resolve_model_path(&args).await?;
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;

    let add_special = bundle.add_bos_default || bundle.add_eos_default;
    let encoding = bundle
        .tokenizer
        .encode(args.prompt.as_str(), add_special)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    if let Some(cap) = args.prompt_tokens {
        prompt_tokens.truncate(cap as usize);
    }
    let prefill_tokens = prompt_tokens.len();
    if prefill_tokens == 0 {
        return Err("tokenized prompt is empty".into());
    }

    let mut engine = Engine::new(args.ubatch_size, args.batch_size)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened");
    let weights = engine.upload_weights(&gguf)?;
    let model = crate::models::open(&gguf, weights, bundle)?;

    let max_seq_len = (prefill_tokens as u32)
        .saturating_add(args.warmup)
        .saturating_add(args.decode_tokens);
    // Size the scratch (compute buffer) for this model + n_ubatch before the
    // prefill, replacing the Engine::new placeholder.
    engine.allocate_scratch(model.scratch_bytes_estimate(
        args.ubatch_size,
        max_seq_len,
        args.cache_type_k,
        args.cache_type_v,
    ))?;
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
    // Hybrid models (qwen35moe etc.) need persistent SSM/GDN recurrent state
    // carried across forwards — including across chunked-prefill ubatches and
    // decode steps. Without this the SSM state resets every forward (decode
    // produces garbage; chunked prefill diverges from single-pass). Mirrors
    // run.rs / chat.rs.
    if let Some(ssm) = model.ssm_state_dims() {
        cache.allocate_ssm_state(
            &engine.device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
    }

    // Greedy sampler — bench is for tok/s and golden-output stability, not
    // sampling behavior. Temperature 0, no penalties, no top-k/p/min-p.
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_penalty: 1.0,
        penalty_last_n: 0,
        seed: 0,
    });

    // ── Prefill ─────────────────────────────────────────────────────────
    //
    // With --dump-logits we go through `engine.forward()` so we can read
    // the last-token logits buffer back; we then greedy-sample on the CPU.
    // Without it we go through `forward_sampled` (the path production
    // decode uses), which keeps the sampler chain on the GPU and reads
    // back just the chosen token id. Either way the model's `record_forward`
    // emits identical compute work. The dumped logits are last-token-only
    // (count == n_vocab) — the correct comparison point vs llama.cpp.
    let position_offset = cache.position;
    let t_prefill = Instant::now();
    let next_id = if args.dump_logits {
        // Mirror forward_sampled's chunking so long --dump-logits prompts stay
        // within the scratch budget: feed every ubatch but the last as a
        // KV-only pass, then run the final chunk through engine.forward() for
        // the last-token logits readback.
        let ub = engine.n_ubatch as usize;
        if ub != 0 {
            let mut s = 0usize;
            while s + ub < prompt_tokens.len() {
                let pos = cache.position;
                engine.forward_kv_only(&*model, &mut cache, &prompt_tokens[s..s + ub], pos)?;
                s += ub;
            }
        }
        let tail_start = cache.position as usize;
        let tail = &prompt_tokens[tail_start..];
        let tail_pos = cache.position;
        let logits = engine.forward(model.weights(), |ctx| {
            model
                .record_forward(ctx, &mut cache, tail, tail_pos, /*compute_logits=*/ true)
                .map(|view| {
                    view.expect("compute_logits=true must return logits")
                        .range()
                })
        })?;
        let prefill_elapsed = t_prefill.elapsed();
        let mut stderr = std::io::stderr().lock();
        use std::io::Write as _;
        writeln!(stderr, "LOGITS_BEGIN count={}", logits.len())?;
        for (i, &v) in logits.iter().enumerate() {
            writeln!(stderr, "LOGIT {i} {v:.8e}")?;
        }
        writeln!(stderr, "LOGITS_END")?;
        let next = greedy_argmax(&logits);
        sampler.accept(next);
        eprintln!(
            "prefill: {prefill_tokens} tok in {:.3}s ({:.1} tok/s)",
            prefill_elapsed.as_secs_f64(),
            prefill_tokens as f64 / prefill_elapsed.as_secs_f64().max(1e-9),
        );
        next
    } else {
        engine.forward_sampled(&*model, &mut cache, &prompt_tokens, position_offset, &mut sampler)?
    };
    let prefill_secs = t_prefill.elapsed().as_secs_f64();

    // ── Warm-up decode ──────────────────────────────────────────────────
    let mut cur = next_id;
    for _ in 0..args.warmup {
        let position_offset = cache.position;
        cur = engine.forward_sampled(&*model, &mut cache, &[cur], position_offset, &mut sampler)?;
    }

    // ── Timed decode ────────────────────────────────────────────────────
    let mut per_token_ns: Vec<u128> = Vec::with_capacity(args.decode_tokens as usize);
    for _ in 0..args.decode_tokens {
        let position_offset = cache.position;
        let t0 = Instant::now();
        cur = engine.forward_sampled(&*model, &mut cache, &[cur], position_offset, &mut sampler)?;
        per_token_ns.push(t0.elapsed().as_nanos());
    }

    let decode_secs: f64 = per_token_ns.iter().sum::<u128>() as f64 / 1e9;
    let mean_ms = decode_secs * 1000.0 / per_token_ns.len().max(1) as f64;
    let median_ms = {
        let mut sorted = per_token_ns.clone();
        sorted.sort_unstable();
        let mid = sorted.len() / 2;
        if sorted.is_empty() {
            0.0
        } else if sorted.len() % 2 == 1 {
            sorted[mid] as f64 / 1e6
        } else {
            (sorted[mid - 1] + sorted[mid]) as f64 / 2.0 / 1e6
        }
    };
    let min_ms = per_token_ns.iter().min().copied().unwrap_or(0) as f64 / 1e6;
    let max_ms = per_token_ns.iter().max().copied().unwrap_or(0) as f64 / 1e6;

    let prefill_tps = prefill_tokens as f64 / prefill_secs.max(1e-9);
    let decode_tps = args.decode_tokens as f64 / decode_secs.max(1e-9);

    println!(
        "prefill: {prefill_tokens} tok in {:.3}s -> {prefill_tps:.1} tok/s",
        prefill_secs
    );
    println!(
        "decode:  {decode_n} tok in {decode_secs:.3}s -> {decode_tps:.1} tok/s  \
         (mean {mean_ms:.2} ms, median {median_ms:.2} ms, min {min_ms:.2} ms, max {max_ms:.2} ms, warmup {warmup_n})",
        decode_n = args.decode_tokens,
        warmup_n = args.warmup,
    );
    Ok(())
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

async fn resolve_model_path(args: &BenchArgs) -> Result<PathBuf, Box<dyn Error>> {
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
