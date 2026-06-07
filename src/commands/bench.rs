//! `seeker bench` — llama.cpp `llama-bench`-style performance benchmark.
//!
//! Reports prompt-processing (pp / prefill) and token-generation (tg / decode)
//! throughput as tokens/second, swept across a ladder of context depths so you
//! can watch throughput degrade as the KV cache fills. Output is an aligned
//! table of `mean ± stddev` tok/s over `--reps` repetitions, in the spirit of
//! `llama-bench`.
//!
//! The workload uses SYNTHETIC tokens (a fixed in-vocab id), so this measures
//! raw engine throughput only — NOT output quality. The compute graph and
//! memory traffic are token-value-independent, so synthetic tokens give the
//! same timing as a real prompt. For correctness/quality diagnostics
//! (perplexity, golden logits) use `seeker probe`.
//!
//! Per `(test, depth)`: the KV cache is reset, `depth` tokens are prefilled
//! UNTIMED, then either `--pp` tokens are processed (pp test, pure forward) or
//! `--tg` tokens are generated one-at-a-time (tg test, the real decode path)
//! under timing. Each config runs `--warmup` untimed iterations first (to prime
//! JIT / shader compilation), then `--reps` timed iterations. The cache is
//! fully reset and re-prefilled every iteration — the only state-reset strategy
//! that is correct for hybrid SSM/recurrent models (e.g. qwen35moe), whose
//! recurrent state has no per-position rewind.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

use crate::commands::download::{HfResolveArgs, resolve_hf};
use crate::gguf::{GgmlType, GgufFile, MetadataValue};
use crate::inference::Engine;
use crate::inference::kv_cache::{KvCache, KvCacheConfig, parse_dtype};
use crate::inference::sample::{Sampler, SamplerConfig};
use crate::tokenizer::build_tokenizer;

/// Default depth ladder (sparse): base case plus a doubling-ish climb to 64K.
const DEFAULT_LADDER: &[u32] = &[0, 4096, 16384, 65536];
/// The auto-ladder is capped here; deeper sweeps require an explicit `--depths`.
const DEFAULT_MAX_DEPTH: u32 = 65536;
/// Fixed synthetic token id fed for every pp/tg token. `0` is always in-vocab.
const DUMMY: u32 = 0;

#[derive(Args)]
pub struct BenchArgs {
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]".
    #[arg(
        long = "hf-repo",
        required_unless_present = "model",
        conflicts_with = "model"
    )]
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

    // ─── Workload ───────────────────────────────────────────────────────
    /// Prompt-processing batch size timed at each depth (llama-bench -p).
    #[arg(long = "pp", default_value_t = 512)]
    pp: u32,

    /// Tokens generated one-at-a-time and timed at each depth (llama-bench -n).
    #[arg(long = "tg", default_value_t = 128)]
    tg: u32,

    /// Repetitions per (test, depth); reported as mean ± stddev (llama-bench -r).
    #[arg(long = "reps", default_value_t = 5)]
    reps: u32,

    /// Untimed warmup iterations per (test, depth), before the timed reps.
    #[arg(long = "warmup", default_value_t = 1)]
    warmup: u32,

    /// Comma-separated depth list, e.g. "0,4096,16384". Overrides the default
    /// ladder (0,4K,16K,64K) AND is the opt-in for depths beyond 64K — explicit
    /// entries are only bounded by the model's trained context.
    #[arg(long)]
    depths: Option<String>,

    /// Run only the pp (prefill) tests.
    #[arg(long = "pp-only", conflicts_with = "tg_only")]
    pp_only: bool,

    /// Run only the tg (decode) tests.
    #[arg(long = "tg-only")]
    tg_only: bool,

    // ─── KV cache / batching ────────────────────────────────────────────
    /// KV cache K dtype.
    #[arg(long = "cache-type-k", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_k: GgmlType,

    /// KV cache V dtype.
    #[arg(long = "cache-type-v", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_v: GgmlType,

    /// Logical batch size (max tokens per submit).
    #[arg(short = 'b', long = "batch-size", default_value_t = 2048)]
    batch_size: u32,

    /// Physical micro-batch size: prefill is split into ≤ this many tokens per
    /// GPU pass so scratch memory stays bounded. 0 = unbounded. (short: -ub)
    #[arg(long = "ubatch-size", default_value_t = 512)]
    ubatch_size: u32,
}

fn parse_dtype_arg(s: &str) -> Result<GgmlType, String> {
    parse_dtype(s)
}

pub async fn run(args: BenchArgs) -> Result<(), Box<dyn Error>> {
    let path = resolve_model_path(&args).await?;
    let gguf = GgufFile::open(&path)?;
    let bundle = build_tokenizer(&gguf)?;

    let do_pp = !args.tg_only;
    let do_tg = !args.pp_only;
    // Headroom needed on top of each depth — count only the ENABLED tests so
    // `--tg-only --pp <big>` (or `--pp-only --tg <big>`) doesn't over-reserve
    // KV/scratch or reject depths for a mode that won't run.
    let span = match (do_pp, do_tg) {
        (true, true) => args.pp.max(args.tg),
        (true, false) => args.pp,
        (false, true) => args.tg,
        (false, false) => unreachable!("clap: --pp-only conflicts with --tg-only"),
    };

    let trained_ctx = gguf.trained_ctx_len();
    let (depths, skipped) = build_depths(&args, trained_ctx, span);
    if let Some(tc) = trained_ctx {
        eprintln!("trained context: {tc} tokens");
    } else {
        eprintln!("trained context: unknown (model metadata missing context_length)");
    }
    for (d, reason) in &skipped {
        eprintln!("  skipping depth {d}: {reason}");
    }
    if depths.is_empty() {
        return Err(format!(
            "no depths fit: need {span} tokens of headroom on top of each depth \
             within the trained context (lower --pp/--tg or pick smaller --depths)"
        )
        .into());
    }

    // ── Engine + model ──────────────────────────────────────────────────
    let mut engine = Engine::new(args.ubatch_size, args.batch_size)?;
    let backend = engine.device.name();
    tracing::info!(device = %backend, "vulkan device opened");
    let weights = engine.upload_weights(&gguf)?;
    let model = crate::models::open(&gguf, weights, bundle, false)?;

    // Size scratch + KV for the DEEPEST config (deepest live length is
    // max_depth + max(pp, tg)). Allocated once for the whole sweep.
    let max_depth = *depths.iter().max().expect("non-empty");
    let max_seq_len = max_depth.saturating_add(span).saturating_add(1);
    engine.allocate_scratch(model.scratch_bytes_estimate(
        args.ubatch_size,
        max_seq_len,
        args.cache_type_k,
        args.cache_type_v,
    ))?;
    let dims = model.cache_dims();
    let cache_config = KvCacheConfig {
        k_dtype: args.cache_type_k,
        v_dtype: args.cache_type_v,
        max_seq_len,
        n_head: dims.n_head,
    };
    let mut cache = match model.cache_per_layer_dims() {
        Some((hd, nkv)) => engine.allocate_kv_cache_per_layer(&hd, &nkv, cache_config)?,
        None => {
            engine.allocate_kv_cache(dims.n_layer, dims.head_dim, dims.n_head_kv, cache_config)?
        }
    };
    eprintln!(
        "KV cache: k={:?} v={:?} max_seq_len={} \u{2192} {:.1} MiB",
        args.cache_type_k,
        args.cache_type_v,
        max_seq_len,
        cache.kv_bytes() as f64 / (1024.0 * 1024.0),
    );
    // Hybrid models (qwen35moe etc.) carry persistent SSM/GDN recurrent state
    // across forwards; allocate it so cache.reset() has a region to zero.
    if let Some(ssm) = model.ssm_state_dims() {
        cache.allocate_ssm_state(
            &engine.device,
            ssm.n_ssm_layers,
            ssm.conv_state_floats,
            ssm.gdn_state_floats,
        )?;
    }

    // Greedy sampler — bench measures tok/s, not sampling behavior.
    let mut sampler = Sampler::new(greedy_config());
    // Reusable buffer of synthetic tokens; any chunk we feed is a prefix of it.
    let dummies = vec![DUMMY; max_seq_len as usize];

    // ── Table metadata (constant across rows; repeated per row, grep-able) ──
    let model_name = model_name(&gguf);
    let size_str = human_size(gguf.file_size() as u64);
    let params_str = human_params(count_params(&gguf));

    eprintln!(
        "model: {model_name} | {size_str} | {params_str} params | backend {backend}\n\
         sweep: pp={} tg={} reps={} warmup={} depths={:?}",
        args.pp, args.tg, args.reps, args.warmup, depths,
    );

    // ── Sweep: test-major (all pp depths, then all tg depths) ────────────
    let mut tests: Vec<(bool, u32)> = Vec::new(); // (is_pp, n)
    if do_pp {
        tests.push((true, args.pp));
    }
    if do_tg {
        tests.push((false, args.tg));
    }

    let mut rows: Vec<[String; 6]> = Vec::new();
    for (is_pp, n) in tests {
        let kind = if is_pp { "pp" } else { "tg" };
        for &depth in &depths {
            // Prefill to `depth` ONCE (untimed), then snapshot the recurrent
            // state and restore it between reps — far cheaper than re-prefilling
            // `depth` tokens every rep, and correct for hybrid SSM models (the
            // timed window only writes attention K/V at [depth, depth+n), which
            // the restored `position` hides; the SSM region is restored exactly).
            cache.reset();
            sampler.reset_recent();
            feed_kv(&mut engine, &*model, &mut cache, &dummies, depth)?; // UNTIMED prefill
            let snap = cache.snapshot_state();

            let mut samples = Vec::with_capacity(args.reps as usize);
            for it in 0..(args.warmup + args.reps) {
                cache.restore_state(&snap);
                sampler.reset_recent();
                let t0 = Instant::now();
                if is_pp {
                    feed_kv(&mut engine, &*model, &mut cache, &dummies, n)?;
                } else {
                    for _ in 0..n {
                        let pos = cache.position;
                        engine.forward_sampled(
                            &*model,
                            &mut cache,
                            &dummies[..1],
                            pos,
                            &mut sampler,
                        )?;
                    }
                }
                let secs = t0.elapsed().as_secs_f64();
                if it >= args.warmup {
                    samples.push(n as f64 / secs.max(1e-9));
                }
            }
            let m = mean(&samples);
            let sd = stddev(&samples);
            rows.push([
                model_name.clone(),
                size_str.clone(),
                params_str.clone(),
                backend.clone(),
                test_label(kind, n, depth),
                format!("{m:.2} \u{00b1} {sd:.2}"),
            ]);
        }
    }

    print_table(
        ["model", "size", "params", "backend", "test", "t/s"],
        [false, true, true, false, false, true], // right-align numeric cols
        &rows,
    );
    Ok(())
}

/// Feed `n` synthetic tokens into the cache as a pure KV forward (no logits /
/// sampling), chunked by `n_ubatch` so scratch stays bounded on long prefills.
/// `forward_kv_only` does not chunk internally, so we do it here (the same
/// pattern `probe.rs` uses for chunked --dump-logits prefill).
fn feed_kv(
    engine: &mut Engine,
    model: &dyn crate::models::Model,
    cache: &mut KvCache,
    dummies: &[u32],
    n: u32,
) -> Result<(), Box<dyn Error>> {
    let total = n as usize;
    if total == 0 {
        return Ok(());
    }
    let ub = engine.n_ubatch as usize;
    let chunk = if ub == 0 { total } else { ub };
    let mut done = 0usize;
    while done < total {
        let take = chunk.min(total - done);
        let pos = cache.position;
        engine.forward_kv_only(model, cache, &dummies[..take], pos)?;
        done += take;
    }
    Ok(())
}

/// Build the depth list, filtered + sorted + deduped, plus the list of skipped
/// depths with a reason (for the warning printout).
fn build_depths(
    args: &BenchArgs,
    trained_ctx: Option<u32>,
    span: u32,
) -> (Vec<u32>, Vec<(u32, String)>) {
    let explicit = args.depths.is_some();
    let raw: Vec<u32> = match &args.depths {
        Some(s) => s
            .split(',')
            .filter_map(|x| x.trim().parse::<u32>().ok())
            .collect(),
        None => DEFAULT_LADDER.to_vec(),
    };

    let mut kept = Vec::new();
    let mut skipped = Vec::new();
    for d in raw {
        // The 64K cap applies only to the auto-ladder; an explicit --depths
        // list is treated as an intentional opt-in past it.
        if !explicit && d > DEFAULT_MAX_DEPTH {
            skipped.push((
                d,
                "above 64K default cap (use --depths to opt in)".to_string(),
            ));
            continue;
        }
        if let Some(tc) = trained_ctx
            && d.saturating_add(span) > tc
        {
            skipped.push((d, format!("d+{span} exceeds trained context {tc}")));
            continue;
        }
        kept.push(d);
    }
    kept.sort_unstable();
    kept.dedup();
    (kept, skipped)
}

fn greedy_config() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_penalty: 1.0,
        penalty_last_n: 0,
        seed: 0,
        logit_bias: Vec::new(),
    }
}

fn test_label(kind: &str, n: u32, depth: u32) -> String {
    if depth == 0 {
        format!("{kind}{n}")
    } else {
        format!("{kind}{n} @ d{depth}")
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Sample standard deviation (n-1), matching llama-bench's `stdev`.
fn stddev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

/// Total parameter count = Σ (product of each tensor's dims). llama.cpp counts
/// the same way (sum of `ggml_nelements`); no metadata key is reliably present.
fn count_params(gguf: &GgufFile) -> u64 {
    gguf.tensors()
        .iter()
        .map(|t| t.dims.iter().product::<u64>())
        .sum()
}

fn model_name(gguf: &GgufFile) -> String {
    match gguf.get("general.name") {
        Some(MetadataValue::String(s)) => s.clone(),
        _ => gguf.architecture().unwrap_or("unknown").to_string(),
    }
}

fn human_params(n: u64) -> String {
    let nf = n as f64;
    if nf >= 1e9 {
        format!("{:.2} B", nf / 1e9)
    } else if nf >= 1e6 {
        format!("{:.2} M", nf / 1e6)
    } else if nf >= 1e3 {
        format!("{:.2} K", nf / 1e3)
    } else {
        format!("{n}")
    }
}

fn human_size(bytes: u64) -> String {
    let b = bytes as f64;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else {
        format!("{:.2} MiB", b / MIB)
    }
}

/// Print a markdown-style aligned table. `right` marks columns to right-align.
/// Widths are computed by char count so the `±` (one char) aligns correctly.
fn print_table(headers: [&str; 6], right: [bool; 6], rows: &[[String; 6]]) {
    let mut w = [0usize; 6];
    for (wi, h) in w.iter_mut().zip(headers.iter()) {
        *wi = h.chars().count();
    }
    for r in rows {
        for (wi, c) in w.iter_mut().zip(r.iter()) {
            *wi = (*wi).max(c.chars().count());
        }
    }
    let line = |cells: &[String; 6]| {
        let mut out = String::from("|");
        for ((cell, &width), &r) in cells.iter().zip(w.iter()).zip(right.iter()) {
            out.push(' ');
            out.push_str(&pad(cell, width, r));
            out.push_str(" |");
        }
        out
    };
    let header_cells: [String; 6] = std::array::from_fn(|i| headers[i].to_string());
    let sep_cells: [String; 6] = std::array::from_fn(|i| "-".repeat(w[i]));
    println!("{}", line(&header_cells));
    println!("{}", line(&sep_cells));
    for r in rows {
        println!("{}", line(r));
    }
}

fn pad(s: &str, w: usize, right: bool) -> String {
    let len = s.chars().count();
    let fill = " ".repeat(w.saturating_sub(len));
    if right {
        format!("{fill}{s}")
    } else {
        format!("{s}{fill}")
    }
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
