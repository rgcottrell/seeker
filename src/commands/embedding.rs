//! `seeker embedding` — produce text embeddings, like llama.cpp's
//! `llama-embedding`. Runs the transformer forward, applies the final
//! `output_norm` to every position, pools (last / mean / cls / none), and
//! normalizes (L2 by default). First target: Qwen3-Embedding-0.6B (dense
//! `qwen3`, last-token pooling). Server/HTTP integration is a later step.

use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

use clap::{Args, ValueEnum};

use crate::commands::download::{self, HfResolveArgs, Resolved};
use crate::gguf::GgufFile;
use crate::inference::Engine;
use crate::inference::kv_cache::{KvCacheConfig, parse_dtype};
use crate::tokenizer::build_tokenizer;
use crate::{gguf::GgmlType, models};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Pooling {
    /// Hidden state of the last token (Qwen3-Embedding default).
    Last,
    /// Mean over all token positions.
    Mean,
    /// First token ([CLS]).
    Cls,
    /// No pooling — emit one (L2-normalized) vector per token.
    None,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// JSON-style 2D array `[[...],[...]]` (matches llama-embedding `--embd-output-format array`).
    Array,
    /// OpenAI-style `{"object":"list","data":[{"embedding":[...]},...]}`.
    Json,
}

#[derive(Args)]
pub struct EmbeddingArgs {
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

    /// HF auth token (defaults to HF_TOKEN env / ~/.cache/huggingface/token).
    #[arg(long = "hf-token", requires = "hf_repo", conflicts_with = "model")]
    hf_token: Option<String>,

    /// Resolve files from the local cache only; never hit the network.
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    offline: bool,

    /// Path to a local .gguf model file.
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,

    /// Text to embed. Repeat for multiple inputs.
    #[arg(long = "prompt", short = 'p')]
    prompt: Vec<String>,

    /// Read inputs from a file, one per line (`-` = stdin). Combined with any
    /// `--prompt` values.
    #[arg(long = "prompt-file", short = 'f')]
    prompt_file: Option<PathBuf>,

    /// Pooling over token hidden states. Defaults to the GGUF `*.pooling_type`
    /// (Qwen3-Embedding = last).
    #[arg(long = "pooling", value_enum)]
    pooling: Option<Pooling>,

    /// Embedding normalization (llama.cpp `--embd-normalize`): -1 none, 0 max-abs,
    /// 1 taxicab/L1, 2 euclidean/L2 (default), p>2 p-norm.
    #[arg(
        long = "embd-normalize",
        default_value_t = 2,
        allow_negative_numbers = true
    )]
    embd_normalize: i32,

    /// Output format.
    #[arg(long = "embd-output-format", value_enum, default_value_t = OutputFormat::Array)]
    embd_output_format: OutputFormat,

    /// Separator printed between embeddings.
    #[arg(long = "embd-separator", default_value = "\n")]
    embd_separator: String,

    /// Print an N×N cosine-similarity matrix instead of the raw embeddings.
    #[arg(long = "sim")]
    sim: bool,

    /// KV cache K dtype (f16 default).
    #[arg(long = "cache-type-k", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_k: GgmlType,

    /// KV cache V dtype (f16 default).
    #[arg(long = "cache-type-v", default_value = "f16", value_parser = parse_dtype_arg)]
    cache_type_v: GgmlType,

    /// Physical micro-batch size for prefill (0 = single pass). Short: -ub.
    #[arg(long = "ubatch-size", default_value_t = 512)]
    ubatch_size: u32,
}

fn parse_dtype_arg(s: &str) -> Result<GgmlType, String> {
    parse_dtype(s)
}

pub async fn run(args: EmbeddingArgs) -> Result<(), Box<dyn Error>> {
    // ---- resolve model + load ----
    let resolved: Resolved = match (args.hf_repo.clone(), args.model.clone()) {
        (Some(repo), None) => {
            download::resolve_hf(
                &HfResolveArgs {
                    repo,
                    file: args.hf_file.clone(),
                    token: args.hf_token.clone(),
                    offline: args.offline,
                },
                /*want_mmproj=*/ false,
            )
            .await?
        }
        (None, Some(model)) => Resolved {
            main: model,
            mmproj: None,
        },
        _ => unreachable!("clap group invariant"),
    };

    let gguf = GgufFile::open(&resolved.main)?;
    let bundle = build_tokenizer(&gguf)?;
    if !bundle.add_eos_default {
        tracing::warn!(
            "model does not set tokenizer.ggml.add_eos_token; last-token pooling may not land \
             on the intended EOS position"
        );
    }

    let mut engine = Engine::new(args.ubatch_size, 2048)?;
    tracing::info!(device = %engine.device.name(), "vulkan device opened for embedding");
    let weights = engine.upload_weights(&gguf)?;
    let model = models::open(&gguf, weights, bundle, /*spec_enabled=*/ false)?;

    // Final-norm weights + eps, read host-side (norm tensors are F32 in both the
    // F16 and Q8_0 GGUFs). The arch prefix is whatever the GGUF declares.
    let arch = gguf.architecture().unwrap_or("");
    let n_embd = gguf
        .meta_u32(&format!("{arch}.embedding_length"))
        .ok_or("missing <arch>.embedding_length")? as usize;
    let rms_eps = gguf
        .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    let on_bytes = gguf
        .tensor_data("output_norm.weight")
        .ok_or("missing output_norm.weight")?;
    if on_bytes.len() != n_embd * 4 {
        return Err(format!(
            "output_norm.weight is {} bytes, expected {} (F32 [{n_embd}]); non-F32 norm tensors \
             are not supported",
            on_bytes.len(),
            n_embd * 4
        )
        .into());
    }
    let output_norm: Vec<f32> = on_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // ---- pooling default from GGUF (3 = last for Qwen3-Embedding) ----
    let pooling = args.pooling.unwrap_or_else(|| {
        match gguf.meta_u32(&format!("{arch}.pooling_type")) {
            Some(1) => Pooling::Mean,
            Some(2) => Pooling::Cls,
            Some(0) => Pooling::None,
            _ => Pooling::Last, // 3 (last) or unspecified
        }
    });

    // ---- collect inputs ----
    let mut inputs: Vec<String> = args.prompt.clone();
    if let Some(pf) = &args.prompt_file {
        let text = if pf.as_os_str() == "-" {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        } else {
            std::fs::read_to_string(pf)?
        };
        // Keep every line (including blanks) so output rows / --sim indices stay
        // aligned with the source file; a blank line still tokenizes to its
        // special tokens and yields a (degenerate) embedding.
        inputs.extend(text.lines().map(str::to_string));
    }
    if inputs.is_empty() {
        return Err("no input — pass --prompt/-p (repeatable) or --prompt-file/-f".into());
    }

    // Tokenize all inputs first so scratch + KV can be sized for the longest.
    let token_seqs: Vec<Vec<u32>> = inputs
        .iter()
        .map(|t| {
            model
                .tokenizer()
                .tokenizer
                .encode(t.as_str(), /*add_special=*/ true)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| -> Box<dyn Error> { format!("tokenize failed: {e}").into() })
        })
        .collect::<Result<_, _>>()?;
    let max_len = token_seqs.iter().map(|t| t.len()).max().unwrap_or(0).max(1) as u32;

    // ---- allocate scratch + KV once at the max input length ----
    let scratch = model.scratch_bytes_estimate(
        args.ubatch_size,
        max_len,
        args.cache_type_k,
        args.cache_type_v,
    );
    engine.allocate_scratch(scratch)?;
    let dims = model.cache_dims();
    let cache_config = KvCacheConfig {
        k_dtype: args.cache_type_k,
        v_dtype: args.cache_type_v,
        max_seq_len: max_len,
        n_head: dims.n_head,
    };
    let mut cache =
        engine.allocate_kv_cache(dims.n_layer, dims.head_dim, dims.n_head_kv, cache_config)?;

    // ---- per-input forward → pool → normalize ----
    // For Pooling::None each input expands to L vectors.
    let mut all: Vec<Vec<f32>> = Vec::new();
    for tokens in &token_seqs {
        if tokens.is_empty() {
            return Err("empty input after tokenization".into());
        }
        cache.reset();
        let (_logits, residual) = engine
            .forward_full_readback(&*model, &mut cache, tokens, 0, /*full_logits=*/ false)?;
        let l = residual.len() / n_embd;
        // output_norm applied per position (llama.cpp order: norm-all then pool).
        let normed: Vec<Vec<f32>> = (0..l)
            .map(|t| {
                rmsnorm_col(
                    &residual[t * n_embd..(t + 1) * n_embd],
                    &output_norm,
                    rms_eps,
                )
            })
            .collect();
        for mut v in pool(&normed, pooling) {
            normalize(&mut v, args.embd_normalize);
            all.push(v);
        }
    }

    // ---- output ----
    if args.sim {
        print_similarity_matrix(&all);
    } else {
        print_embeddings(&all, args.embd_output_format, &args.embd_separator);
    }
    eprintln!(
        "embeddings: {} vector(s) of dim {} (pooling {:?}, normalize {})",
        all.len(),
        n_embd,
        pooling,
        args.embd_normalize
    );
    Ok(())
}

// ─── pure host-side math (unit-tested) ───────────────────────────────

/// RMSNorm a single hidden-state column with the learned weight (matches the
/// `rms_norm.slang` kernel: `x / sqrt(mean(x²)+eps) * w`).
fn rmsnorm_col(col: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = col.len() as f32;
    let ms = col.iter().map(|x| x * x).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    col.iter().zip(weight).map(|(x, w)| x * inv * w).collect()
}

/// Pool the per-position (already output_norm'd) vectors into the embedding(s).
fn pool(normed: &[Vec<f32>], pooling: Pooling) -> Vec<Vec<f32>> {
    match pooling {
        Pooling::Last => vec![normed.last().cloned().unwrap_or_default()],
        Pooling::Cls => vec![normed.first().cloned().unwrap_or_default()],
        Pooling::Mean => {
            let l = normed.len().max(1);
            let dim = normed.first().map(Vec::len).unwrap_or(0);
            let mut acc = vec![0.0f32; dim];
            for v in normed {
                for (a, x) in acc.iter_mut().zip(v) {
                    *a += *x;
                }
            }
            for a in &mut acc {
                *a /= l as f32;
            }
            vec![acc]
        }
        Pooling::None => normed.to_vec(),
    }
}

/// In-place embedding normalization, matching llama.cpp `common_embd_normalize`:
/// p<0 none, 0 max-abs, 1 L1/taxicab, 2 L2/euclidean, p>2 p-norm.
fn normalize(v: &mut [f32], p: i32) {
    let sum: f64 = match p {
        i32::MIN..=-1 => 1.0,
        0 => v.iter().fold(0.0f64, |m, x| m.max(x.abs() as f64)),
        1 => v.iter().map(|x| x.abs() as f64).sum(),
        2 => v
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt(),
        p => {
            let pf = p as f64;
            v.iter()
                .map(|x| (x.abs() as f64).powf(pf))
                .sum::<f64>()
                .powf(1.0 / pf)
        }
    };
    let norm = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for x in v.iter_mut() {
        *x = (*x as f64 * norm) as f32;
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn print_embeddings(all: &[Vec<f32>], fmt: OutputFormat, sep: &str) {
    match fmt {
        OutputFormat::Array => {
            let rows: Vec<String> = all
                .iter()
                .map(|v| {
                    let body = v
                        .iter()
                        .map(|x| format!("{x:.6}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("[{body}]")
                })
                .collect();
            println!("[{}]", rows.join(&format!(",{sep}")));
        }
        OutputFormat::Json => {
            let data: Vec<String> = all
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let body = v
                        .iter()
                        .map(|x| format!("{x:.6}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("{{\"object\":\"embedding\",\"index\":{i},\"embedding\":[{body}]}}")
                })
                .collect();
            println!("{{\"object\":\"list\",\"data\":[{}]}}", data.join(","));
        }
    }
}

fn print_similarity_matrix(all: &[Vec<f32>]) {
    for a in all {
        let row: Vec<String> = all.iter().map(|b| format!("{:.4}", cosine(a, b))).collect();
        println!("{}", row.join(" "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_col_matches_hand_compute() {
        // x=[1,2,3], w=[1,1,1], eps=0 → ms=14/3, inv=1/sqrt(14/3).
        let out = rmsnorm_col(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 0.0);
        let inv = 1.0f32 / (14.0f32 / 3.0).sqrt();
        for (o, x) in out.iter().zip([1.0, 2.0, 3.0]) {
            assert!((o - x * inv).abs() < 1e-6, "{o} vs {}", x * inv);
        }
    }

    #[test]
    fn rmsnorm_col_applies_weight() {
        let out = rmsnorm_col(&[1.0, 1.0], &[2.0, 4.0], 0.0);
        // ms=1, inv=1 → out = [2,4].
        assert!((out[0] - 2.0).abs() < 1e-6 && (out[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn pool_selects_right_column() {
        let cols = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0]];
        assert_eq!(pool(&cols, Pooling::Last)[0], vec![3.0, 0.0]);
        assert_eq!(pool(&cols, Pooling::Cls)[0], vec![1.0, 0.0]);
        assert_eq!(pool(&cols, Pooling::Mean)[0], vec![2.0, 0.0]);
        assert_eq!(pool(&cols, Pooling::None).len(), 3);
    }

    #[test]
    fn normalize_l2_is_unit() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v, 2);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_l1_and_none() {
        let mut v = vec![1.0, 3.0];
        normalize(&mut v, 1);
        assert!((v[0] - 0.25).abs() < 1e-6 && (v[1] - 0.75).abs() < 1e-6);
        let mut w = vec![1.0, 3.0];
        normalize(&mut w, -1);
        assert_eq!(w, vec![1.0, 3.0]);
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 1.0], &[2.0, 2.0]) - 1.0).abs() < 1e-6);
    }
}
