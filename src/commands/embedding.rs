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
use crate::inference::embed::{self, Pooling};
use crate::inference::kv_cache::{KvCacheConfig, parse_dtype};
use crate::tokenizer::build_tokenizer;
use crate::{gguf::GgmlType, models};

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
    let pooling = args
        .pooling
        .unwrap_or_else(|| Pooling::from_gguf(gguf.meta_u32(&format!("{arch}.pooling_type"))));

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
        // output_norm per position, pool, normalize (the shared host-side path).
        all.extend(embed::pool_and_normalize(
            &residual,
            n_embd,
            &output_norm,
            rms_eps,
            pooling,
            args.embd_normalize,
        ));
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

// ─── CLI-only output helpers ─────────────────────────────────────────

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
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 1.0], &[2.0, 2.0]) - 1.0).abs() < 1e-6);
    }
}
