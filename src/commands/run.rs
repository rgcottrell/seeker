//! `seeker run` — single-shot forward pass: feed the prompt through the
//! model, take logits at the last position, argmax → print the predicted
//! next token. Exits.
//!
//! This is the CLI shim. The actual inference flow lives in
//! `crate::inference` (Vulkan runtime) and `crate::models` (architecture).

use std::error::Error;
use std::path::PathBuf;

use clap::Args;

use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::GgufFile;
use crate::inference::device::Device;
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

    let device = Device::new()?;
    tracing::info!(device = %device.name(), "vulkan device opened");

    let weights = crate::inference::weights::upload(&device, &gguf)?;
    tracing::info!(
        tensors = weights.views.len(),
        bytes = weights.region.cursor,
        "weights uploaded to GPU",
    );

    let _model = crate::models::open(&gguf, weights, bundle)?;

    // TODO: build DispatchContext, model.record_forward(ctx, &tokens),
    //       submit, fence-wait, readback logits, argmax, print.
    println!("prompt: {}", args.prompt);
    println!("tokens: {tokens:?}");
    println!("next:   (forward pass not yet implemented — engine scaffold only)");

    Ok(())
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
