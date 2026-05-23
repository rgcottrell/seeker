//! Build a `tokenizers::Tokenizer` from a GGUF file's embedded `tokenizer.ggml.*`
//! metadata. The GGUF is self-sufficient — we don't fetch an external
//! `tokenizer.json` — at the cost of mapping the GGUF schema onto the
//! `tokenizers` crate's component model ourselves.
//!
//! Two paths today:
//!   - `gpt2`  → `BPE` + `ByteLevel` pre-tokenizer/decoder (GPT-2 family,
//!     Mistral / Phi / Qwen / SmolLM-style models).  See [`bpe`].
//!   - `llama` → `Unigram` + `Metaspace` pre-tokenizer/decoder
//!     (SentencePiece-style; LLaMA, Llama 2, CodeLlama, Mistral v0.1, …).
//!     See [`unigram`].
//!
//! Anything else returns [`TokenizerError::UnsupportedModel`] so callers can render
//! a useful message rather than panic.

mod bpe;
mod bundle;
mod error;
mod metadata;
mod unigram;

pub use bundle::{Tokenizer, TokenizerBundle};
pub use error::TokenizerError;

use tokenizers::processors::template::TemplateProcessing;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::tokenizer::AddedToken;

use crate::gguf::GgufFile;
use crate::tokenizer::metadata::{
    read_optional_bool, read_optional_string, read_optional_u32, read_string, read_string_array,
};

pub fn build_tokenizer(gguf: &GgufFile) -> Result<TokenizerBundle, TokenizerError> {
    let model_kind = read_string(gguf, "tokenizer.ggml.model")?;
    let tokens = read_string_array(gguf, "tokenizer.ggml.tokens")?;
    if tokens.is_empty() {
        return Err(TokenizerError::EmptyVocab);
    }

    let bos_id = read_optional_u32(gguf, "tokenizer.ggml.bos_token_id");
    let eos_id = read_optional_u32(gguf, "tokenizer.ggml.eos_token_id");
    let unk_id = read_optional_u32(gguf, "tokenizer.ggml.unknown_token_id");
    let add_bos_default = read_optional_bool(gguf, "tokenizer.ggml.add_bos_token").unwrap_or(false);
    let add_eos_default = read_optional_bool(gguf, "tokenizer.ggml.add_eos_token").unwrap_or(false);

    let bos_token = bos_id.and_then(|i| tokens.get(i as usize).cloned());
    let eos_token = eos_id.and_then(|i| tokens.get(i as usize).cloned());
    let chat_template = read_optional_string(gguf, "tokenizer.chat_template");

    let mut tokenizer = match model_kind.as_str() {
        "gpt2" => bpe::build(&tokens, gguf)?,
        "llama" => unigram::build(&tokens, gguf, unk_id)?,
        other => return Err(TokenizerError::UnsupportedModel(other.to_string())),
    };

    install_specials(&mut tokenizer, &tokens, bos_id, eos_id, unk_id);

    Ok(TokenizerBundle {
        tokenizer,
        model_kind,
        bos_id,
        eos_id,
        add_bos_default,
        add_eos_default,
        chat_template,
        bos_token,
        eos_token,
    })
}

/// Register BOS/EOS/UNK as added special tokens (so `decode(skip_special=true)`
/// strips them) and install a `TemplateProcessing` post-processor so that
/// `encode(add_special_tokens=true)` actually prepends BOS / appends EOS.
/// Without the post-processor, the boolean is silently a no-op.
fn install_specials(
    tokenizer: &mut Tokenizer,
    tokens: &[String],
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
) {
    let bos_str = bos_id.and_then(|i| tokens.get(i as usize).cloned());
    let eos_str = eos_id.and_then(|i| tokens.get(i as usize).cloned());
    let unk_str = unk_id.and_then(|i| tokens.get(i as usize).cloned());

    let specials: Vec<AddedToken> = [&bos_str, &eos_str, &unk_str]
        .into_iter()
        .flatten()
        .map(|s| AddedToken::from(s.clone(), true))
        .collect();
    if !specials.is_empty() {
        tokenizer.add_special_tokens(&specials);
    }

    if bos_str.is_none() && eos_str.is_none() {
        return;
    }

    let mut single = String::new();
    let mut special_pairs: Vec<(String, u32)> = Vec::new();
    if let (Some(s), Some(i)) = (bos_str.as_ref(), bos_id) {
        single.push_str(s);
        single.push(' ');
        special_pairs.push((s.clone(), i));
    }
    single.push_str("$A");
    if let (Some(s), Some(i)) = (eos_str.as_ref(), eos_id) {
        single.push(' ');
        single.push_str(s);
        special_pairs.push((s.clone(), i));
    }

    match TemplateProcessing::builder()
        .try_single(single.as_str())
        .map(|b| b.special_tokens(special_pairs))
        .and_then(|b| b.build().map_err(|e| e.to_string()))
    {
        Ok(tp) => {
            tokenizer.with_post_processor(Some(PostProcessorWrapper::Template(tp)));
        }
        Err(e) => {
            tracing::debug!(error = %e, "skipping post-processor (template build failed)");
        }
    }
}
