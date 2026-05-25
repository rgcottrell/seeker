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
    read_optional_bool, read_optional_i32_array, read_optional_string, read_optional_u32,
    read_string, read_string_array,
};

/// GGUF `tokenizer.ggml.token_type` values we must register as added tokens so
/// the BPE/Unigram pipeline matches them verbatim instead of shredding them
/// into sub-pieces. Mirrors llama.cpp's `LLAMA_TOKEN_TYPE_*`.
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

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
    let token_types = read_optional_i32_array(gguf, "tokenizer.ggml.token_type");

    let mut tokenizer = match model_kind.as_str() {
        "gpt2" => bpe::build(&tokens, gguf)?,
        "llama" => unigram::build(&tokens, gguf, unk_id)?,
        other => return Err(TokenizerError::UnsupportedModel(other.to_string())),
    };

    install_specials(&mut tokenizer, &tokens, token_types.as_deref(), bos_id, eos_id, unk_id);

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

/// Register the model's special tokens as *added tokens* so the BPE/Unigram
/// pipeline matches them verbatim (a single id) instead of shredding them into
/// sub-pieces, and install a `TemplateProcessing` post-processor so that
/// `encode(add_special_tokens=true)` actually prepends BOS / appends EOS.
///
/// Two sources of "special" tokens:
///   - BOS/EOS/UNK by id (always, even when `token_type` is absent).
///   - Every token flagged CONTROL or USER_DEFINED in `tokenizer.ggml.token_type`.
///     Without this, control tokens like Llama-3's `<|start_header_id|>` /
///     `<|end_header_id|>` get split into ~9 junk sub-tokens and the model never
///     sees its chat-header structure.
///
/// CONTROL tokens are registered `special` (so `decode(skip_special=true)`
/// strips them from user-visible output); USER_DEFINED tokens are matched
/// verbatim but kept in decoded output since they carry content.
fn install_specials(
    tokenizer: &mut Tokenizer,
    tokens: &[String],
    token_types: Option<&[i32]>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
) {
    let bos_str = bos_id.and_then(|i| tokens.get(i as usize).cloned());
    let eos_str = eos_id.and_then(|i| tokens.get(i as usize).cloned());
    let unk_str = unk_id.and_then(|i| tokens.get(i as usize).cloned());

    let mut specials: Vec<AddedToken> = [&bos_str, &eos_str, &unk_str]
        .into_iter()
        .flatten()
        .map(|s| AddedToken::from(s.clone(), true))
        .collect();

    // Sweep token_type for CONTROL / USER_DEFINED tokens. add_special_tokens
    // dedups by content, so re-adding BOS/EOS (themselves CONTROL) is harmless.
    if let Some(types) = token_types {
        for (tok, &ty) in tokens.iter().zip(types).filter(|(t, _)| !t.is_empty()) {
            match ty {
                TOKEN_TYPE_CONTROL => specials.push(AddedToken::from(tok.clone(), true)),
                TOKEN_TYPE_USER_DEFINED => {
                    specials.push(AddedToken::from(tok.clone(), false).normalized(false))
                }
                _ => {}
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::{Vocab as BpeVocab, BPE};
    use tokenizers::models::ModelWrapper;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;

    /// Build a ByteLevel+BPE tokenizer with one token per vocab entry and no
    /// merges — mirrors the `gpt2` path closely enough to exercise special-token
    /// extraction. Printable ASCII maps to itself under ByteLevel, so the chars
    /// used below survive round-trip.
    fn bytelevel_bpe(tokens: &[&str]) -> Tokenizer {
        let vocab: BpeVocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap();
        let mut tk = Tokenizer::new(ModelWrapper::BPE(bpe));
        let bl = ByteLevel::new(false, true, true);
        tk.with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(bl)));
        tk
    }

    fn ids(tk: &Tokenizer, text: &str) -> Vec<u32> {
        tk.encode(text, false).unwrap().get_ids().to_vec()
    }

    /// A control-typed token (token_type == 3) must encode to its single vocab
    /// id rather than being shredded into sub-pieces by the BPE pipeline. This
    /// is the Llama-3 `<|start_header_id|>` bug in miniature.
    #[test]
    fn control_token_encodes_as_single_id() {
        let tokens = ["<", "|", "x", ">", "<|x|>"];
        let token_types = [1, 1, 1, 1, 3];

        // Baseline: without registering the control token it shreds.
        let plain = bytelevel_bpe(&tokens);
        assert!(
            ids(&plain, "<|x|>").len() > 1,
            "precondition: unregistered control token should shred"
        );

        // With install_specials it should collapse to the single id 4.
        let mut tk = bytelevel_bpe(&tokens);
        let owned: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
        install_specials(&mut tk, &owned, Some(&token_types), None, None, None);
        assert_eq!(ids(&tk, "<|x|>"), vec![4]);
    }

    /// End-to-end against the real Llama-3.2 GGUF: the chat header control
    /// tokens must each be a single id (128006/128007), not 9 shredded pieces.
    /// `#[ignore]` to keep `cargo test` hermetic — run with
    /// `cargo test -- --ignored` after the HF cache is populated.
    #[test]
    #[ignore]
    fn llama3_header_tokens_are_single_ids() {
        let home = std::env::var("HOME").unwrap();
        let dir = format!(
            "{home}/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots"
        );
        let snapshot = std::fs::read_dir(&dir)
            .expect("cached snapshots dir missing")
            .next()
            .expect("no snapshot")
            .expect("dir entry")
            .path();
        let path = snapshot.join("Llama-3.2-1B-Instruct-Q8_0.gguf");
        let gguf = crate::gguf::GgufFile::open(&path).expect("open gguf");
        let bundle = build_tokenizer(&gguf).expect("build tokenizer");

        let enc = bundle.tokenizer.encode("<|start_header_id|>", false).unwrap();
        assert_eq!(enc.get_ids(), &[128006], "start_header_id must be one token");
        let enc = bundle.tokenizer.encode("<|end_header_id|>", false).unwrap();
        assert_eq!(enc.get_ids(), &[128007], "end_header_id must be one token");
    }
}
