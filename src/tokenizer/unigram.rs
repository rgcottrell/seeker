//! SentencePiece Unigram path: each token has a score, and `▁`-prefixed
//! pieces are split by a Metaspace pre-tokenizer / re-joined by the matching
//! decoder. Used when `tokenizer.ggml.model == "llama"` — Llama / Llama 2
//! / CodeLlama / Mistral v0.1 / etc.

use std::error::Error;

use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::models::unigram::Unigram;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

use crate::gguf::GgufFile;
use crate::tokenizer::bundle::Tokenizer;
use crate::tokenizer::error::TokenizerError;
use crate::tokenizer::metadata::{read_f32_array, read_optional_i32_array};

pub(super) fn build(
    tokens: &[String],
    gguf: &GgufFile,
    unk_id: Option<u32>,
) -> Result<Tokenizer, TokenizerError> {
    let scores = read_f32_array(gguf, "tokenizer.ggml.scores")?;
    if scores.len() != tokens.len() {
        return Err(TokenizerError::WrongFieldType("tokenizer.ggml.scores"));
    }
    let token_types = read_optional_i32_array(gguf, "tokenizer.ggml.token_type");
    let byte_fallback = token_types.as_ref().is_some_and(|tt| tt.contains(&6));

    let vocab: Vec<(String, f64)> = tokens
        .iter()
        .zip(scores.iter())
        .map(|(t, s)| (t.clone(), *s as f64))
        .collect();

    let unigram = Unigram::from(vocab, unk_id.map(|u| u as usize), byte_fallback)
        .map_err(|e| TokenizerError::Inner(e as Box<dyn Error + Send + Sync>))?;

    let mut tokenizer = Tokenizer::new(ModelWrapper::Unigram(unigram));
    let metaspace = Metaspace::new('▁', PrependScheme::Always, /* split */ true);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Metaspace(metaspace.clone())));
    tokenizer.with_decoder(Some(DecoderWrapper::Metaspace(metaspace)));
    Ok(tokenizer)
}
