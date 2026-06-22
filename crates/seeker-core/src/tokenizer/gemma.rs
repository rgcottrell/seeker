//! Gemma 4 path (`tokenizer.ggml.model == "gemma4"`): a SentencePiece-style
//! BPE — it uses BPE *merges* (not unigram scores) but tokenizes on raw UTF-8
//! with `▁`-escaped spaces (no GPT-2 ByteLevel byte-encoding), falling back to
//! `<0xXX>` byte tokens. So it needs a third route distinct from both the
//! `gpt2` (BPE + ByteLevel) and `llama` (Unigram + Metaspace-Always) paths:
//!   - BPE model with `byte_fallback` (`<0xXX>` matches llama.cpp's fallback).
//!   - `Metaspace('▁', Never, split=false)` — `add_space_prefix=false` (no
//!     leading `▁`), `escape_whitespaces=true` (space→`▁`), no word-splitting
//!     (merges run across the whole string).
//!
//! Mirrors llama.cpp's `LLAMA_VOCAB_TYPE_BPE` + `tokenizer_pre == "gemma4"`
//! (`llama-vocab.cpp`): merges load-bearing, scores ignored.

use std::error::Error;

use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::models::bpe::{BPE, Vocab as BpeVocab};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

use crate::gguf::GgufFile;
use crate::tokenizer::bundle::Tokenizer;
use crate::tokenizer::error::TokenizerError;
use crate::tokenizer::metadata::read_string_array;

pub(super) fn build(tokens: &[String], gguf: &GgufFile) -> Result<Tokenizer, TokenizerError> {
    let merges_raw = read_string_array(gguf, "tokenizer.ggml.merges").unwrap_or_default();
    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| TokenizerError::BadMerge(m.clone()))
        })
        .collect::<Result<_, _>>()?;

    let vocab: BpeVocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    // byte_fallback: unknown bytes map to the `<0xXX>` tokens (token_type==6,
    // always present in gemma vocabs). The `tokenizers` crate emits exactly
    // `<{b:#04X}>` = uppercase hex, byte-identical to llama.cpp.
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .build()
        .map_err(|e| TokenizerError::Inner(e as Box<dyn Error + Send + Sync>))?;

    let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(bpe));
    // Never => add_space_prefix=false; split=false => no per-`▁` word split.
    let metaspace = Metaspace::new('▁', PrependScheme::Never, /* split */ false);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Metaspace(metaspace.clone())));
    tokenizer.with_decoder(Some(DecoderWrapper::Metaspace(metaspace)));
    Ok(tokenizer)
}
