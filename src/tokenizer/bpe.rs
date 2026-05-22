//! GPT-2-style BPE path: vocab as token→id map, merges as `(left, right)`
//! pairs, ByteLevel pre-tokenizer/decoder. Used for GPT-2 / GPT-J / Mistral
//! / Phi / Qwen / SmolLM and friends — anything where
//! `tokenizer.ggml.model == "gpt2"`.

use std::error::Error;

use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{Vocab as BpeVocab, BPE};
use tokenizers::models::ModelWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;

use crate::gguf::GgufFile;
use crate::tokenizer::bundle::Tokenizer;
use crate::tokenizer::error::BuildError;
use crate::tokenizer::metadata::read_string_array;

pub(super) fn build(tokens: &[String], gguf: &GgufFile) -> Result<Tokenizer, BuildError> {
    let merges_raw = read_string_array(gguf, "tokenizer.ggml.merges").unwrap_or_default();
    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| BuildError::BadMerge(m.clone()))
        })
        .collect::<Result<_, _>>()?;

    // tokenizers' BPE vocab is an `AHashMap<String, u32>` aliased as `BpeVocab`;
    // collect directly into that to avoid a HashMap→AHashMap conversion that
    // doesn't impl `From`.
    let vocab: BpeVocab = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| BuildError::Inner(e as Box<dyn Error + Send + Sync>))?;

    let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(bpe));
    let bl = ByteLevel::new(/* add_prefix_space */ false, /* trim_offsets */ true, /* use_regex */ true);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(bl.clone())));
    tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(bl)));
    Ok(tokenizer)
}
