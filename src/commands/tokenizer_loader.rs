//! Build a `tokenizers::Tokenizer` from a GGUF file's embedded `tokenizer.ggml.*`
//! metadata. The GGUF is self-sufficient — we don't fetch an external
//! `tokenizer.json` — at the cost of having to map the GGUF schema onto the
//! `tokenizers` crate's component model ourselves.
//!
//! Two paths today:
//!   - `gpt2`  → `BPE` + `ByteLevel` pre-tokenizer/decoder (GPT-2 family,
//!     Mistral / Phi / Qwen-style models that follow GPT-2 conventions).
//!   - `llama` → `Unigram` + `Metaspace` pre-tokenizer/decoder
//!     (SentencePiece-style; LLaMA, Llama 2, CodeLlama, Mistral v0.1, …).
//!
//! Anything else returns an explanatory error so the caller can render a
//! useful message rather than panic.
//!
//! The returned `TokenizerBundle` carries the assembled `Tokenizer` plus the
//! BOS/EOS ids and the GGUF's "should we add BOS/EOS by default" flags. The
//! callers decide how to honor those.

use std::error::Error;
use std::fmt;

use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{Vocab as BpeVocab, BPE};
use tokenizers::models::unigram::Unigram;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::tokenizer::AddedToken;
use tokenizers::TokenizerImpl;

use crate::gguf::{GgufFile, MetadataValue};

/// `tokenizers::Tokenizer` is just `TokenizerImpl<Wrappers…>`; the concrete
/// alias keeps signatures readable without dragging the bound everywhere.
pub type Tokenizer = TokenizerImpl<
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// Output of `build_tokenizer`. `tokenizer` is ready to encode/decode. The
/// `add_bos_default`/`add_eos_default` mirror the GGUF flags so the caller
/// can pick a sensible default when the user didn't pass `--add-special`.
/// `bos_id`/`eos_id` are informational — exposed for future callers that
/// want to surface them in output.
#[allow(dead_code)]
pub struct TokenizerBundle {
    pub tokenizer: Tokenizer,
    pub model_kind: String,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub add_bos_default: bool,
    pub add_eos_default: bool,
}

#[derive(Debug)]
pub enum BuildError {
    MissingField(&'static str),
    WrongFieldType(&'static str),
    UnsupportedModel(String),
    EmptyVocab,
    BadMerge(String),
    Inner(Box<dyn Error + Send + Sync>),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(k) => write!(f, "GGUF is missing required field `{k}`"),
            Self::WrongFieldType(k) => write!(f, "GGUF field `{k}` has unexpected type"),
            Self::UnsupportedModel(m) => write!(
                f,
                "unsupported tokenizer.ggml.model: `{m}`; only `gpt2` (BPE) and `llama` (Unigram) are wired up"
            ),
            Self::EmptyVocab => write!(f, "tokenizer.ggml.tokens is empty"),
            Self::BadMerge(m) => write!(f, "malformed BPE merge `{m}` (expected `left right`)"),
            Self::Inner(e) => write!(f, "tokenizers crate error: {e}"),
        }
    }
}

impl Error for BuildError {}

pub fn build_tokenizer(gguf: &GgufFile) -> Result<TokenizerBundle, BuildError> {
    let model_kind = read_string(gguf, "tokenizer.ggml.model")?;
    let tokens = read_string_array(gguf, "tokenizer.ggml.tokens")?;
    if tokens.is_empty() {
        return Err(BuildError::EmptyVocab);
    }

    let bos_id = read_optional_u32(gguf, "tokenizer.ggml.bos_token_id");
    let eos_id = read_optional_u32(gguf, "tokenizer.ggml.eos_token_id");
    let unk_id = read_optional_u32(gguf, "tokenizer.ggml.unknown_token_id");
    let add_bos_default = read_optional_bool(gguf, "tokenizer.ggml.add_bos_token").unwrap_or(false);
    let add_eos_default = read_optional_bool(gguf, "tokenizer.ggml.add_eos_token").unwrap_or(false);

    let mut tokenizer = match model_kind.as_str() {
        "gpt2" => build_bpe(&tokens, gguf)?,
        "llama" => build_unigram(&tokens, gguf, unk_id)?,
        other => return Err(BuildError::UnsupportedModel(other.to_string())),
    };

    // Mark BOS/EOS/UNK as special tokens by id so they survive
    // `decode(skip_special_tokens = true)` correctly.
    let bos_str = bos_id.and_then(|i| tokens.get(i as usize).cloned());
    let eos_str = eos_id.and_then(|i| tokens.get(i as usize).cloned());
    let unk_str = unk_id.and_then(|i| tokens.get(i as usize).cloned());
    let mut specials: Vec<AddedToken> = Vec::new();
    for s in [&bos_str, &eos_str, &unk_str].into_iter().flatten() {
        specials.push(AddedToken::from(s.clone(), true));
    }
    if !specials.is_empty() {
        tokenizer.add_special_tokens(&specials);
    }

    // Set up a post-processor so encode(add_special_tokens=true) actually
    // prepends BOS / appends EOS. Without this, the boolean is silently a
    // no-op. We register specials by string id (matching tokens[i]) so the
    // template piece looks up the right vocab entry.
    if bos_str.is_some() || eos_str.is_some() {
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
            .and_then(|b| Ok(b.special_tokens(special_pairs.clone())))
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

    Ok(TokenizerBundle {
        tokenizer,
        model_kind,
        bos_id,
        eos_id,
        add_bos_default,
        add_eos_default,
    })
}

fn build_bpe(tokens: &[String], gguf: &GgufFile) -> Result<Tokenizer, BuildError> {
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

fn build_unigram(
    tokens: &[String],
    gguf: &GgufFile,
    unk_id: Option<u32>,
) -> Result<Tokenizer, BuildError> {
    let scores = read_f32_array(gguf, "tokenizer.ggml.scores")?;
    if scores.len() != tokens.len() {
        return Err(BuildError::WrongFieldType("tokenizer.ggml.scores"));
    }
    let token_types = read_optional_i32_array(gguf, "tokenizer.ggml.token_type");
    let byte_fallback = token_types
        .as_ref()
        .is_some_and(|tt| tt.iter().any(|t| *t == 6 /* BYTE */));

    let vocab: Vec<(String, f64)> = tokens
        .iter()
        .zip(scores.iter())
        .map(|(t, s)| (t.clone(), *s as f64))
        .collect();

    let unigram = Unigram::from(vocab, unk_id.map(|u| u as usize), byte_fallback)
        .map_err(|e| BuildError::Inner(e as Box<dyn Error + Send + Sync>))?;

    let mut tokenizer = Tokenizer::new(ModelWrapper::Unigram(unigram));
    let metaspace = Metaspace::new('▁', PrependScheme::Always, /* split */ true);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Metaspace(metaspace.clone())));
    tokenizer.with_decoder(Some(DecoderWrapper::Metaspace(metaspace)));
    Ok(tokenizer)
}

// ── Small GGUF-metadata accessor helpers ─────────────────────────────────────

fn read_string(gguf: &GgufFile, key: &'static str) -> Result<String, BuildError> {
    match gguf.get(key) {
        Some(MetadataValue::String(s)) => Ok(s.clone()),
        Some(_) => Err(BuildError::WrongFieldType(key)),
        None => Err(BuildError::MissingField(key)),
    }
}

fn read_string_array(gguf: &GgufFile, key: &'static str) -> Result<Vec<String>, BuildError> {
    let arr = match gguf.get(key) {
        Some(MetadataValue::Array(a)) => a,
        Some(_) => return Err(BuildError::WrongFieldType(key)),
        None => return Err(BuildError::MissingField(key)),
    };
    arr.iter()
        .map(|v| match v {
            MetadataValue::String(s) => Ok(s.clone()),
            _ => Err(BuildError::WrongFieldType(key)),
        })
        .collect()
}

fn read_f32_array(gguf: &GgufFile, key: &'static str) -> Result<Vec<f32>, BuildError> {
    let arr = match gguf.get(key) {
        Some(MetadataValue::Array(a)) => a,
        Some(_) => return Err(BuildError::WrongFieldType(key)),
        None => return Err(BuildError::MissingField(key)),
    };
    arr.iter()
        .map(|v| match v {
            MetadataValue::F32(f) => Ok(*f),
            MetadataValue::F64(f) => Ok(*f as f32),
            _ => Err(BuildError::WrongFieldType(key)),
        })
        .collect()
}

fn read_optional_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    match gguf.get(key)? {
        MetadataValue::U32(v) => Some(*v),
        MetadataValue::I32(v) if *v >= 0 => Some(*v as u32),
        MetadataValue::U64(v) => u32::try_from(*v).ok(),
        _ => None,
    }
}

fn read_optional_bool(gguf: &GgufFile, key: &str) -> Option<bool> {
    match gguf.get(key)? {
        MetadataValue::Bool(b) => Some(*b),
        _ => None,
    }
}

fn read_optional_i32_array(gguf: &GgufFile, key: &str) -> Option<Vec<i32>> {
    let arr = match gguf.get(key)? {
        MetadataValue::Array(a) => a,
        _ => return None,
    };
    arr.iter()
        .map(|v| match v {
            MetadataValue::I32(i) => Some(*i),
            MetadataValue::U32(u) => Some(*u as i32),
            _ => None,
        })
        .collect()
}
