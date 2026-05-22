use std::error::Error;

use thiserror::Error as ThisError;

/// Failure modes for `tokenizer::build_tokenizer`. The "wrong" / "missing"
/// variants point at GGUF metadata fields by name so the CLI can tell the
/// user exactly which key was off. `Inner` wraps anything that bubbles up
/// from the `tokenizers` crate itself (model construction, template build).
#[derive(Debug, ThisError)]
pub enum TokenizerError {
    #[error("GGUF is missing required field `{0}`")]
    MissingField(&'static str),

    #[error("GGUF field `{0}` has unexpected type")]
    WrongFieldType(&'static str),

    #[error("unsupported tokenizer.ggml.model: `{0}`; only `gpt2` (BPE) and `llama` (Unigram) are wired up")]
    UnsupportedModel(String),

    #[error("tokenizer.ggml.tokens is empty")]
    EmptyVocab,

    #[error("malformed BPE merge `{0}` (expected `left right`)")]
    BadMerge(String),

    #[error("tokenizers crate error: {0}")]
    Inner(#[source] Box<dyn Error + Send + Sync>),
}
