//! Small typed accessors over `GgufFile::get` that surface a `TokenizerError` with
//! the offending field name. The tokenizer builder calls these instead of
//! matching on `MetadataValue` directly so the dispatch + error message lives
//! in one place.

use crate::gguf::{GgufFile, MetadataValue};
use crate::tokenizer::error::TokenizerError;

pub(super) fn read_string(gguf: &GgufFile, key: &'static str) -> Result<String, TokenizerError> {
    match gguf.get(key) {
        Some(MetadataValue::String(s)) => Ok(s.clone()),
        Some(_) => Err(TokenizerError::WrongFieldType(key)),
        None => Err(TokenizerError::MissingField(key)),
    }
}

pub(super) fn read_string_array(
    gguf: &GgufFile,
    key: &'static str,
) -> Result<Vec<String>, TokenizerError> {
    let arr = match gguf.get(key) {
        Some(MetadataValue::Array(a)) => a,
        Some(_) => return Err(TokenizerError::WrongFieldType(key)),
        None => return Err(TokenizerError::MissingField(key)),
    };
    arr.iter()
        .map(|v| match v {
            MetadataValue::String(s) => Ok(s.clone()),
            _ => Err(TokenizerError::WrongFieldType(key)),
        })
        .collect()
}

pub(super) fn read_f32_array(gguf: &GgufFile, key: &'static str) -> Result<Vec<f32>, TokenizerError> {
    let arr = match gguf.get(key) {
        Some(MetadataValue::Array(a)) => a,
        Some(_) => return Err(TokenizerError::WrongFieldType(key)),
        None => return Err(TokenizerError::MissingField(key)),
    };
    arr.iter()
        .map(|v| match v {
            MetadataValue::F32(f) => Ok(*f),
            MetadataValue::F64(f) => Ok(*f as f32),
            _ => Err(TokenizerError::WrongFieldType(key)),
        })
        .collect()
}

pub(super) fn read_optional_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    match gguf.get(key)? {
        MetadataValue::U32(v) => Some(*v),
        MetadataValue::I32(v) if *v >= 0 => Some(*v as u32),
        MetadataValue::U64(v) => u32::try_from(*v).ok(),
        _ => None,
    }
}

pub(super) fn read_optional_bool(gguf: &GgufFile, key: &str) -> Option<bool> {
    match gguf.get(key)? {
        MetadataValue::Bool(b) => Some(*b),
        _ => None,
    }
}

pub(super) fn read_optional_i32_array(gguf: &GgufFile, key: &str) -> Option<Vec<i32>> {
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
