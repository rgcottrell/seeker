use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("not a GGUF file: bad magic {0:?}")]
    BadMagic([u8; 4]),

    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),

    #[error("truncated file at byte {at}")]
    Truncated { at: usize },

    #[error("invalid UTF-8 in {field}")]
    BadUtf8 { field: &'static str },

    #[error("unknown metadata value type tag {0}")]
    UnknownValueType(u32),

    #[error("unknown ggml tensor type {0}")]
    UnknownGgmlType(u32),

    #[error("tensor {name:?}: dim product {elements} is not a multiple of block size {block}")]
    BadTensorShape {
        name: String,
        elements: u64,
        block: usize,
    },
}
