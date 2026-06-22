//! Backend-neutral shared layer for seeker.
//!
//! Everything here is independent of any compute backend (no Vulkan, no
//! `ash`): GGUF parsing, tokenizers, chat templating, runtime flags, and
//! host-side image/audio decode. The Vulkan backend (`seeker-vulkan`) and the
//! CLI app (`seeker-cli`) build on top of this; a future non-Vulkan backend
//! becomes a sibling crate that reuses it.

pub mod chat_template;
pub mod gguf;
pub mod runtime_flags;
pub mod tokenizer;

pub mod vision {
    //! Host-side vision preprocessing (the GPU encoder lives in `seeker-vulkan`).
    pub mod preprocess;
}

pub mod audio {
    //! Host-side audio decode (the GPU encoder lives in `seeker-vulkan`).
    pub mod decode;
}
