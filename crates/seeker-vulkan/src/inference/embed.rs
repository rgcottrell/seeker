//! Embedding post-processing + the [`TextEmbedder`](seeker_core::embed::TextEmbedder)
//! trait moved to `seeker-core` (backend-neutral, shared with future backends).
//!
//! This re-export keeps existing `crate::inference::embed::*` paths (CLI, serve)
//! resolving unchanged. The Vulkan implementation of `TextEmbedder` lives in
//! [`crate::inference::embedder`].
pub use seeker_core::embed::*;
