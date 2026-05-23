//! HTTP handlers grouped by API surface. Each handler is a thin shim:
//! `axum::extract::Json<Request>` in, typed response out. The actual stub
//! payloads live on each response struct's `stub()` constructor.

pub mod anthropic;
pub mod llama;
pub mod ops;
pub mod openai;
