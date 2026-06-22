//! Wire-level DTOs grouped by API surface. Every endpoint deserializes its
//! request body into one of these and serializes one back — no `serde_json::Value`
//! escape hatches in handler signatures, so future inference work has typed
//! anchor points to bind to.

pub mod anthropic;
pub mod common;
pub mod llama;
pub mod openai;
