//! HTTP server shared state.
//!
//! Today this is just the chat-template + special-token strings needed by
//! `/apply-template`. As more handlers grow real implementations (the
//! OpenAI / llama-server stubs eventually call into inference) they'll
//! pull additional resources — engine, model, sampler defaults — through
//! this same struct.

use std::sync::Arc;

/// Cheaply cloneable per-request handle. `Arc`-wrapping keeps the inner
/// strings cheap to share across the threadpool that axum's
/// `tokio::spawn`-per-request model produces.
#[derive(Clone, Default)]
pub struct AppState {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    /// Raw jinja2 chat template from the loaded model's GGUF, if any.
    chat_template: Option<String>,
    /// String form of the BOS / EOS tokens — most chat templates reference
    /// these as `{{ bos_token }}` / `{{ eos_token }}`.
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl AppState {
    pub fn new(
        chat_template: Option<String>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                chat_template,
                bos_token,
                eos_token,
            }),
        }
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.inner.chat_template.as_deref()
    }

    pub fn bos_token(&self) -> Option<&str> {
        self.inner.bos_token.as_deref()
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.inner.eos_token.as_deref()
    }
}
