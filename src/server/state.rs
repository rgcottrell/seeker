//! HTTP server shared state.
//!
//! Holds everything the handlers need that outlives a single request: the
//! shared tokenizer + chat template (for synchronous render/encode/tokenize),
//! the [`InferenceHandle`] to the GPU worker thread, and the CLI-provided
//! generation defaults that individual API requests override. Cheaply
//! cloneable (`Arc`-wrapped) so axum can hand a copy to every request.

use std::sync::Arc;

use crate::inference::sample::SamplerConfig;
use crate::server::inference::InferenceHandle;
use crate::tokenizer::TokenizerBundle;

#[derive(Clone, Default)]
pub struct AppState {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    /// Shared tokenizer + chat template + special tokens. `None` until a model
    /// is loaded (`serve` can start without `--model` for `/health` etc.).
    tokenizer: Option<Arc<TokenizerBundle>>,
    /// Handle to the GPU worker. `None` ⇒ generation endpoints return 503.
    inference: Option<InferenceHandle>,
    /// Extra template-context variables from `serve --chat-template-kwargs`,
    /// merged into every render (override built-ins).
    template_kwargs: serde_json::Map<String, serde_json::Value>,
    /// CLI sampling flags as the per-request base (requests override fields).
    default_sampler: SamplerConfig,
    /// `--max-tokens` default for requests that omit `max_tokens`/`n_predict`.
    default_max_tokens: u32,
    /// `--ignore-eos` (not exposed per-request).
    default_ignore_eos: bool,
    /// `--system-prompt`, injected as the leading system turn for chat/messages
    /// requests that carry no system message of their own.
    default_system_prompt: Option<String>,
    /// `--ctx-size`, reported in `/props`.
    ctx_size: u32,
    /// Number of cache slots (`--parallel`), reported by `/props` + `/slots`.
    n_slots: u32,
    /// Model id reported by `/models` and echoed in responses (file stem).
    model_id: String,
    /// Absolute model path, reported in `/props`.
    model_path: Option<String>,
    /// Vision projector config (when an mmproj was resolved). The chat handler
    /// CPU-preprocesses image content with it; the worker holds the encoder.
    /// `None` ⇒ image requests are rejected.
    vision_config: Option<crate::vision::VisionConfig>,
    /// Audio projector config (gemma4ua), when the mmproj carries an audio
    /// encoder. The chat handler uses it to size the `<|audio|>` block;
    /// `None` ⇒ audio requests are rejected.
    audio_config: Option<crate::audio::AudioConfig>,
}

/// Everything needed to build an `AppState` with a loaded model. Built by
/// `serve::run` once the worker reports ready.
pub struct AppStateInit {
    pub tokenizer: Arc<TokenizerBundle>,
    pub inference: InferenceHandle,
    pub template_kwargs: serde_json::Map<String, serde_json::Value>,
    pub default_sampler: SamplerConfig,
    pub default_max_tokens: u32,
    pub default_ignore_eos: bool,
    pub default_system_prompt: Option<String>,
    pub ctx_size: u32,
    pub n_slots: u32,
    pub model_id: String,
    pub model_path: String,
    pub vision_config: Option<crate::vision::VisionConfig>,
    pub audio_config: Option<crate::audio::AudioConfig>,
}

impl AppState {
    /// Build state for a loaded model (the full inference path).
    pub fn new(init: AppStateInit) -> Self {
        Self {
            inner: Arc::new(Inner {
                tokenizer: Some(init.tokenizer),
                inference: Some(init.inference),
                template_kwargs: init.template_kwargs,
                default_sampler: init.default_sampler,
                default_max_tokens: init.default_max_tokens,
                default_ignore_eos: init.default_ignore_eos,
                default_system_prompt: init.default_system_prompt,
                ctx_size: init.ctx_size,
                n_slots: init.n_slots,
                model_id: init.model_id,
                model_path: Some(init.model_path),
                vision_config: init.vision_config,
                audio_config: init.audio_config,
            }),
        }
    }

    /// The vision projector config, if an mmproj was loaded (image input).
    pub fn vision_config(&self) -> Option<&crate::vision::VisionConfig> {
        self.inner.vision_config.as_ref()
    }

    /// The audio projector config, if the mmproj has an audio encoder.
    pub fn audio_config(&self) -> Option<&crate::audio::AudioConfig> {
        self.inner.audio_config.as_ref()
    }

    /// The loaded tokenizer bundle, if any (`/tokenize`, `/detokenize`,
    /// render+encode for chat endpoints, `count_tokens`).
    pub fn tokenizer(&self) -> Option<&TokenizerBundle> {
        self.inner.tokenizer.as_deref()
    }

    /// Handle to the GPU worker, if a model is loaded.
    pub fn inference(&self) -> Option<&InferenceHandle> {
        self.inner.inference.as_ref()
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.inner
            .tokenizer
            .as_ref()
            .and_then(|t| t.chat_template.as_deref())
    }

    pub fn bos_token(&self) -> Option<&str> {
        self.inner
            .tokenizer
            .as_ref()
            .and_then(|t| t.bos_token.as_deref())
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.inner
            .tokenizer
            .as_ref()
            .and_then(|t| t.eos_token.as_deref())
    }

    pub fn template_kwargs(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.inner.template_kwargs
    }

    pub fn default_sampler(&self) -> &SamplerConfig {
        &self.inner.default_sampler
    }

    pub fn default_max_tokens(&self) -> u32 {
        self.inner.default_max_tokens
    }

    pub fn default_ignore_eos(&self) -> bool {
        self.inner.default_ignore_eos
    }

    pub fn default_system_prompt(&self) -> Option<&str> {
        self.inner.default_system_prompt.as_deref()
    }

    pub fn ctx_size(&self) -> u32 {
        self.inner.ctx_size
    }

    /// Number of KV-cache slots (`--parallel`); 0 when no model is loaded.
    pub fn n_slots(&self) -> u32 {
        self.inner.n_slots
    }

    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    pub fn model_path(&self) -> Option<&str> {
        self.inner.model_path.as_deref()
    }
}
