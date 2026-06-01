//! OpenAI-compatible handlers. Chat + legacy completions run real inference;
//! embeddings / rerank / responses / audio return 501 (no embedding/pooling
//! path on a causal-LM logits engine).

use axum::Json;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::server::convert::{
    self, openai_messages_to_chat_mm, parse_stop, prompt_value_to_tokens,
};
use crate::server::error;
use crate::server::inference::{GenConfig, GenOutput, ServeImage, collect};
use crate::server::state::AppState;
use crate::server::stream::{gen_id, openai_chat_stream, openai_completion_stream, unix_now};
use crate::server::types::common::Usage;
use crate::server::types::openai::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice,
    CompletionRequest, CompletionResponse, Model, ModelListResponse,
};

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let Some(handle) = state.inference() else {
        return error::no_model_openai();
    };
    if req.n.unwrap_or(1) > 1 {
        return error::bad_request("`n` > 1 is not supported (single-sequence engine)");
    }
    // Multimodal-aware: collect any `image_url` content (the message text keeps
    // the `<__media__>` marker where each image sat) and render+splice the
    // vision block. `image` is `None` for a text-only request.
    let (messages, images) = match openai_messages_to_chat_mm(&req.messages) {
        Ok(m) => m,
        Err(e) => return error::bad_request(e),
    };
    let (tokens, image_parts) = match convert::render_and_encode_mm(&state, messages, &images) {
        Ok(t) => t,
        Err(e) => return error::bad_request(e),
    };
    let image = image_parts.map(|(pimg, image_start, nx, ny)| ServeImage {
        pimg,
        image_start,
        nx,
        ny,
    });
    let config = GenConfig {
        sampler: req.sampler_config(state.default_sampler()),
        max_tokens: req.max_tokens.unwrap_or(state.default_max_tokens()),
        stop: parse_stop(&req.stop),
        ignore_eos: state.default_ignore_eos(),
        id_slot: None,
    };
    let rx = match handle.start_mm(tokens, config, image).await {
        Ok(rx) => rx,
        Err(e) => return error::internal(e),
    };
    let model = state.model_id().to_string();
    if req.stream.unwrap_or(false) {
        Sse::new(openai_chat_stream(rx, model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match collect(rx).await {
            Ok(out) => Json(chat_response(model, out)).into_response(),
            Err(e) => error::internal(e),
        }
    }
}

fn chat_response(model: String, out: GenOutput) -> ChatCompletionResponse {
    let finish_reason = out.stop_reason.openai_finish();
    ChatCompletionResponse {
        id: gen_id("chatcmpl"),
        object: "chat.completion",
        created: unix_now(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: Value::String(out.text),
                name: None,
                tool_call_id: None,
            },
            finish_reason,
        }],
        usage: usage(out.prompt_tokens, out.completion_tokens),
    }
}

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let Some(handle) = state.inference() else {
        return error::no_model_openai();
    };
    if req.n.unwrap_or(1) > 1 {
        return error::bad_request("`n` > 1 is not supported (single-sequence engine)");
    }
    let Some(bundle) = state.tokenizer() else {
        return error::no_model_openai();
    };
    let add_special = bundle.add_bos_default || bundle.add_eos_default;
    let tokens = match prompt_value_to_tokens(&req.prompt, bundle, add_special) {
        Ok(t) => t,
        Err(e) => return error::bad_request(e),
    };
    let config = GenConfig {
        sampler: req.sampler_config(state.default_sampler()),
        max_tokens: req.max_tokens.unwrap_or(state.default_max_tokens()),
        stop: parse_stop(&req.stop),
        ignore_eos: state.default_ignore_eos(),
        id_slot: None,
    };
    let rx = match handle.start(tokens, config).await {
        Ok(rx) => rx,
        Err(e) => return error::internal(e),
    };
    let model = state.model_id().to_string();
    if req.stream.unwrap_or(false) {
        Sse::new(openai_completion_stream(rx, model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match collect(rx).await {
            Ok(out) => Json(completion_response(model, out)).into_response(),
            Err(e) => error::internal(e),
        }
    }
}

fn completion_response(model: String, out: GenOutput) -> CompletionResponse {
    let finish_reason = out.stop_reason.openai_finish();
    CompletionResponse {
        id: gen_id("cmpl"),
        object: "text_completion",
        created: unix_now(),
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text: out.text,
            finish_reason,
            logprobs: None,
        }],
        usage: usage(out.prompt_tokens, out.completion_tokens),
    }
}

fn usage(prompt_tokens: u32, completion_tokens: u32) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    }
}

pub async fn models(State(state): State<AppState>) -> Response {
    let data = if state.tokenizer().is_some() {
        vec![Model {
            id: state.model_id().to_string(),
            object: "model",
            created: 0,
            owned_by: "seeker".to_string(),
        }]
    } else {
        Vec::new()
    };
    Json(ModelListResponse {
        object: "list",
        data,
    })
    .into_response()
}

pub async fn embeddings() -> Response {
    error::not_supported(
        "embeddings are not supported by seeker serve (causal-LM with no embedding/pooling path)",
    )
}

pub async fn rerank() -> Response {
    error::not_supported("rerank is not supported by seeker serve")
}

pub async fn responses() -> Response {
    error::not_supported("the responses API is not supported by seeker serve")
}

pub async fn audio_transcriptions() -> Response {
    error::not_supported("audio transcription is not supported by seeker serve")
}
