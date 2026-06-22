//! OpenAI-compatible handlers. Chat + legacy completions run real inference;
//! embeddings / rerank / responses / audio return 501 (no embedding/pooling
//! path on a causal-LM logits engine).

use axum::Json;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::server::convert::{
    self, MediaParts, openai_messages_to_chat_mm, parse_stop, prompt_value_to_tokens,
};
use crate::server::error;
use crate::server::inference::{GenConfig, GenOutput, ServeAudio, ServeImage, collect};
use crate::server::state::AppState;
use crate::server::stream::{gen_id, openai_chat_stream, openai_completion_stream, unix_now};
use crate::server::types::common::Usage;
use crate::server::types::openai::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice,
    CompletionRequest, CompletionResponse, EmbeddingObject, EmbeddingRequest, EmbeddingResponse,
    Model, ModelListResponse,
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
    // Multimodal-aware: collect any `image_url` / `input_audio` content (the
    // message text keeps the `<__media__>` marker where each item sat) and
    // render+splice the projector block. Both are `None` for a text request.
    let (messages, images, audios) = match openai_messages_to_chat_mm(&req.messages) {
        Ok(m) => m,
        Err(e) => return error::bad_request(e),
    };
    let (tokens, media) = match convert::render_and_encode_mm(&state, messages, &images, &audios) {
        Ok(t) => t,
        Err(e) => return error::bad_request(e),
    };
    let (image, audio) = match media {
        Some(MediaParts::Image {
            pimg,
            image_start,
            nx,
            ny,
        }) => (
            Some(ServeImage {
                pimg,
                image_start,
                nx,
                ny,
            }),
            None,
        ),
        Some(MediaParts::Audio {
            samples,
            audio_start,
            n_tok,
        }) => (
            None,
            Some(ServeAudio {
                samples,
                audio_start,
                n_tok,
            }),
        ),
        None => (None, None),
    };
    let config = GenConfig {
        sampler: req.sampler_config(state.default_sampler()),
        max_tokens: req.max_tokens.unwrap_or(state.default_max_tokens()),
        stop: parse_stop(&req.stop),
        ignore_eos: state.default_ignore_eos(),
        id_slot: None,
    };
    let rx = match handle.start_mm(tokens, config, image, audio).await {
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

/// `POST /v1/embeddings` — OpenAI-compatible. One pooled vector per input; the
/// `{object:"list", data, model, usage}` envelope. `--pooling none` is rejected
/// (not OAI-compatible). `encoding_format: "base64"` returns LE-f32 base64.
pub async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    // Embeddings disabled (or no model) → llama.cpp's "start with --embeddings".
    if !state.embeddings_enabled() {
        return error::not_supported(
            "This server does not support embeddings. Start it with `--embeddings`",
        );
    }
    let (Some(handle), Some(bundle)) = (state.inference(), state.tokenizer()) else {
        return error::no_model_openai();
    };
    let Some(input) = req.input.as_ref().or(req.content.as_ref()) else {
        return error::bad_request("\"input\" must be provided");
    };
    let base64 = match req.encoding_format.as_deref() {
        Some("base64") => true,
        Some("float") | None => false,
        Some(_) => return error::bad_request("encoding_format must be either float or base64"),
    };
    let inputs = match convert::embedding_inputs_to_tokens(bundle, input) {
        Ok(t) => t,
        Err(e) => return error::bad_request(e),
    };
    let n_tokens: u32 = inputs.iter().map(|t| t.len() as u32).sum();
    let outs = match handle.embed(inputs, req.embd_normalize).await {
        Ok(o) => o,
        Err(e) => return error::internal(e),
    };
    let mut data = Vec::with_capacity(outs.len());
    for (i, out) in outs.into_iter().enumerate() {
        // OpenAI returns a single pooled vector per input.
        if out.vectors.len() != 1 {
            return error::bad_request(
                "Pooling type 'none' is not OAI compatible. Please use a different pooling type",
            );
        }
        let v = &out.vectors[0];
        let embedding = if base64 {
            Value::String(f32_base64(v))
        } else {
            serde_json::json!(v)
        };
        data.push(EmbeddingObject {
            object: "embedding",
            index: i as u32,
            embedding,
            encoding_format: base64.then_some("base64"),
        });
    }
    Json(EmbeddingResponse {
        object: "list",
        data,
        model: state.model_id().to_string(),
        usage: Usage {
            prompt_tokens: n_tokens,
            completion_tokens: 0,
            total_tokens: n_tokens,
        },
    })
    .into_response()
}

/// Base64 of an embedding's little-endian f32 bytes (OpenAI `encoding_format`).
pub(crate) fn f32_base64(v: &[f32]) -> String {
    use base64::Engine;
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(&bytes)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_base64_round_trips_le() {
        use base64::Engine;
        let v = [1.0f32, -2.5, 0.0, 3.25];
        let b64 = f32_base64(&v);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64.as_bytes())
            .unwrap();
        assert_eq!(bytes.len(), v.len() * 4);
        let back: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(back, v);
    }
}
