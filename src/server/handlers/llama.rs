//! llama-server native handlers: completion, infill, tokenize, detokenize,
//! embedding, apply-template.

use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::chat_template;
use crate::server::convert::{self, prompt_value_to_tokens, value_messages_to_chat};
use crate::server::error;
use crate::server::inference::{collect, GenConfig};
use crate::server::state::AppState;
use crate::server::stream::llama_completion_stream;
use crate::server::types::llama::{
    ApplyTemplateRequest, ApplyTemplateResponse, CompletionRequest, CompletionResponse,
    DetokenizeRequest, DetokenizeResponse, InfillRequest, TokenizeRequest, TokenizeResponse,
};

pub async fn completion(State(state): State<AppState>, Json(req): Json<CompletionRequest>) -> Response {
    let Some(handle) = state.inference() else {
        return error::no_model_openai();
    };
    let Some(bundle) = state.tokenizer() else {
        return error::no_model_openai();
    };
    let add_special = bundle.add_bos_default || bundle.add_eos_default;
    let tokens = match prompt_value_to_tokens(&req.prompt, bundle, add_special) {
        Ok(t) => t,
        Err(e) => return error::bad_request(e),
    };
    // llama-server's `n_predict <= 0` means "until EOS/context"; we cap at the
    // CLI `--max-tokens` ceiling either way.
    let max_tokens = match req.n_predict {
        Some(n) if n > 0 => n as u32,
        _ => state.default_max_tokens(),
    };
    let config = GenConfig {
        sampler: req.sampler_config(state.default_sampler()),
        max_tokens,
        stop: req.stop.clone().unwrap_or_default(),
        ignore_eos: state.default_ignore_eos(),
    };
    let rx = match handle.start(tokens, config).await {
        Ok(rx) => rx,
        Err(e) => return error::internal(e),
    };
    let model = state.model_id().to_string();
    if req.stream.unwrap_or(false) {
        Sse::new(llama_completion_stream(rx, model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match collect(rx).await {
            Ok(out) => Json(CompletionResponse {
                content: out.text,
                stop: true,
                model,
                tokens_predicted: out.completion_tokens,
                tokens_evaluated: out.prompt_tokens,
            })
            .into_response(),
            Err(e) => error::internal(e),
        }
    }
}

pub async fn infill(State(state): State<AppState>, Json(req): Json<InfillRequest>) -> Response {
    let Some(handle) = state.inference() else {
        return error::no_model_openai();
    };
    let Some(bundle) = state.tokenizer() else {
        return error::no_model_openai();
    };
    // Detect a known FIM token triple (prefix, suffix, middle). Without one we
    // can't build a fill-in-the-middle prompt, so fail clearly.
    let triples = [
        ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
        ("<PRE>", "<SUF>", "<MID>"),
        ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    ];
    let tok = &bundle.tokenizer;
    let fim = triples.iter().find_map(|(p, s, m)| {
        match (tok.token_to_id(p), tok.token_to_id(s), tok.token_to_id(m)) {
            (Some(pi), Some(si), Some(mi)) => Some((pi, si, mi)),
            _ => None,
        }
    });
    let Some((pre, suf, mid)) = fim else {
        return error::not_supported("this model has no FIM/infill tokens");
    };
    let prefix = req.input_prefix.clone().unwrap_or_default();
    let suffix = req.input_suffix.clone().unwrap_or_default();
    // llama.cpp prefix-suffix-middle order: <PRE> prefix <SUF> suffix <MID>.
    let mut tokens = vec![pre];
    match convert::encode(bundle, &prefix, false) {
        Ok(ids) => tokens.extend(ids),
        Err(e) => return error::bad_request(e),
    }
    tokens.push(suf);
    match convert::encode(bundle, &suffix, false) {
        Ok(ids) => tokens.extend(ids),
        Err(e) => return error::bad_request(e),
    }
    tokens.push(mid);

    let max_tokens = match req.n_predict {
        Some(n) if n > 0 => n as u32,
        _ => state.default_max_tokens(),
    };
    let config = GenConfig {
        sampler: state.default_sampler().clone(),
        max_tokens,
        stop: Vec::new(),
        ignore_eos: state.default_ignore_eos(),
    };
    let rx = match handle.start(tokens, config).await {
        Ok(rx) => rx,
        Err(e) => return error::internal(e),
    };
    let model = state.model_id().to_string();
    if req.stream.unwrap_or(false) {
        Sse::new(llama_completion_stream(rx, model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match collect(rx).await {
            Ok(out) => Json(CompletionResponse {
                content: out.text,
                stop: true,
                model,
                tokens_predicted: out.completion_tokens,
                tokens_evaluated: out.prompt_tokens,
            })
            .into_response(),
            Err(e) => error::internal(e),
        }
    }
}

pub async fn tokenize(State(state): State<AppState>, Json(req): Json<TokenizeRequest>) -> Response {
    let Some(bundle) = state.tokenizer() else {
        return error::no_model_openai();
    };
    let content = req.content.clone().unwrap_or_default();
    let add_special = req.add_special.unwrap_or(false);
    let ids = match convert::encode(bundle, &content, add_special) {
        Ok(i) => i,
        Err(e) => return error::bad_request(e),
    };
    if req.with_pieces.unwrap_or(false) {
        let pairs = ids
            .iter()
            .map(|&id| {
                let piece = bundle.tokenizer.decode(&[id], false).unwrap_or_default();
                (id, piece)
            })
            .collect();
        Json(TokenizeResponse::pieces(pairs)).into_response()
    } else {
        Json(TokenizeResponse::ids(ids)).into_response()
    }
}

pub async fn detokenize(
    State(state): State<AppState>,
    Json(req): Json<DetokenizeRequest>,
) -> Response {
    let Some(bundle) = state.tokenizer() else {
        return error::no_model_openai();
    };
    let content = bundle
        .tokenizer
        .decode(&req.tokens, /*skip_special=*/ false)
        .unwrap_or_default();
    Json(DetokenizeResponse { content }).into_response()
}

pub async fn embedding() -> Response {
    error::not_supported("embeddings are not supported by seeker serve")
}

pub async fn apply_template(
    State(state): State<AppState>,
    Json(req): Json<ApplyTemplateRequest>,
) -> Response {
    let Some(template) = state.chat_template() else {
        return error::no_model_openai();
    };
    let messages = match value_messages_to_chat(&req.messages) {
        Ok(m) => m,
        Err(e) => return error::bad_request(e),
    };
    match chat_template::render(
        template,
        &messages,
        req.add_generation_prompt.unwrap_or(true),
        state.bos_token().unwrap_or(""),
        state.eos_token().unwrap_or(""),
        state.template_kwargs(),
    ) {
        Ok(prompt) => Json(ApplyTemplateResponse { prompt }).into_response(),
        Err(e) => error::bad_request(e.to_string()),
    }
}
