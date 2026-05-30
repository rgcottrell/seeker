//! Operational endpoints: health, props, slots, metrics, lora-adapters.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

use crate::server::state::AppState;
use crate::server::types::llama::{LoraAdapter, PropsResponse, PropsUpdateRequest, SlotState};

pub async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

pub async fn props_get(State(state): State<AppState>) -> impl IntoResponse {
    let loaded = state.tokenizer().is_some();
    let s = state.default_sampler();
    let default_generation_settings = json!({
        "temperature": s.temperature,
        "top_k": s.top_k,
        "top_p": s.top_p,
        "min_p": s.min_p,
        "presence_penalty": s.presence_penalty,
        "frequency_penalty": s.frequency_penalty,
        "repeat_penalty": s.repeat_penalty,
        "repeat_last_n": s.penalty_last_n,
        "seed": s.seed,
        "n_predict": state.default_max_tokens(),
        "n_ctx": state.ctx_size(),
        "ignore_eos": state.default_ignore_eos(),
    });
    Json(PropsResponse {
        model_path: state.model_path().map(|p| p.to_string()),
        chat_template: state.chat_template().map(|t| t.to_string()),
        build_info: env!("CARGO_PKG_VERSION"),
        default_generation_settings,
        total_slots: if loaded { 1 } else { 0 },
    })
}

pub async fn props_post(Json(_req): Json<PropsUpdateRequest>) -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"success": true})))
}

pub async fn slots() -> impl IntoResponse {
    Json::<Vec<SlotState>>(Vec::new())
}

pub async fn metrics() -> impl IntoResponse {
    let body = "# HELP seeker_stub_info Static marker indicating the seeker stub server is running.\n\
                # TYPE seeker_stub_info gauge\n\
                seeker_stub_info 1\n";
    ([("content-type", "text/plain; version=0.0.4")], body)
}

pub async fn lora_adapters_get() -> impl IntoResponse {
    Json::<Vec<LoraAdapter>>(Vec::new())
}

pub async fn lora_adapters_post(Json(_req): Json<Vec<LoraAdapter>>) -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"success": true})))
}
