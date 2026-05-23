//! Operational endpoints: health, props, slots, metrics, lora-adapters.

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

use crate::server::types::llama::{LoraAdapter, PropsResponse, PropsUpdateRequest, SlotState};

pub async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

pub async fn props_get() -> impl IntoResponse {
    Json(PropsResponse::stub())
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
