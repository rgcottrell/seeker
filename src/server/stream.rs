//! SSE stream stubs for the two streaming-capable surfaces.
//!
//! Both helpers return owned `Stream`s of `Result<Event, Infallible>` so they
//! plug straight into `axum::response::sse::Sse::new(...)`. Each yields a
//! short, deterministic sequence of frames and then terminates — no timers,
//! no real inference work.

use std::convert::Infallible;

use axum::response::sse::Event;
use futures_util::stream::{self, Stream};
use serde_json::json;

use super::types::anthropic::STUB_TEXT as ANTHROPIC_STUB_TEXT;
use super::types::openai::STUB_TEXT as OPENAI_STUB_TEXT;

/// OpenAI-style chat completions stream: three unnamed `data:` frames, the
/// last being the literal sentinel `[DONE]`.
pub fn openai_stub_stream(model: String) -> impl Stream<Item = Result<Event, Infallible>> {
    let role_frame = json!({
        "id": "chatcmpl-seeker-stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": null,
        }],
    });
    let content_frame = json!({
        "id": "chatcmpl-seeker-stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": OPENAI_STUB_TEXT},
            "finish_reason": null,
        }],
    });
    let stop_frame = json!({
        "id": "chatcmpl-seeker-stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    });

    let events = vec![
        Ok(Event::default().data(role_frame.to_string())),
        Ok(Event::default().data(content_frame.to_string())),
        Ok(Event::default().data(stop_frame.to_string())),
        // OpenAI's literal terminator — not JSON.
        Ok(Event::default().data("[DONE]")),
    ];
    stream::iter(events)
}

/// Anthropic-style Messages stream: six named events in canonical order. Each
/// uses `Event::default().event("name").data(json)` so axum emits both the
/// `event:` and `data:` SSE lines per Anthropic's wire format.
pub fn anthropic_stub_stream(model: String) -> impl Stream<Item = Result<Event, Infallible>> {
    let message_id = "msg_seeker_stub";

    let message_start = json!({
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    });
    let content_block_start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    });
    let content_block_delta = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": ANTHROPIC_STUB_TEXT},
    });
    let content_block_stop = json!({
        "type": "content_block_stop",
        "index": 0,
    });
    let message_delta = json!({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": null},
        "usage": {"output_tokens": 1},
    });
    let message_stop = json!({"type": "message_stop"});

    let events = vec![
        Ok(Event::default().event("message_start").data(message_start.to_string())),
        Ok(Event::default()
            .event("content_block_start")
            .data(content_block_start.to_string())),
        Ok(Event::default()
            .event("content_block_delta")
            .data(content_block_delta.to_string())),
        Ok(Event::default()
            .event("content_block_stop")
            .data(content_block_stop.to_string())),
        Ok(Event::default().event("message_delta").data(message_delta.to_string())),
        Ok(Event::default().event("message_stop").data(message_stop.to_string())),
    ];
    stream::iter(events)
}
