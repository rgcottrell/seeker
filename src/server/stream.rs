//! SSE stream generators for the streaming-capable surfaces (OpenAI chat +
//! legacy completions, Anthropic Messages, llama-native completion).
//!
//! Each turns the per-job `mpsc::Receiver<GenEvent>` from the inference worker
//! into a `Stream<Item = Result<Event, Infallible>>` that plugs straight into
//! `axum::response::sse::Sse::new(...)`. The async↔sync bridge is
//! `futures_util::stream::unfold` over the receiver (no tokio-stream dep); a
//! small queue in the unfold state lets one `GenEvent` expand to several SSE
//! frames (e.g. the final chunk plus the literal `[DONE]`).

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::Event;
use futures_util::stream::{self, Stream};
use serde_json::json;
use tokio::sync::mpsc::Receiver;

use super::inference::{GenEvent, StopReason};

static REQ_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A process-unique id with the given prefix (e.g. `chatcmpl-…`).
pub fn gen_id(prefix: &str) -> String {
    let n = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{n:016x}")
}

/// Seconds since the Unix epoch (the `created` field). 0 on clock error.
pub fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn data(value: serde_json::Value) -> Event {
    Event::default().data(value.to_string())
}

fn named(name: &str, value: serde_json::Value) -> Event {
    Event::default().event(name).data(value.to_string())
}

fn done_sentinel() -> Event {
    Event::default().data("[DONE]")
}

/// Generic adapter: drive an SSE stream from the worker's reply channel,
/// mapping each `GenEvent` to zero-or-more SSE frames via `frame`. Ends the
/// stream after the terminal event's frames drain.
fn event_stream<F>(rx: Receiver<GenEvent>, frame: F) -> impl Stream<Item = Result<Event, Infallible>>
where
    F: FnMut(GenEvent) -> Vec<Event> + Send + 'static,
{
    struct St<F> {
        rx: Receiver<GenEvent>,
        queue: VecDeque<Event>,
        done: bool,
        frame: F,
    }
    let st = St {
        rx,
        queue: VecDeque::new(),
        done: false,
        frame,
    };
    stream::unfold(st, |mut st| async move {
        loop {
            if let Some(ev) = st.queue.pop_front() {
                return Some((Ok(ev), st));
            }
            if st.done {
                return None;
            }
            match st.rx.recv().await {
                Some(gen_ev) => {
                    let terminal = matches!(gen_ev, GenEvent::Done { .. } | GenEvent::Error(_));
                    for e in (st.frame)(gen_ev) {
                        st.queue.push_back(e);
                    }
                    if terminal {
                        st.done = true;
                    }
                }
                None => return None,
            }
        }
    })
}

/// OpenAI `chat.completion.chunk` stream: a role frame, content-delta frames,
/// a final `finish_reason` frame, then the literal `[DONE]`.
pub fn openai_chat_stream(
    rx: Receiver<GenEvent>,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let id = gen_id("chatcmpl");
    let created = unix_now();
    event_stream(rx, move |ev| {
        let base = |delta: serde_json::Value, finish: serde_json::Value| {
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })
        };
        match ev {
            GenEvent::Started { .. } => {
                vec![data(base(json!({"role": "assistant", "content": ""}), json!(null)))]
            }
            GenEvent::Delta(t) => vec![data(base(json!({"content": t}), json!(null)))],
            GenEvent::Done { stop_reason, .. } => vec![
                data(base(json!({}), json!(stop_reason.openai_finish()))),
                done_sentinel(),
            ],
            GenEvent::Error(e) => vec![
                data(json!({"error": {"message": e, "type": "server_error"}})),
                done_sentinel(),
            ],
        }
    })
}

/// OpenAI legacy `text_completion` stream.
pub fn openai_completion_stream(
    rx: Receiver<GenEvent>,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let id = gen_id("cmpl");
    let created = unix_now();
    event_stream(rx, move |ev| {
        let base = |text: &str, finish: serde_json::Value| {
            json!({
                "id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": null}],
            })
        };
        match ev {
            GenEvent::Started { .. } => vec![],
            GenEvent::Delta(t) => vec![data(base(&t, json!(null)))],
            GenEvent::Done { stop_reason, .. } => {
                vec![data(base("", json!(stop_reason.openai_finish()))), done_sentinel()]
            }
            GenEvent::Error(e) => vec![
                data(json!({"error": {"message": e, "type": "server_error"}})),
                done_sentinel(),
            ],
        }
    })
}

/// llama-server native completion stream: unnamed `data:` frames of
/// `{content, stop:false}`, then a final `{content:"", stop:true, …}`. No
/// `[DONE]` sentinel (llama-server doesn't emit one).
pub fn llama_completion_stream(
    rx: Receiver<GenEvent>,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    event_stream(rx, move |ev| match ev {
        GenEvent::Started { .. } => vec![],
        GenEvent::Delta(t) => vec![data(json!({"content": t, "stop": false}))],
        GenEvent::Done {
            stop_reason,
            prompt_tokens,
            completion_tokens,
        } => {
            let matched = stop_reason.matched_sequence();
            vec![data(json!({
                "content": "",
                "stop": true,
                "model": model,
                "tokens_predicted": completion_tokens,
                "tokens_evaluated": prompt_tokens,
                "stopped_eos": matches!(&stop_reason, StopReason::Eos),
                "stopped_word": matched.is_some(),
                "stopped_limit": matches!(
                    &stop_reason,
                    StopReason::MaxTokens | StopReason::ContextFull
                ),
                "stopping_word": matched.unwrap_or(""),
            }))]
        }
        GenEvent::Error(e) => vec![data(json!({"content": "", "stop": true, "error": e}))],
    })
}

/// Anthropic Messages stream: the canonical 6-event sequence, driven by real
/// deltas. `message_start` carries the real `input_tokens` from `Started`;
/// `message_delta` carries the stop reason + `output_tokens`.
pub fn anthropic_messages_stream(
    rx: Receiver<GenEvent>,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let id = gen_id("msg");
    let mut block_open = false;
    event_stream(rx, move |ev| match ev {
        GenEvent::Started { prompt_tokens } => {
            block_open = true;
            vec![
                named(
                    "message_start",
                    json!({
                        "type": "message_start",
                        "message": {
                            "id": id, "type": "message", "role": "assistant", "content": [],
                            "model": model, "stop_reason": null, "stop_sequence": null,
                            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                        },
                    }),
                ),
                named(
                    "content_block_start",
                    json!({"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}}),
                ),
            ]
        }
        GenEvent::Delta(t) => vec![named(
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": t}}),
        )],
        GenEvent::Done {
            stop_reason,
            completion_tokens,
            ..
        } => {
            let mut out = Vec::new();
            // Guard against a Done before any Started (e.g. an error path).
            if block_open {
                out.push(named(
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": 0}),
                ));
            }
            out.push(named(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason.anthropic_reason(),
                        "stop_sequence": stop_reason.matched_sequence(),
                    },
                    "usage": {"output_tokens": completion_tokens},
                }),
            ));
            out.push(named("message_stop", json!({"type": "message_stop"})));
            out
        }
        GenEvent::Error(e) => vec![named(
            "error",
            json!({"type": "error", "error": {"type": "api_error", "message": e}}),
        )],
    })
}
