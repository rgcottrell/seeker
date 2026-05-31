//! Request → inference conversions shared across the API surfaces: turning the
//! various message shapes (OpenAI, Anthropic, raw `Value`) into the engine's
//! [`ChatMessage`], assembling raw-completion prompts into token ids, and
//! normalizing the `stop` field.

use serde_json::Value;

use crate::chat_template::ChatMessage;
use crate::server::state::AppState;
use crate::server::types::anthropic::MessagesInputMessage;
use crate::server::types::openai::ChatMessage as OpenAiMessage;
use crate::tokenizer::TokenizerBundle;

/// Flatten a message `content` field into plain text. Accepts a bare string or
/// the structured array form (`[{type:"text", text:"…"}, …]`) that OpenAI and
/// Anthropic both use; non-text parts (images, audio) are rejected with a clear
/// error since this is a text-only LM.
pub fn content_to_text(content: &Value) -> Result<String, String> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        let t = part
                            .get("text")
                            .and_then(|t| t.as_str())
                            .ok_or_else(|| "content text part missing string `text`".to_string())?;
                        out.push_str(t);
                    }
                    Some(other) => {
                        return Err(format!(
                            "unsupported content part type {other:?} — this server is text-only"
                        ))
                    }
                    None => return Err("content array part missing `type`".to_string()),
                }
            }
            Ok(out)
        }
        other => Err(format!("unsupported message content: {other}")),
    }
}

/// OpenAI chat messages → engine `ChatMessage`s. Roles pass through unchanged.
pub fn openai_messages_to_chat(messages: &[OpenAiMessage]) -> Result<Vec<ChatMessage>, String> {
    messages
        .iter()
        .map(|m| {
            Ok(ChatMessage {
                role: m.role.clone(),
                content: content_to_text(&m.content)?,
                reasoning_content: None,
            })
        })
        .collect()
}

/// Anthropic Messages request → engine `ChatMessage`s. The top-level `system`
/// field (string or array of text blocks) becomes a leading system turn.
pub fn anthropic_to_chat(
    system: &Option<Value>,
    messages: &[MessagesInputMessage],
) -> Result<Vec<ChatMessage>, String> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if let Some(sys) = system {
        let text = content_to_text(sys)?;
        if !text.is_empty() {
            out.push(ChatMessage::system(text));
        }
    }
    for m in messages {
        out.push(ChatMessage {
            role: m.role.clone(),
            content: content_to_text(&m.content)?,
            reasoning_content: None,
        });
    }
    Ok(out)
}

/// Raw `serde_json::Value` messages (the `/apply-template` shape) → engine
/// `ChatMessage`s. Requires string `role`; accepts the structured content forms
/// and an optional `reasoning_content` (so reasoning templates can re-render
/// prior thinking).
pub fn value_messages_to_chat(messages: &[Value]) -> Result<Vec<ChatMessage>, String> {
    messages
        .iter()
        .map(|v| {
            let role = v
                .get("role")
                .and_then(|r| r.as_str())
                .ok_or_else(|| "message missing string `role`".to_string())?;
            let content = v
                .get("content")
                .map(content_to_text)
                .transpose()?
                .unwrap_or_default();
            let reasoning_content = v
                .get("reasoning_content")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());
            Ok(ChatMessage {
                role: role.to_string(),
                content,
                reasoning_content,
            })
        })
        .collect()
}

/// Inject the CLI `--system-prompt` as the leading system turn, but only when
/// the request supplied no system message of its own (matches `chat`'s
/// "one system message, CLI wins when absent" and llama-server's behavior).
pub fn apply_default_system(messages: &mut Vec<ChatMessage>, default: Option<&str>) {
    if let Some(sys) = default {
        let has_system = messages.first().map(|m| m.role == "system").unwrap_or(false);
        if !has_system {
            messages.insert(0, ChatMessage::system(sys));
        }
    }
}

/// Normalize a `stop` field (string, array of strings, or absent) into a list
/// of stop sequences. Non-string array entries are ignored.
pub fn parse_stop(stop: &Option<Value>) -> Vec<String> {
    match stop {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

/// Turn a raw-completion `prompt` field into token ids. Accepts a string
/// (encoded with `add_special`), an array of strings (encoded as their
/// concatenation), or an array of integers (used directly as token ids).
pub fn prompt_value_to_tokens(
    prompt: &Option<Value>,
    bundle: &TokenizerBundle,
    add_special: bool,
) -> Result<Vec<u32>, String> {
    let value = prompt
        .as_ref()
        .ok_or_else(|| "request is missing `prompt`".to_string())?;
    match value {
        Value::String(s) => encode(bundle, s, add_special),
        Value::Array(arr) => {
            if arr.iter().all(|v| v.is_u64() || v.is_i64()) {
                // Already token ids.
                arr.iter()
                    .map(|v| {
                        v.as_u64()
                            .filter(|n| *n <= u32::MAX as u64)
                            .map(|n| n as u32)
                            .ok_or_else(|| "prompt token id out of range".to_string())
                    })
                    .collect()
            } else {
                // Array of string segments — encode the concatenation.
                let mut joined = String::new();
                for v in arr {
                    let s = v
                        .as_str()
                        .ok_or_else(|| "prompt array must be all strings or all ints".to_string())?;
                    joined.push_str(s);
                }
                encode(bundle, &joined, add_special)
            }
        }
        other => Err(format!("unsupported `prompt`: {other}")),
    }
}

/// Encode text to token ids via the bundle's tokenizer.
pub fn encode(bundle: &TokenizerBundle, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
    bundle
        .tokenizer
        .encode(text, add_special)
        .map(|enc| enc.get_ids().to_vec())
        .map_err(|e| format!("tokenize failed: {e}"))
}

/// Apply the chat template to `messages` (injecting the CLI default system
/// prompt when absent) and encode the result to token ids — the prompt the
/// engine prefills for chat / messages requests. Errors (no template, render
/// failure, unsupported content) bubble up for a `400`.
pub fn render_and_encode(state: &AppState, mut messages: Vec<ChatMessage>) -> Result<Vec<u32>, String> {
    let template = state
        .chat_template()
        .ok_or("this model has no chat template — use the completion endpoints")?;
    let bundle = state.tokenizer().ok_or("no tokenizer loaded")?;
    apply_default_system(&mut messages, state.default_system_prompt());
    let rendered = crate::chat_template::render(
        template,
        &messages,
        /* add_generation_prompt = */ true,
        state.bos_token().unwrap_or(""),
        state.eos_token().unwrap_or(""),
        state.template_kwargs(),
    )
    .map_err(|e| e.to_string())?;
    encode(bundle, &rendered, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_string_and_array_forms() {
        assert_eq!(content_to_text(&json!("hi")).unwrap(), "hi");
        assert_eq!(
            content_to_text(&json!([{"type":"text","text":"a"},{"type":"text","text":"b"}])).unwrap(),
            "ab"
        );
        assert!(content_to_text(&json!([{"type":"image_url","image_url":{}}])).is_err());
        assert_eq!(content_to_text(&Value::Null).unwrap(), "");
    }

    #[test]
    fn parse_stop_forms() {
        assert_eq!(parse_stop(&None), Vec::<String>::new());
        assert_eq!(parse_stop(&Some(json!("STOP"))), vec!["STOP"]);
        assert_eq!(
            parse_stop(&Some(json!(["a", "b", 3]))),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn default_system_only_when_absent() {
        let mut msgs = vec![ChatMessage::user("hi")];
        apply_default_system(&mut msgs, Some("be terse"));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "be terse");

        let mut msgs = vec![ChatMessage::system("client sys"), ChatMessage::user("hi")];
        apply_default_system(&mut msgs, Some("be terse"));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "client sys"); // client's system wins
    }

    #[test]
    fn anthropic_system_becomes_leading_turn() {
        let msgs = anthropic_to_chat(
            &Some(json!("sys")),
            &[MessagesInputMessage {
                role: "user".into(),
                content: json!("hello"),
            }],
        )
        .unwrap();
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "sys");
        assert_eq!(msgs[1].role, "user");
    }
}
