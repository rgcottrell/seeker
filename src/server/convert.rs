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
use crate::vision::preprocess::{PreprocessConfig, PreprocessedImage};

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
                        ));
                    }
                    None => return Err("content array part missing `type`".to_string()),
                }
            }
            Ok(out)
        }
        other => Err(format!("unsupported message content: {other}")),
    }
}

/// The media placeholder the chat template embeds; the worker later replaces it
/// with the vision block (`<|vision_start|><|image_pad|>×n_tok<|vision_end|>`).
/// Same marker llama.cpp's mtmd uses; matches `commands::run` / chat `/image`.
pub const MEDIA_MARKER: &str = "<__media__>";

/// Decode an OpenAI `image_url` value into raw encoded image bytes. Accepts a
/// `data:<mime>;base64,<…>` data URL (the common local case). Remote `http(s)`
/// URLs are rejected — the server does not fetch external resources.
pub fn decode_image_url(url: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let rest = url.strip_prefix("data:").ok_or_else(|| {
        "image_url must be a base64 data URL (data:<mime>;base64,…); remote URLs are not fetched"
            .to_string()
    })?;
    let comma = rest
        .find(',')
        .ok_or("malformed image_url data URL: missing comma")?;
    if !rest[..comma].contains("base64") {
        return Err("image_url data URL must be base64-encoded".into());
    }
    base64::engine::general_purpose::STANDARD
        .decode(rest[comma + 1..].trim())
        .map_err(|e| format!("image_url base64 decode failed: {e}"))
}

/// Like [`content_to_text`] but also extracts images: each `image_url` part
/// emits the [`MEDIA_MARKER`] into the text (where the image sits in the turn)
/// and its decoded bytes are collected, in order. Used by the vision-capable
/// chat path; `decode_image_url` rejects anything but base64 data URLs.
pub fn content_to_text_and_images(content: &Value) -> Result<(String, Vec<Vec<u8>>), String> {
    let mut text = String::new();
    let mut images = Vec::new();
    match content {
        Value::Null => {}
        Value::String(s) => text.push_str(s),
        Value::Array(parts) => {
            for part in parts {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        let t = part
                            .get("text")
                            .and_then(|t| t.as_str())
                            .ok_or("content text part missing string `text`")?;
                        text.push_str(t);
                    }
                    Some("image_url") => {
                        let url = part
                            .get("image_url")
                            .and_then(|o| o.get("url"))
                            .and_then(|u| u.as_str())
                            .ok_or("image_url part missing string `image_url.url`")?;
                        images.push(decode_image_url(url)?);
                        text.push_str(MEDIA_MARKER);
                    }
                    Some(other) => return Err(format!("unsupported content part type {other:?}")),
                    None => return Err("content array part missing `type`".to_string()),
                }
            }
        }
        other => return Err(format!("unsupported message content: {other}")),
    }
    Ok((text, images))
}

/// OpenAI chat messages → (engine `ChatMessage`s, collected images in order).
/// The image-bearing turn's content carries the [`MEDIA_MARKER`] where each
/// image sat, so the chat template renders it in place. The multimodal sibling
/// of [`openai_messages_to_chat`].
pub fn openai_messages_to_chat_mm(
    messages: &[OpenAiMessage],
) -> Result<(Vec<ChatMessage>, Vec<Vec<u8>>), String> {
    let mut out = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    for m in messages {
        let (content, imgs) = content_to_text_and_images(&m.content)?;
        images.extend(imgs);
        out.push(ChatMessage {
            role: m.role.clone(),
            content,
            reasoning_content: None,
        });
    }
    Ok((out, images))
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
        let has_system = messages
            .first()
            .map(|m| m.role == "system")
            .unwrap_or(false);
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
                    let s = v.as_str().ok_or_else(|| {
                        "prompt array must be all strings or all ints".to_string()
                    })?;
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
pub fn render_and_encode(
    state: &AppState,
    mut messages: Vec<ChatMessage>,
) -> Result<Vec<u32>, String> {
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

/// As [`render_and_encode`] but for a chat request that may carry images: the
/// rendered prompt has a [`MEDIA_MARKER`] where each image sits, which we replace
/// with the vision block (`<|vision_start|><|image_pad|>×n_tok<|vision_end|>`).
/// Returns the token ids plus, for an image request, the preprocessed image +
/// its placement `(image, image_start, nx, ny)` (the worker encodes + splices
/// it). First cut: at most one image per request.
#[allow(clippy::type_complexity)]
pub fn render_and_encode_mm(
    state: &AppState,
    mut messages: Vec<ChatMessage>,
    images: &[Vec<u8>],
) -> Result<(Vec<u32>, Option<(PreprocessedImage, usize, usize, usize)>), String> {
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
    if images.is_empty() {
        return Ok((encode(bundle, &rendered, false)?, None));
    }
    if images.len() > 1 {
        return Err("only one image per request is supported".into());
    }
    let vcfg = state
        .vision_config()
        .ok_or("this server has no vision model (mmproj); image input is unsupported")?;
    let pcfg = PreprocessConfig::qwen3vl_default(
        vcfg.patch_size,
        vcfg.spatial_merge_size,
        vcfg.image_mean,
        vcfg.image_std,
    );
    let pimg = crate::vision::preprocess::preprocess_bytes(&images[0], &pcfg)
        .map_err(|e| format!("image preprocess failed: {e}"))?;
    let merge = vcfg.spatial_merge_size as usize;
    let (nx, ny) = (pimg.grid_w as usize / merge, pimg.grid_h as usize / merge);
    let n_tok = pimg.n_tokens as usize;
    let (before, after) = rendered
        .split_once(MEDIA_MARKER)
        .ok_or("rendered chat prompt lost the <__media__> marker")?;
    let tid = |s: &str| -> Result<u32, String> {
        bundle
            .tokenizer
            .token_to_id(s)
            .ok_or_else(|| format!("tokenizer has no {s} token — this model is not vision-capable"))
    };
    let mut tokens = encode(bundle, before, false)?;
    tokens.push(tid("<|vision_start|>")?);
    let image_start = tokens.len();
    tokens.resize(tokens.len() + n_tok, tid("<|image_pad|>")?);
    tokens.push(tid("<|vision_end|>")?);
    tokens.extend(encode(bundle, after, false)?);
    Ok((tokens, Some((pimg, image_start, nx, ny))))
}

/// Shared leading-prefix tokens to PIN for the leading-prefix cache: the longest
/// common token prefix of two synthetic `[system, user_X]` renders. That's
/// exactly the system block + user-turn opening every real `[system, user…]`
/// request begins with — taking the LCP (rather than rendering the system alone)
/// makes it robust to tokenization-boundary merges, since both synthetic renders
/// go through the SAME path real requests use ([`render_and_encode`]:
/// `apply_default_system` + template + `add_generation_prompt`). `None` if the
/// model has no chat template or the renders share no leading tokens. A mismatch
/// is harmless (no seed, not a correctness bug), but the LCP makes it reliable.
pub fn compute_pin_prefix(
    bundle: &TokenizerBundle,
    system_prompt: &str,
    template_kwargs: &serde_json::Map<String, Value>,
) -> Option<Vec<u32>> {
    let template = bundle.chat_template.as_deref()?;
    let bos = bundle.bos_token.as_deref().unwrap_or("");
    let eos = bundle.eos_token.as_deref().unwrap_or("");
    let render_user = |user: &str| -> Option<Vec<u32>> {
        let mut messages = vec![ChatMessage::user(user)];
        apply_default_system(&mut messages, Some(system_prompt));
        let rendered = crate::chat_template::render(
            template,
            &messages,
            /* add_generation_prompt = */ true,
            bos,
            eos,
            template_kwargs,
        )
        .ok()?;
        encode(bundle, &rendered, false).ok()
    };
    // Two different user contents → diverge at the user content, so the LCP is
    // the shared system + user-turn-open prefix.
    let a = render_user("a")?;
    let b = render_user("the quick brown fox jumps over")?;
    let p = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    (p > 0).then(|| a[..p].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_string_and_array_forms() {
        assert_eq!(content_to_text(&json!("hi")).unwrap(), "hi");
        assert_eq!(
            content_to_text(&json!([{"type":"text","text":"a"},{"type":"text","text":"b"}]))
                .unwrap(),
            "ab"
        );
        assert!(content_to_text(&json!([{"type":"image_url","image_url":{}}])).is_err());
        assert_eq!(content_to_text(&Value::Null).unwrap(), "");
    }

    #[test]
    fn decode_image_url_data_url() {
        // base64("hi") == "aGk=".
        assert_eq!(
            decode_image_url("data:image/png;base64,aGk=").unwrap(),
            b"hi"
        );
        // Remote URLs and non-base64 data URLs are rejected.
        assert!(decode_image_url("https://example.com/x.png").is_err());
        assert!(decode_image_url("data:image/png,rawbytes").is_err());
    }

    #[test]
    fn content_mm_interleaves_marker_and_collects_images() {
        let content = json!([
            {"type": "text", "text": "look: "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": " what is it?"}
        ]);
        let (text, images) = content_to_text_and_images(&content).unwrap();
        assert_eq!(text, format!("look: {MEDIA_MARKER} what is it?"));
        assert_eq!(images, vec![b"hi".to_vec()]);
        // A bare string yields no images.
        let (text, images) = content_to_text_and_images(&json!("plain")).unwrap();
        assert_eq!(text, "plain");
        assert!(images.is_empty());
    }

    #[test]
    fn openai_mm_collects_across_messages() {
        let msgs = vec![OpenAiMessage {
            role: "user".into(),
            content: json!([
                {"type": "image_url", "image_url": {"url": "data:x;base64,aGk="}},
                {"type": "text", "text": "hi"}
            ]),
            name: None,
            tool_call_id: None,
        }];
        let (chat, images) = openai_messages_to_chat_mm(&msgs).unwrap();
        assert_eq!(chat[0].content, format!("{MEDIA_MARKER}hi"));
        assert_eq!(images.len(), 1);
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
