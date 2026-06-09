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

/// Decode an OpenAI `input_audio` content part into raw audio file bytes. The
/// part is `{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}`;
/// `data` is base64-encoded audio (wav/mp3/flac/…). A `data:<mime>;base64,<…>`
/// data URL is also accepted. The container is sniffed at decode time, so the
/// `format` field is advisory and not required.
pub fn decode_input_audio(part: &Value) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let data = part
        .get("input_audio")
        .and_then(|o| o.get("data"))
        .and_then(|d| d.as_str())
        .ok_or("input_audio part missing string `input_audio.data`")?;
    // Accept a raw base64 string (the OpenAI shape) or a data URL.
    let b64 = match data.strip_prefix("data:") {
        Some(rest) => {
            let comma = rest
                .find(',')
                .ok_or("malformed input_audio data URL: missing comma")?;
            &rest[comma + 1..]
        }
        None => data,
    };
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("input_audio base64 decode failed: {e}"))
}

/// Like [`content_to_text`] but also extracts media: each `image_url` /
/// `input_audio` part emits the [`MEDIA_MARKER`] into the text (where the media
/// sits in the turn) and its decoded bytes are collected, in order. Returns
/// `(text, images, audios)`. Used by the multimodal chat path.
#[allow(clippy::type_complexity)] // (text, images, audios) is clearer inline than an alias
pub fn content_to_text_and_media(
    content: &Value,
) -> Result<(String, Vec<Vec<u8>>, Vec<Vec<u8>>), String> {
    let mut text = String::new();
    let mut images = Vec::new();
    let mut audios = Vec::new();
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
                    Some("input_audio") => {
                        audios.push(decode_input_audio(part)?);
                        text.push_str(MEDIA_MARKER);
                    }
                    Some(other) => return Err(format!("unsupported content part type {other:?}")),
                    None => return Err("content array part missing `type`".to_string()),
                }
            }
        }
        other => return Err(format!("unsupported message content: {other}")),
    }
    Ok((text, images, audios))
}

/// OpenAI chat messages → (engine `ChatMessage`s, images, audios — in order).
/// The media-bearing turn's content carries the [`MEDIA_MARKER`] where each item
/// sat, so the chat template renders it in place. The multimodal sibling of
/// [`openai_messages_to_chat`].
#[allow(clippy::type_complexity)] // (messages, images, audios) is clearer inline than an alias
pub fn openai_messages_to_chat_mm(
    messages: &[OpenAiMessage],
) -> Result<(Vec<ChatMessage>, Vec<Vec<u8>>, Vec<Vec<u8>>), String> {
    let mut out = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    let mut audios = Vec::new();
    for m in messages {
        let (content, imgs, auds) = content_to_text_and_media(&m.content)?;
        images.extend(imgs);
        audios.extend(auds);
        out.push(ChatMessage {
            role: m.role.clone(),
            content,
            reasoning_content: None,
        });
    }
    Ok((out, images, audios))
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

/// Parse an embeddings `input`/`content` JSON value into one tokenized sequence
/// per input. Accepts a string, an array of strings, a pre-tokenized array of
/// ints, or an array of int arrays (mixed string/array elements allowed).
/// Strings are encoded with special tokens (BOS/EOS per the model), matching
/// llama.cpp.
pub fn embedding_inputs_to_tokens(
    bundle: &TokenizerBundle,
    value: &serde_json::Value,
) -> Result<Vec<Vec<u32>>, String> {
    use serde_json::Value;
    // An all-integer array → a single pre-tokenized input.
    let as_tokens = |arr: &[Value]| -> Option<Vec<u32>> {
        arr.iter()
            .map(|v| v.as_u64().map(|n| n as u32))
            .collect::<Option<Vec<u32>>>()
    };
    match value {
        Value::String(s) => Ok(vec![encode(bundle, s, true)?]),
        Value::Array(items) if items.is_empty() => Err("`input` is empty".into()),
        Value::Array(items) => {
            if let Some(toks) = as_tokens(items) {
                return Ok(vec![toks]); // array of ints → one pre-tokenized input
            }
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    Value::String(s) => out.push(encode(bundle, s, true)?),
                    Value::Array(inner) => {
                        out.push(as_tokens(inner).ok_or(
                            "`input` array elements must be strings or integer token arrays",
                        )?)
                    }
                    other => return Err(format!("unsupported `input` element: {other}")),
                }
            }
            Ok(out)
        }
        other => Err(format!("unsupported `input`: {other}")),
    }
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

/// The one media item (image XOR audio) extracted from a chat request, ready
/// for the worker to encode + splice. The handler maps this to the worker's
/// `ServeImage` / `ServeAudio`.
pub enum MediaParts {
    Image {
        pimg: PreprocessedImage,
        image_start: usize,
        nx: usize,
        ny: usize,
    },
    Audio {
        /// 16 kHz mono f32 (decoded on the handler thread).
        samples: Vec<f32>,
        audio_start: usize,
        n_tok: usize,
    },
}

/// As [`render_and_encode`] but for a chat request that may carry an image or an
/// audio clip: the rendered prompt has a [`MEDIA_MARKER`] where the media sits,
/// which we replace with the projector's block (vision
/// `<|vision_start|><|image_pad|>×n_tok<|vision_end|>`, or audio
/// `<|audio><|audio|>×n_tok<audio|>`). Returns the token ids plus the
/// [`MediaParts`] for the worker to encode + splice. First cut: at most one
/// media item per request (image and audio are mutually exclusive).
pub fn render_and_encode_mm(
    state: &AppState,
    mut messages: Vec<ChatMessage>,
    images: &[Vec<u8>],
    audios: &[Vec<u8>],
) -> Result<(Vec<u32>, Option<MediaParts>), String> {
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
    if !images.is_empty() && !audios.is_empty() {
        return Err("a request may carry an image or audio, not both".into());
    }
    if images.is_empty() && audios.is_empty() {
        return Ok((encode(bundle, &rendered, false)?, None));
    }
    let tid = |s: &str| -> Result<u32, String> {
        bundle
            .tokenizer
            .token_to_id(s)
            .ok_or_else(|| format!("tokenizer has no {s} token — this model lacks that modality"))
    };
    let (before, after) = rendered
        .split_once(MEDIA_MARKER)
        .ok_or("rendered chat prompt lost the <__media__> marker")?;

    if !images.is_empty() {
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
        let mut tokens = encode(bundle, before, false)?;
        tokens.push(tid("<|vision_start|>")?);
        let image_start = tokens.len();
        tokens.resize(tokens.len() + n_tok, tid("<|image_pad|>")?);
        tokens.push(tid("<|vision_end|>")?);
        tokens.extend(encode(bundle, after, false)?);
        return Ok((
            tokens,
            Some(MediaParts::Image {
                pimg,
                image_start,
                nx,
                ny,
            }),
        ));
    }

    // Audio.
    if audios.len() > 1 {
        return Err("only one audio clip per request is supported".into());
    }
    let acfg = state.audio_config().ok_or(
        "this server has no audio model (mmproj audio encoder); audio input is unsupported",
    )?;
    let samples = crate::audio::decode::decode_audio_bytes(audios[0].clone(), None)
        .map_err(|e| format!("audio decode failed: {e}"))?;
    let n_tok = samples.len().div_ceil(acfg.frame_size as usize);
    let mut tokens = encode(bundle, before, false)?;
    tokens.push(tid("<|audio>")?);
    let audio_start = tokens.len();
    tokens.resize(tokens.len() + n_tok, tid("<|audio|>")?);
    tokens.push(tid("<audio|>")?);
    tokens.extend(encode(bundle, after, false)?);
    Ok((
        tokens,
        Some(MediaParts::Audio {
            samples,
            audio_start,
            n_tok,
        }),
    ))
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
    fn content_mm_interleaves_marker_and_collects_media() {
        let content = json!([
            {"type": "text", "text": "look: "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": " what is it?"}
        ]);
        let (text, images, audios) = content_to_text_and_media(&content).unwrap();
        assert_eq!(text, format!("look: {MEDIA_MARKER} what is it?"));
        assert_eq!(images, vec![b"hi".to_vec()]);
        assert!(audios.is_empty());
        // A bare string yields no media.
        let (text, images, audios) = content_to_text_and_media(&json!("plain")).unwrap();
        assert_eq!(text, "plain");
        assert!(images.is_empty());
        assert!(audios.is_empty());
        // An `input_audio` part emits the marker and collects the decoded bytes.
        let content = json!([
            {"type": "input_audio", "input_audio": {"data": "aGk=", "format": "wav"}},
            {"type": "text", "text": " transcribe"}
        ]);
        let (text, images, audios) = content_to_text_and_media(&content).unwrap();
        assert_eq!(text, format!("{MEDIA_MARKER} transcribe"));
        assert!(images.is_empty());
        assert_eq!(audios, vec![b"hi".to_vec()]);
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
        let (chat, images, audios) = openai_messages_to_chat_mm(&msgs).unwrap();
        assert_eq!(chat[0].content, format!("{MEDIA_MARKER}hi"));
        assert_eq!(images.len(), 1);
        assert!(audios.is_empty());
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
