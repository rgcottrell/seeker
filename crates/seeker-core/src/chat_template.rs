//! Render a model's embedded jinja2 chat template against an in-memory
//! conversation. Templates live in `tokenizer.chat_template` in the GGUF;
//! they reference `messages`, `add_generation_prompt`, `bos_token`, and
//! `eos_token` and use plain jinja2 control flow (`{% for %}`, `{% if %}`,
//! `loop.first`, dict indexing). Powered by `minijinja`.
//!
//! Used by `seeker chat`'s REPL between turns and by the `/apply-template`
//! HTTP endpoint.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use minijinja::value::Value;
use minijinja::{Environment, Error as MjError, ErrorKind, State};
use serde::{Deserialize, Serialize};

/// Handle Python-style string methods that real-world chat templates rely
/// on but MiniJinja doesn't ship by default. Qwen3's template uses
/// `content.startswith(...)`, `content.endswith(...)`, `content.split(...)`,
/// `content.rstrip(...)`, `content.lstrip(...)`, `content.strip(...)`,
/// `content.replace(...)`. Registered via
/// `Environment::set_unknown_method_callback`.
fn unknown_method_callback(
    _state: &State,
    receiver: &Value,
    name: &str,
    args: &[Value],
) -> Result<Value, MjError> {
    // Python dict `.get(key[, default])` on a map receiver (e.g. gemma4's chat
    // template does `message.get('reasoning')`). minijinja maps support
    // indexing but not the `.get` method, so shim it: return the item, or the
    // default (or Undefined ⇒ falsy) when the key is absent.
    if name == "get" && receiver.as_str().is_none() {
        let key = args
            .first()
            .ok_or_else(|| MjError::new(ErrorKind::InvalidOperation, "get: expected key"))?;
        let item = receiver.get_item(key).unwrap_or(Value::UNDEFINED);
        if item.is_undefined() {
            return Ok(args.get(1).cloned().unwrap_or(Value::UNDEFINED));
        }
        return Ok(item);
    }
    let s = match receiver.as_str() {
        Some(s) => s,
        None => return Err(MjError::from(ErrorKind::UnknownMethod)),
    };
    match name {
        "startswith" => {
            let prefix = args.first().and_then(|a| a.as_str()).ok_or_else(|| {
                MjError::new(
                    ErrorKind::InvalidOperation,
                    "startswith: expected string arg",
                )
            })?;
            Ok(Value::from(s.starts_with(prefix)))
        }
        "endswith" => {
            let suffix = args.first().and_then(|a| a.as_str()).ok_or_else(|| {
                MjError::new(ErrorKind::InvalidOperation, "endswith: expected string arg")
            })?;
            Ok(Value::from(s.ends_with(suffix)))
        }
        "split" => {
            // .split(sep) — like Python: splits at every occurrence.
            // .split() — like Python: splits on whitespace, collapsing runs.
            let parts: Vec<String> = match args.first().and_then(|a| a.as_str()) {
                Some(sep) if !sep.is_empty() => s.split(sep).map(|p| p.to_string()).collect(),
                _ => s.split_whitespace().map(|p| p.to_string()).collect(),
            };
            Ok(Value::from(parts))
        }
        "strip" => Ok(Value::from(strip_str(s, args.first(), true, true))),
        "lstrip" => Ok(Value::from(strip_str(s, args.first(), true, false))),
        "rstrip" => Ok(Value::from(strip_str(s, args.first(), false, true))),
        "replace" => {
            let old = args.first().and_then(|a| a.as_str()).ok_or_else(|| {
                MjError::new(ErrorKind::InvalidOperation, "replace: expected old")
            })?;
            let new = args.get(1).and_then(|a| a.as_str()).ok_or_else(|| {
                MjError::new(ErrorKind::InvalidOperation, "replace: expected new")
            })?;
            Ok(Value::from(s.replace(old, new)))
        }
        _ => Err(MjError::from(ErrorKind::UnknownMethod)),
    }
}

/// Helper for strip/lstrip/rstrip. When `chars` is None, strip whitespace
/// (matches Python); otherwise strip any character contained in `chars`.
fn strip_str(s: &str, chars: Option<&Value>, left: bool, right: bool) -> String {
    let trim_chars: Option<Vec<char>> = chars.and_then(|v| v.as_str()).map(|c| c.chars().collect());
    let is_trim = |ch: char| match &trim_chars {
        Some(set) => set.contains(&ch),
        None => ch.is_whitespace(),
    };
    let mut start = 0;
    let mut end = s.len();
    if left {
        while let Some(c) = s[start..].chars().next() {
            if is_trim(c) {
                start += c.len_utf8();
            } else {
                break;
            }
        }
    }
    if right {
        while let Some(c) = s[start..end].chars().next_back() {
            if is_trim(c) {
                end -= c.len_utf8();
            } else {
                break;
            }
        }
    }
    s[start..end].to_string()
}

/// `strftime`-style formatter over the current **local** wall-clock time. Chat
/// templates (e.g. Llama-3.x) call `strftime_now("%d %b %Y")` to stamp "Today
/// Date" into the system prompt; without the function defined they fall back to
/// a hardcoded stale date. Because the stamp goes into the prompt the model
/// sees, it should reflect the user's local date — so we format in the system
/// timezone via `jiff`, matching llama.cpp / transformers. On the unreachable
/// chance the template passes a specifier jiff rejects, falls back to empty
/// rather than panicking.
fn strftime_now(fmt: &str) -> String {
    jiff::fmt::strtime::format(fmt, &jiff::Zoned::now()).unwrap_or_default()
}

/// A single conversation turn. Mirrors the OpenAI-flavored shape that real
/// chat templates iterate over. `tool_calls` / multi-modal content are out
/// of scope here — add them when the first model that needs them lands.
///
/// `reasoning_content` holds a thinking/reasoning model's chain-of-thought
/// separately from the final `content`. Reasoning templates (e.g. Qwen3)
/// read `message.reasoning_content` directly and decide per-turn whether to
/// re-include it. It's omitted from serialization when `None`, so messages
/// without reasoning render exactly as a plain `{role, content}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// A system turn — the leading instruction message. Most chat templates
    /// render it specially (and skip their built-in default system prompt when
    /// one is present).
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            reasoning_content: None,
        }
    }

    /// A user turn (never carries reasoning).
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            reasoning_content: None,
        }
    }

    /// An assistant turn, optionally with its reasoning split out.
    pub fn assistant(content: impl Into<String>, reasoning_content: Option<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            reasoning_content,
        }
    }
}

#[derive(Debug)]
pub struct RenderError(minijinja::Error);

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chat template render failed: {}", self.0)
    }
}

impl Error for RenderError {}

impl From<minijinja::Error> for RenderError {
    fn from(e: minijinja::Error) -> Self {
        RenderError(e)
    }
}

/// Render `template` (raw jinja2 source from `tokenizer.chat_template`)
/// against the conversation. `add_generation_prompt` typically appends the
/// assistant-role opener that the model expects to continue from.
///
/// `bos_token` and `eos_token` are the *string* forms of the special
/// tokens (looked up from the GGUF's `tokenizer.ggml.tokens` array). They
/// may be empty strings if the model has no BOS/EOS — most chat templates
/// don't reference them and rendering still succeeds.
///
/// `extra_context` carries caller-supplied template variables (from
/// `--chat-template-kwargs`). Its entries are merged in **last**, so a kwarg
/// key overrides the built-in variable of the same name. This is also the
/// only way to set template-specific switches like `enable_thinking` — e.g.
/// `{"enable_thinking": false}`; otherwise the template's own default applies.
pub fn render(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
    extra_context: &serde_json::Map<String, serde_json::Value>,
) -> Result<String, RenderError> {
    let mut env = Environment::new();
    // Permissive: chat templates frequently use `{% generation %}` blocks,
    // attribute access on dicts, etc. The defaults already allow most of
    // this; if a template later needs filters we don't ship, register
    // them here.
    //
    // Defining `strftime_now` is the parity win: Llama-3.x templates gate the
    // "Today Date" stamp on `strftime_now is defined`, so without it they fall
    // back to a hardcoded 2024 date and the prompt drifts from what llama.cpp
    // feeds the model. Formats in the system-local timezone (via jiff) so the
    // stamped date matches the user's wall clock.
    env.add_function("strftime_now", |fmt: String| strftime_now(&fmt));
    // Python-style string methods (.startswith, .endswith, .split, .strip,
    // .lstrip, .rstrip, .replace) that Qwen3 / Llama / DeepSeek chat
    // templates call directly on strings.
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;
    // Built dynamically (rather than via the compile-time `context!` macro)
    // so `extra_context` kwargs can be merged in. `Value::from_serialize` is
    // infallible — serialization errors surface as an invalid value that
    // only errors if the template actually touches it.
    let mut ctx: BTreeMap<String, Value> = BTreeMap::new();
    ctx.insert("messages".into(), Value::from_serialize(messages));
    ctx.insert(
        "add_generation_prompt".into(),
        Value::from(add_generation_prompt),
    );
    ctx.insert("bos_token".into(), Value::from(bos_token));
    ctx.insert("eos_token".into(), Value::from(eos_token));
    // Merged last: kwargs win on key collision (see doc comment). This is the
    // only source of switches like `enable_thinking`.
    for (k, v) in extra_context {
        ctx.insert(k.clone(), Value::from_serialize(v));
    }
    let rendered = tmpl.render(ctx)?;
    if crate::runtime_flags::chat_debug() {
        eprintln!("=== rendered chat prompt ===\n{rendered}\n=== end ===");
    }
    Ok(rendered)
}

/// clap `value_parser` for `--chat-template-kwargs`: parse a JSON **object**
/// string into the map merged into the template context by [`render`].
/// Mirrors llama.cpp's `--chat-template-kwargs '{"key":"value",...}'`.
pub fn parse_template_kwargs(
    s: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    match serde_json::from_str::<serde_json::Value>(s).map_err(|e| format!("invalid JSON: {e}"))? {
        serde_json::Value::Object(m) => Ok(m),
        _ => Err("must be a JSON object, e.g. '{\"key1\":\"value1\",\"key2\":\"value2\"}'".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim SmolLM2-Instruct chat template from
    /// `tokenizer.chat_template` in the GGUF. Newlines inside the literals
    /// are intentional (they encode `\n` in the rendered output).
    const SMOLLM2_TEMPLATE: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    #[test]
    fn smollm2_template_injects_default_system_and_assistant_opener() {
        let messages = vec![ChatMessage::user("Hello")];
        let out = render(
            SMOLLM2_TEMPLATE,
            &messages,
            true,
            "<|im_start|>",
            "<|im_end|>",
            &serde_json::Map::new(),
        )
        .expect("render");
        assert!(
            out.contains("<|im_start|>system\nYou are a helpful AI assistant"),
            "missing default system block: {out:?}"
        );
        assert!(
            out.contains("<|im_start|>user\nHello<|im_end|>"),
            "missing user turn: {out:?}"
        );
        assert!(
            out.ends_with("<|im_start|>assistant\n"),
            "missing assistant opener: {out:?}"
        );
    }

    /// An assistant turn's `reasoning_content` must not leak into a
    /// content-only history render (this is exactly how reasoning models like
    /// Qwen3 drop the previous turn's chain-of-thought) — which is the root of
    /// the prefix-reuse divergence: the generated `<think>` tokens are in the
    /// KV cache but absent from the re-rendered prompt, forcing a re-prefill.
    #[test]
    fn reasoning_content_not_rendered_by_content_only_template() {
        let messages = vec![
            ChatMessage::user("q1"),
            ChatMessage::assistant("the final answer", Some("hidden chain of thought".into())),
            ChatMessage::user("q2"),
        ];
        let out = render(
            SMOLLM2_TEMPLATE,
            &messages,
            true,
            "",
            "",
            &serde_json::Map::new(),
        )
        .expect("render");
        assert!(
            out.contains("<|im_start|>assistant\nthe final answer<|im_end|>"),
            "answer should round-trip into history: {out:?}"
        );
        assert!(
            !out.contains("hidden chain of thought"),
            "reasoning_content leaked into the rendered history: {out:?}"
        );
    }

    #[test]
    fn system_constructor_renders_as_system_turn() {
        let messages = vec![ChatMessage::system("Be terse."), ChatMessage::user("hi")];
        let out = render(
            SMOLLM2_TEMPLATE,
            &messages,
            true,
            "",
            "",
            &serde_json::Map::new(),
        )
        .expect("render");
        assert!(
            out.contains("<|im_start|>system\nBe terse.<|im_end|>"),
            "{out:?}"
        );
        assert!(
            !out.contains("You are a helpful AI assistant"),
            "default not skipped: {out:?}"
        );
    }

    #[test]
    fn explicit_system_message_skips_default() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "Be terse.".into(),
                reasoning_content: None,
            },
            ChatMessage::user("hi"),
        ];
        let out = render(
            SMOLLM2_TEMPLATE,
            &messages,
            true,
            "",
            "",
            &serde_json::Map::new(),
        )
        .expect("render");
        assert!(
            out.contains("<|im_start|>system\nBe terse.<|im_end|>"),
            "{out:?}"
        );
        assert!(!out.contains("You are a helpful AI assistant"), "{out:?}");
    }

    #[test]
    fn add_generation_prompt_false_omits_assistant_opener() {
        let messages = vec![ChatMessage::user("hi")];
        let out = render(
            SMOLLM2_TEMPLATE,
            &messages,
            false,
            "",
            "",
            &serde_json::Map::new(),
        )
        .expect("render");
        assert!(!out.contains("<|im_start|>assistant\n"), "{out:?}");
        assert!(out.ends_with("<|im_start|>user\nhi<|im_end|>\n"), "{out:?}");
    }

    /// The strftime specifiers real templates use must format as expected.
    /// Tested against a fixed instant in a fixed (UTC) zone so it's
    /// machine-independent; `strftime_now` uses the same jiff formatter on the
    /// system-local zone.
    #[test]
    fn strftime_formats_known_specifiers() {
        // 1_700_000_000 = 2023-11-14 22:13:20 UTC.
        let z = jiff::Timestamp::from_second(1_700_000_000)
            .unwrap()
            .to_zoned(jiff::tz::TimeZone::UTC);
        let f = |fmt: &str| jiff::fmt::strtime::format(fmt, &z).unwrap();
        assert_eq!(f("%d %b %Y"), "14 Nov 2023");
        assert_eq!(f("%Y-%m-%d %H:%M:%S"), "2023-11-14 22:13:20");
        assert_eq!(f("%B"), "November");
    }

    #[test]
    fn strftime_now_function_is_available_to_templates() {
        // A template gating on `strftime_now is defined` must take the true branch.
        let tmpl =
            "{% if strftime_now is defined %}{{ strftime_now('%Y') }}{% else %}NONE{% endif %}";
        let out = render(tmpl, &[], false, "", "", &serde_json::Map::new()).expect("render");
        assert_ne!(out, "NONE", "strftime_now should be defined in the env");
        assert_eq!(out.len(), 4, "expected a 4-digit year, got {out:?}");
    }

    #[test]
    fn missing_template_string_is_distinguished_from_render_error() {
        let err = render("", &[], true, "", "", &serde_json::Map::new())
            .unwrap_or_else(|_| String::new());
        // Empty template just renders an empty string — not an error.
        assert_eq!(err, "");
    }
}
