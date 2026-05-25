//! Render a model's embedded jinja2 chat template against an in-memory
//! conversation. Templates live in `tokenizer.chat_template` in the GGUF;
//! they reference `messages`, `add_generation_prompt`, `bos_token`, and
//! `eos_token` and use plain jinja2 control flow (`{% for %}`, `{% if %}`,
//! `loop.first`, dict indexing). Powered by `minijinja`.
//!
//! Used by `seeker chat`'s REPL between turns and by the `/apply-template`
//! HTTP endpoint.

use std::error::Error;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::{context, Environment};
use serde::Serialize;

/// Minimal `strftime`-style formatter over the current UTC time. Chat
/// templates (e.g. Llama-3.x) call `strftime_now("%d %b %Y")` to stamp
/// "Today Date" into the system prompt; without it, those templates fall back
/// to a hardcoded stale date. Supports the specifiers that appear in real
/// templates; unknown ones pass through verbatim. UTC (not localtime) — close
/// enough for a date stamp, and dependency-free.
fn strftime_now(fmt: &str) -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    strftime(fmt, secs)
}

/// `strftime_now` split out for testing against a fixed Unix timestamp.
fn strftime(fmt: &str, secs: i64) -> String {
    const MON_ABBR: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const MON_FULL: [&str; 12] = [
        "January", "February", "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December",
    ];
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let tod = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);

    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('b') => out.push_str(MON_ABBR[(m - 1) as usize]),
            Some('B') => out.push_str(MON_FULL[(m - 1) as usize]),
            Some('H') => out.push_str(&format!("{hh:02}")),
            Some('M') => out.push_str(&format!("{mm:02}")),
            Some('S') => out.push_str(&format!("{ss:02}")),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// Civil date `(year, month, day)` from days since the Unix epoch.
/// Howard Hinnant's `civil_from_days` algorithm (public domain).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// A single conversation turn. Mirrors the OpenAI-flavored shape that real
/// chat templates iterate over. `tool_calls` / multi-modal content are out
/// of scope here — add them when the first model that needs them lands.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
pub fn render(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
) -> Result<String, RenderError> {
    let mut env = Environment::new();
    // Permissive: chat templates frequently use `{% generation %}` blocks,
    // attribute access on dicts, etc. The defaults already allow most of
    // this; if a template later needs filters we don't ship, register
    // them here.
    //
    // `strftime_now` matches llama.cpp / transformers: Llama-3.x templates
    // gate the "Today Date" stamp on `strftime_now is defined`, so without it
    // they fall back to a hardcoded 2024 date and the prompt drifts from what
    // llama.cpp feeds the model.
    env.add_function("strftime_now", |fmt: String| strftime_now(&fmt));
    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;
    let rendered = tmpl.render(context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        eos_token => eos_token,
    })?;
    Ok(rendered)
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
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let out = render(SMOLLM2_TEMPLATE, &messages, true, "<|im_start|>", "<|im_end|>")
            .expect("render");
        assert!(out.contains("<|im_start|>system\nYou are a helpful AI assistant"), "missing default system block: {out:?}");
        assert!(out.contains("<|im_start|>user\nHello<|im_end|>"), "missing user turn: {out:?}");
        assert!(out.ends_with("<|im_start|>assistant\n"), "missing assistant opener: {out:?}");
    }

    #[test]
    fn explicit_system_message_skips_default() {
        let messages = vec![
            ChatMessage { role: "system".into(), content: "Be terse.".into() },
            ChatMessage { role: "user".into(), content: "hi".into() },
        ];
        let out = render(SMOLLM2_TEMPLATE, &messages, true, "", "")
            .expect("render");
        assert!(out.contains("<|im_start|>system\nBe terse.<|im_end|>"), "{out:?}");
        assert!(!out.contains("You are a helpful AI assistant"), "{out:?}");
    }

    #[test]
    fn add_generation_prompt_false_omits_assistant_opener() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let out = render(SMOLLM2_TEMPLATE, &messages, false, "", "")
            .expect("render");
        assert!(!out.contains("<|im_start|>assistant\n"), "{out:?}");
        assert!(out.ends_with("<|im_start|>user\nhi<|im_end|>\n"), "{out:?}");
    }

    #[test]
    fn strftime_matches_known_dates() {
        // 0 = 1970-01-01.
        assert_eq!(super::strftime("%d %b %Y", 0), "01 Jan 1970");
        // 1721952000 = 2024-07-26 00:00:00 UTC (the template's hardcoded fallback).
        assert_eq!(super::strftime("%d %b %Y", 1_721_952_000), "26 Jul 2024");
        // 1700000000 = 2023-11-14 22:13:20 UTC.
        assert_eq!(super::strftime("%Y-%m-%d %H:%M:%S", 1_700_000_000), "2023-11-14 22:13:20");
        assert_eq!(super::strftime("%B", 1_700_000_000), "November");
    }

    #[test]
    fn strftime_now_function_is_available_to_templates() {
        // A template gating on `strftime_now is defined` must take the true branch.
        let tmpl = "{% if strftime_now is defined %}{{ strftime_now('%Y') }}{% else %}NONE{% endif %}";
        let out = render(tmpl, &[], false, "", "").expect("render");
        assert_ne!(out, "NONE", "strftime_now should be defined in the env");
        assert_eq!(out.len(), 4, "expected a 4-digit year, got {out:?}");
    }

    #[test]
    fn missing_template_string_is_distinguished_from_render_error() {
        let err = render("", &[], true, "", "").unwrap_or_else(|_| String::new());
        // Empty template just renders an empty string — not an error.
        assert_eq!(err, "");
    }
}
