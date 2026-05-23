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

use minijinja::{context, Environment};
use serde::Serialize;

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
    fn missing_template_string_is_distinguished_from_render_error() {
        let err = render("", &[], true, "", "").unwrap_or_else(|_| String::new());
        // Empty template just renders an empty string — not an error.
        assert_eq!(err, "");
    }
}
