use tokenizers::TokenizerImpl;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;

/// `tokenizers::Tokenizer` is just `TokenizerImpl<Wrappers…>`; this concrete
/// alias keeps every signature in the module short and uniform.
pub type Tokenizer = TokenizerImpl<
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// Output of `tokenizer::build_tokenizer`. `tokenizer` is ready to encode /
/// decode. `add_bos_default` / `add_eos_default` mirror the GGUF's
/// `tokenizer.ggml.add_{bos,eos}_token` flags so the caller can pick a
/// sensible default when the user didn't pass `--add-special`. `bos_id` /
/// `eos_id` are informational — exposed for callers that want to surface
/// them in output later.
#[allow(dead_code)]
pub struct TokenizerBundle {
    pub tokenizer: Tokenizer,
    pub model_kind: String,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub add_bos_default: bool,
    pub add_eos_default: bool,
    /// Raw `tokenizer.chat_template` from the GGUF (jinja2 source). `None`
    /// for base / non-chat-tuned models.
    pub chat_template: Option<String>,
    /// String form of the BOS/EOS tokens — chat templates reference these
    /// via `{{ bos_token }}` / `{{ eos_token }}`. Resolved from the
    /// `tokens` array using the corresponding `*_token_id`.
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    /// Every end-of-generation token id (EOS + EOT + EOM + well-known turn
    /// terminators), matching llama.cpp's `llama_token_is_eog`. Generation
    /// should stop on **any** of these; stopping on only `eos_id` lets a model
    /// whose turn terminator differs from its EOS run to the token budget.
    pub eog_ids: Vec<u32>,
}
