//! Audio (mmproj) support: load the gemma4 `gemma4ua` audio projector from an
//! mmproj GGUF and parse its `clip.audio.*` metadata into an [`AudioConfig`].
//!
//! The gemma4 "any-to-any" mmproj carries an audio encoder alongside the vision
//! one (`clip.has_audio_encoder = true`). The `gemma4ua` ("unified audio")
//! encoder is intentionally minimal — raw 16 kHz mono samples are chunked into
//! `n_mel_bins`-sample frames (no FFT/mel), RMS-normalized, and projected once
//! by `mm.a.input_projection.weight` into the decoder's embedding space. There
//! is no transformer/conformer: the `clip.audio.block_count /
//! attention.head_count / feed_forward_length` keys are present but **unused**
//! (faithful to llama.cpp's `clip_graph_gemma4ua::build`). The decoder layers do
//! the actual audio understanding, exactly like the `gemma4uv` vision path.

use std::error::Error;

use crate::gguf::GgufFile;
use crate::vision::{get_bool, get_str};

pub mod encoder;
// Host-side decode now lives in seeker-core; re-export it here so
// `crate::audio::decode` paths in this crate resolve unchanged.
pub use seeker_core::audio::decode;

/// The kind of audio projector shipped in an mmproj GGUF
/// (`clip.audio.projector_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioProjectorType {
    /// Gemma 4 unified audio (`"gemma4ua"`): RMSNorm → single linear projection
    /// of raw `n_mel_bins`-sample frames. No encoder tower.
    Gemma4Ua,
}

impl AudioProjectorType {
    /// Parse the `clip.audio.projector_type` metadata string. Errors clearly for
    /// any audio projector seeker doesn't (yet) support — e.g. `"gemma4a"`, the
    /// full Conformer variant.
    pub fn parse(s: &str) -> Result<AudioProjectorType, Box<dyn Error>> {
        match s {
            "gemma4ua" => Ok(AudioProjectorType::Gemma4Ua),
            other => Err(format!("unsupported clip.audio.projector_type: {other:?}").into()),
        }
    }
}

/// Parsed `clip.audio.*` configuration for an audio projector.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// The detected audio projector type.
    pub projector_type: AudioProjectorType,
    /// Frame size in samples (640 for gemma4ua = 40 ms @ 16 kHz). Sourced from
    /// the projection weight's input (K) dim — the authoritative value. NOT from
    /// `clip.audio.num_mel_bins`: that key is 128 in the real mmproj (a vestigial
    /// value from the conformer variant), but the `gemma4ua` projection takes raw
    /// 640-sample frames. llama.cpp likewise hardcodes 640 and ignores the key.
    pub frame_size: u32,
    /// RMSNorm epsilon. Hardcoded to `1e-6` for `gemma4ua` to match llama.cpp's
    /// `clip.cpp` (which overrides any metadata value for this projector).
    pub eps: f32,
}

/// Whether an mmproj GGUF advertises an audio encoder (`clip.has_audio_encoder`).
pub fn has_audio(gguf: &GgufFile) -> bool {
    get_bool(gguf, "clip.has_audio_encoder").unwrap_or(false)
}

/// Parse an [`AudioConfig`] from an mmproj GGUF's `clip.audio.*` metadata.
/// Returns `Err` if the GGUF has no (recognized) audio encoder.
pub fn parse_config(gguf: &GgufFile) -> Result<AudioConfig, Box<dyn Error>> {
    if !has_audio(gguf) {
        return Err("mmproj has no audio encoder (clip.has_audio_encoder != true)".into());
    }
    let proj_str = get_str(gguf, "clip.audio.projector_type")
        .ok_or("mmproj missing clip.audio.projector_type")?;
    let projector_type = AudioProjectorType::parse(proj_str)?;
    // The frame size = the projection weight's input (K) dim (640 for gemma4ua).
    // Read it straight from the GGUF tensor shape (no GPU needed) rather than
    // trusting `clip.audio.num_mel_bins` (128 here — vestigial, see `frame_size`).
    let frame_size = gguf
        .tensor("mm.a.input_projection.weight")
        .and_then(|t| t.dims.first().copied())
        .filter(|&k| k > 0)
        .ok_or("mmproj missing/empty mm.a.input_projection.weight (no gemma4ua audio encoder)")?
        as u32;
    Ok(AudioConfig {
        projector_type,
        frame_size,
        eps: 1e-6,
    })
}
