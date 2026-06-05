//! Audio file/byte decoding for the gemma4 `gemma4ua` audio encoder.
//!
//! The encoder wants **16 kHz mono f32 PCM** (raw samples, no mel/FFT — see
//! [`super::encoder`]). This module decodes any container/codec symphonia is
//! built with (wav/mp3/flac/aac/alac/ogg-vorbis/mp4), downmixes to mono, and
//! resamples to 16 kHz with `rubato` when the source rate differs.
//!
//! A clip already at **16 kHz mono** skips the resampler entirely, so it lands
//! byte-for-byte identical to what `llama-mtmd-cli` feeds its encoder (miniaudio
//! likewise no-ops when the input is already at the target rate) — the path used
//! for cross-engine token-parity validation.

use std::error::Error;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Target sample rate of the gemma4 audio encoder (`clip.audio.*` is fixed at
/// 16 kHz; see `mtmd-audio.cpp`).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Decode an audio file to 16 kHz mono f32 PCM.
pub fn decode_audio_file(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let file = File::open(path).map_err(|e| format!("open audio {}: {e}", path.display()))?;
    let ext = path.extension().and_then(|e| e.to_str());
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    decode_mss(mss, ext)
}

/// Decode in-memory audio bytes (e.g. a base64-decoded `input_audio` part from
/// an OpenAI request) to 16 kHz mono f32 PCM. `format_hint` is the container
/// extension/format string if known (e.g. `"wav"`, `"mp3"`).
pub fn decode_audio_bytes(
    bytes: Vec<u8>,
    format_hint: Option<&str>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    decode_mss(mss, format_hint)
}

/// Probe + decode a media stream to interleaved f32, then downmix + resample to
/// 16 kHz mono.
fn decode_mss(
    mss: MediaSourceStream<'static>,
    ext_hint: Option<&str>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut hint = Hint::new();
    if let Some(ext) = ext_hint {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("probe audio: {e}"))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or("audio stream has no audio track")?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or("audio track has no audio codec parameters")?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| format!("make audio decoder: {e}"))?;

    let mut interleaved: Vec<f32> = Vec::new();
    let mut frame: Vec<f32> = Vec::new();
    let mut sample_rate: u32 = 0;
    let mut channels: usize = 0;

    while let Some(packet) = format
        .next_packet()
        .map_err(|e| format!("read packet: {e}"))?
    {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = audio_buf.spec();
                sample_rate = spec.rate();
                channels = spec.channels().count();
                frame.resize(audio_buf.samples_interleaved(), 0.0);
                audio_buf.copy_to_slice_interleaved(&mut frame);
                interleaved.extend_from_slice(&frame);
            }
            // Skip recoverable decode errors (matches the symphonia example +
            // llama.cpp's lenient decode), bail on anything fatal.
            Err(SymphoniaError::DecodeError(_)) => {}
            Err(e) => return Err(format!("decode audio: {e}").into()),
        }
    }

    if channels == 0 || interleaved.is_empty() {
        return Err("decoded no audio samples".into());
    }

    let mono = downmix_to_mono(&interleaved, channels);
    if sample_rate == TARGET_SAMPLE_RATE {
        Ok(mono)
    } else {
        resample_to_16k(&mono, sample_rate)
    }
}

/// Average interleaved channels down to a single mono track.
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    let inv = 1.0 / channels as f32;
    for f in 0..frames {
        let base = f * channels;
        let sum: f32 = interleaved[base..base + channels].iter().sum();
        mono.push(sum * inv);
    }
    mono
}

/// Resample a mono f32 track from `src_rate` to 16 kHz with a windowed-sinc
/// resampler. Offline, whole-buffer (`process_all_into_buffer` trims the
/// resampler's startup delay for us).
fn resample_to_16k(mono: &[f32], src_rate: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::audioadapter_buffers::owned::InterleavedOwned;
    use rubato::{
        Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    if src_rate == 0 {
        return Err("audio source sample rate is 0".into());
    }

    let ratio = TARGET_SAMPLE_RATE as f64 / src_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = Async::<f32>::new_sinc(ratio, 1.1, &params, 1024, 1, FixedAsync::Input)
        .map_err(|e| format!("build resampler: {e}"))?;

    let in_frames = mono.len();
    let out_cap = resampler.process_all_needed_output_len(in_frames);
    let input =
        InterleavedSlice::new(mono, 1, in_frames).map_err(|e| format!("resample input: {e:?}"))?;
    let mut output = InterleavedOwned::<f32>::new(0.0f32, 1, out_cap);
    let (_in_used, out_used) = resampler
        .process_all_into_buffer(&input, &mut output, in_frames, None)
        .map_err(|e| format!("resample: {e}"))?;

    let mut data = output.take_data();
    data.truncate(out_used);
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_channels() {
        // 2 frames, stereo: frame0 = (1,3) -> 2, frame1 = (0,1) -> 0.5
        let interleaved = [1.0, 3.0, 0.0, 1.0];
        assert_eq!(downmix_to_mono(&interleaved, 2), vec![2.0, 0.5]);
    }

    #[test]
    fn downmix_mono_is_identity() {
        let mono = [0.1, 0.2, 0.3];
        assert_eq!(downmix_to_mono(&mono, 1), mono.to_vec());
    }

    #[test]
    fn resample_changes_length_by_ratio() {
        // 8 kHz -> 16 kHz roughly doubles the sample count.
        let src = vec![0.0f32; 8000];
        let out = resample_to_16k(&src, 8000).unwrap();
        let ratio = out.len() as f64 / src.len() as f64;
        assert!((ratio - 2.0).abs() < 0.05, "ratio {ratio} not ~2.0");
    }
}
