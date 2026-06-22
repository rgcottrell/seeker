//! Gemma 4 `gemma4ua` audio encoder.
//!
//! A faithful port of llama.cpp's `clip_graph_gemma4ua::build`
//! (`tools/mtmd/models/gemma4ua.cpp`), which is just two ops:
//!
//! ```text
//!   raw 16 kHz mono samples, chunked into [frame_size, n_tok]
//!     → RMSNorm (eps 1e-6, no weight, over frame_size)
//!     → matmul(mm.a.input_projection.weight)         → [proj_dim, n_tok]
//! ```
//!
//! No FFT/mel, no attention, no encoder tower (`clip.audio.block_count` etc. are
//! ignored). The preprocessor frames `n_samples` raw samples into `n_tok =
//! ceil(n_samples / frame_size)` columns of `frame_size` samples (zero-padding
//! the tail). llama.cpp stores those frames "mel-major" then `permute(1,0,2,3)`s
//! to `[frame_size, n_tok]`; that permuted contiguous buffer is *exactly* the
//! raw sample array zero-padded to `n_tok * frame_size`, so we build it directly
//! and skip the permute. `proj_dim` is read from the projection weight — it must
//! equal the decoder's `n_embd` (the embeddings are spliced straight into the
//! residual stream, like the `gemma4uv` vision path).

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::Engine;
use crate::inference::ops::{matmul, rms_norm};
use crate::inference::weights::WeightsHandle;
use crate::vision::encoder::{alloc_scratch_write, dense_view, f32_to_bytes};

use super::AudioConfig;

/// Tensor name of the gemma4ua audio input projection (`TN_A_MM_INP_PROJ`).
const INPUT_PROJECTION: &str = "mm.a.input_projection.weight";

/// Encode 16 kHz mono f32 `samples` through the gemma4 `gemma4ua` projector.
/// Returns `(embeddings [proj_dim · n_tok] column-major, n_tok)`.
pub fn encode_audio_gemma4(
    engine: &mut Engine,
    weights: &WeightsHandle,
    cfg: &AudioConfig,
    samples: &[f32],
) -> Result<(Vec<f32>, usize), Box<dyn Error>> {
    let frame = cfg.frame_size as usize;
    if frame == 0 {
        return Err("audio frame_size is 0".into());
    }
    if samples.is_empty() {
        return Err("encode_audio_gemma4: empty samples".into());
    }
    let n_tok = samples.len().div_ceil(frame);

    // Build the [frame_size, n_tok] input directly (== raw samples zero-padded
    // to n_tok*frame; column t holds samples[t*frame .. t*frame+frame]).
    let mut buf = samples.to_vec();
    buf.resize(n_tok * frame, 0.0);

    let proj = weights.view(INPUT_PROJECTION)?;
    if proj.dims[0] != frame as u64 {
        return Err(format!(
            "{INPUT_PROJECTION} K dim {} != frame_size {frame}",
            proj.dims[0]
        )
        .into());
    }
    let proj_dim = proj.dims[1];
    let frame_u = frame as u64;
    let n_tok_u = n_tok as u64;
    let eps = cfg.eps;

    let out = engine.forward(weights, |ctx| {
        let in_r = alloc_scratch_write(ctx, &f32_to_bytes(&buf))?;
        let in_v = dense_view(&in_r, [frame_u, n_tok_u, 1, 1]);

        // RMSNorm over the frame_size dim (ne0), no learned weight.
        let normed = ctx.alloc_tensor([frame_u, n_tok_u, 1, 1], GgmlType::F32)?;
        rms_norm::record_noweight(ctx, in_v, normed, eps)?;

        // Project each frame into decoder embedding space: [proj_dim, n_tok].
        let outp = ctx.alloc_tensor([proj_dim, n_tok_u, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, proj, normed, outp)?;
        Ok(outp.range())
    })?;

    Ok((out, n_tok))
}

#[cfg(test)]
mod tests {
    #[test]
    fn n_tok_is_ceil_div_frame() {
        let frame = 640usize;
        let cases: [(usize, usize); 6] = [
            (1, 1),
            (640, 1),
            (641, 2),
            (1280, 2),
            (1281, 3),
            (16000, 25),
        ];
        for (n, want) in cases {
            assert_eq!(n.div_ceil(frame), want, "n_samples={n}");
        }
    }
}
