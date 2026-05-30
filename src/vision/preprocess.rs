//! CPU image preprocessing for the Qwen3-VL vision tower (mmproj).
//!
//! This is a faithful port of llama.cpp's qwen3vl preprocessing path. The two
//! reference functions are in
//! `/home/bob/tools/llama.cpp/src/tools/mtmd/mtmd-image.cpp`:
//!
//!   * `img_tool::calc_size_preserved_ratio(inp, align, min_px, max_px)`
//!     (the 4-arg "smart_resize", line 156) — ported as [`smart_resize`].
//!   * `img_tool::resize_bilinear` (line 212) — ported as [`resize_bilinear_u8`].
//!     llama.cpp dispatches QWEN3VL to `mtmd_image_preprocessor_dyn_size`,
//!     which calls `calc_size_preserved_ratio` then `img_tool::resize(...,
//!     RESIZE_ALGO_BILINEAR, PAD_NONE)`. The resize filter for QWEN3VL is
//!     **bilinear** (clip.cpp:1398-1399 sets `image_resize_algo =
//!     RESIZE_ALGO_BILINEAR`).
//!
//! Critically, llama.cpp's `resize_bilinear` is **corner-aligned** (it maps
//! `px = x * (src-1)/(dst-1)`, NOT the half-pixel convention used by most image
//! libraries), so we cannot delegate the resize to the `image` crate's
//! `imageops` — we reimplement the exact loop here. The `image` crate is used
//! only to *decode* the file to interleaved RGB8.
//!
//! Normalization (`img_u8_to_f32`) is `(px/255 - mean)/std` per channel, and
//! the final pixel tensor is laid out **planar / channel-major** exactly as
//! llama.cpp packs `inp_raw` (clip.cpp:3403-3420): `[R plane (H*W, row-major),
//! G plane, B plane]`. See [`PreprocessedImage::pixels`].

use std::error::Error;
use std::path::Path;

/// Default min image tokens (after the 2x2 spatial merge) for the QWEN3VL
/// preprocessor when the mmproj GGUF carries no `clip.vision.image_min_pixels`.
///
/// Source: llama.cpp `clip.cpp` QWEN2VL/QWEN25VL/QWEN3VL case calls
/// `hparams.set_limit_image_tokens(8, 4096)`. The 8 is the min token count,
/// the 4096 the max.
pub const QWEN3VL_DEFAULT_MIN_TOKENS: u32 = 8;
/// Default max image tokens (after merge) for QWEN3VL — see above.
pub const QWEN3VL_DEFAULT_MAX_TOKENS: u32 = 4096;

/// Configuration for the smart-resize + normalize preprocessing step.
///
/// `min_pixels`/`max_pixels` are in *raw resized-image pixels* (W*H before the
/// 2x2 merge), matching llama.cpp's `hparams.image_min_pixels` /
/// `image_max_pixels`. They are derived from the GGUF if it carries
/// `clip.vision.image_min_pixels` / `..._max_pixels`; otherwise they fall back
/// to the QWEN3VL defaults via [`PreprocessConfig::qwen3vl_default`].
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Vision patch size in pixels (`clip.vision.patch_size`, =16 for Qwen3-VL).
    pub patch_size: u32,
    /// Spatial merge factor (`clip.vision.spatial_merge_size`, =2).
    pub spatial_merge_size: u32,
    /// Per-channel normalization mean (`clip.vision.image_mean`, =[0.5;3]).
    pub image_mean: [f32; 3],
    /// Per-channel normalization std (`clip.vision.image_std`, =[0.5;3]).
    pub image_std: [f32; 3],
    /// Lower clamp on resized W*H (raw pixels).
    pub min_pixels: u32,
    /// Upper clamp on resized W*H (raw pixels).
    pub max_pixels: u32,
}

impl PreprocessConfig {
    /// Build a config from vision params, deriving min/max pixels from the
    /// llama.cpp QWEN3VL token limits when the GGUF doesn't specify them.
    ///
    /// llama.cpp `clip-model.h::set_limit_image_tokens`:
    /// `patch_area = patch_size² * n_merge²`; `image_{min,max}_pixels =
    /// {min,max}_tokens * patch_area`. For Qwen3-VL (patch=16, merge=2):
    /// patch_area = 16²·2² = 1024 ⇒ min_pixels = 8·1024 = 8192,
    /// max_pixels = 4096·1024 = 4_194_304.
    pub fn qwen3vl_default(
        patch_size: u32,
        spatial_merge_size: u32,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    ) -> PreprocessConfig {
        let patch_area = patch_size * patch_size * spatial_merge_size * spatial_merge_size;
        PreprocessConfig {
            patch_size,
            spatial_merge_size,
            image_mean,
            image_std,
            min_pixels: QWEN3VL_DEFAULT_MIN_TOKENS * patch_area,
            max_pixels: QWEN3VL_DEFAULT_MAX_TOKENS * patch_area,
        }
    }

    /// The smart-resize alignment = `patch_size * spatial_merge_size` (=32 for
    /// Qwen3-VL). Both output dims are multiples of this.
    #[inline]
    pub fn align(&self) -> u32 {
        self.patch_size * self.spatial_merge_size
    }
}

/// A preprocessed image ready to upload to the GPU vision tower.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// Normalized pixels in **planar / channel-major** layout, matching the
    /// `inp_raw` tensor llama.cpp feeds to the patch-embed conv
    /// (clip.cpp:3403-3420):
    ///
    /// ```text
    /// pixels = [ R[0..W*H] , G[0..W*H] , B[0..W*H] ]   (each plane row-major)
    /// index(c, y, x) = c * (W*H) + y * W + x
    /// ```
    ///
    /// where W = `resized_w`, H = `resized_h`. Values are
    /// `(u8/255 - mean[c]) / std[c]`.
    pub pixels: Vec<f32>,
    /// Patch-grid width = `resized_w / patch_size` (before merge).
    pub grid_w: u32,
    /// Patch-grid height = `resized_h / patch_size` (before merge).
    pub grid_h: u32,
    /// Resized image width in pixels (multiple of `align`).
    pub resized_w: u32,
    /// Resized image height in pixels (multiple of `align`).
    pub resized_h: u32,
    /// Number of merged image tokens =
    /// `(resized_w / (patch*merge)) * (resized_h / (patch*merge))`.
    pub n_tokens: u32,
}

/// Round `x / f` to the nearest integer, then multiply back by `f`.
///
/// Matches C++ `static_cast<int>(std::round(x / f)) * f`. `f32::round` uses
/// round-half-away-from-zero, identical to `std::round`.
#[inline]
fn round_by_factor(x: f32, f: u32) -> u32 {
    ((x / f as f32).round() as i64 * f as i64).max(0) as u32
}

/// `ceil(x / f) * f`.
#[inline]
fn ceil_by_factor(x: f32, f: u32) -> u32 {
    ((x / f as f32).ceil() as i64 * f as i64).max(0) as u32
}

/// `floor(x / f) * f`.
#[inline]
fn floor_by_factor(x: f32, f: u32) -> u32 {
    ((x / f as f32).floor() as i64 * f as i64).max(0) as u32
}

/// Compute the resized `(width, height)`, preserving aspect ratio, both aligned
/// to `cfg.align()` and clamped so `min_pixels <= W*H <= max_pixels`.
///
/// Exact port of llama.cpp `calc_size_preserved_ratio` (the 4-arg "smart_resize"
/// variant, mtmd-image.cpp:156-180):
///
/// 1. Align both dims up first: `bar = max(align, round_by_factor(dim))`.
/// 2. If `h_bar*w_bar > max_pixels`: `beta = sqrt(h*w / max_pixels)`, then
///    `bar = max(align, floor_by_factor(dim / beta))`.
/// 3. Else if `< min_pixels`: `beta = sqrt(min_pixels / (h*w))`, then
///    `bar = ceil_by_factor(dim * beta)`.
///
/// Returns `(w_bar, h_bar)`.
pub fn smart_resize(w: u32, h: u32, cfg: &PreprocessConfig) -> (u32, u32) {
    let align = cfg.align();
    debug_assert!(align > 0);
    let width = w as f32;
    let height = h as f32;

    // always align up first
    let mut h_bar = align.max(round_by_factor(height, align));
    let mut w_bar = align.max(round_by_factor(width, align));

    // llama.cpp compares `h_bar * w_bar` as plain `int`; use u64 here to avoid
    // overflow on pathological inputs while preserving the comparison.
    let area = h_bar as u64 * w_bar as u64;
    if area > cfg.max_pixels as u64 {
        let beta = ((height * width) / cfg.max_pixels as f32).sqrt();
        h_bar = align.max(floor_by_factor(height / beta, align));
        w_bar = align.max(floor_by_factor(width / beta, align));
    } else if area < cfg.min_pixels as u64 {
        let beta = (cfg.min_pixels as f32 / (height * width)).sqrt();
        h_bar = ceil_by_factor(height * beta, align);
        w_bar = ceil_by_factor(width * beta, align);
    }

    (w_bar, h_bar)
}

/// Corner-aligned bilinear resize of an interleaved RGB8 buffer.
///
/// Exact port of llama.cpp `img_tool::resize_bilinear` (mtmd-image.cpp:212-243).
/// `src` is `src_w*src_h*3` bytes interleaved RGB; returns `dst_w*dst_h*3`
/// interleaved RGB. Uses `x_ratio = (src_w-1)/(dst_w-1)` (corner alignment) and
/// rounds the interpolated value half-away-from-zero (`std::lround`) — this
/// differs from the half-pixel convention in most image libraries, hence the
/// hand-rolled loop. All arithmetic is in `f32` to byte-match the C++.
fn resize_bilinear_u8(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let sw = src_w as i64;
    let sh = src_h as i64;
    let dw = dst_w.max(1) as i64;
    let dh = dst_h.max(1) as i64;

    if src_w == 0 || src_h == 0 {
        return Vec::new();
    }
    let mut dst = vec![0u8; (dw * dh * 3) as usize];

    let x_ratio = if dw > 1 {
        (sw - 1) as f32 / (dw - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if dh > 1 {
        (sh - 1) as f32 / (dh - 1) as f32
    } else {
        0.0
    };

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;

    for y in 0..dh {
        let py = y as f32 * y_ratio;
        let y0 = ((py as i64).min(sh - 1)).max(0);
        let y1 = (y0 + 1).min(sh - 1);
        let yf = py - y0 as f32;
        for x in 0..dw {
            let px = x as f32 * x_ratio;
            let x0 = ((px as i64).min(sw - 1)).max(0);
            let x1 = (x0 + 1).min(sw - 1);
            let xf = px - x0 as f32;
            let i00 = (3 * (y0 * sw + x0)) as usize;
            let i01 = (3 * (y0 * sw + x1)) as usize;
            let i10 = (3 * (y1 * sw + x0)) as usize;
            let i11 = (3 * (y1 * sw + x1)) as usize;
            let idst = (3 * (y * dw + x)) as usize;
            for c in 0..3 {
                let top = lerp(src[i00 + c] as f32, src[i01 + c] as f32, xf);
                let bot = lerp(src[i10 + c] as f32, src[i11 + c] as f32, xf);
                // std::lround: round half away from zero, then clamp to u8.
                let v = lerp(top, bot, yf).round();
                dst[idst + c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// Decode an image, smart-resize it, normalize, and lay out the planar pixel
/// tensor — the full Qwen3-VL CPU preprocessing pipeline.
///
/// Pipeline (matching `mtmd_image_preprocessor_dyn_size::preprocess`):
/// decode → [`smart_resize`] → [`resize_bilinear_u8`] →
/// `(px/255 - mean)/std` → planar RGB layout.
pub fn preprocess(
    img_path: &Path,
    cfg: &PreprocessConfig,
) -> Result<PreprocessedImage, Box<dyn Error>> {
    let dyn_img = image::open(img_path)?;
    let rgb = dyn_img.to_rgb8();
    let (src_w, src_h) = rgb.dimensions();
    preprocess_rgb8(rgb.as_raw(), src_w, src_h, cfg)
}

/// Core preprocessing on an already-decoded interleaved RGB8 buffer. Factored
/// out so tests (and cross-validation) can feed deterministic raw pixels
/// without round-tripping through a file decoder.
pub fn preprocess_rgb8(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    cfg: &PreprocessConfig,
) -> Result<PreprocessedImage, Box<dyn Error>> {
    if src.len() != (src_w as usize * src_h as usize * 3) {
        return Err(format!(
            "preprocess_rgb8: buffer is {} bytes, expected {}x{}x3 = {}",
            src.len(),
            src_w,
            src_h,
            src_w as usize * src_h as usize * 3
        )
        .into());
    }

    let (rw, rh) = smart_resize(src_w, src_h, cfg);
    let resized = resize_bilinear_u8(src, src_w, src_h, rw, rh);

    let n = (rw as usize) * (rh as usize);
    let mut pixels = vec![0f32; n * 3];
    // Planar / channel-major: pixels[c*n + y*W + x] = (u8/255 - mean[c])/std[c].
    // The interleaved source index is 3*(y*W + x) + c.
    for p in 0..n {
        let base = 3 * p;
        for c in 0..3 {
            let v = (resized[base + c] as f32 / 255.0 - cfg.image_mean[c]) / cfg.image_std[c];
            pixels[c * n + p] = v;
        }
    }

    let grid_w = rw / cfg.patch_size;
    let grid_h = rh / cfg.patch_size;
    let merge = cfg.spatial_merge_size;
    let n_tokens = (rw / (cfg.patch_size * merge)) * (rh / (cfg.patch_size * merge));

    Ok(PreprocessedImage {
        pixels,
        grid_w,
        grid_h,
        resized_w: rw,
        resized_h: rh,
        n_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real Qwen3-VL mmproj config: patch=16, merge=2, mean=std=0.5, and
    /// the QWEN3VL default pixel limits (min=8192, max=4_194_304).
    fn qwen3vl_cfg() -> PreprocessConfig {
        PreprocessConfig::qwen3vl_default(16, 2, [0.5; 3], [0.5; 3])
    }

    /// Assert both dims are multiples of `align` (=32) and at least one tile.
    fn assert_aligned(w: u32, h: u32, cfg: &PreprocessConfig) {
        let a = cfg.align();
        assert_eq!(w % a, 0, "w={w} not multiple of {a}");
        assert_eq!(h % a, 0, "h={h} not multiple of {a}");
        assert!(w >= a && h >= a, "dims below one tile: {w}x{h}");
    }

    #[test]
    fn smart_resize_square() {
        let cfg = qwen3vl_cfg();
        // 512x512 = 262144 px, within [8192, 4194304]. round(512/32)*32 = 512.
        let (w, h) = smart_resize(512, 512, &cfg);
        assert_aligned(w, h, &cfg);
        assert_eq!((w, h), (512, 512));
    }

    #[test]
    fn smart_resize_wide() {
        let cfg = qwen3vl_cfg();
        // 1000x300: round(1000/32)*32 = 992, round(300/32)*32 = 288.
        // 992*288 = 285_696, within bounds, so no clamp.
        let (w, h) = smart_resize(1000, 300, &cfg);
        assert_aligned(w, h, &cfg);
        assert_eq!((w, h), (992, 288));
        assert!((w as u64 * h as u64) <= cfg.max_pixels as u64);
    }

    #[test]
    fn smart_resize_tall() {
        let cfg = qwen3vl_cfg();
        // 300x1000 — the transpose of `wide`.
        let (w, h) = smart_resize(300, 1000, &cfg);
        assert_aligned(w, h, &cfg);
        assert_eq!((w, h), (288, 992));
    }

    #[test]
    fn smart_resize_tiny_clamped_up() {
        let cfg = qwen3vl_cfg();
        // 16x16: round(16/32)*32 = 0 -> max(32, 0) = 32 each => 32*32 = 1024 px,
        // below min_pixels (8192). beta = sqrt(8192 / 256) = sqrt(32) ≈ 5.657.
        // ceil_by_factor(16 * 5.657) = ceil(90.5/32)*32 = 96. => 96x96 = 9216.
        let (w, h) = smart_resize(16, 16, &cfg);
        assert_aligned(w, h, &cfg);
        assert!(
            (w as u64 * h as u64) >= cfg.min_pixels as u64,
            "tiny not clamped up: {w}x{h} = {}",
            w as u64 * h as u64
        );
        assert_eq!((w, h), (96, 96));
    }

    #[test]
    fn smart_resize_huge_clamped_down() {
        let cfg = qwen3vl_cfg();
        // 8000x8000: aligned -> 8000 (8000%32==0). 8000*8000 = 64_000_000 >>
        // max_pixels (4_194_304). beta = sqrt(64e6 / 4194304) ≈ 3.9063.
        // floor_by_factor(8000 / 3.9063) = floor(2048/32)*32 = 2048. => 2048².
        let (w, h) = smart_resize(8000, 8000, &cfg);
        assert_aligned(w, h, &cfg);
        assert!(
            (w as u64 * h as u64) <= cfg.max_pixels as u64,
            "huge not clamped down: {w}x{h} = {}",
            w as u64 * h as u64
        );
        assert_eq!((w, h), (2048, 2048));
    }

    #[test]
    fn smart_resize_exact_multiple() {
        let cfg = qwen3vl_cfg();
        // 640x384 are already multiples of 32; area 245_760 within bounds.
        let (w, h) = smart_resize(640, 384, &cfg);
        assert_aligned(w, h, &cfg);
        assert_eq!((w, h), (640, 384));
    }

    #[test]
    fn n_tokens_and_layout() {
        let cfg = qwen3vl_cfg();
        // Build a deterministic 200x150 buffer (matches the cross-validation
        // test image) and check the derived dims + layout invariants.
        let (sw, sh) = (200u32, 150u32);
        let mut buf = vec![0u8; (sw * sh * 3) as usize];
        for y in 0..sh {
            for x in 0..sw {
                let i = (3 * (y * sw + x)) as usize;
                buf[i] = (x % 256) as u8;
                buf[i + 1] = (y % 256) as u8;
                buf[i + 2] = ((x + y) % 256) as u8;
            }
        }
        let out = preprocess_rgb8(&buf, sw, sh, &cfg).unwrap();
        // smart_resize(200,150): round(200/32)*32=192, round(150/32)*32=160.
        // 192*160=30720 in-bounds. tokens = (192/32)*(160/32) = 6*5 = 30.
        assert_eq!((out.resized_w, out.resized_h), (192, 160));
        assert_eq!(out.n_tokens, 30);
        assert_eq!(out.grid_w, 192 / 16);
        assert_eq!(out.grid_h, 160 / 16);
        // Planar layout: pixels.len() == 3 * W * H.
        assert_eq!(
            out.pixels.len(),
            3 * out.resized_w as usize * out.resized_h as usize
        );
        // n_tokens == (resized_w/(patch*merge)) * (resized_h/(patch*merge)).
        let pm = cfg.patch_size * cfg.spatial_merge_size;
        assert_eq!(out.n_tokens, (out.resized_w / pm) * (out.resized_h / pm));
    }

    #[test]
    fn normalize_endpoints() {
        // With mean=std=0.5: 0 -> -1, 255 -> +1.
        let cfg = qwen3vl_cfg();
        let n = 32 * 32;
        let mut buf = vec![0u8; n * 3];
        let out_b = preprocess_rgb8(&buf, 32, 32, &cfg).unwrap();
        assert!((out_b.pixels[0] - (-1.0)).abs() < 1e-6);
        for v in buf.iter_mut() {
            *v = 255;
        }
        let out_w = preprocess_rgb8(&buf, 32, 32, &cfg).unwrap();
        assert!((out_w.pixels[0] - 1.0).abs() < 1e-6);
    }

    /// Cross-validation against the Python reference + llama-mtmd-cli. Ignored
    /// by default; run with
    /// `cargo test --bin seeker -- --ignored xvalidate --nocapture`.
    ///
    /// This test is the *single source of truth* for the raw test-image bytes:
    /// it writes a deterministic 200x150 gradient PNG to /tmp/viz_test.png (via
    /// the `image` crate) using the exact same per-pixel formula as the Python
    /// harness's `make_test_image` and the `n_tokens_and_layout` unit test, then
    /// preprocesses it. Both `scripts/vision_ref.py` and `llama-mtmd-cli` are
    /// pointed at that same PNG, so all three decode identical bytes. Set
    /// `SEEKER_XVAL_DUMP=1` to also write the planar f32 tensor to
    /// /tmp/rs_pixels.f32 for an element-wise diff against the Python harness.
    #[test]
    #[ignore]
    fn xvalidate_test_image() {
        let cfg = qwen3vl_cfg();
        let path = std::path::Path::new("/tmp/viz_test.png");

        // Deterministic gradient: R=x, G=y, B=(x+y) (mod 256).
        let (w, h) = (200u32, 150u32);
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]),
                );
            }
        }
        img.save(path).unwrap();

        let out = preprocess(path, &cfg).unwrap();
        let sum: f64 = out.pixels.iter().map(|&v| v as f64).sum();
        eprintln!(
            "SEEKER xvalidate: resized={}x{} grid={}x{} n_tokens={} pixels_len={} checksum_sum_f64={:.6}",
            out.resized_w, out.resized_h, out.grid_w, out.grid_h, out.n_tokens, out.pixels.len(), sum
        );
        eprintln!(
            "SEEKER first5=[{}]",
            out.pixels.iter().take(5).map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(", ")
        );
        let last5: Vec<_> = out.pixels.iter().rev().take(5).cloned().collect();
        eprintln!(
            "SEEKER last5=[{}]",
            last5.iter().rev().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(", ")
        );
        // Write little-endian f32 for exact element-wise diff vs the Python ref.
        if std::env::var("SEEKER_XVAL_DUMP").is_ok() {
            let mut bytes = Vec::with_capacity(out.pixels.len() * 4);
            for &v in &out.pixels {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write("/tmp/rs_pixels.f32", &bytes).unwrap();
            eprintln!("SEEKER wrote /tmp/rs_pixels.f32 ({} bytes)", bytes.len());
        }
    }
}
