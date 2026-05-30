//! Qwen3-VL vision-tower **patch-embedding front-end** (Phase 2, Slice 2).
//!
//! This is the input stage of the ViT: from a [`PreprocessedImage`] to the
//! sequence of patch tokens (in 2x2-block order) with the learned, bilinearly
//! resized absolute position embedding added — i.e. the tensor that feeds ViT
//! block 0. The transformer blocks and the merger MLP land in Slices 3-4.
//!
//! It is a faithful port of llama.cpp's `clip_graph_qwen3vl::build` (the part
//! BEFORE the block loop) in
//! `/home/bob/tools/llama.cpp/src/tools/mtmd/models/qwen3vl.cpp`, plus
//! `clip_graph::resize_position_embeddings` in `clip.cpp`. The exact source
//! lines and the derived index maps are documented inline.
//!
//! ## What the front-end computes
//!
//! 1. **Dual conv2d patch-embed** (`v.patch_embd.weight` + `v.patch_embd.weight.1`,
//!    each ggml shape `[kw=16, kh=16, Cin=3, Cout=1152]`, stride 16, no
//!    padding) applied to the planar pixel tensor and **summed**. With
//!    stride == kernel == patch_size and no overlap, each output patch is an
//!    independent dot product of its 16x16x3 receptive field against each of
//!    the 1152 filters — i.e. a plain GEMM over an im2col matrix. We exploit
//!    this: build the im2col matrix `[768, n_patches]` on the host (a gather of
//!    the already-host pixels) and reuse the GEMM op (`matmul::record`),
//!    avoiding any new shader. `ggml_conv_2d` reshapes the kernel
//!    `[16,16,3,1152]` to `[768, 1152]` with element order
//!    `kw + 16*(kh + 16*ic)`; the F32 kernel is already contiguous in that
//!    order, so the reshape is a no-op view.
//! 2. **Add patch bias** `v.patch_embd.bias` `[1152]` (broadcast over tokens).
//! 3. **2x2 spatial-merge interleave**: the `permute/cont/reshape` sequence in
//!    qwen3vl.cpp (lines 27-38) reorders the `n_patches` patch columns so that
//!    each consecutive group of 4 columns is a 2x2 spatial neighborhood. It
//!    does **not** reduce the token count or widen n_embd here — the x4 concat
//!    happens at the merger in Slice 4. We fold this permutation into the
//!    *column order* of the im2col matrix (and of the pos-embd matrix), so the
//!    GPU matmul output is already in 2x2-block order.
//! 4. **Learned absolute position embedding** `v.position_embd.weight`
//!    `[1152, 2304]` (2304 = 48x48 base grid). Bilinearly resized (ggml
//!    `GGML_SCALE_MODE_BILINEAR`, half-pixel / non-align-corners) to the
//!    `n_patches_x x n_patches_y` grid, reordered with the SAME 2x2 map, and
//!    **added**.
//!
//! Output of Slice 2 = `[n_embd=1152, n_patches]` ready for block 0.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::buffer::BufferRange;
use crate::inference::context::DispatchContext;
use crate::inference::ops::elementwise::record_add;
use crate::inference::ops::matmul;
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::vision::preprocess::PreprocessedImage;

/// Number of input channels (RGB).
pub const N_CHANNELS: usize = 3;
/// Base side length of the learned position-embd grid: sqrt(2304) = 48.
pub const POS_EMBD_BASE_SIDE: usize = 48;

/// Parsed weight views for the patch-embed front-end. Holds the two conv-kernel
/// [`TensorView`]s (which are `Copy`) into the mmproj [`WeightsHandle`] — these
/// drive the on-GPU GEMM. The bias and position-embedding are applied from
/// host-side slices ([`HostWeights`]) instead, because they're combined on the
/// CPU (bias broadcast + bilinear pos-embd resize) and the uploaded GPU copies
/// are device-local (not host-readable). Structured so Slices 3-4 can extend it
/// with the block / merger weights.
///
/// In the real `mmproj-F16.gguf` the conv kernels and position_embd are **F16**
/// and the bias is **F32**. The GEMM accepts F16 A directly (`matmul::record`
/// wires the `mul_mat_vec_f16` / coop-matrix F16 variants); the CPU reference
/// up-converts F16 to f32. F32 weights are also accepted (the BF16 mmproj
/// variant ships some tensors F32).
pub struct VisionEncoder {
    /// First patch-embed conv kernel `[16,16,3,1152]` (F16 or F32).
    pub patch_embd_0: TensorView,
    /// Second patch-embed conv kernel `[16,16,3,1152]` (F16 or F32).
    pub patch_embd_1: TensorView,
    /// Vision embedding dim (= 1152).
    pub n_embd: usize,
    /// Patch size in pixels (= 16).
    pub patch_size: usize,
}

impl VisionEncoder {
    /// Parse the patch-embed weight views out of an mmproj [`WeightsHandle`].
    ///
    /// Tensor names match llama.cpp's `clip` GGUF naming and the verified
    /// `mmproj-F16.gguf`. `n_embd` / `patch_size` come from the already-parsed
    /// config. Validates that the conv kernels + pos-embd are F16/F32 and the
    /// bias is F32 (the GEMM A side handles F16/F32; the host-add side
    /// up-converts).
    pub fn new(
        weights: &WeightsHandle,
        n_embd: usize,
        patch_size: usize,
    ) -> Result<VisionEncoder, Box<dyn Error>> {
        let patch_embd_0 = weights.view("v.patch_embd.weight")?;
        let patch_embd_1 = weights.view("v.patch_embd.weight.1")?;
        let patch_bias = weights.view("v.patch_embd.bias")?;
        let position_embd = weights.view("v.position_embd.weight")?;

        // Conv kernels + pos-embd may be F16 (mmproj-F16) or F32 (mmproj-BF16);
        // the GEMM A side and the host up-conversion both handle either. The
        // bias is F32 in both. Reject anything else loudly.
        for (name, v) in [
            ("v.patch_embd.weight", &patch_embd_0),
            ("v.patch_embd.weight.1", &patch_embd_1),
            ("v.position_embd.weight", &position_embd),
        ] {
            if v.dtype != GgmlType::F16 && v.dtype != GgmlType::F32 {
                return Err(format!(
                    "vision patch-embed expects F16/F32 {name}, got {:?}",
                    v.dtype
                )
                .into());
            }
        }
        if patch_bias.dtype != GgmlType::F32 {
            return Err(format!(
                "vision patch-embed expects F32 v.patch_embd.bias, got {:?}",
                patch_bias.dtype
            )
            .into());
        }

        Ok(VisionEncoder {
            patch_embd_0,
            patch_embd_1,
            n_embd,
            patch_size,
        })
    }

    /// The conv-kernel as a GEMM matrix view `[K=patch²·3, M=n_embd]`, keeping
    /// the kernel's native dtype (F16 or F32).
    ///
    /// `ggml_conv_2d` reshapes the `[kw,kh,Cin,Cout]` kernel to
    /// `[kw*kh*Cin, Cout]`; the tensor is contiguous in `kw + KW*(kh + KH*ic)`
    /// order, so this is a pure reinterpret of the same bytes — only the
    /// logical shape changes. Element strides are dtype-independent (counts),
    /// byte strides scale by the element size.
    fn kernel_as_gemm(&self, kernel: &TensorView) -> TensorView {
        let k = (self.patch_size * self.patch_size * N_CHANNELS) as u64;
        let m = self.n_embd as u64;
        let esz = match kernel.dtype {
            GgmlType::F16 => 2u64,
            _ => 4u64,
        };
        TensorView {
            buffer: kernel.buffer,
            byte_offset: kernel.byte_offset,
            byte_size: kernel.byte_size,
            dims: [k, m, 1, 1],
            byte_stride: [esz, k * esz, k * m * esz, k * m * esz],
            element_stride: [1, k, k * m, k * m],
            dtype: kernel.dtype,
        }
    }

    /// Record the patch-embed front-end on the GPU and return the result view
    /// `[n_embd, n_patches]` (F32, in 2x2-block token order, pos-embd added).
    ///
    /// Reuses existing op wrappers only: `matmul::record` (conv-as-GEMM, x2)
    /// and `record_add` (sum the two convs, add the per-token bias+pos-embd
    /// matrix). The im2col matrix and the combined `bias + resized pos-embd`
    /// matrix are computed on the host (deterministic gathers of already-host
    /// data, no broadcast dependency) and staged into scratch via
    /// [`alloc_scratch_write`].
    pub fn record_patch_embed(
        &self,
        ctx: &mut DispatchContext,
        img: &PreprocessedImage,
        host_weights: &HostWeights,
    ) -> Result<TensorView, Box<dyn Error>> {
        let n_embd = self.n_embd;
        let gw = img.grid_w as usize;
        let gh = img.grid_h as usize;
        let n_patches = gw * gh;
        let k = self.patch_size * self.patch_size * N_CHANNELS;

        // Host-built im2col `[K, n_patches]` in 2x2-block token order.
        let im2col = build_im2col_reordered(img, self.patch_size);
        debug_assert_eq!(im2col.len(), k * n_patches);
        let im2col_range = alloc_scratch_write(ctx, &f32_to_bytes(&im2col))?;
        let im2col_view = dense_view(&im2col_range, [k as u64, n_patches as u64, 1, 1]);

        // out0 = conv0ᵀ · im2col  -> [n_embd, n_patches].
        // `matmul::record` is the public GEMM entry point (A=[K,M], B=[K,N],
        // D=[M,N], K contracting). A is the conv kernel in its native dtype
        // (F16 or F32 — both wired); B/D are F32. The conv-as-GEMM is exactly
        // this shape. Patch-embed cost is negligible vs the ViT blocks.
        let out_view = ctx.alloc_tensor([n_embd as u64, n_patches as u64, 1, 1], GgmlType::F32)?;
        let kern0 = self.kernel_as_gemm(&self.patch_embd_0);
        matmul::record(ctx, kern0, im2col_view, out_view)?;

        // out1 = conv1ᵀ · im2col, then out = out0 + out1.
        let tmp_view = ctx.alloc_tensor([n_embd as u64, n_patches as u64, 1, 1], GgmlType::F32)?;
        let kern1 = self.kernel_as_gemm(&self.patch_embd_1);
        matmul::record(ctx, kern1, im2col_view, tmp_view)?;
        record_add(ctx, out_view, tmp_view, out_view)?;

        // + (patch bias broadcast over tokens) + (resized+reordered pos-embd).
        // Both are precomputed on the host into one `[n_embd, n_patches]`
        // matrix, so a single full-shape `record_add` applies them — no
        // reliance on the binary shader's broadcast path. The pos-embd weight
        // is read from `host_weights` (up-converted from the mmap'd GGUF), since
        // the uploaded GPU copy is device-local and not host-readable.
        let bias_pos = bias_plus_pos_host(host_weights, n_embd, img);
        debug_assert_eq!(bias_pos.len(), n_embd * n_patches);
        let bp_range = alloc_scratch_write(ctx, &f32_to_bytes(&bias_pos))?;
        let bp_view = dense_view(&bp_range, [n_embd as u64, n_patches as u64, 1, 1]);
        record_add(ctx, out_view, bp_view, out_view)?;

        Ok(out_view)
    }
}

/// Host-side **F32** copies of the patch-embed weights, up-converted from the
/// mmproj GGUF (which stores the conv kernels + pos-embd as F16, the bias as
/// F32). Used by both the GPU pos-embd staging and the CPU reference, because
/// the uploaded GPU weights are device-local and not host-readable. Build via
/// [`HostWeights::from_gguf`].
#[derive(Clone)]
pub struct HostWeights {
    /// `v.patch_embd.weight` as `[K=768, M=1152]` (idx = kk + K*m).
    pub patch_embd_0: Vec<f32>,
    /// `v.patch_embd.weight.1` (same layout).
    pub patch_embd_1: Vec<f32>,
    /// `v.patch_embd.bias` `[1152]`.
    pub patch_bias: Vec<f32>,
    /// `v.position_embd.weight` `[1152, 2304]` (idx = c + n_embd*(gx+48*gy)).
    pub position_embd: Vec<f32>,
}

impl HostWeights {
    /// Read the four patch-embed tensors out of the mmap'd mmproj GGUF and
    /// up-convert each to f32 (F16 -> f32 or F32 passthrough).
    pub fn from_gguf(gguf: &crate::gguf::GgufFile) -> Result<HostWeights, Box<dyn Error>> {
        Ok(HostWeights {
            patch_embd_0: read_tensor_as_f32(gguf, "v.patch_embd.weight")?,
            patch_embd_1: read_tensor_as_f32(gguf, "v.patch_embd.weight.1")?,
            patch_bias: read_tensor_as_f32(gguf, "v.patch_embd.bias")?,
            position_embd: read_tensor_as_f32(gguf, "v.position_embd.weight")?,
        })
    }
}

/// Read a named F16/F32 tensor from the GGUF, returning its elements as a
/// contiguous `Vec<f32>` (logical order, F16 up-converted via [`f16_to_f32`]).
fn read_tensor_as_f32(
    gguf: &crate::gguf::GgufFile,
    name: &str,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = gguf
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("mmproj missing {name}"))?;
    let bytes = gguf
        .tensor_data(name)
        .ok_or_else(|| format!("{name} has no data slice"))?;
    match t.ggml_type {
        GgmlType::F32 => {
            let n = bytes.len() / 4;
            let mut out = vec![0f32; n];
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes([
                    bytes[4 * i],
                    bytes[4 * i + 1],
                    bytes[4 * i + 2],
                    bytes[4 * i + 3],
                ]);
            }
            Ok(out)
        }
        GgmlType::F16 => {
            let n = bytes.len() / 2;
            let mut out = vec![0f32; n];
            for (i, o) in out.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                *o = f16_to_f32(bits);
            }
            Ok(out)
        }
        other => Err(format!("{name}: expected F16/F32, got {other:?}").into()),
    }
}

/// IEEE-754 half-precision (binary16) -> f32. Handles subnormals, inf, NaN.
/// Matches `ggml_fp16_to_fp32` numerically (exact, no rounding loss widening).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let val: f32 = if exp == 0 {
        // zero or subnormal: value = mant * 2^-24
        (mant as f32) * (2.0f32).powi(-24)
    } else if exp == 0x1f {
        // inf / NaN
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        // normal: (1 + mant/1024) * 2^(exp-15)
        (1.0f32 + (mant as f32) / 1024.0) * (2.0f32).powi(exp as i32 - 15)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// `bias[c] + pos_embd_resized_reordered[c, tok]` as a flat `[n_embd,
/// n_patches]` matrix (index = `c + n_embd*tok`).
fn bias_plus_pos_host(hw: &HostWeights, n_embd: usize, img: &PreprocessedImage) -> Vec<f32> {
    let mut out = resize_position_embeddings_reordered(
        &hw.position_embd,
        n_embd,
        img.grid_w as usize,
        img.grid_h as usize,
    );
    let n_patches = out.len() / n_embd;
    for tok in 0..n_patches {
        for c in 0..n_embd {
            out[c + n_embd * tok] += hw.patch_bias[c];
        }
    }
    out
}

/// Pure CPU reference for [`VisionEncoder::record_patch_embed`] — implements the
/// SAME math directly from the (up-converted f32) host weight slices with plain
/// loops. This is the numerical oracle the GPU path is validated against.
pub fn patch_embed_cpu(
    hw: &HostWeights,
    n_embd: usize,
    patch_size: usize,
    img: &PreprocessedImage,
) -> Vec<f32> {
    let gw = img.grid_w as usize;
    let gh = img.grid_h as usize;
    let n_patches = gw * gh;
    let ps = patch_size;
    let k = ps * ps * N_CHANNELS;
    let w = img.resized_w as usize;
    let h = img.resized_h as usize;

    let pos = resize_position_embeddings_reordered(&hw.position_embd, n_embd, gw, gh);

    let mut out = vec![0f32; n_embd * n_patches];
    for tok in 0..n_patches {
        // token -> source patch (pw, ph), via the 2x2-block map.
        let (pw, ph) = token_to_patch(tok, gw);
        // im2col column for this patch.
        let mut col = vec![0f32; k];
        for ic in 0..N_CHANNELS {
            for kh in 0..ps {
                for kw in 0..ps {
                    let x = pw * ps + kw;
                    let y = ph * ps + kh;
                    let kidx = kw + ps * (kh + ps * ic);
                    col[kidx] = img.pixels[ic * (w * h) + y * w + x];
                }
            }
        }
        for m in 0..n_embd {
            let mut acc = 0f32;
            for kk in 0..k {
                let wv = hw.patch_embd_0[kk + k * m] + hw.patch_embd_1[kk + k * m];
                acc += wv * col[kk];
            }
            acc += hw.patch_bias[m];
            acc += pos[m + n_embd * tok];
            out[m + n_embd * tok] = acc;
        }
    }
    out
}

/// Map an output token index (in 2x2-block order) back to its source patch
/// `(pw, ph)` in the raster patch grid. Inverse of qwen3vl.cpp's
/// permute/reshape/cont merge sequence (models/qwen3vl.cpp:27-38).
///
/// Forward (derived from ggml contiguous semantics — see the simulation in the
/// commit message):
/// ```text
///   tok = p + 2*hr + 4*bw + 2*npx*bh
///   with  bw = pw/2,  p = pw%2,  bh = ph/2,  hr = ph%2
/// ```
/// so each consecutive group of 4 tokens is one 2x2 block (intra-block offset
/// `p + 2*hr`), blocks raster-ordered by `(bw, bh)`. The inverse below.
#[inline]
pub fn token_to_patch(tok: usize, npx: usize) -> (usize, usize) {
    let bh = tok / (2 * npx);
    let rem1 = tok % (2 * npx);
    let bw = rem1 / 4;
    let rem2 = rem1 % 4;
    let hr = rem2 / 2;
    let p = rem2 % 2;
    (2 * bw + p, 2 * bh + hr)
}

/// Forward direction of [`token_to_patch`]: patch `(pw, ph)` -> token index.
#[inline]
pub fn patch_to_token(pw: usize, ph: usize, npx: usize) -> usize {
    let bw = pw / 2;
    let p = pw % 2;
    let bh = ph / 2;
    let hr = ph % 2;
    p + 2 * hr + 4 * bw + 2 * npx * bh
}

/// Build the im2col matrix `[K=patch²·3, n_patches]` (column = token, in
/// 2x2-block order) as a flat row-major `Vec<f32>` (index = `kidx + K*tok`).
/// Element order within a column is `kw + patch*(kh + patch*ic)` to match
/// `ggml_conv_2d`'s kernel reshape.
pub fn build_im2col_reordered(img: &PreprocessedImage, patch_size: usize) -> Vec<f32> {
    let gw = img.grid_w as usize;
    let gh = img.grid_h as usize;
    let n_patches = gw * gh;
    let ps = patch_size;
    let k = ps * ps * N_CHANNELS;
    let w = img.resized_w as usize;
    let h = img.resized_h as usize;
    let mut out = vec![0f32; k * n_patches];
    for tok in 0..n_patches {
        let (pw, ph) = token_to_patch(tok, gw);
        for ic in 0..N_CHANNELS {
            for kh in 0..ps {
                for kw in 0..ps {
                    let x = pw * ps + kw;
                    let y = ph * ps + kh;
                    let kidx = kw + ps * (kh + ps * ic);
                    out[kidx + k * tok] = img.pixels[ic * (w * h) + y * w + x];
                }
            }
        }
    }
    out
}

/// Resize the learned position embedding to the patch grid and reorder it into
/// 2x2-block token order, returning `[n_embd, n_patches]` row-major
/// (index = `c + n_embd*tok`).
///
/// Faithful port of `clip_graph::resize_position_embeddings` (clip.cpp:278-298)
/// followed by the same merge reorder. `pos_base` is the raw `[n_embd, 2304]`
/// weight (row = channel, col = `gx + 48*gy`). The bilinear resize uses ggml's
/// non-align-corners formula (ggml-cpu/ops.cpp `ggml_compute_forward_upscale_f32`,
/// `GGML_SCALE_MODE_BILINEAR`, lines 7683-7723): pixel_offset=0.5,
/// `sf = dst/src`, `x = (i+0.5)/sf - 0.5`, floor + clamp, 4-tap blend.
pub fn resize_position_embeddings_reordered(
    pos_base: &[f32],
    n_embd: usize,
    npx: usize,
    npy: usize,
) -> Vec<f32> {
    let side = POS_EMBD_BASE_SIDE; // 48
    debug_assert_eq!(pos_base.len(), n_embd * side * side);
    let n_patches = npx * npy;
    let mut out = vec![0f32; n_embd * n_patches];

    // resize_position_embeddings short-circuits when no resize is needed.
    let need_resize = !(npx == side && npy == side);

    // ggml sf = dst/src (NOT align-corners): sf0 over width, sf1 over height.
    let sf0 = npx as f32 / side as f32;
    let sf1 = npy as f32 / side as f32;
    let po = 0.5f32;

    for tok in 0..n_patches {
        let (pw, ph) = token_to_patch(tok, npx);
        let (dx, dy) = (pw, ph); // raster grid coords in the npx x npy grid.
        for c in 0..n_embd {
            let val = if need_resize {
                // bilinear sample base 48x48 grid at the dst pixel (dx, dy).
                let xf = (dx as f32 + po) / sf0 - po;
                let yf = (dy as f32 + po) / sf1 - po;
                let x0 = (xf.floor() as i64).clamp(0, side as i64 - 1) as usize;
                let x1 = ((xf.floor() as i64) + 1).clamp(0, side as i64 - 1) as usize;
                let y0 = (yf.floor() as i64).clamp(0, side as i64 - 1) as usize;
                let y1 = ((yf.floor() as i64) + 1).clamp(0, side as i64 - 1) as usize;
                let dx_frac = (xf - xf.floor()).clamp(0.0, 1.0);
                let dy_frac = (yf - yf.floor()).clamp(0.0, 1.0);
                let at = |gx: usize, gy: usize| pos_base[c + n_embd * (gx + side * gy)];
                let a = at(x0, y0);
                let b = at(x1, y0);
                let cc = at(x0, y1);
                let d = at(x1, y1);
                a * (1.0 - dx_frac) * (1.0 - dy_frac)
                    + b * dx_frac * (1.0 - dy_frac)
                    + cc * (1.0 - dx_frac) * dy_frac
                    + d * dx_frac * dy_frac
            } else {
                pos_base[c + n_embd * (dx + side * dy)]
            };
            out[c + n_embd * tok] = val;
        }
    }
    out
}

/// Reinterpret a `&[f32]` as native-endian bytes.
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

/// Allocate a scratch slot and host-write `bytes` into it. The scratch
/// [`Region`](crate::inference::memory::Region) backs decode inputs/outputs and
/// is host-visible, so this is a direct memcpy into the mapped pointer — used
/// to stage the host-computed im2col / pos-embd matrices into the GPU graph.
fn alloc_scratch_write(
    ctx: &mut DispatchContext,
    bytes: &[u8],
) -> Result<BufferRange, Box<dyn Error>> {
    let range = ctx.alloc_scratch(bytes.len() as u64)?;
    let base = ctx
        .scratch
        .host_ptr
        .ok_or("vision: scratch region is not host-visible; cannot stage inputs")?;
    unsafe {
        let dst = base.add(range.offset as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
    Ok(range)
}

/// Build a dense (contiguous) F32 [`TensorView`] over a scratch range.
fn dense_view(range: &BufferRange, dims: [u64; 4]) -> TensorView {
    let es = [1u64, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]];
    TensorView {
        buffer: range.buffer,
        byte_offset: range.offset,
        byte_size: range.size,
        dims,
        byte_stride: [es[0] * 4, es[1] * 4, es[2] * 4, es[3] * 4],
        element_stride: es,
        dtype: GgmlType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 2x2 merge map must be a bijection over `[0, npx*npy)` and group each
    /// consecutive 4 tokens into one 2x2 spatial block. Checked on a small grid
    /// (npx=4, npy=4) against the hand-derived qwen3vl.cpp index formula.
    #[test]
    fn merge_map_is_bijection_and_2x2_blocks() {
        let npx = 4;
        let npy = 4;
        let n = npx * npy;
        let mut seen = vec![false; n];
        for tok in 0..n {
            let (pw, ph) = token_to_patch(tok, npx);
            assert!(pw < npx && ph < npy, "tok {tok} -> ({pw},{ph}) out of grid");
            let back = patch_to_token(pw, ph, npx);
            assert_eq!(back, tok, "round-trip failed for tok {tok}");
            assert!(!seen[tok], "tok {tok} produced twice");
            seen[tok] = true;
        }
        assert!(seen.iter().all(|&b| b), "map is not surjective");

        // Each consecutive group of 4 tokens is a single 2x2 block: same
        // (bw, bh) = (pw/2, ph/2) and the four intra-block (p,hr) combos.
        for blk in 0..(n / 4) {
            let mut combos = vec![];
            let mut block_id = None;
            for off in 0..4 {
                let tok = blk * 4 + off;
                let (pw, ph) = token_to_patch(tok, npx);
                let id = (pw / 2, ph / 2);
                match block_id {
                    None => block_id = Some(id),
                    Some(b) => assert_eq!(b, id, "block {blk} not a single 2x2 region"),
                }
                combos.push((pw % 2, ph % 2));
            }
            combos.sort();
            assert_eq!(
                combos,
                vec![(0, 0), (0, 1), (1, 0), (1, 1)],
                "block {blk} missing a 2x2 corner"
            );
        }
    }

    /// Hand-checked exact token order for a 4x4 patch grid (npx=npy=4). Derived
    /// from `tok = p + 2*hr + 4*bw + 2*npx*bh`:
    ///   tok 0..3  -> block (bw=0,bh=0): (0,0)(1,0)(0,1)(1,1)
    ///   tok 4..7  -> block (bw=1,bh=0): (2,0)(3,0)(2,1)(3,1)
    ///   tok 8..11 -> block (bw=0,bh=1): (0,2)(1,2)(0,3)(1,3)
    ///   tok 12..15-> block (bw=1,bh=1): (2,2)(3,2)(2,3)(3,3)
    #[test]
    fn merge_map_exact_4x4() {
        let npx = 4;
        let expected = [
            (0, 0), (1, 0), (0, 1), (1, 1),
            (2, 0), (3, 0), (2, 1), (3, 1),
            (0, 2), (1, 2), (0, 3), (1, 3),
            (2, 2), (3, 2), (2, 3), (3, 3),
        ];
        for (tok, &want) in expected.iter().enumerate() {
            assert_eq!(token_to_patch(tok, npx), want, "tok {tok}");
        }
    }

    /// Non-rectangular grid (npx=6, npy=4 -> the 192x128 test case scaled
    /// down): still a bijection and 2x2-block-grouped.
    #[test]
    fn merge_map_bijection_6x4() {
        let npx = 6;
        let npy = 4;
        let n = npx * npy;
        let mut seen = vec![false; n];
        for tok in 0..n {
            let (pw, ph) = token_to_patch(tok, npx);
            assert!(pw < npx && ph < npy, "tok {tok} -> ({pw},{ph})");
            assert_eq!(patch_to_token(pw, ph, npx), tok);
            assert!(!seen[tok]);
            seen[tok] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    /// When the patch grid already equals the 48x48 base, the resize is a
    /// no-op and the only transform is the 2x2 reorder.
    #[test]
    fn pos_embd_no_resize_is_identity_reorder() {
        let n_embd = 1;
        let side = POS_EMBD_BASE_SIDE;
        // pos_base[gx + side*gy] = gx + 100*gy  (channel 0)
        let mut base = vec![0f32; n_embd * side * side];
        for gy in 0..side {
            for gx in 0..side {
                base[gx + side * gy] = (gx + 100 * gy) as f32;
            }
        }
        let out = resize_position_embeddings_reordered(&base, n_embd, side, side);
        for tok in 0..(side * side) {
            let (pw, ph) = token_to_patch(tok, side);
            assert_eq!(out[tok], (pw + 100 * ph) as f32, "tok {tok}");
        }
    }

    /// Exact ggml half-pixel bilinear sampler on a hand-checkable 2x2 -> 4x4
    /// upscale of a single channel. Re-derives the same arithmetic as the
    /// production inner loop with a local `side=2` so the values are tiny and
    /// hand-verifiable. (The production path always uses side=48; the formula
    /// is identical.)
    #[test]
    fn bilinear_sampler_2x2_to_4x4() {
        // base 2x2, channel 0: values (gx,gy): (0,0)=0 (1,0)=2 (0,1)=8 (1,1)=10.
        let side = 2usize;
        let base = vec![0.0f32, 2.0, 8.0, 10.0];
        let n = 4usize;
        let sf = n as f32 / side as f32; // 2.0
        let po = 0.5f32;
        let sample = |dx: usize, dy: usize| -> f32 {
            let xf = (dx as f32 + po) / sf - po;
            let yf = (dy as f32 + po) / sf - po;
            let x0 = (xf.floor() as i64).clamp(0, side as i64 - 1) as usize;
            let x1 = ((xf.floor() as i64) + 1).clamp(0, side as i64 - 1) as usize;
            let y0 = (yf.floor() as i64).clamp(0, side as i64 - 1) as usize;
            let y1 = ((yf.floor() as i64) + 1).clamp(0, side as i64 - 1) as usize;
            let dxf = (xf - xf.floor()).clamp(0.0, 1.0);
            let dyf = (yf - yf.floor()).clamp(0.0, 1.0);
            let at = |gx: usize, gy: usize| base[gx + side * gy];
            at(x0, y0) * (1.0 - dxf) * (1.0 - dyf)
                + at(x1, y0) * dxf * (1.0 - dyf)
                + at(x0, y1) * (1.0 - dxf) * dyf
                + at(x1, y1) * dxf * dyf
        };
        // dst pixel i -> src (i+0.5)/2 - 0.5:
        //   i=0 -> -0.25 floor -1 clamp0, dx 0 effectively -> col0
        //   i=1 ->  0.25 -> x0=0,x1=1, dx=0.25
        //   i=3 ->  1.25 -> x0=1,x1=1 (clamp), dx=0.25 (but x0==x1 -> col1)
        assert!((sample(0, 0) - 0.0).abs() < 1e-6, "got {}", sample(0, 0));
        // (1,0): xf=0.25 over row0 [0,2] -> 0.5.
        assert!((sample(1, 0) - 0.5).abs() < 1e-6, "got {}", sample(1, 0));
        // (3,0): x clamps to col1 -> 2.0.
        assert!((sample(3, 0) - 2.0).abs() < 1e-6, "got {}", sample(3, 0));
        // (0,3): y clamps to row1, col0 -> 8.0.
        assert!((sample(0, 3) - 8.0).abs() < 1e-6, "got {}", sample(0, 3));
        // (3,3): clamps to (1,1) -> 10.0.
        assert!((sample(3, 3) - 10.0).abs() < 1e-6, "got {}", sample(3, 3));
        // interior (1,1): xf=yf=0.25, bilinear of [[0,2],[8,10]]:
        //   = 0*0.75*0.75 + 2*0.25*0.75 + 8*0.75*0.25 + 10*0.25*0.25 = 2.5
        assert!((sample(1, 1) - 2.5).abs() < 1e-6, "got {}", sample(1, 1));
    }
}
