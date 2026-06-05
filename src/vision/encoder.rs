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

use ash::vk;

use crate::gguf::GgmlType;
use crate::inference::Engine;
use crate::inference::buffer::BufferRange;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::memory::Region;
use crate::inference::ops::bind_and_dispatch;
use crate::inference::ops::elementwise::{record_add, record_get_rows, record_mul};
use crate::inference::ops::{matmul, rms_norm};
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::{TensorView, WeightsHandle};
use crate::shaders;
use crate::vision::preprocess::PreprocessedImage;

/// Byte size of `GenericParams` (`shaders/include/generic_head.slang`):
/// 6 × 4 = 24 bytes (KX, KY, param1..4).
const GENERIC_PARAMS_BYTES: u32 = 6 * 4;

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
    /// Attention head count (= 16).
    pub n_head: usize,
    /// Per-head dim (= n_embd / n_head = 72).
    pub head_dim: usize,
    /// Feed-forward hidden dim (= 4304).
    pub n_ff: usize,
    /// LayerNorm epsilon (= 1e-6).
    pub eps: f32,
    /// Per-block GPU weight views, indexed by layer 0..n_layer.
    pub blocks: Vec<BlockWeights>,
    /// Post-encoder LayerNorm-affine + merger MLP weight views.
    pub merger: MergerWeights,
    /// Merger output dim = text n_embd (= clip.vision.projection_dim = 2048),
    /// derived from `mm.2.weight`'s M dim.
    pub projection_dim: usize,
}

/// GPU [`TensorView`]s for the post-encoder norm + merger MLP
/// (`qwen3vl_merger`): `post_ln` (LayerNorm-affine, weight+bias F32) then
/// `mm.0` `[4*n_embd -> 4*n_embd]` -> GELU -> `mm.2` `[4*n_embd -> proj_dim]`
/// (matmul weights BF16/F16, biases F32). The `4*n_embd` input is the 2x2
/// spatial-merge group, formed by a contiguous reshape of the post_ln output
/// (Slice 2 already made the 4 patches of each 2x2 block consecutive tokens).
/// `mm.1` is the GELU between the two linears (hence the `mm.0`/`mm.2` naming).
#[derive(Clone, Copy)]
pub struct MergerWeights {
    /// `v.post_ln.weight` `[n_embd]` F32.
    pub post_ln_w: TensorView,
    /// `v.post_ln.bias` `[n_embd]` F32.
    pub post_ln_b: TensorView,
    /// `mm.0.weight` `[K=4*n_embd, M=4*n_embd]` BF16/F16.
    pub mm0_w: TensorView,
    /// `mm.0.bias` `[4*n_embd]` F32.
    pub mm0_b: TensorView,
    /// `mm.2.weight` `[K=4*n_embd, M=proj_dim]` BF16/F16.
    pub mm2_w: TensorView,
    /// `mm.2.bias` `[proj_dim]` F32.
    pub mm2_b: TensorView,
}

/// GPU [`TensorView`]s for one ViT transformer block. The four matmul weights
/// (`attn_qkv`/`attn_out`/`ffn_up`/`ffn_down`) are BF16 in the real
/// `mmproj-BF16.gguf` (F16 in the F16 variant); biases + LN weight/bias pairs
/// are F32. Both dtypes flow through the GPU path unchanged (`matmul::record`
/// wires BF16/F16 A; biases/LN are F32).
#[derive(Clone, Copy)]
pub struct BlockWeights {
    /// `v.blk.{i}.ln1.weight` `[n_embd]` F32.
    pub ln1_w: TensorView,
    /// `v.blk.{i}.ln1.bias` `[n_embd]` F32.
    pub ln1_b: TensorView,
    /// `v.blk.{i}.attn_qkv.weight` `[n_embd, 3*n_embd]` (GEMM A: K=n_embd,
    /// M=3*n_embd) BF16/F16.
    pub qkv_w: TensorView,
    /// `v.blk.{i}.attn_qkv.bias` `[3*n_embd]` F32.
    pub qkv_b: TensorView,
    /// `v.blk.{i}.attn_out.weight` `[n_embd, n_embd]` BF16/F16.
    pub out_w: TensorView,
    /// `v.blk.{i}.attn_out.bias` `[n_embd]` F32.
    pub out_b: TensorView,
    /// `v.blk.{i}.ln2.weight` `[n_embd]` F32.
    pub ln2_w: TensorView,
    /// `v.blk.{i}.ln2.bias` `[n_embd]` F32.
    pub ln2_b: TensorView,
    /// `v.blk.{i}.ffn_up.weight` `[n_embd, n_ff]` BF16/F16.
    pub ffn_up_w: TensorView,
    /// `v.blk.{i}.ffn_up.bias` `[n_ff]` F32.
    pub ffn_up_b: TensorView,
    /// `v.blk.{i}.ffn_down.weight` `[n_ff, n_embd]` BF16/F16.
    pub ffn_down_w: TensorView,
    /// `v.blk.{i}.ffn_down.bias` `[n_embd]` F32.
    pub ffn_down_b: TensorView,
}

/// Host-side F32 copies of one block's weights, up-converted from the mmproj
/// GGUF (BF16/F16 matmul weights, F32 biases/LN). The numerical oracle
/// ([`block_cpu`]) reads these. Matmul weights are stored as `[K, M]`
/// row-major (idx = `k + K*m`), matching `read_tensor_as_f32`'s logical order
/// and the GEMM contraction.
#[derive(Clone)]
pub struct BlockHostWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    /// `[K=n_embd, M=3*n_embd]` (idx = k + n_embd*m).
    pub qkv_w: Vec<f32>,
    pub qkv_b: Vec<f32>,
    /// `[K=n_embd, M=n_embd]`.
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    /// `[K=n_embd, M=n_ff]`.
    pub ffn_up_w: Vec<f32>,
    pub ffn_up_b: Vec<f32>,
    /// `[K=n_ff, M=n_embd]`.
    pub ffn_down_w: Vec<f32>,
    pub ffn_down_b: Vec<f32>,
}

impl BlockHostWeights {
    /// Read all twelve tensors for block `layer_idx` from the mmproj GGUF and
    /// up-convert to f32 (BF16/F16 -> f32 for the matmul weights, F32
    /// passthrough for biases/LN).
    pub fn from_gguf(
        gguf: &crate::gguf::GgufFile,
        layer_idx: u32,
    ) -> Result<BlockHostWeights, Box<dyn Error>> {
        let p = |s: &str| format!("v.blk.{layer_idx}.{s}");
        Ok(BlockHostWeights {
            ln1_w: read_tensor_as_f32(gguf, &p("ln1.weight"))?,
            ln1_b: read_tensor_as_f32(gguf, &p("ln1.bias"))?,
            qkv_w: read_tensor_as_f32(gguf, &p("attn_qkv.weight"))?,
            qkv_b: read_tensor_as_f32(gguf, &p("attn_qkv.bias"))?,
            out_w: read_tensor_as_f32(gguf, &p("attn_out.weight"))?,
            out_b: read_tensor_as_f32(gguf, &p("attn_out.bias"))?,
            ln2_w: read_tensor_as_f32(gguf, &p("ln2.weight"))?,
            ln2_b: read_tensor_as_f32(gguf, &p("ln2.bias"))?,
            ffn_up_w: read_tensor_as_f32(gguf, &p("ffn_up.weight"))?,
            ffn_up_b: read_tensor_as_f32(gguf, &p("ffn_up.bias"))?,
            ffn_down_w: read_tensor_as_f32(gguf, &p("ffn_down.weight"))?,
            ffn_down_b: read_tensor_as_f32(gguf, &p("ffn_down.bias"))?,
        })
    }
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
        n_head: usize,
        n_ff: usize,
        n_layer: usize,
        eps: f32,
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

        // Parse the per-block weight views. Each block's four matmul weights
        // (qkv/out/ffn_up/ffn_down) must be BF16/F16/F32 (the GEMM A side);
        // biases + LN weight/bias must be F32. Validate loudly.
        let head_dim = n_embd / n_head;
        let mut blocks = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let p = |s: &str| format!("v.blk.{il}.{s}");
            let mat = |name: String| -> Result<TensorView, Box<dyn Error>> {
                let v = weights.view(&name)?;
                if v.dtype != GgmlType::BF16 && v.dtype != GgmlType::F16 && v.dtype != GgmlType::F32
                {
                    return Err(format!(
                        "vision block {il}: expected BF16/F16/F32 {name}, got {:?}",
                        v.dtype
                    )
                    .into());
                }
                Ok(v)
            };
            let f32v = |name: String| -> Result<TensorView, Box<dyn Error>> {
                let v = weights.view(&name)?;
                if v.dtype != GgmlType::F32 {
                    return Err(format!(
                        "vision block {il}: expected F32 {name}, got {:?}",
                        v.dtype
                    )
                    .into());
                }
                Ok(v)
            };
            blocks.push(BlockWeights {
                ln1_w: f32v(p("ln1.weight"))?,
                ln1_b: f32v(p("ln1.bias"))?,
                qkv_w: mat(p("attn_qkv.weight"))?,
                qkv_b: f32v(p("attn_qkv.bias"))?,
                out_w: mat(p("attn_out.weight"))?,
                out_b: f32v(p("attn_out.bias"))?,
                ln2_w: f32v(p("ln2.weight"))?,
                ln2_b: f32v(p("ln2.bias"))?,
                ffn_up_w: mat(p("ffn_up.weight"))?,
                ffn_up_b: f32v(p("ffn_up.bias"))?,
                ffn_down_w: mat(p("ffn_down.weight"))?,
                ffn_down_b: f32v(p("ffn_down.bias"))?,
            });
        }

        // Post-encoder norm + merger MLP. mm.0/mm.2 weights are BF16/F16/F32
        // (GEMM A side); post_ln + biases are F32. Validate loudly, mirroring
        // the per-block parsing above.
        let mat = |name: &str| -> Result<TensorView, Box<dyn Error>> {
            let v = weights.view(name)?;
            if v.dtype != GgmlType::BF16 && v.dtype != GgmlType::F16 && v.dtype != GgmlType::F32 {
                return Err(format!(
                    "vision merger: expected BF16/F16/F32 {name}, got {:?}",
                    v.dtype
                )
                .into());
            }
            Ok(v)
        };
        let f32v = |name: &str| -> Result<TensorView, Box<dyn Error>> {
            let v = weights.view(name)?;
            if v.dtype != GgmlType::F32 {
                return Err(
                    format!("vision merger: expected F32 {name}, got {:?}", v.dtype).into(),
                );
            }
            Ok(v)
        };
        let merger = MergerWeights {
            post_ln_w: f32v("v.post_ln.weight")?,
            post_ln_b: f32v("v.post_ln.bias")?,
            mm0_w: mat("mm.0.weight")?,
            mm0_b: f32v("mm.0.bias")?,
            mm2_w: mat("mm.2.weight")?,
            mm2_b: f32v("mm.2.bias")?,
        };
        // proj_dim = mm.2.weight's M (output) dim. Sanity-check the merger input
        // dim equals 4*n_embd (the 2x2 spatial-merge group).
        let projection_dim = merger.mm2_w.dims[1] as usize;
        let merge_in = merger.mm0_w.dims[0] as usize;
        if merge_in != 4 * n_embd {
            return Err(format!(
                "vision merger: mm.0 input dim {merge_in} != 4*n_embd {}",
                4 * n_embd
            )
            .into());
        }

        Ok(VisionEncoder {
            patch_embd_0,
            patch_embd_1,
            n_embd,
            patch_size,
            n_head,
            head_dim,
            n_ff,
            eps,
            blocks,
            merger,
            projection_dim,
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

    /// Record ONE Qwen3-VL ViT transformer block (layer `layer_idx`) on the
    /// GPU. Input/output are both F32 `[n_embd, n_patches]` in the 2x2-block
    /// token order Slice 2 produces. `positions` is the staged vision-RoPE
    /// positions buffer ([`stage_vision_positions`]); it depends only on the
    /// patch grid (not the layer), so the caller builds it once and passes the
    /// same range to every block. Keeping it out of `record_block` is also what
    /// lets [`VisionEncoder::encode_image`] reclaim each block's scratch between
    /// layers without a host-write-into-reused-scratch hazard (the positions are
    /// the only host-staged input inside the block loop).
    ///
    /// Faithful port of the per-block body of `clip_graph_qwen3vl::build`
    /// (`/home/bob/tools/llama.cpp/src/tools/mtmd/models/qwen3vl.cpp:80-148`)
    /// reusing `clip_graph::build_norm` / `build_attn` / `build_ffn`
    /// (`clip.cpp:531-714`):
    ///
    /// * **LayerNorm-affine** (`build_norm`, NORM_TYPE_NORMAL, clip.cpp:539-551):
    ///   `ggml_norm` (subtract row mean, divide by `sqrt(var+eps)` over the
    ///   n_embd row) then `*ln.weight + ln.bias`. Composed from `norm.slang`
    ///   (mean-subtract+inv_std, no affine) -> `record_mul` weight -> `record_add`
    ///   bias (both broadcast over the n_pos columns). No new shader.
    /// * **Fused QKV** (qwen3vl.cpp:88-104): `qkv = qkv_w @ h + qkv_b`,
    ///   `[3*n_embd, n_pos]`. Q/K/V at row ranges 0/n_embd/2*n_embd, each
    ///   `n_head x head_dim`.
    /// * **Vision M-RoPE** (qwen3vl.cpp:111-116): `ggml_rope_multi(..., d_head/2,
    ///   {18,18,18,18}, GGML_ROPE_TYPE_VISION, n_ctx=32768, freq_base=10000,
    ///   ...)` on Q,K only. Positions built by [`build_vision_positions`]
    ///   (clip.cpp:3705-3730), read by `rope_vision` (rope_funcs.slang:155) as
    ///   `pos[i2]` (row) / `pos[i2+ne02]` (col).
    /// * **Attention** (`build_attn`, clip.cpp:685-701): no mask
    ///   (bidirectional), `scale = 1/sqrt(d_head)` (clip.cpp:253). Run via the
    ///   project's `flash_attn` op with `mask=None`.
    /// * **GELU FFN** (`build_ffn` FFN_GELU no-gate, clip.cpp:597-604): `ffn_down
    ///   @ gelu(ffn_up @ h2 + ffn_up_b) + ffn_down_b`. `ggml_gelu` is the
    ///   **tanh** approximation (gelu.slang).
    pub fn record_block(
        &self,
        ctx: &mut DispatchContext,
        x: TensorView,
        layer_idx: u32,
        positions: BufferRange,
    ) -> Result<TensorView, Box<dyn Error>> {
        let n_embd = self.n_embd;
        let n_head = self.n_head;
        let head_dim = self.head_dim;
        let n_ff = self.n_ff;
        let eps = self.eps;
        let n_pos = x.dims[1] as usize;
        let blk = self.blocks[layer_idx as usize];
        let f32 = GgmlType::F32;
        let nemb = n_embd as u64;
        let np = n_pos as u64;

        // Optional per-stage cutoff for the gpu_debug block smoke test: when
        // `SEEKER_VISION_BLOCK_STAGE=<name>` is set, return that intermediate so
        // the harness can localize a GPU-vs-CPU divergence.
        #[cfg(feature = "gpu_debug")]
        let stage = std::env::var("SEEKER_VISION_BLOCK_STAGE").ok();
        #[cfg(not(feature = "gpu_debug"))]
        let stage: Option<String> = None;
        let want = |s: &str| stage.as_deref() == Some(s);

        // --- 1. h = LayerNorm_affine(x, ln1) ---
        let h = ctx.alloc_tensor([nemb, np, 1, 1], f32)?;
        record_layernorm_affine(ctx, x, blk.ln1_w, blk.ln1_b, h, eps)?;
        if want("ln1") {
            return Ok(h);
        }

        // --- 2. qkv = qkv_w @ h + qkv_b   [3*n_embd, n_pos] ---
        let qkv = ctx.alloc_tensor([3 * nemb, np, 1, 1], f32)?;
        matmul::record(ctx, blk.qkv_w, h, qkv)?;
        record_add_bias_broadcast(ctx, qkv, blk.qkv_b)?;
        if want("qkv") {
            return Ok(qkv);
        }

        // --- 3. vision M-RoPE on Q (rows 0..n_embd) and K (n_embd..2*n_embd) ---
        let q_view = qkv_section_view(&qkv, head_dim, n_head, n_pos, 0);
        let k_view = qkv_section_view(&qkv, head_dim, n_head, n_pos, n_embd);
        record_vision_rope(ctx, q_view, positions, head_dim)?;
        record_vision_rope(ctx, k_view, positions, head_dim)?;
        if want("rope") {
            return Ok(qkv);
        }

        // --- 4. Full/bidirectional attention via flash_attn (mask=None) ---
        // flash_attn indexes Q/K/V as `data[d + i*nb01 + iq2*nb02]` with `i`
        // = ne1 = the query/token row and `iq2` = ne2 = the head
        // (flash_attn.slang:131-148), i.e. it wants the layout
        // `[head_dim, n_pos, n_head]` (TOKEN in dim1, HEAD in dim2). It writes
        // its output as `[hidden = head_dim*n_head, n_pos]` contiguous
        // (head-major within hidden), which is exactly attn_concat's
        // `[n_embd, n_pos]` (row = d + head_dim*head). Materialize Q/K/V into
        // that token-major layout from the strided fused-qkv sections
        // (copy.slang honors the source strides). Using flash_attn — the
        // project's validated fused QKᵀ->softmax->V kernel.
        let hd_pad = head_dim.next_multiple_of(32); // 72 -> 96
        let fa_dims = [hd_pad as u64, np, n_head as u64, 1];
        let q_c = ctx.alloc_tensor(fa_dims, f32)?;
        let k_c = ctx.alloc_tensor(fa_dims, f32)?;
        let v_c = ctx.alloc_tensor(fa_dims, f32)?;
        record_fill_zero(ctx, q_c)?;
        record_fill_zero(ctx, k_c)?;
        record_fill_zero(ctx, v_c)?;
        record_copy_pad_head(
            ctx,
            qkv_fa_view(&qkv, head_dim, n_head, n_pos, 0),
            &q_c,
            head_dim,
            hd_pad,
        )?;
        record_copy_pad_head(
            ctx,
            qkv_fa_view(&qkv, head_dim, n_head, n_pos, n_embd),
            &k_c,
            head_dim,
            hd_pad,
        )?;
        record_copy_pad_head(
            ctx,
            qkv_fa_view(&qkv, head_dim, n_head, n_pos, 2 * n_embd),
            &v_c,
            head_dim,
            hd_pad,
        )?;
        if want("qcopy") {
            return Ok(q_c);
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let fa_out = ctx.alloc_tensor([(hd_pad * n_head) as u64, np, 1, 1], f32)?;
        crate::inference::ops::flash_attn::record(
            ctx,
            q_c,
            k_c,
            v_c,
            /*mask=*/ None,
            fa_out,
            crate::inference::ops::flash_attn::FlashAttnParams {
                scale,
                head_dim_k: hd_pad as u32,
                head_dim_v: hd_pad as u32,
                gqa_ratio: 1,
                swa_window: 0,
            },
            /*kv_actual=*/ n_pos as u32,
        )?;
        let attn_concat = ctx.alloc_tensor([nemb, np, 1, 1], f32)?;
        record_gather_unpad_heads(ctx, &fa_out, &attn_concat, head_dim, hd_pad, n_head, n_pos)?;
        if want("attn_concat") {
            return Ok(attn_concat);
        }

        // --- 5. attn = out_w @ attn_concat + out_b ; x = x + attn ---
        let attn = ctx.alloc_tensor([nemb, np, 1, 1], f32)?;
        matmul::record(ctx, blk.out_w, attn_concat, attn)?;
        record_add_bias_broadcast(ctx, attn, blk.out_b)?;
        let resid1 = ctx.alloc_tensor([nemb, np, 1, 1], f32)?;
        record_add(ctx, x, attn, resid1)?;

        // --- 6. h2 = LayerNorm_affine(resid1, ln2) ---
        let h2 = ctx.alloc_tensor([nemb, np, 1, 1], f32)?;
        record_layernorm_affine(ctx, resid1, blk.ln2_w, blk.ln2_b, h2, eps)?;

        // --- 7. ffn = ffn_down @ gelu(ffn_up @ h2 + ffn_up_b) + ffn_down_b ---
        let up = ctx.alloc_tensor([n_ff as u64, np, 1, 1], f32)?;
        matmul::record(ctx, blk.ffn_up_w, h2, up)?;
        record_add_bias_broadcast(ctx, up, blk.ffn_up_b)?;
        let act = ctx.alloc_tensor([n_ff as u64, np, 1, 1], f32)?;
        record_gelu(ctx, up, act)?;
        let ffn = ctx.alloc_tensor([nemb, np, 1, 1], f32)?;
        matmul::record(ctx, blk.ffn_down_w, act, ffn)?;
        record_add_bias_broadcast(ctx, ffn, blk.ffn_down_b)?;

        // --- 8. x = resid1 + ffn ---
        let out = ctx.alloc_tensor([nemb, np, 1, 1], f32)?;
        record_add(ctx, resid1, ffn, out)?;
        Ok(out)
    }

    /// Record the post-encoder LayerNorm + merger MLP on the GPU. Input `x` is
    /// the final block output `[n_embd, n_pos]` (F32, 2x2-block token order);
    /// output is the image embeddings `[proj_dim, n_pos/4]` (F32) — one token
    /// per 2x2 merged patch, in text n_embd.
    ///
    /// Faithful port of the merger tail of `clip_graph_qwen3vl::build`
    /// (`/home/bob/tools/llama.cpp/src/tools/mtmd/models/qwen3vl.cpp`, the
    /// post-block section) + `clip.cpp build_norm`/`build_ffn`:
    /// * `post_ln`: LayerNorm-affine (`build_norm` NORM_TYPE_NORMAL), reusing
    ///   [`record_layernorm_affine`].
    /// * **2x2 merge**: `ggml_reshape(emb, n_embd*4, n_pos/4)` — a pure
    ///   contiguous reinterpret, since Slice 2 already placed each 2x2 block's
    ///   four patches at consecutive tokens (so the 4*n_embd row = the 4
    ///   stacked patch vectors). No data movement.
    /// * `mm.0` `@ x + mm.0.bias` -> **GELU** (`mm.1`) -> `mm.2 @ · + mm.2.bias`
    ///   (`build_ffn` FFN_GELU, no gate). Output dim = `proj_dim`.
    ///
    /// Deepstack is disabled in this mmproj (no `v.deepstack.*` tensors); a
    /// deepstack-enabled file would concat its features onto this output here.
    pub fn record_merger(
        &self,
        ctx: &mut DispatchContext,
        x: TensorView,
    ) -> Result<TensorView, Box<dyn Error>> {
        let n_embd = self.n_embd;
        let n_pos = x.dims[1] as usize;
        debug_assert_eq!(
            n_pos % 4,
            0,
            "merger needs n_pos divisible by 4 (2x2 merge)"
        );
        let n_merged = n_pos / 4;
        let merge_in = 4 * n_embd;
        let proj = self.projection_dim;
        let f32 = GgmlType::F32;
        let nm = n_merged as u64;
        let m = self.merger;

        // post_ln (affine) over the [n_embd, n_pos] rows.
        let pln = ctx.alloc_tensor([n_embd as u64, n_pos as u64, 1, 1], f32)?;
        record_layernorm_affine(ctx, x, m.post_ln_w, m.post_ln_b, pln, self.eps)?;

        // 2x2 merge = contiguous reshape [n_embd, n_pos] -> [4*n_embd, n_pos/4].
        let merged = dense_view(&pln.range(), [merge_in as u64, nm, 1, 1]);

        // fc1: out0 = mm0_w @ merged + mm0_b   [4*n_embd, n_merged]
        let out0 = ctx.alloc_tensor([merge_in as u64, nm, 1, 1], f32)?;
        matmul::record(ctx, m.mm0_w, merged, out0)?;
        record_add_bias_broadcast(ctx, out0, m.mm0_b)?;

        // gelu
        let act = ctx.alloc_tensor([merge_in as u64, nm, 1, 1], f32)?;
        record_gelu(ctx, out0, act)?;

        // fc2: out = mm2_w @ act + mm2_b   [proj_dim, n_merged]
        let out = ctx.alloc_tensor([proj as u64, nm, 1, 1], f32)?;
        matmul::record(ctx, m.mm2_w, act, out)?;
        record_add_bias_broadcast(ctx, out, m.mm2_b)?;
        Ok(out)
    }

    /// Record the FULL Qwen3-VL vision tower on the GPU: patch-embed -> 27 ViT
    /// blocks -> post_ln -> merger. Returns the image embeddings
    /// `[proj_dim, n_tokens]` (F32) ready to splice into the LLM decoder
    /// residual stream (Phase 3). `host_weights` supplies the host-side
    /// patch-embed conv/pos/bias (the device-local GPU copies aren't
    /// host-readable for the pos-embd resize).
    pub fn encode_image(
        &self,
        ctx: &mut DispatchContext,
        img: &PreprocessedImage,
        host_weights: &HostWeights,
    ) -> Result<TensorView, Box<dyn Error>> {
        let n_embd = self.n_embd as u64;
        let n_pos = (img.grid_w as u64) * (img.grid_h as u64);
        let f32 = GgmlType::F32;

        // Persistent buffers, allocated up front so resetting the scratch bump
        // cursor after each stage never frees them: two `[n_embd, n_pos]`
        // ping-pong carriers for the residual stream, plus the grid-invariant
        // vision-RoPE positions (built once; identical for every block).
        let mut cur = ctx.alloc_tensor([n_embd, n_pos, 1, 1], f32)?;
        let mut nxt = ctx.alloc_tensor([n_embd, n_pos, 1, 1], f32)?;
        let positions = stage_vision_positions(ctx, img.grid_w, img.grid_h)?;

        // Everything allocated past this checkpoint is per-stage scratch we
        // reclaim once the stage's result is copied into a carrier. This keeps
        // the working set O(n_pos) instead of O(n_layer · n_pos): without it the
        // bump allocator accumulates every block's intermediates and a
        // max-resolution image would reserve tens of GB of scratch — past the
        // device's max buffer size, losing the device. Reclaim is safe because
        // the command buffer runs in recorded order with compute barriers
        // between ops, and `positions` (the only host-staged input in the loop)
        // lives in the persistent region above, never in reclaimed scratch.
        let cp = ctx.scratch_checkpoint();

        let embed = self.record_patch_embed(ctx, img, host_weights)?;
        record_copy_contiguous(ctx, embed, cur)?;
        ctx.scratch_restore(cp);

        for il in 0..self.blocks.len() as u32 {
            let out = self.record_block(ctx, cur, il, positions)?;
            record_copy_contiguous(ctx, out, nxt)?;
            ctx.scratch_restore(cp);
            std::mem::swap(&mut cur, &mut nxt);
        }

        self.record_merger(ctx, cur)
    }
}

/// Build the grid-invariant Qwen3-VL vision-RoPE positions for a
/// `grid_w × grid_h` patch grid and stage them into scratch. The positions
/// depend only on the grid (every layer reuses them), so callers build them
/// once and hand the range to each [`VisionEncoder::record_block`].
pub fn stage_vision_positions(
    ctx: &mut DispatchContext,
    grid_w: u32,
    grid_h: u32,
) -> Result<BufferRange, Box<dyn Error>> {
    let positions = build_vision_positions(grid_w as usize, grid_h as usize);
    alloc_scratch_write(ctx, &i32_to_bytes(&positions))
}

/// Per-submit attention-work budget (in `n_pos² · n_blocks_per_submit` units)
/// for the chunked vision encode. The vision attention is full/bidirectional,
/// so each block's flash-attn dispatch costs ~`n_pos²`; past a per-submit
/// threshold a RADV/Strix-Halo compute submit faults (the fence still signals,
/// but the output is corrupted and a device loss is deferred to the next
/// submit). Empirically a submit survives ~250M `n_pos²·blocks`; we budget 150M
/// for margin. NOTE: a *single* block at n_pos ≳ 14.3k (n_pos² ≳ 205M) faults
/// on its own — beyond that even one-block-per-submit can't help (needs a
/// KV-chunked attention, or capping `SEEKER_IMG_MAX_TOKENS`). Override the
/// budget with `SEEKER_VISION_SUBMIT_BUDGET`.
fn vision_submit_budget() -> u64 {
    std::env::var("SEEKER_VISION_SUBMIT_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&b| b > 0)
        .unwrap_or(150_000_000)
}

/// Encode an image through the vision tower, splitting the ViT block stack
/// across MULTIPLE GPU submits when the whole forward would exceed
/// [`vision_submit_budget`] (large images). The residual `[n_embd, n_pos]` is
/// carried across submits in a dedicated persistent buffer — `engine.forward`
/// resets the scratch region each call, so it can't hold cross-submit state.
///
/// Numerically identical to the single-submit [`VisionEncoder::encode_image`]:
/// the same ops in the same order, just fenced into smaller submits (the carry
/// round-trips through the persistent buffer via exact F32 copies). Falls back
/// to the single submit when the whole encode fits one. Returns the merged
/// embeddings `[proj_dim, n_tok]` as host f32.
pub fn encode_image_chunked(
    engine: &mut Engine,
    weights: &WeightsHandle,
    encoder: &VisionEncoder,
    img: &PreprocessedImage,
    host_weights: &HostWeights,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let n_pos = (img.grid_w as u64) * (img.grid_h as u64);
    let nblocks = encoder.blocks.len();

    // Whole tower fits one submit (small/medium image): use the validated
    // single-submit path (byte-identical, fewer fence waits). Attention work
    // ~ n_pos² per block.
    if n_pos * n_pos * nblocks as u64 <= vision_submit_budget() {
        return engine.forward(weights, |ctx| {
            Ok(encoder.encode_image(ctx, img, host_weights)?.range())
        });
    }

    let n_embd = encoder.n_embd as u64;
    // Persistent residual carry, separate from the per-submit scratch.
    let carry = Region::new(
        &engine.device,
        n_embd * n_pos * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let carry_range = BufferRange {
        buffer: carry.buffer,
        offset: 0,
        size: n_embd * n_pos * 4,
    };
    let result = encode_chunked_submits(
        engine,
        weights,
        encoder,
        img,
        host_weights,
        carry_range,
        nblocks,
    );
    let mut carry = carry;
    carry.destroy(&engine.device.device);
    result
}

/// The multi-submit body of [`encode_image_chunked`] (split out so the `engine`
/// borrow is released before the caller frees the carry buffer): patch-embed
/// (one submit) → block groups (one submit each, ≤ budget/n_pos blocks) →
/// merger (final submit, read back to host). `carry` is the persistent
/// `[n_embd, n_pos]` residual buffer.
#[allow(clippy::too_many_arguments)]
fn encode_chunked_submits(
    engine: &mut Engine,
    weights: &WeightsHandle,
    encoder: &VisionEncoder,
    img: &PreprocessedImage,
    host_weights: &HostWeights,
    carry: BufferRange,
    nblocks: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let n_pos = (img.grid_w as u64) * (img.grid_h as u64);
    let n_embd = encoder.n_embd as u64;
    let f32t = GgmlType::F32;
    let carry_view = || dense_view(&carry, [n_embd, n_pos, 1, 1]);
    // Blocks per submit so that `n_pos² · group` stays under the budget (≥1).
    let group = ((vision_submit_budget() / (n_pos * n_pos)).max(1)) as usize;

    // Submit 1: patch-embed → carry.
    engine.forward(weights, |ctx| {
        let embed = encoder.record_patch_embed(ctx, img, host_weights)?;
        record_copy_contiguous(ctx, embed, carry_view())?;
        ctx.alloc_scratch(4) // dummy readback range (unused)
    })?;

    // Block groups: load carry → run ≤`group` blocks (scratch ping-pong) → store carry.
    let mut il = 0usize;
    while il < nblocks {
        let end = (il + group).min(nblocks);
        engine.forward(weights, |ctx| {
            let positions = stage_vision_positions(ctx, img.grid_w, img.grid_h)?;
            let mut cur = ctx.alloc_tensor([n_embd, n_pos, 1, 1], f32t)?;
            let mut nxt = ctx.alloc_tensor([n_embd, n_pos, 1, 1], f32t)?;
            record_copy_contiguous(ctx, carry_view(), cur)?;
            let cp = ctx.scratch_checkpoint();
            for l in il..end {
                let out = encoder.record_block(ctx, cur, l as u32, positions)?;
                record_copy_contiguous(ctx, out, nxt)?;
                ctx.scratch_restore(cp);
                std::mem::swap(&mut cur, &mut nxt);
            }
            record_copy_contiguous(ctx, cur, carry_view())?;
            ctx.alloc_scratch(4) // dummy readback range (unused)
        })?;
        il = end;
    }

    // Final submit: merger reads carry → embeddings, read back to host.
    engine.forward(weights, |ctx| {
        Ok(encoder.record_merger(ctx, carry_view())?.range())
    })
}

/// View the Q/K/V section (base_row 0 / n_embd / 2*n_embd) of the fused `qkv`
/// `[3*n_embd, n_pos]` tensor in the layout flash_attn wants:
/// `[head_dim, n_pos, n_head]` (TOKEN in dim1, HEAD in dim2). Element strides:
/// dim0 (head_dim) = 1, dim1 (token) = the full `3*n_embd` qkv row pitch,
/// dim2 (head) = head_dim. flash_attn reads `data[d + token*nb01 + head*nb02]`.
fn qkv_fa_view(
    qkv: &TensorView,
    head_dim: usize,
    n_head: usize,
    n_pos: usize,
    base_row: usize,
) -> TensorView {
    let row_pitch = qkv.element_stride[1]; // = 3*n_embd
    TensorView {
        buffer: qkv.buffer,
        byte_offset: qkv.byte_offset + (base_row as u64) * qkv.byte_stride[0],
        byte_size: qkv.byte_size,
        dims: [head_dim as u64, n_pos as u64, n_head as u64, 1],
        byte_stride: [
            qkv.byte_stride[0],
            qkv.byte_stride[0] * row_pitch,
            qkv.byte_stride[0] * head_dim as u64,
            qkv.byte_stride[0] * head_dim as u64 * n_head as u64,
        ],
        element_stride: [
            1,
            row_pitch,
            head_dim as u64,
            head_dim as u64 * n_head as u64,
        ],
        dtype: GgmlType::F32,
    }
}

/// Fill an F32 tensor with zeros via `fill.slang` (`data_d[i] = param1`,
/// i < KX). Used to zero the head-dim padding of the flash_attn Q/K/V buffers.
fn record_fill_zero(ctx: &mut DispatchContext, dst: TensorView) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    let nelements: u32 = dst.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes()); // KX
    // param1 (offset 8) = 0.0 -> already zero.
    let key = PipelineKey::dense("fill_f32", 1, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::FILL_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    bind_and_dispatch(ctx, &pipeline, &[0], &[dst.range()], &push, workgroups)?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// Gather the real `head_dim` channels of each head from the padded flash_attn
/// output `fa_out` `[hd_pad*n_head, n_pos]` (row `head*hd_pad + d`) into
/// `attn_concat` `[n_embd, n_pos]` (row `head*head_dim + d`). copy.slang reads
/// the strided source (skipping the `head_dim..hd_pad` pad rows) and writes the
/// contiguous dst.
fn record_gather_unpad_heads(
    ctx: &mut DispatchContext,
    fa_out: &TensorView,
    attn_concat: &TensorView,
    head_dim: usize,
    hd_pad: usize,
    n_head: usize,
    n_pos: usize,
) -> Result<(), Box<dyn Error>> {
    // Source view over fa_out as [head_dim, n_head, n_pos]: dim0 (d) stride 1,
    // dim1 (head) stride hd_pad (skip pad rows), dim2 (token) stride
    // hd_pad*n_head (full fa_out column pitch).
    let pad_col = (hd_pad * n_head) as u64;
    let src = TensorView {
        buffer: fa_out.buffer,
        byte_offset: fa_out.byte_offset,
        byte_size: fa_out.byte_size,
        dims: [head_dim as u64, n_head as u64, n_pos as u64, 1],
        byte_stride: [
            4,
            4 * hd_pad as u64,
            4 * pad_col,
            4 * pad_col * n_pos as u64,
        ],
        element_stride: [1, hd_pad as u64, pad_col, pad_col * n_pos as u64],
        dtype: GgmlType::F32,
    };
    // Dst is contiguous attn_concat reinterpreted as [head_dim, n_head, n_pos]
    // (= [n_embd, n_pos], row = head*head_dim + d).
    let dst = TensorView {
        buffer: attn_concat.buffer,
        byte_offset: attn_concat.byte_offset,
        byte_size: attn_concat.byte_size,
        dims: [head_dim as u64, n_head as u64, n_pos as u64, 1],
        byte_stride: [
            4,
            4 * head_dim as u64,
            4 * head_dim as u64 * n_head as u64,
            4 * head_dim as u64 * n_head as u64 * n_pos as u64,
        ],
        element_stride: [
            1,
            head_dim as u64,
            head_dim as u64 * n_head as u64,
            head_dim as u64 * n_head as u64 * n_pos as u64,
        ],
        dtype: GgmlType::F32,
    };
    record_copy_contiguous(ctx, src, dst)
}

/// LayerNorm-affine on `[n_embd, n_pos]`: `ggml_norm` (subtract row mean,
/// divide by `sqrt(var+eps)` over the n_embd row) -> `*weight` -> `+bias`.
/// Composed from `norm.slang` (mean-subtract+inv_std, NO affine) + `record_mul`
/// by `weight` + `record_add` by `bias`, both broadcast over the n_pos columns
/// via `[n_embd, 1]` views. No new shader. `weight`/`bias` are `[n_embd]` F32.
fn record_layernorm_affine(
    ctx: &mut DispatchContext,
    src: TensorView,
    weight: TensorView,
    bias: TensorView,
    dst: TensorView,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    let ne00 = src.dims[0] as u32; // n_embd (row length)
    // norm.slang: GenericParams { KX, KY, param1=eps, ... }, one WG per row,
    // row index decoded as z*262144 + y*512 + x.
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&ne00.to_ne_bytes()); // KX
    push[8..12].copy_from_slice(&eps.to_ne_bytes()); // param1 = eps
    let key = PipelineKey::dense("norm_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::NORM_F32_SPV.as_bytes())?;
    let rows = (src.dims[1].max(1) * src.dims[2].max(1) * src.dims[3].max(1)) as u32;
    let workgroups = row_workgroups(rows);
    bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());

    let w_bcast = broadcast_col_view(&weight, ne00 as u64);
    record_mul(ctx, dst, w_bcast, dst)?;
    let b_bcast = broadcast_col_view(&bias, ne00 as u64);
    record_add(ctx, dst, b_bcast, dst)?;
    Ok(())
}

/// Workgroup decomposition for a one-WG-per-row dispatch whose shader decodes
/// `row = z*262144 + y*512 + x` (norm.slang, soft_max.slang).
fn row_workgroups(rows: u32) -> [u32; 3] {
    if rows <= 512 {
        [rows, 1, 1]
    } else if rows <= 512 * 512 {
        [512, rows.div_ceil(512), 1]
    } else {
        [512, 512, rows.div_ceil(262144)]
    }
}

/// View a `[n]` F32 weight/bias as `[n, 1, 1, 1]` so the binary add/mul shader
/// broadcasts it across the dst's columns (dim1=1 folds the column index to 0
/// via the fastdiv path).
fn broadcast_col_view(v: &TensorView, n: u64) -> TensorView {
    TensorView {
        buffer: v.buffer,
        byte_offset: v.byte_offset,
        byte_size: v.byte_size,
        dims: [n, 1, 1, 1],
        byte_stride: [4, 4 * n, 4 * n, 4 * n],
        element_stride: [1, n, n, n],
        dtype: GgmlType::F32,
    }
}

/// Add a per-row bias vector `[M]` to every column of a `[M, n_pos]` F32 tensor
/// in place, via `record_add` with the bias presented as `[M, 1]`.
fn record_add_bias_broadcast(
    ctx: &mut DispatchContext,
    dst: TensorView,
    bias: TensorView,
) -> Result<(), Box<dyn Error>> {
    let m = dst.dims[0];
    let b = broadcast_col_view(&bias, m);
    record_add(ctx, dst, b, dst)
}

/// Apply Qwen3-VL vision M-RoPE in place to a `[head_dim, n_head, n_pos]` F32
/// view (Q or K). Thin wrapper over `rope_vision.slang`
/// (`rope_funcs.slang::rope_vision`, GGML_ROPE_TYPE_VISION). Matches
/// `ggml_rope_multi(..., d_head/2, {18,18,18,18}, VISION, 32768, 10000, 1, ...)`
/// (qwen3vl.cpp:111-116). `n_dims = d_head/2`; the shader reads `sections[0]`/
/// `[1]` only (`sect_dims = s0+s1`), uses `pos[i2]` for the first section
/// (patch row) and `pos[i2+ne02]` for the second (patch col).
fn record_vision_rope(
    ctx: &mut DispatchContext,
    qk: TensorView,
    positions: BufferRange,
    head_dim: usize,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(qk.dtype, GgmlType::F32);
    let ne00 = qk.dims[0] as u32; // head_dim
    let ne01 = qk.dims[1] as u32; // n_head
    let ne02 = qk.dims[2] as u32; // n_pos
    let nrows: u32 = ne01 * ne02 * qk.dims[3].max(1) as u32;
    let n_dims = (head_dim / 2) as u32; // d_head/2 = 36
    let sec = n_dims / 2; // 18 per (h,w) section
    let freq_base = 10000.0f32;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);

    // rope_params layout (rope_head.slang), 29 u32 slots = 116 bytes.
    const ROPE_PARAMS_BYTES: u32 = 116;
    let mut push = [0u8; ROPE_PARAMS_BYTES as usize];
    let mut w = 0usize;
    let put_u = |out: &mut [u8], w: &mut usize, v: u32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };
    let put_f = |out: &mut [u8], w: &mut usize, v: f32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };
    let put_i = |out: &mut [u8], w: &mut usize, v: i32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };
    put_u(&mut push, &mut w, 0); // rope_mode (unused by rope_vision)
    put_u(&mut push, &mut w, nrows);
    put_u(&mut push, &mut w, n_dims);
    put_f(&mut push, &mut w, 1.0); // freq_scale
    put_f(&mut push, &mut w, freq_base);
    put_f(&mut push, &mut w, 0.0); // ext_factor
    put_f(&mut push, &mut w, 1.0); // attn_factor
    put_f(&mut push, &mut w, 0.0); // corr_dims[0]
    put_f(&mut push, &mut w, 0.0); // corr_dims[1]
    put_f(&mut push, &mut w, theta_scale);
    put_u(&mut push, &mut w, 0); // has_ff
    put_i(&mut push, &mut w, sec as i32); // sections[0..4]
    put_i(&mut push, &mut w, sec as i32);
    put_i(&mut push, &mut w, sec as i32);
    put_i(&mut push, &mut w, sec as i32);
    put_u(&mut push, &mut w, 0); // is_imrope
    put_u(&mut push, &mut w, 0); // is_back
    put_u(&mut push, &mut w, 0); // set_rows_stride
    put_u(&mut push, &mut w, ne00);
    put_u(&mut push, &mut w, ne01);
    put_u(&mut push, &mut w, ne02);
    put_u(&mut push, &mut w, qk.element_stride[1] as u32); // nb01 (a)
    put_u(&mut push, &mut w, qk.element_stride[2] as u32); // nb02 (a)
    put_u(&mut push, &mut w, qk.element_stride[3] as u32); // nb03 (a)
    put_u(&mut push, &mut w, qk.element_stride[1] as u32); // nb11 (d)
    put_u(&mut push, &mut w, qk.element_stride[2] as u32); // nb12 (d)
    put_u(&mut push, &mut w, qk.element_stride[3] as u32); // nb13 (d)
    put_u(&mut push, &mut w, 0); // a_offset
    put_u(&mut push, &mut w, 0); // d_offset
    debug_assert_eq!(w, ROPE_PARAMS_BYTES as usize);

    let key = PipelineKey::dense("rope_vision_f32", 4, ROPE_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::ROPE_VISION_F32_SPV.as_bytes())?;
    // Shader: numthreads(1,256,1); i0 = 2*y so y spans n_dims/2 pairs; row = x +
    // 32768*z. Bindings: 0=A, 1=pos, 2=freq_factor (unused; bind pos), 3=D.
    let pairs = (ne00 / 2).max(1);
    let workgroups = [nrows, pairs.div_ceil(256), 1];
    bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3],
        &[qk.range(), positions, positions, qk.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, qk.range());
    Ok(())
}

/// View the Q (base_row=0) / K (n_embd) / V (2*n_embd) section of the fused
/// `qkv` `[3*n_embd, n_pos]` tensor as `[head_dim, n_head, n_pos]`. The
/// section's `n_embd = head_dim*n_head` rows are reinterpreted as `head_dim`
/// (dim0) × `n_head` (dim1); the token (dim2) stride is the full `3*n_embd`
/// qkv row pitch. Used both for the in-place vision-RoPE (Q/K) and the
/// flash_attn contiguous materialize (Q/K/V).
fn qkv_section_view(
    qkv: &TensorView,
    head_dim: usize,
    n_head: usize,
    n_pos: usize,
    base_row: usize,
) -> TensorView {
    let row_pitch = qkv.element_stride[1]; // = 3*n_embd
    TensorView {
        buffer: qkv.buffer,
        byte_offset: qkv.byte_offset + (base_row as u64) * qkv.byte_stride[0],
        byte_size: qkv.byte_size,
        dims: [head_dim as u64, n_head as u64, n_pos as u64, 1],
        byte_stride: [
            qkv.byte_stride[0],
            qkv.byte_stride[0] * head_dim as u64,
            qkv.byte_stride[0] * row_pitch,
            qkv.byte_stride[0] * row_pitch * n_pos as u64,
        ],
        element_stride: [1, head_dim as u64, row_pitch, row_pitch * n_pos as u64],
        dtype: GgmlType::F32,
    }
}

/// Copy a (possibly strided) F32 source into a contiguous F32 `dst` of the same
/// logical shape. Reuses the generic-unary `copy.slang` (`f32` variant), which
/// reads the source via its per-dim element strides (UnaryParams) and writes
/// the dst contiguously.
fn record_copy_contiguous(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    let push = crate::inference::ops::unary_params_bytes(&src, &dst, 0.0, 0.0);
    let key = PipelineKey::dense(
        "copy_f32",
        2,
        crate::inference::ops::UNARY_PARAMS_BYTES,
        Vec::new(),
    );
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::COPY_F32_SPV.as_bytes())?;
    let nelements: u32 = dst.dims.iter().product::<u64>() as u32;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// Copy a strided F32 `src` `[head_dim, n_pos, n_head]` into the first
/// `head_dim` channels of a padded contiguous `dst_pad`
/// `[hd_pad, n_pos, n_head]`, leaving channels `head_dim..hd_pad` (pre-zeroed)
/// untouched. The dst is a strided `[head_dim, n_pos, n_head]` view whose
/// dim1/dim2 strides step by `hd_pad` (skipping the pad rows).
fn record_copy_pad_head(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst_pad: &TensorView,
    head_dim: usize,
    hd_pad: usize,
) -> Result<(), Box<dyn Error>> {
    let n_pos = src.dims[1];
    let n_head = src.dims[2];
    let dst_slice = TensorView {
        buffer: dst_pad.buffer,
        byte_offset: dst_pad.byte_offset,
        byte_size: dst_pad.byte_size,
        dims: [head_dim as u64, n_pos, n_head, 1],
        byte_stride: [
            4,
            4 * hd_pad as u64,
            4 * hd_pad as u64 * n_pos,
            4 * hd_pad as u64 * n_pos * n_head,
        ],
        element_stride: [
            1,
            hd_pad as u64,
            hd_pad as u64 * n_pos,
            hd_pad as u64 * n_pos * n_head,
        ],
        dtype: GgmlType::F32,
    };
    record_copy_contiguous(ctx, src, dst_slice)
}

/// Element-wise GELU (tanh approximation) — matches `ggml_gelu` used by
/// `build_ffn`'s `FFN_GELU` (clip.cpp:602). `gelu.slang` is the same
/// tanh-approx kernel ggml-vulkan uses.
fn record_gelu(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes()); // KX
    let key = PipelineKey::dense("gelu_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::GELU_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// Build the Qwen3-VL vision M-RoPE positions buffer: a flat `[n_pos*4]` i32
/// array, axis-major. Faithful port of clip.cpp:3705-3730 (QWEN3VL):
///
/// ```text
/// ptr = 0;
/// for y in (0..ph step 2): for x in (0..pw step 2):
///     for dy in 0..2: for dx in 0..2:
///         positions[0*n_pos + ptr] = y + dy;   // axis 0 (patch ROW)
///         positions[1*n_pos + ptr] = x + dx;   // axis 1 (patch COL)
///         positions[2*n_pos + ptr] = y + dy;   // axis 2 (= row)
///         positions[3*n_pos + ptr] = x + dx;   // axis 3 (= col)
///         ptr++;
/// ```
///
/// The 2x2-block iteration order matches Slice 2's `token_to_patch`.
/// `rope_vision` reads axis0 (`pos[i2]`, row) and axis1 (`pos[i2+ne02]`, col).
pub fn build_vision_positions(npx: usize, npy: usize) -> Vec<i32> {
    let n_pos = npx * npy;
    let mut positions = vec![0i32; n_pos * 4];
    let merge = 2usize;
    let mut ptr = 0usize;
    let mut y = 0usize;
    while y < npy {
        let mut x = 0usize;
        while x < npx {
            for dy in 0..merge {
                for dx in 0..merge {
                    positions[ptr] = (y + dy) as i32;
                    positions[n_pos + ptr] = (x + dx) as i32;
                    positions[2 * n_pos + ptr] = (y + dy) as i32;
                    positions[3 * n_pos + ptr] = (x + dx) as i32;
                    ptr += 1;
                }
            }
            x += merge;
        }
        y += merge;
    }
    positions
}

/// Reinterpret a `&[i32]` as native-endian bytes.
fn i32_to_bytes(data: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

/// Pure CPU reference for [`VisionEncoder::record_block`] — the numerical
/// oracle. Implements the SAME math from the (up-converted f32)
/// [`BlockHostWeights`] with plain loops. `x` is `[n_embd, n_pos]`
/// (idx = c + n_embd*t).
#[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
pub fn block_cpu(
    bw: &BlockHostWeights,
    x: &[f32],
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    eps: f32,
    grid_w: usize,
    grid_h: usize,
) -> Vec<f32> {
    block_cpu_stage(bw, x, n_embd, n_head, n_ff, eps, grid_w, grid_h, None)
}

/// As [`block_cpu`] but returns the named intermediate stage instead of the
/// final output, mirroring `record_block`'s `SEEKER_VISION_BLOCK_STAGE` hook.
/// Stages: `"ln1"`, `"qkv"`, `"rope"`, `"qcopy"`, `"attn_concat"`. `None` =
/// full block.
#[allow(clippy::too_many_arguments)]
pub fn block_cpu_stage(
    bw: &BlockHostWeights,
    x: &[f32],
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    eps: f32,
    grid_w: usize,
    grid_h: usize,
    stage: Option<&str>,
) -> Vec<f32> {
    let n_pos = x.len() / n_embd;
    let head_dim = n_embd / n_head;
    debug_assert_eq!(grid_w * grid_h, n_pos);

    // --- 1. h = LayerNorm_affine(x, ln1) ---
    let h = layernorm_affine_cpu(x, &bw.ln1_w, &bw.ln1_b, n_embd, n_pos, eps);
    if stage == Some("ln1") {
        return h;
    }

    // --- 2. qkv = qkv_w @ h + qkv_b   [3*n_embd, n_pos] ---
    let m3 = 3 * n_embd;
    let mut qkv = vec![0f32; m3 * n_pos];
    for t in 0..n_pos {
        for mm in 0..m3 {
            let mut acc = 0f32;
            for k in 0..n_embd {
                acc += bw.qkv_w[k + n_embd * mm] * h[k + n_embd * t];
            }
            qkv[mm + m3 * t] = acc + bw.qkv_b[mm];
        }
    }
    if stage == Some("qkv") {
        return qkv;
    }

    // --- 3. vision M-RoPE on Q (rows 0..n_embd) and K (n_embd..2n_embd) ---
    let positions = build_vision_positions(grid_w, grid_h);
    let n_dims = head_dim / 2; // 36
    let sec = n_dims / 2; // 18
    let freq_base = 10000.0f32;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    for &base in &[0usize, n_embd] {
        for t in 0..n_pos {
            let pos_row = positions[t] as f32; // axis 0
            let pos_col = positions[n_pos + t] as f32; // axis 1
            for hd in 0..n_head {
                let off = base + hd * head_dim + m3 * t;
                for j in 0..n_dims {
                    let theta_base = if j < sec {
                        pos_row * theta_scale.powi(j as i32)
                    } else {
                        pos_col * theta_scale.powi((j - sec) as i32)
                    };
                    let cos_t = theta_base.cos();
                    let sin_t = theta_base.sin();
                    let x0 = qkv[off + j];
                    let x1 = qkv[off + j + n_dims];
                    qkv[off + j] = x0 * cos_t - x1 * sin_t;
                    qkv[off + j + n_dims] = x0 * sin_t + x1 * cos_t;
                }
            }
        }
    }
    if stage == Some("rope") {
        return qkv;
    }

    // --- 4. bidirectional scaled-softmax attention per head ---
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    if stage == Some("qcopy") {
        // q_c in the PADDED flash_attn layout [hd_pad, n_pos, n_head]: flat idx
        // = d + hd_pad*token + hd_pad*n_pos*head, with channels head_dim..hd_pad
        // zero. Matches the GPU's zero-filled + copied q_c exactly.
        let hd_pad = head_dim.next_multiple_of(32);
        let mut out = vec![0f32; hd_pad * n_pos * n_head];
        for hd in 0..n_head {
            for t in 0..n_pos {
                for d in 0..head_dim {
                    out[d + hd_pad * t + hd_pad * n_pos * hd] = qkv[hd * head_dim + d + m3 * t];
                }
            }
        }
        return out;
    }
    // attn_concat[c, t] with c = hd*head_dim + d.
    let mut attn_concat = vec![0f32; n_embd * n_pos];
    for hd in 0..n_head {
        let q_base = hd * head_dim;
        let k_base = n_embd + hd * head_dim;
        let v_base = 2 * n_embd + hd * head_dim;
        for qt in 0..n_pos {
            let mut logits = vec![0f32; n_pos];
            let mut maxv = f32::NEG_INFINITY;
            for kt in 0..n_pos {
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += qkv[q_base + d + m3 * qt] * qkv[k_base + d + m3 * kt];
                }
                let s = dot * scale;
                logits[kt] = s;
                if s > maxv {
                    maxv = s;
                }
            }
            let mut sum = 0f32;
            for l in logits.iter_mut() {
                *l = (*l - maxv).exp();
                sum += *l;
            }
            let inv = 1.0f32 / sum;
            for d in 0..head_dim {
                let mut acc = 0f32;
                for kt in 0..n_pos {
                    acc += logits[kt] * inv * qkv[v_base + d + m3 * kt];
                }
                attn_concat[hd * head_dim + d + n_embd * qt] = acc;
            }
        }
    }
    if stage == Some("attn_concat") {
        return attn_concat;
    }

    // --- 5. attn = out_w @ attn_concat + out_b ; x = x + attn ---
    let mut resid1 = vec![0f32; n_embd * n_pos];
    for t in 0..n_pos {
        for mm in 0..n_embd {
            let mut acc = 0f32;
            for k in 0..n_embd {
                acc += bw.out_w[k + n_embd * mm] * attn_concat[k + n_embd * t];
            }
            resid1[mm + n_embd * t] = x[mm + n_embd * t] + acc + bw.out_b[mm];
        }
    }

    // --- 6. h2 = LayerNorm_affine(resid1, ln2) ---
    let h2 = layernorm_affine_cpu(&resid1, &bw.ln2_w, &bw.ln2_b, n_embd, n_pos, eps);

    // --- 7. ffn = ffn_down @ gelu(ffn_up @ h2 + ffn_up_b) + ffn_down_b ---
    let mut act = vec![0f32; n_ff * n_pos];
    for t in 0..n_pos {
        for mm in 0..n_ff {
            let mut acc = 0f32;
            for k in 0..n_embd {
                acc += bw.ffn_up_w[k + n_embd * mm] * h2[k + n_embd * t];
            }
            act[mm + n_ff * t] = gelu_tanh(acc + bw.ffn_up_b[mm]);
        }
    }
    let mut out = vec![0f32; n_embd * n_pos];
    for t in 0..n_pos {
        for mm in 0..n_embd {
            let mut acc = 0f32;
            for k in 0..n_ff {
                acc += bw.ffn_down_w[k + n_ff * mm] * act[k + n_ff * t];
            }
            // --- 8. x = resid1 + ffn ---
            out[mm + n_embd * t] = resid1[mm + n_embd * t] + acc + bw.ffn_down_b[mm];
        }
    }
    out
}

/// CPU LayerNorm-affine over each `[n_embd]` column of `x` `[n_embd, n_pos]`:
/// `(x - mean)/sqrt(var + eps) * w + b`. ggml_norm uses the **population**
/// variance (`/n_embd`) — matching norm.slang:41.
fn layernorm_affine_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n_embd: usize,
    n_pos: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; n_embd * n_pos];
    for t in 0..n_pos {
        let base = n_embd * t;
        let mut mean = 0f32;
        for c in 0..n_embd {
            mean += x[base + c];
        }
        mean /= n_embd as f32;
        let mut var = 0f32;
        for c in 0..n_embd {
            let d = x[base + c] - mean;
            var += d * d;
        }
        var /= n_embd as f32;
        let inv_std = 1.0f32 / (var + eps).sqrt();
        for c in 0..n_embd {
            out[base + c] = (x[base + c] - mean) * inv_std * w[c] + b[c];
        }
    }
    out
}

/// Tanh-approximation GELU, matching `ggml_gelu` / `gelu.slang`:
/// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEF_A: f32 = 0.044715;
    let inner = SQRT_2_OVER_PI * (x + COEF_A * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
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

/// Host-side **F32** copies of the post-encoder norm + merger MLP weights
/// (up-converted from the mmproj GGUF). The numerical oracle ([`merger_cpu`])
/// reads these. Matmul weights are stored `[K, M]` row-major (idx = `k + K*m`).
#[derive(Clone)]
pub struct MergerHostWeights {
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    /// `[K=4*n_embd, M=4*n_embd]`.
    pub mm0_w: Vec<f32>,
    pub mm0_b: Vec<f32>,
    /// `[K=4*n_embd, M=proj_dim]`.
    pub mm2_w: Vec<f32>,
    pub mm2_b: Vec<f32>,
}

impl MergerHostWeights {
    pub fn from_gguf(gguf: &crate::gguf::GgufFile) -> Result<MergerHostWeights, Box<dyn Error>> {
        Ok(MergerHostWeights {
            post_ln_w: read_tensor_as_f32(gguf, "v.post_ln.weight")?,
            post_ln_b: read_tensor_as_f32(gguf, "v.post_ln.bias")?,
            mm0_w: read_tensor_as_f32(gguf, "mm.0.weight")?,
            mm0_b: read_tensor_as_f32(gguf, "mm.0.bias")?,
            mm2_w: read_tensor_as_f32(gguf, "mm.2.weight")?,
            mm2_b: read_tensor_as_f32(gguf, "mm.2.bias")?,
        })
    }
}

/// Pure CPU reference for [`VisionEncoder::record_merger`]. `x` is the final
/// block output `[n_embd, n_pos]` (idx = c + n_embd*t); returns
/// `[proj_dim, n_pos/4]` (idx = m + proj_dim*b).
pub fn merger_cpu(mh: &MergerHostWeights, x: &[f32], n_embd: usize, eps: f32) -> Vec<f32> {
    let n_pos = x.len() / n_embd;
    debug_assert_eq!(n_pos % 4, 0);
    let n_merged = n_pos / 4;
    let merge_in = 4 * n_embd;
    let proj = mh.mm2_b.len();

    // post_ln (affine), then the 2x2 merge is a no-op reinterpret: the
    // contiguous [n_embd, n_pos] f32 buffer IS the [4*n_embd, n_pos/4] buffer
    // (idx c + n_embd*t == idx (c + n_embd*(t%4)) + 4*n_embd*(t/4)).
    let merged = layernorm_affine_cpu(x, &mh.post_ln_w, &mh.post_ln_b, n_embd, n_pos, eps);

    // fc1 + bias -> GELU
    let mut act = vec![0f32; merge_in * n_merged];
    for b in 0..n_merged {
        for m in 0..merge_in {
            let mut acc = 0f32;
            for k in 0..merge_in {
                acc += mh.mm0_w[k + merge_in * m] * merged[k + merge_in * b];
            }
            act[m + merge_in * b] = gelu_tanh(acc + mh.mm0_b[m]);
        }
    }

    // fc2 + bias
    let mut out = vec![0f32; proj * n_merged];
    for b in 0..n_merged {
        for m in 0..proj {
            let mut acc = 0f32;
            for k in 0..merge_in {
                acc += mh.mm2_w[k + merge_in * m] * act[k + merge_in * b];
            }
            out[m + proj * b] = acc + mh.mm2_b[m];
        }
    }
    out
}

/// Pure CPU reference for the FULL vision tower ([`VisionEncoder::encode_image`]):
/// patch-embed -> 27 blocks -> post_ln -> merger. Returns `[proj_dim, n_tokens]`
/// (idx = m + proj_dim*tok). The numerical oracle the GPU path is diffed
/// against, and the faithful llama.cpp port the Slice-4 anchor checks.
#[allow(clippy::too_many_arguments)]
pub fn encode_image_cpu(
    hw: &HostWeights,
    blocks: &[BlockHostWeights],
    mh: &MergerHostWeights,
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    eps: f32,
    patch_size: usize,
    img: &PreprocessedImage,
) -> Vec<f32> {
    let gw = img.grid_w as usize;
    let gh = img.grid_h as usize;
    let mut x = patch_embed_cpu(hw, n_embd, patch_size, img);
    for bw in blocks {
        x = block_cpu(bw, &x, n_embd, n_head, n_ff, eps, gw, gh);
    }
    merger_cpu(mh, &x, n_embd, eps)
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
        GgmlType::BF16 => {
            let n = bytes.len() / 2;
            let mut out = vec![0f32; n];
            for (i, o) in out.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                *o = bf16_to_f32(bits);
            }
            Ok(out)
        }
        other => Err(format!("{name}: expected BF16/F16/F32, got {other:?}").into()),
    }
}

/// bfloat16 (truncated IEEE-754 binary32 top 16 bits) -> f32. Exact: bf16 is
/// the high half of an f32, so zero-extend into the mantissa. Matches
/// `ggml_bf16_to_fp32`.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        // normal: (1 + mant/1024) * 2^(exp-15)
        (1.0f32 + (mant as f32) / 1024.0) * (2.0f32).powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
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
            #[allow(clippy::needless_range_loop)]
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
/// Faithful port of `clip_graph::resize_position_embeddings` (clip.cpp:278-296)
/// followed by the same merge reorder. `pos_base` is the raw `[n_embd, 2304]`
/// weight (row = channel, col = `gx + 48*gy`, gx = width/x index, gy =
/// height/y index — matching the `ggml_reshape_3d(n_embd, 48, 48)` +
/// `permute(2,0,1)` that feeds `ggml_interpolate`).
///
/// **Antialiased** bilinear: the default mode is
/// `GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS` (clip-graph.h:12), NOT
/// plain bilinear. For the typical >1x downscale (48 -> ~10..14) this is a wide
/// triangle-filter low-pass, not a 4-tap blend — using plain bilinear here adds
/// a wrong positional bias to every patch that then compounds through all 27
/// ViT blocks. Exact port of the antialias branch of
/// `ggml_compute_forward_upscale_f32` (ggml-cpu/ops.cpp:7624-7680), itself
/// modeled on `F.interpolate(mode="bilinear", align_corners=False,
/// antialias=True)`: `pixel_offset=0.5`, `sf=dst/src`,
/// `support=max(1, 1/sf)`, `invscale=1/support`, source coord
/// `x=(i+0.5)/sf`, and a normalized sum of `triangle((s - x + 0.5)*invscale)`
/// weights over `s in [x-support+0.5, x+support+0.5) ∩ [0, src)`.
pub fn resize_position_embeddings_reordered(
    pos_base: &[f32],
    n_embd: usize,
    npx: usize,
    npy: usize,
) -> Vec<f32> {
    let side = POS_EMBD_BASE_SIDE; // 48 (= sqrt(2304))
    debug_assert_eq!(pos_base.len(), n_embd * side * side);
    let n_patches = npx * npy;
    let mut out = vec![0f32; n_embd * n_patches];

    // resize_position_embeddings short-circuits when no resize is needed
    // (height == n_per_side && width == n_per_side); only the 2x2 reorder runs.
    if npx == side && npy == side {
        for tok in 0..n_patches {
            let (pw, ph) = token_to_patch(tok, npx);
            for c in 0..n_embd {
                out[c + n_embd * tok] = pos_base[c + n_embd * (pw + side * ph)];
            }
        }
        return out;
    }

    // ggml antialiased bilinear. sf = dst/src; pixel_offset = 0.5.
    let sf0 = npx as f32 / side as f32; // width  (x)
    let sf1 = npy as f32 / side as f32; // height (y)
    let po = 0.5f32;
    let support0 = (1.0f32 / sf0).max(1.0);
    let support1 = (1.0f32 / sf1).max(1.0);
    let invscale0 = 1.0 / support0;
    let invscale1 = 1.0 / support1;
    let tri = |t: f32| (1.0 - t.abs()).max(0.0);

    for tok in 0..n_patches {
        let (dx, dy) = token_to_patch(tok, npx); // dst x (col), dst y (row)
        let x = (dx as f32 + po) / sf0;
        let y = (dy as f32 + po) / sf1;
        // Contributing source range (int64 trunc-toward-zero then clamp, as ggml).
        let x_min = ((x - support0 + po) as i64).max(0) as usize;
        let x_max = ((x + support0 + po) as i64).min(side as i64) as usize;
        let y_min = ((y - support1 + po) as i64).max(0) as usize;
        let y_max = ((y + support1 + po) as i64).min(side as i64) as usize;
        // Precompute the per-axis triangle weights (independent of channel).
        let xs: Vec<(usize, f32)> = (x_min..x_max)
            .map(|sx| (sx, tri((sx as f32 - x + po) * invscale0)))
            .collect();
        let ys: Vec<(usize, f32)> = (y_min..y_max)
            .map(|sy| (sy, tri((sy as f32 - y + po) * invscale1)))
            .collect();
        for c in 0..n_embd {
            // Accumulate val + total_weight in ggml's (sy outer, sx inner) order,
            // skipping non-positive weights, then normalize.
            let mut val = 0f32;
            let mut tw = 0f32;
            for &(sy, wy) in &ys {
                for &(sx, wx) in &xs {
                    let w = wx * wy;
                    if w <= 0.0 {
                        continue;
                    }
                    val += pos_base[c + n_embd * (sx + side * sy)] * w;
                    tw += w;
                }
            }
            if tw > 0.0 {
                val /= tw;
            }
            out[c + n_embd * tok] = val;
        }
    }
    out
}

/// Reinterpret a `&[f32]` as native-endian bytes.
pub(crate) fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
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
pub(crate) fn alloc_scratch_write(
    ctx: &mut DispatchContext,
    bytes: &[u8],
) -> Result<BufferRange, Box<dyn Error>> {
    let range = ctx.alloc_scratch(bytes.len() as u64)?;
    let base = ctx
        .scratch
        .host_ptr
        .ok_or("scratch region is not host-visible; cannot stage media inputs")?;
    unsafe {
        let dst = base.add(range.offset as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
    Ok(range)
}

/// Build the gemma4 im2col matrix `[K=patch²·3, n_patches]` in **raster** patch
/// order (token = gy·npx + gx), flat row-major (index = `kidx + K·tok`). Element
/// order within a column is `kw + patch·(kh + patch·ic)` to match ggml's im2col
/// (which the `patch_embd` weight was converted for). gemma4uv feeds the pixels
/// directly (px/255, no ×2−1 — unlike gemma4v).
fn build_gemma_im2col(img: &PreprocessedImage, patch: usize) -> Vec<f32> {
    let w = img.resized_w as usize;
    let h = img.resized_h as usize;
    let npx = w / patch;
    let npy = h / patch;
    let n_patches = npx * npy;
    let k = patch * patch * N_CHANNELS;
    let mut out = vec![0f32; k * n_patches];
    for tok in 0..n_patches {
        let gx = tok % npx;
        let gy = tok / npx;
        for ic in 0..N_CHANNELS {
            for kh in 0..patch {
                for kw in 0..patch {
                    let x = gx * patch + kw;
                    let y = gy * patch + kh;
                    let kidx = kw + patch * (kh + patch * ic);
                    out[kidx + k * tok] = img.pixels[ic * (w * h) + y * w + x];
                }
            }
        }
    }
    out
}

/// View one of the two stacked `[n_embd, pos_size]` position-embedding lookup
/// tables inside `v.position_embd.weight [n_embd, pos_size, 2]` (idx 0 = x, 1 = y).
fn slice_pos_table(posembd: &TensorView, n_embd: u64, pos_size: u64, idx: u64) -> TensorView {
    let elem = posembd.byte_stride[0];
    TensorView {
        buffer: posembd.buffer,
        byte_offset: posembd.byte_offset + idx * n_embd * pos_size * elem,
        byte_size: n_embd * pos_size * elem,
        dims: [n_embd, pos_size, 1, 1],
        byte_stride: [
            elem,
            elem * n_embd,
            elem * n_embd * pos_size,
            elem * n_embd * pos_size,
        ],
        element_stride: [1, n_embd, n_embd * pos_size, n_embd * pos_size],
        dtype: posembd.dtype,
    }
}

fn u32_to_bytes(data: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(data.len() * 4);
    for &x in data {
        v.extend_from_slice(&x.to_ne_bytes());
    }
    v
}

/// Result of [`encode_image_gemma4`]: `(embeddings [projection_dim · n_tok]
/// column-major, n_tok, npx, npy)`.
pub type Gemma4Encoded = (Vec<f32>, usize, usize, usize);

/// Encode an image through the gemma4 `gemma4uv` "no-tower" projector. Returns
/// `(embeddings [projection_dim · n_tok] column-major, n_tok, npx, npy)`.
///
/// Pipeline (`clip_graph_gemma4uv::build`): im2col(48,stride48) →
/// LayerNorm(patch_norm_1) → matmul(patch_embd)+patch_bias →
/// LayerNorm(patch_norm_2) → +pos_x +pos_y (2D lookup) → LayerNorm(patch_norm_3,
/// "pos_norm") → RMSNorm(embedding_pre_projection_norm) →
/// matmul(input_projection). No transformer tower, no pooler. The 3 LayerNorms
/// use eps=1e-5 (pytorch default); the final RMSNorm uses the clip eps (1e-6).
pub fn encode_image_gemma4(
    engine: &mut crate::inference::Engine,
    weights: &WeightsHandle,
    cfg: &crate::vision::VisionConfig,
    img: &PreprocessedImage,
) -> Result<Gemma4Encoded, Box<dyn Error>> {
    let patch = (cfg.patch_size * cfg.n_merge) as usize; // 16·3 = 48
    let npx = img.resized_w as usize / patch;
    let npy = img.resized_h as usize / patch;
    let n_patches = npx * npy;
    let k = (patch * patch * N_CHANNELS) as u64; // 6912
    let n_embd = cfg.n_embd as u64; // 3840
    let proj_dim = cfg.projection_dim as u64;
    let np = n_patches as u64;

    let im2col = build_gemma_im2col(img, patch);
    let pos_x: Vec<u32> = (0..n_patches).map(|t| (t % npx) as u32).collect();
    let pos_y: Vec<u32> = (0..n_patches).map(|t| (t / npx) as u32).collect();

    let pe = weights.view("v.patch_embd.weight")?;
    let pe_b = weights.view("v.patch_embd.bias")?;
    let pn1w = weights.view("v.patch_norm.1.weight")?;
    let pn1b = weights.view("v.patch_norm.1.bias")?;
    let pn2w = weights.view("v.patch_norm.2.weight")?;
    let pn2b = weights.view("v.patch_norm.2.bias")?;
    let pn3w = weights.view("v.patch_norm.3.weight")?;
    let pn3b = weights.view("v.patch_norm.3.bias")?;
    let posembd = weights.view("v.position_embd.weight")?;
    let inproj = weights.view("mm.input_projection.weight")?;
    let pos_size = posembd.dims[1];
    let tbl_x = slice_pos_table(&posembd, n_embd, pos_size, 0);
    let tbl_y = slice_pos_table(&posembd, n_embd, pos_size, 1);

    let ln_eps = 1e-5f32; // gemma4uv embedder LayerNorms (pytorch default)
    let rms_eps = cfg.eps; // embedding_pre_projection_norm (clip eps, 1e-6)

    let out = engine.forward(weights, |ctx| {
        let im_r = alloc_scratch_write(ctx, &f32_to_bytes(&im2col))?;
        let im_v = dense_view(&im_r, [k, np, 1, 1]);
        let ln1 = ctx.alloc_tensor([k, np, 1, 1], GgmlType::F32)?;
        record_layernorm_affine(ctx, im_v, pn1w, pn1b, ln1, ln_eps)?;

        let embd = ctx.alloc_tensor([n_embd, np, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, pe, ln1, embd)?;
        record_add(ctx, embd, broadcast_col_view(&pe_b, n_embd), embd)?;

        let ln2 = ctx.alloc_tensor([n_embd, np, 1, 1], GgmlType::F32)?;
        record_layernorm_affine(ctx, embd, pn2w, pn2b, ln2, ln_eps)?;

        let px_r = alloc_scratch_write(ctx, &u32_to_bytes(&pos_x))?;
        let py_r = alloc_scratch_write(ctx, &u32_to_bytes(&pos_y))?;
        let emx = ctx.alloc_tensor([n_embd, np, 1, 1], GgmlType::F32)?;
        record_get_rows(ctx, tbl_x, px_r, n_patches as u32, emx)?;
        record_add(ctx, ln2, emx, ln2)?;
        let emy = ctx.alloc_tensor([n_embd, np, 1, 1], GgmlType::F32)?;
        record_get_rows(ctx, tbl_y, py_r, n_patches as u32, emy)?;
        record_add(ctx, ln2, emy, ln2)?;

        let ln3 = ctx.alloc_tensor([n_embd, np, 1, 1], GgmlType::F32)?;
        record_layernorm_affine(ctx, ln2, pn3w, pn3b, ln3, ln_eps)?;

        let rms = ctx.alloc_tensor([n_embd, np, 1, 1], GgmlType::F32)?;
        rms_norm::record_noweight(ctx, ln3, rms, rms_eps)?;

        let outp = ctx.alloc_tensor([proj_dim, np, 1, 1], GgmlType::F32)?;
        matmul::record(ctx, inproj, rms, outp)?;
        Ok(outp.range())
    })?;
    Ok((out, n_patches, npx, npy))
}

/// Build a dense (contiguous) F32 [`TensorView`] over a scratch range.
pub(crate) fn dense_view(range: &BufferRange, dims: [u64; 4]) -> TensorView {
    let es = [
        1u64,
        dims[0],
        dims[0] * dims[1],
        dims[0] * dims[1] * dims[2],
    ];
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
        #[allow(clippy::needless_range_loop)]
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
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (3, 0),
            (2, 1),
            (3, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 2),
            (3, 2),
            (2, 3),
            (3, 3),
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
        #[allow(clippy::needless_range_loop)]
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
        #[allow(clippy::needless_range_loop)]
        for tok in 0..(side * side) {
            let (pw, ph) = token_to_patch(tok, side);
            assert_eq!(out[tok], (pw + 100 * ph) as f32, "tok {tok}");
        }
    }

    /// Hand-checked ggml **antialiased** bilinear downscale (the production
    /// path; `GGML_SCALE_FLAG_ANTIALIAS`) on a 4x1 -> 2x1 single-channel row.
    /// Re-derives the same arithmetic as the production inner loop on a tiny
    /// case so the weights and normalization are hand-verifiable.
    ///
    /// src = [10,20,30,40], sf0 = 2/4 = 0.5, support0 = max(1,1/0.5)=2,
    /// invscale0 = 0.5, po = 0.5. triangle(t) = max(1-|t|,0).
    ///   dst 0: x=(0+.5)/.5=1.0; src sx∈{0,1,2}; wx=tri((sx-1+.5)*.5)=
    ///          {.75,.75,.25}; val=(10*.75+20*.75+30*.25)/1.75 = 30/1.75
    ///   dst 1: x=(1+.5)/.5=3.0; src sx∈{1,2,3}; wx={.25,.75,.75};
    ///          val=(20*.25+30*.75+40*.75)/1.75 = 57.5/1.75
    #[test]
    fn antialias_bilinear_downscale_4_to_2() {
        let src = [10.0f32, 20.0, 30.0, 40.0];
        let (src_w, dst_w) = (4usize, 2usize);
        let sf0 = dst_w as f32 / src_w as f32;
        let po = 0.5f32;
        let support0 = (1.0f32 / sf0).max(1.0);
        let invscale0 = 1.0 / support0;
        let tri = |t: f32| (1.0f32 - t.abs()).max(0.0);
        let sample = |i0: usize| -> f32 {
            let x = (i0 as f32 + po) / sf0;
            let x_min = ((x - support0 + po) as i64).max(0) as usize;
            let x_max = ((x + support0 + po) as i64).min(src_w as i64) as usize;
            let (mut val, mut tw) = (0f32, 0f32);
            #[allow(clippy::needless_range_loop)]
            for sx in x_min..x_max {
                let w = tri((sx as f32 - x + po) * invscale0);
                if w <= 0.0 {
                    continue;
                }
                val += src[sx] * w;
                tw += w;
            }
            if tw > 0.0 { val / tw } else { 0.0 }
        };
        assert!(
            (sample(0) - 30.0 / 1.75).abs() < 1e-5,
            "dst0 got {}",
            sample(0)
        );
        assert!(
            (sample(1) - 57.5 / 1.75).abs() < 1e-5,
            "dst1 got {}",
            sample(1)
        );
    }

    // ---- Slice 3 (ViT block) CPU unit tests ----

    /// LayerNorm-affine on a hand-checkable row. For `x=[1,2,3,4]`: mean=2.5,
    /// population var=1.25, normalized=[-1.5,-0.5,0.5,1.5]/sqrt(1.25). With
    /// weight=2, bias=10 the output is `2*norm + 10`.
    #[test]
    fn layernorm_affine_hand_checked() {
        let n_embd = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![2.0f32; 4];
        let b = vec![10.0f32; 4];
        let out = layernorm_affine_cpu(&x, &w, &b, n_embd, 1, 0.0);
        let inv = 1.0f32 / 1.25f32.sqrt();
        let expect = [
            2.0 * (-1.5 * inv) + 10.0,
            2.0 * (-0.5 * inv) + 10.0,
            2.0 * (0.5 * inv) + 10.0,
            2.0 * (1.5 * inv) + 10.0,
        ];
        for (o, e) in out.iter().zip(expect.iter()) {
            assert!((o - e).abs() < 1e-5, "got {o}, want {e}");
        }
        // Population variance: normalized is zero-mean.
        let mean_norm: f32 = out.iter().map(|v| (v - 10.0) / 2.0).sum::<f32>() / 4.0;
        assert!(mean_norm.abs() < 1e-5, "norm not zero-mean: {mean_norm}");
    }

    /// GELU (tanh approx) endpoints/anchors.
    #[test]
    fn gelu_tanh_endpoints() {
        assert!(gelu_tanh(0.0).abs() < 1e-7);
        assert!((gelu_tanh(10.0) - 10.0).abs() < 1e-3);
        assert!(gelu_tanh(-10.0).abs() < 1e-4);
        assert!(
            (gelu_tanh(1.0) - 0.841_192).abs() < 1e-4,
            "got {}",
            gelu_tanh(1.0)
        );
        assert!(gelu_tanh(0.5) > gelu_tanh(-0.5));
    }

    /// bf16->f32 round-trips the high 16 bits of an f32 exactly.
    #[test]
    fn bf16_roundtrip_exact_when_representable() {
        for &v in &[1.0f32, -2.0, 0.5, 0.0, 4.0] {
            let bits = (v.to_bits() >> 16) as u16;
            assert_eq!(bf16_to_f32(bits), v, "bf16 roundtrip failed for {v}");
        }
    }

    /// The fused QKV split: Q rows `0..n_embd`, K `n_embd..2*n_embd`, V
    /// `2*n_embd..3*n_embd`; head `hd` is `hd*head_dim..(hd+1)*head_dim`. Verify
    /// the offset arithmetic block_cpu uses is a non-overlapping partition.
    #[test]
    fn qkv_split_offsets_partition() {
        let n_embd = 1152;
        let n_head = 16;
        let head_dim = n_embd / n_head;
        assert_eq!(head_dim, 72);
        let mut seen = vec![0u8; 3 * n_embd];
        for hd in 0..n_head {
            for d in 0..head_dim {
                for base in [
                    hd * head_dim,
                    n_embd + hd * head_dim,
                    2 * n_embd + hd * head_dim,
                ] {
                    assert!(base + d < 3 * n_embd);
                    seen[base + d] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "qkv split not a partition");
    }

    /// The vision-rope position map for a 4x4 patch grid. Asserts axis0[tok]=ph
    /// and axis1[tok]=pw where `(pw,ph)=token_to_patch(tok, npx)`, i.e. the
    /// positions match clip.cpp's 2x2-block iteration AND Slice 2's token order.
    #[test]
    fn vision_positions_match_token_order_4x4() {
        let npx = 4;
        let npy = 4;
        let n_pos = npx * npy;
        let pos = build_vision_positions(npx, npy);
        assert_eq!(pos.len(), n_pos * 4);
        for tok in 0..n_pos {
            let (pw, ph) = token_to_patch(tok, npx);
            assert_eq!(pos[tok], ph as i32, "tok {tok} axis0 (row)");
            assert_eq!(pos[n_pos + tok], pw as i32, "tok {tok} axis1 (col)");
            assert_eq!(pos[2 * n_pos + tok], ph as i32, "tok {tok} axis2");
            assert_eq!(pos[3 * n_pos + tok], pw as i32, "tok {tok} axis3");
        }
        // First 2x2 block (clip.cpp:3713-3724, y=0,x=0; dy,dx in 0..2):
        //   tok0=(r0,c0) tok1=(r0,c1) tok2=(r1,c0) tok3=(r1,c1).
        assert_eq!((pos[0], pos[n_pos]), (0, 0));
        assert_eq!((pos[1], pos[n_pos + 1]), (0, 1));
        assert_eq!((pos[2], pos[n_pos + 2]), (1, 0));
        assert_eq!((pos[3], pos[n_pos + 3]), (1, 1));
    }

    /// block_cpu with all matmul weights = 0 and biases = 0: LN1->qkv=0->attn=0
    /// ->residual1=x; LN2->ffn(0)=0->out=residual1=x. A zero-weight block is the
    /// identity on the residual stream.
    #[test]
    fn block_cpu_zero_weights_is_identity() {
        let n_embd = 8;
        let n_head = 2;
        let n_ff = 16;
        let (gw, gh) = (2usize, 2usize);
        let n_pos = gw * gh;
        let bw = BlockHostWeights {
            ln1_w: vec![1.0; n_embd],
            ln1_b: vec![0.0; n_embd],
            qkv_w: vec![0.0; n_embd * 3 * n_embd],
            qkv_b: vec![0.0; 3 * n_embd],
            out_w: vec![0.0; n_embd * n_embd],
            out_b: vec![0.0; n_embd],
            ln2_w: vec![1.0; n_embd],
            ln2_b: vec![0.0; n_embd],
            ffn_up_w: vec![0.0; n_embd * n_ff],
            ffn_up_b: vec![0.0; n_ff],
            ffn_down_w: vec![0.0; n_ff * n_embd],
            ffn_down_b: vec![0.0; n_embd],
        };
        let x: Vec<f32> = (0..n_embd * n_pos)
            .map(|i| (i as f32) * 0.1 - 1.0)
            .collect();
        let out = block_cpu(&bw, &x, n_embd, n_head, n_ff, 1e-6, gw, gh);
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!(
                (o - xi).abs() < 1e-5,
                "zero-weight block not identity: {o} vs {xi}"
            );
        }
    }

    /// merger_cpu shape + bias invariant: with both matmul weights zeroed, the
    /// merger collapses to a constant — fc1 = mm0_b, gelu(mm0_b), fc2 = mm2_b —
    /// so every output token equals `mm2_b` regardless of the input or post_ln,
    /// and the output is `[proj, n_pos/4]`. Catches a wrong output shape, a
    /// mis-broadcast bias, or a 2x2-merge stride bug (which would still have to
    /// reproduce mm2_b exactly, so it pins the bias path).
    #[test]
    fn merger_cpu_shape_and_bias() {
        let n_embd = 8;
        let proj = 4;
        let merge_in = 4 * n_embd; // 32
        let n_pos = 12; // -> n_merged = 3
        let n_merged = n_pos / 4;
        let mh = MergerHostWeights {
            post_ln_w: (0..n_embd).map(|i| 1.0 + 0.1 * i as f32).collect(),
            post_ln_b: (0..n_embd).map(|i| -0.2 * i as f32).collect(),
            mm0_w: vec![0.0; merge_in * merge_in],
            mm0_b: (0..merge_in).map(|i| 0.3 * i as f32 - 1.0).collect(),
            mm2_w: vec![0.0; merge_in * proj],
            mm2_b: vec![10.0, 20.0, 30.0, 40.0],
        };
        // Arbitrary, non-degenerate input — output must NOT depend on it.
        let x: Vec<f32> = (0..n_embd * n_pos)
            .map(|i| (i as f32) * 0.07 - 1.3)
            .collect();
        let out = merger_cpu(&mh, &x, n_embd, 1e-6);
        assert_eq!(out.len(), proj * n_merged, "merger output shape");
        for b in 0..n_merged {
            for m in 0..proj {
                let v = out[m + proj * b];
                assert!(
                    (v - mh.mm2_b[m]).abs() < 1e-5,
                    "tok {b} ch {m}: got {v}, want mm2_b {}",
                    mh.mm2_b[m]
                );
            }
        }
    }

    /// Full block_cpu pipeline on a valid 2x2 patch grid (n_pos=4) with identity
    /// QKV, zero out-proj and zero FFN: attention and FFN contribute 0, so the
    /// block is the residual identity `out = x`. Exercises the whole pipeline
    /// (LN->QKV->rope->attention->residual->LN->FFN->residual) without panics.
    #[test]
    fn block_cpu_runs_2x2_grid() {
        let n_embd = 8;
        let n_head = 2;
        let n_ff = 16;
        let (gw, gh) = (2usize, 2usize);
        let n_pos = gw * gh;
        let mut qkv_w = vec![0.0f32; n_embd * 3 * n_embd];
        for sec in 0..3 {
            for c in 0..n_embd {
                qkv_w[c + n_embd * (sec * n_embd + c)] = 1.0;
            }
        }
        let bw = BlockHostWeights {
            ln1_w: vec![1.0; n_embd],
            ln1_b: vec![0.0; n_embd],
            qkv_w,
            qkv_b: vec![0.0; 3 * n_embd],
            out_w: vec![0.0; n_embd * n_embd],
            out_b: vec![0.0; n_embd],
            ln2_w: vec![1.0; n_embd],
            ln2_b: vec![0.0; n_embd],
            ffn_up_w: vec![0.0; n_embd * n_ff],
            ffn_up_b: vec![0.0; n_ff],
            ffn_down_w: vec![0.0; n_ff * n_embd],
            ffn_down_b: vec![0.0; n_embd],
        };
        let x: Vec<f32> = (0..n_embd * n_pos)
            .map(|i| (i as f32) * 0.05 - 1.0)
            .collect();
        let out = block_cpu(&bw, &x, n_embd, n_head, n_ff, 1e-6, gw, gh);
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!((o - xi).abs() < 1e-5, "2x2 residual mismatch: {o} vs {xi}");
        }
    }
}
