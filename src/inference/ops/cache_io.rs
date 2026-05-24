//! Helpers for moving K / V into and out of the [`KvCache`].
//!
//! The cache stores per-token slots packed at offsets that for some
//! quant types aren't aligned to `minStorageBufferOffsetAlignment` (e.g.
//! Q4_0 with `head_dim=64, n_head_kv=3` → 108 bytes per token). Binding
//! a descriptor at a non-aligned offset isn't allowed, so we always go
//! through a contiguous scratch slot:
//!
//! Write path (per layer, per side):
//!   1. Cast F32 new-K (shape `[head_dim, n_head_kv, L]`) → scratch slot
//!      in cache dtype (shape `[head_dim, n_head_kv, L]`, contiguous).
//!   2. `vkCmdCopyBuffer` the scratch slot into the cache at byte offset
//!      `layer_offset + position * per_token_bytes`.
//!
//! Read path (per layer, per side):
//!   1. `vkCmdCopyBuffer` the cache prefix `[0, position+L)` of the
//!      layer into a scratch slot in cache dtype.
//!   2. Cast scratch → F32 scratch (shape `[head_dim, n_head_kv,
//!      position+L]`) for flash_attn to consume.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::buffer::BufferRange;
use crate::inference::command::{record_copy, record_global_barrier};
use crate::inference::context::DispatchContext;
use crate::inference::kv_cache::KvCache;
use crate::inference::weights::TensorView;

use super::cast::record_cast;

/// Stage and stash one layer's K (or V) for the current decode window.
///
/// `new_kv_f32` is the new K (or V) just produced by the matmul + RoPE,
/// shape `[head_dim, n_head_kv, L]` F32.
///
/// `cache_layer` is the per-layer `TensorView` from `cache.k_layers` /
/// `cache.v_layers` (full max-seq-len view).
///
/// `position` is the absolute starting position of the new tokens
/// (= `cache.position` before this call). Emits a trailing
/// transfer→all-stages barrier so the cache write is visible before the
/// next compute pass reads it.
pub fn record_write(
    ctx: &mut DispatchContext,
    new_kv_f32: TensorView,
    cache_layer: TensorView,
    position: u32,
) -> Result<(), Box<dyn Error>> {
    record_write_inner(ctx, new_kv_f32, cache_layer, position, /*fence=*/ true)
}

/// Same as [`record_write`] but skips the trailing global barrier — use
/// for the first of a paired K/V write where the second write's trailing
/// barrier covers both cache buffers ahead of the upcoming flash-attn.
pub fn record_write_nofence(
    ctx: &mut DispatchContext,
    new_kv_f32: TensorView,
    cache_layer: TensorView,
    position: u32,
) -> Result<(), Box<dyn Error>> {
    record_write_inner(ctx, new_kv_f32, cache_layer, position, /*fence=*/ false)
}

fn record_write_inner(
    ctx: &mut DispatchContext,
    new_kv_f32: TensorView,
    cache_layer: TensorView,
    position: u32,
    fence: bool,
) -> Result<(), Box<dyn Error>> {
    let head_dim = new_kv_f32.dims[0];
    let n_head_kv = new_kv_f32.dims[1];
    let l = new_kv_f32.dims[2];
    let dtype = cache_layer.dtype;

    // Scratch slot in cache dtype, contiguous shape [head_dim, n_head_kv, L].
    let scratch_cache = ctx.alloc_tensor([head_dim, n_head_kv, l, 1], dtype)?;
    record_cast(ctx, new_kv_f32, scratch_cache)?;
    // The cast's compute write must be visible before the upcoming TRANSFER read.
    record_global_barrier(ctx.device, ctx.cmd);

    // Copy into the cache at the right byte offset.
    let per_token_bytes = per_token_bytes(head_dim, n_head_kv, dtype);
    let dst_offset = cache_layer.byte_offset + position as u64 * per_token_bytes;
    let copy_size = (l as u64) * per_token_bytes;
    record_copy(
        ctx.device,
        ctx.cmd,
        BufferRange {
            buffer: scratch_cache.buffer,
            offset: scratch_cache.byte_offset,
            size: copy_size,
        },
        BufferRange {
            buffer: cache_layer.buffer,
            offset: dst_offset,
            size: copy_size,
        },
        copy_size,
    );
    if fence {
        // Cache write must complete before the upcoming cache read.
        record_global_barrier(ctx.device, ctx.cmd);
    }
    Ok(())
}

/// Materialize the live prefix `[0, total_len)` of one layer's K (or V)
/// from the cache into a fresh F32 scratch slot. Returns that slot as a
/// `TensorView` shaped `[head_dim, n_head_kv, total_len]` ready for
/// flash_attn's permuted reads.
pub fn record_read(
    ctx: &mut DispatchContext,
    cache_layer: TensorView,
    total_len: u32,
) -> Result<TensorView, Box<dyn Error>> {
    let head_dim = cache_layer.dims[0];
    let n_head_kv = cache_layer.dims[1];
    let total_u = total_len as u64;
    let dtype = cache_layer.dtype;
    let per_token_bytes = per_token_bytes(head_dim, n_head_kv, dtype);

    // Stage: contiguous scratch slot of the cache prefix in cache dtype.
    let scratch_cache = ctx.alloc_tensor([head_dim, n_head_kv, total_u, 1], dtype)?;
    let copy_size = total_u * per_token_bytes;
    record_copy(
        ctx.device,
        ctx.cmd,
        BufferRange {
            buffer: cache_layer.buffer,
            offset: cache_layer.byte_offset,
            size: copy_size,
        },
        BufferRange {
            buffer: scratch_cache.buffer,
            offset: scratch_cache.byte_offset,
            size: copy_size,
        },
        copy_size,
    );
    // TRANSFER write must complete before the upcoming cast (compute read).
    record_global_barrier(ctx.device, ctx.cmd);

    // Cast into F32 for flash_attn.
    let dst_f32 = ctx.alloc_tensor([head_dim, n_head_kv, total_u, 1], GgmlType::F32)?;
    record_cast(ctx, scratch_cache, dst_f32)?;
    Ok(dst_f32)
}

/// Bytes used by one full KV row (head_dim × n_head_kv elements) for
/// the given dtype. Matches `make_view` in `kv_cache.rs`.
pub fn per_token_bytes(head_dim: u64, n_head_kv: u64, dtype: GgmlType) -> u64 {
    let (block_size, type_size) = dtype.block_layout();
    let elements_per_token = head_dim * n_head_kv;
    if block_size > 1 {
        (elements_per_token / block_size as u64) * type_size as u64
    } else {
        elements_per_token * type_size as u64
    }
}

/// Bump the cache's `position` field. Used by the model after recording
/// the writes for L new tokens.
pub fn advance(cache: &mut KvCache, by: u32) {
    cache.position = cache.position.saturating_add(by);
}
