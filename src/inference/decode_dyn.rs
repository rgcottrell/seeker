//! Per-forward dynamic parameters.
//!
//! Holds the small set of scalars that vary between consecutive decode
//! tokens — KV-cache length, split-K count, RoPE write offset, sampler RNG
//! draw, penalty pair count. Lives in a single host-writable scratch slot
//! that shaders bind as a storage buffer instead of receiving these values
//! as push constants. Migrating them out of push lets the recorded decode
//! command buffer be replayed across tokens with only host-side scratch
//! updates between submits (see plan
//! `~/.claude/plans/this-project-was-originally-fluffy-sundae.md`).
//!
//! Values that don't change between consecutive decodes (head_dim, scale,
//! top_p, min_p, inv_temp, repeat_penalty, …) stay on the push path —
//! migrating them adds shader cost for no replay benefit.

use std::error::Error;

use super::buffer::BufferRange;
use super::context::DispatchContext;

/// Mirror of the shader-side struct (matches `decode_dyn.slang`):
/// 8 × 4 bytes laid out contiguously, std430-friendly.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DecodeDyn {
    /// KV cache read length for flash_attn (= position_offset + L).
    pub kv_len: u32,
    /// Number of split-K workgroups for the main flash_attn dispatch.
    pub k_num: u32,
    /// KV blocks per split (each WG sweeps this many `Bc`-sized chunks).
    pub blocks_per_split: u32,
    /// Element offset in the K-cache buffer where the new tokens land.
    pub rope_d_offset: u32,
    /// Uniform `[0, 1)` draw consumed by the stochastic categorical sampler.
    pub uniform_rng: f32,
    /// Count of valid `(token_id, count)` pairs in the penalty buffer.
    pub penalty_count: u32,
    /// Element offset in the V-cache buffer where the new tokens land
    /// (= `position_offset * head_dim_v * n_head_kv`). Read by the
    /// dyn-offset V-cache write shader so the cmdbuf binds the *full*
    /// cache layer instead of a position-baked slice.
    pub v_cache_d_offset: u32,
    /// KV-cache slab index for batched/continuous decode — the flash-attn
    /// kernel reads K/V from `slot * nb13` / `slot * nb23`, so a gathered batch
    /// can address arbitrary (non-contiguous) `BatchKvCache` slabs. `0` (slab 0)
    /// for single-sequence decode, byte-identical to the old `iq3`-strided read.
    pub slot: u32,
    /// Number of query rows (`L_s`) this sequence contributes this step, for the
    /// unified varlen batched flash (`VARLEN` spec constant). Decode = 1; a
    /// prefill chunk = its token count. Rows `>= n_query` are skipped. The
    /// per-row causal bound is `base + i_row + 1` with `base = kv_len - n_query`
    /// (the cached prefix length), so the shader masks causally in-place — no
    /// host mask. Unused when `VARLEN == 0`.
    pub n_query: u32,
    /// Flat query-token offset: index of this sequence's first query row in the
    /// packed `[N_total]` token dimension (`q_start[s] = sum L_i for i<s`).
    /// Decode = the sequence's batch index. Unused when `VARLEN == 0`.
    pub q_start: u32,
}

impl DecodeDyn {
    pub const SIZE: u64 = 40;
}

/// Allocate the DecodeDyn slot in the active scratch region. Pair with
/// [`write`] to populate fields via the host-mapped pointer.
pub fn alloc(ctx: &mut DispatchContext) -> Result<BufferRange, Box<dyn Error>> {
    ctx.alloc_scratch(DecodeDyn::SIZE)
}

/// Allocate a contiguous array of `n_seqs` DecodeDyn entries for batched
/// decode — one entry per sequence (batch element). The flash-attn kernel
/// indexes `data_dyn[iq3].kv_len` by batch element, so each sequence attends
/// to its own cache length; `n_seqs == 1` is byte-identical to [`alloc`].
pub fn alloc_array(ctx: &mut DispatchContext, n_seqs: u32) -> Result<BufferRange, Box<dyn Error>> {
    ctx.alloc_scratch(DecodeDyn::SIZE * (n_seqs.max(1) as u64))
}

/// Host-write a single field of entry `seq_idx` within a DecodeDyn array
/// allocated by [`alloc_array`]. Offsets past entry 0 by `seq_idx * SIZE`.
pub fn write_field_indexed<T: Copy>(
    ctx: &DispatchContext,
    range: BufferRange,
    seq_idx: u32,
    field_offset: usize,
    value: T,
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch not host-visible — DecodeDyn requires mapped memory")?;
    let entry_offset = range.offset + (seq_idx as u64) * DecodeDyn::SIZE;
    write_field(host_ptr, entry_offset, field_offset, value);
    Ok(())
}

/// Write a populated `DecodeDyn` into the scratch slot via the mapped
/// pointer. Cheaper than recording a transfer; the buffer becomes visible
/// to the next submit immediately (scratch is HOST_COHERENT).
pub fn write(
    ctx: &DispatchContext,
    range: BufferRange,
    values: &DecodeDyn,
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch not host-visible — DecodeDyn requires mapped memory")?;
    unsafe {
        let dst = host_ptr.add(range.offset as usize) as *mut DecodeDyn;
        std::ptr::write(dst, *values);
    }
    Ok(())
}

/// Host-write a single field of an already-allocated `DecodeDyn` slot.
/// Two-argument shape used during recording when the dispatch context
/// owns the scratch region. The bare-pointer overload below
/// ([`write_field`] free function) is used by the replay path which
/// has already resolved the mapped pointer.
pub fn write_field_ctx<T: Copy>(
    ctx: &DispatchContext,
    range: BufferRange,
    field_offset: usize,
    value: T,
) -> Result<(), Box<dyn Error>> {
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch not host-visible — DecodeDyn requires mapped memory")?;
    write_field(host_ptr, range.offset, field_offset, value);
    Ok(())
}

/// Host-write a single field of an already-allocated `DecodeDyn` slot.
/// Used by the replay path which has already resolved the mapped
/// pointer once at the top of the call.
pub fn write_field<T: Copy>(
    host_ptr: *mut u8,
    range_offset: u64,
    field_offset: usize,
    value: T,
) {
    debug_assert!(field_offset + std::mem::size_of::<T>() <= DecodeDyn::SIZE as usize);
    unsafe {
        let dst = host_ptr.add(range_offset as usize + field_offset) as *mut T;
        std::ptr::write(dst, value);
    }
}

/// Byte offsets of each field within `DecodeDyn`. Used by the replay
/// path to host-write individual fields between submits.
pub const OFFSET_KV_LEN: usize = 0;
pub const OFFSET_K_NUM: usize = 4;
pub const OFFSET_BLOCKS_PER_SPLIT: usize = 8;
pub const OFFSET_ROPE_D_OFFSET: usize = 12;
pub const OFFSET_UNIFORM_RNG: usize = 16;
pub const OFFSET_PENALTY_COUNT: usize = 20;
pub const OFFSET_V_CACHE_D_OFFSET: usize = 24;
pub const OFFSET_SLOT: usize = 28;
pub const OFFSET_N_QUERY: usize = 32;
pub const OFFSET_Q_START: usize = 36;

/// Snapshot of the scratch offsets and small constants captured during
/// the first decode recording. Lets the host re-populate the same slots
/// (token_buf, positions_buf, decode_dyn, penalty pairs) between
/// subsequent submits of the cached decode command buffer.
///
/// Created by `Engine::forward_sampled` with `decode_dyn_offset` set
/// and the rest of the fields `None`. The model fills `token_buf_offset`
/// and `positions_buf_offset` at the top of its `record_forward`; the
/// sampler fills `sampler_output_offset` (always) and `penalty_pairs`
/// (if the chain recorded penalties). After recording, the Engine
/// validates that every required field is populated.
#[derive(Debug, Clone, Default)]
pub struct ReplayPlan {
    pub decode_dyn_offset: u64,
    pub token_buf_offset: Option<u64>,
    pub positions_buf_offset: Option<u64>,
    pub sampler_output_offset: Option<u64>,
    /// `(offset, max_pairs)` for the penalty-pairs scratch slot. None
    /// if the sampler config has no penalties (apply_penalties wasn't
    /// recorded).
    pub penalty_pairs: Option<(u64, u32)>,
}

/// Per-model constants the host needs to drive the replay path.
/// Returned by `Model::replay_constants`.
#[derive(Debug, Clone, Copy)]
pub struct ModelReplayConstants {
    /// `head_dim_k * n_head_kv` — multiplied by `position_offset` to
    /// get the K-cache write offset (rope_d_offset).
    pub rope_d_offset_per_position: u32,
    /// `head_dim_v * n_head_kv` — multiplied by `position_offset` to
    /// get the V-cache write offset (v_cache_d_offset).
    pub v_cache_d_offset_per_position: u32,
    /// Number of position axes in the M-RoPE positions buffer
    /// (typically 4 for qwen35moe).
    pub mrope_axes: u32,
}
