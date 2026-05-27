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
    pub _pad0: u32,
    pub _pad1: u32,
}

impl DecodeDyn {
    pub const SIZE: u64 = 32;
}

/// Allocate the DecodeDyn slot in the active scratch region. Pair with
/// [`write`] to populate fields via the host-mapped pointer.
pub fn alloc(ctx: &mut DispatchContext) -> Result<BufferRange, Box<dyn Error>> {
    ctx.alloc_scratch(DecodeDyn::SIZE)
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

/// Host-write a single field of an already-allocated `DecodeDyn` slot
/// (cheaper than re-writing the whole struct when only one value
/// changed, e.g. the sampler updating `uniform_rng` after the model
/// finished setting the cache-related fields).
pub fn write_field<T: Copy>(
    ctx: &DispatchContext,
    range: BufferRange,
    field_offset: usize,
    value: T,
) -> Result<(), Box<dyn Error>> {
    debug_assert!(field_offset + std::mem::size_of::<T>() <= DecodeDyn::SIZE as usize);
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch not host-visible — DecodeDyn requires mapped memory")?;
    unsafe {
        let dst = host_ptr.add(range.offset as usize + field_offset) as *mut T;
        std::ptr::write(dst, value);
    }
    Ok(())
}

/// Byte offset of `uniform_rng` within `DecodeDyn`. Used for partial
/// writes from the sampler after the model code has already populated
/// the cache-related fields.
pub const OFFSET_UNIFORM_RNG: usize = 16;

/// Byte offset of `penalty_count`.
pub const OFFSET_PENALTY_COUNT: usize = 20;
