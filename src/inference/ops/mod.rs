//! Per-op dispatch recorders. Each op fills its shader's push-constant
//! struct exactly per llama.cpp's `vk_op_*_push_constants` patterns, binds
//! the right slots, and records a single compute dispatch + barrier into
//! the active command buffer.

pub mod cache_io;
pub mod cast;
pub mod elementwise;
pub mod flash_attn;
pub mod matmul;
pub mod moe;
pub mod rms_norm;
pub mod rope;
pub mod rope_multi;
pub mod sampler;
pub mod ssm;

use std::error::Error;

use crate::inference::buffer::BufferRange;
use crate::inference::command::{record_dispatch, record_dispatch_indirect, record_dispatch_push};
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::CachedPipeline;
use crate::inference::weights::TensorView;

/// Single entry point for "bind buffers + push constants + dispatch".
/// Branches at runtime on `device.push_descriptor` (core in Vulkan 1.4):
/// when present, skips the descriptor-pool round-trip entirely via
/// `vkCmdPushDescriptorSet`; otherwise allocates a set from
/// `DescriptorAllocator`. Pipeline-layout flags are chosen matchingly in
/// `pipeline::compile_compute_pipeline`.
///
/// `binding_indices[i]` is the descriptor-set slot for `bindings[i]`;
/// use the *_indexed shape uniformly — passing
/// `&[0, 1, 2, …, bindings.len() - 1]` covers the dense case.
pub fn bind_and_dispatch(
    ctx: &mut DispatchContext,
    pipeline: &CachedPipeline,
    binding_indices: &[u32],
    bindings: &[BufferRange],
    push: &[u8],
    workgroups: [u32; 3],
) -> Result<(), Box<dyn Error>> {
    if ctx.device.push_descriptor {
        record_dispatch_push(
            ctx.device,
            ctx.cmd,
            pipeline,
            binding_indices,
            bindings,
            push,
            workgroups,
        );
    } else {
        let set = ctx.descriptors.allocate_and_write_indexed(
            ctx.device,
            pipeline.set_layout,
            binding_indices,
            bindings,
        )?;
        record_dispatch(ctx.device, ctx.cmd, pipeline, set, push, workgroups);
    }
    #[cfg(feature = "profile_gpu")]
    {
        ctx.n_dispatches += 1;
    }
    Ok(())
}

/// Indirect-dispatch variant of [`bind_and_dispatch`]. The workgroup
/// count is read from `indirect` (a `BufferRange` holding three packed
/// `u32`s) at submit time, so the dispatch grid can vary per token
/// without re-recording the cmdbuf. Push-descriptor only — the
/// descriptor-pool fallback path doesn't need this yet.
pub fn bind_and_dispatch_indirect(
    ctx: &mut DispatchContext,
    pipeline: &CachedPipeline,
    binding_indices: &[u32],
    bindings: &[BufferRange],
    push: &[u8],
    indirect: BufferRange,
) -> Result<(), Box<dyn Error>> {
    if !ctx.device.push_descriptor {
        return Err(
            "bind_and_dispatch_indirect requires push_descriptor; Vulkan 1.4 path only".into(),
        );
    }
    record_dispatch_indirect(
        ctx.device,
        ctx.cmd,
        pipeline,
        binding_indices,
        bindings,
        push,
        indirect,
    );
    #[cfg(feature = "profile_gpu")]
    {
        ctx.n_dispatches += 1;
    }
    Ok(())
}


/// Byte size of `UnaryParams` (in shaders/include/generic_unary_head.slang):
/// 32 × 4 bytes = 128 bytes.
pub const UNARY_PARAMS_BYTES: u32 = 32 * 4;

/// Byte size of `BinaryParams` (in shaders/include/generic_binary_head.slang):
/// 29 × 4 bytes (28 uints + 2 floats + 1 int = 29 × 4-byte slots) = 116 bytes.
pub const BINARY_PARAMS_BYTES: u32 = 29 * 4;

/// Pack a `BinaryParams` block in the byte order the shader expects. Matches
/// `vk_op_binary_push_constants` in ggml-vulkan.cpp:11322.
pub fn binary_params_bytes(
    src0: &TensorView,
    src1: &TensorView,
    dst: &TensorView,
    param1: f32,
    param2: f32,
    param3: i32,
) -> [u8; BINARY_PARAMS_BYTES as usize] {
    let mut out = [0u8; BINARY_PARAMS_BYTES as usize];
    let mut w = 0;
    let put_u = |out: &mut [u8], w: &mut usize, v: u32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };

    let total_elements: u64 = src0.dims.iter().product();
    put_u(&mut out, &mut w, total_elements as u32);
    // src0 dims + strides (elements)
    for d in src0.dims {
        put_u(&mut out, &mut w, d as u32);
    }
    for s in src0.element_stride {
        put_u(&mut out, &mut w, s as u32);
    }
    // src1 dims + strides
    for d in src1.dims {
        put_u(&mut out, &mut w, d as u32);
    }
    for s in src1.element_stride {
        put_u(&mut out, &mut w, s as u32);
    }
    // dst dims + strides
    for d in dst.dims {
        put_u(&mut out, &mut w, d as u32);
    }
    for s in dst.element_stride {
        put_u(&mut out, &mut w, s as u32);
    }
    // misalign_offsets
    put_u(&mut out, &mut w, 0);
    // param1, param2 (floats), param3 (int)
    out[w..w + 4].copy_from_slice(&param1.to_ne_bytes());
    w += 4;
    out[w..w + 4].copy_from_slice(&param2.to_ne_bytes());
    w += 4;
    out[w..w + 4].copy_from_slice(&param3.to_ne_bytes());
    out
}

/// Pack the 128-byte `UnaryParams` push-constant block. Matches
/// `vk_op_unary_push_constants` in ggml-vulkan.cpp:1167-1281. Two free
/// floats `param1, param2` are exposed for ops like `scale` (α, β).
pub fn unary_params_bytes(
    src: &TensorView,
    dst: &TensorView,
    param1: f32,
    param2: f32,
) -> [u8; UNARY_PARAMS_BYTES as usize] {
    let mut out = [0u8; UNARY_PARAMS_BYTES as usize];
    let mut w = 0usize;
    let put = |out: &mut [u8], w: &mut usize, v: u32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };

    let nelements: u64 = src.dims.iter().product();
    put(&mut out, &mut w, nelements as u32);
    for d in src.dims {
        put(&mut out, &mut w, d as u32);
    }
    for s in src.element_stride {
        put(&mut out, &mut w, s as u32);
    }
    for d in dst.dims {
        put(&mut out, &mut w, d as u32);
    }
    for s in dst.element_stride {
        put(&mut out, &mut w, s as u32);
    }
    put(&mut out, &mut w, 0); // misalign_offsets
    out[w..w + 4].copy_from_slice(&param1.to_ne_bytes());
    w += 4;
    out[w..w + 4].copy_from_slice(&param2.to_ne_bytes());
    w += 4;

    let (mp, l) = fastdiv_values((src.dims[2] * src.dims[1] * src.dims[0]) as u32);
    put(&mut out, &mut w, mp);
    put(&mut out, &mut w, l);
    let (mp, l) = fastdiv_values((src.dims[1] * src.dims[0]) as u32);
    put(&mut out, &mut w, mp);
    put(&mut out, &mut w, l);
    let (mp, l) = fastdiv_values(src.dims[0] as u32);
    put(&mut out, &mut w, mp);
    put(&mut out, &mut w, l);
    let (mp, l) = fastdiv_values((dst.dims[2] * dst.dims[1] * dst.dims[0]) as u32);
    put(&mut out, &mut w, mp);
    put(&mut out, &mut w, l);
    let (mp, l) = fastdiv_values((dst.dims[1] * dst.dims[0]) as u32);
    put(&mut out, &mut w, mp);
    put(&mut out, &mut w, l);
    let (mp, l) = fastdiv_values(dst.dims[0] as u32);
    put(&mut out, &mut w, mp);
    put(&mut out, &mut w, l);
    out
}

/// Compute the magic numbers (mp, L) used by the shader's `fastdiv`
/// helper. Matches `init_fastdiv_values` in ggml-vulkan.cpp:1257.
pub fn fastdiv_values(d: u32) -> (u32, u32) {
    if d == 0 {
        return (0, 0);
    }
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    let mp = (((1u128 << 32) * ((1u128 << l) - d as u128) / d as u128) + 1) as u32;
    (mp, l)
}
