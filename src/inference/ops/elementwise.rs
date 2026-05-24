//! Element-wise ops: add, mul, silu, plus `get_rows` for embedding lookup.
//! All share the same simple dispatch pattern; only push-constant shape and
//! workgroup math differ.
//!
//! - `add` / `mul` use `BinaryParams` and `wg_denoms = {512, 1, 1}` so
//!   workgroups = `ceil(nelements / 512)` in the X direction.
//! - `silu` uses `GenericParams` (KX = nelements).
//! - `get_rows` uses `BinaryParams` and `wg_denoms = {512, 1, 1}` with
//!   workgroups = `(ceil(ne00/512), ne10, ne11*ne12)`.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::buffer::BufferRange;
use crate::inference::command::{record_compute_barrier, record_dispatch};
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::{CachedPipeline, PipelineKey};
use crate::inference::weights::TensorView;
use crate::shaders;

use super::binary_params_bytes;

const GENERIC_PARAMS_BYTES: u32 = 6 * 4;

pub fn record_add(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    record_binary_f32(
        ctx,
        "add_f32",
        shaders::ADD_F32_SPV.as_bytes(),
        a,
        b,
        dst,
    )
}

pub fn record_mul(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    record_binary_f32(
        ctx,
        "mul_f32",
        shaders::MUL_F32_SPV.as_bytes(),
        a,
        b,
        dst,
    )
}

pub fn record_sub(
    ctx: &mut DispatchContext,
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    record_binary_f32(
        ctx,
        "sub_f32",
        shaders::SUB_F32_SPV.as_bytes(),
        a,
        b,
        dst,
    )
}

fn record_binary_f32(
    ctx: &mut DispatchContext,
    name: &str,
    spirv: &[u8],
    a: TensorView,
    b: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    let key = PipelineKey::dense(name, 3, super::BINARY_PARAMS_BYTES, vec![0]);
    let pipeline = *ctx.pipelines.get(ctx.device, key, spirv)?;
    let push = binary_params_bytes(&a, &b, &dst, 0.0, 0.0, 0);

    // Shader: 256 threads × num_iter=2 with `idx += num_threads=256` per
    // iter. Each workgroup covers 512 indices, but adjacent workgroups
    // overlap at the 256-boundary, so the effective non-overlapping per-WG
    // stride is 256. Dispatch ceil(N/256) workgroups to guarantee full
    // coverage with some redundant writes (idempotent for add/mul).
    let nelements: u64 = dst.dims.iter().product();
    let workgroups = [(nelements as u32).div_ceil(256), 1, 1];

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[a.range(), b.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}

pub fn record_silu(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());
    // KY, param1..4 all zero — leave as default.

    let key = PipelineKey::dense("silu_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SILU_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}

/// `get_rows`: dst[col, row] = src[col, indices[row]]. src has shape
/// `[hidden, vocab]` (ggml: ne[0]=hidden, ne[1]=vocab), indices is `[L]` of
/// u32, dst is `[hidden, L]`.
pub fn record_get_rows(
    ctx: &mut DispatchContext,
    src: TensorView,
    indices: BufferRange,
    indices_len: u32,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    // Synthetic TensorView for the indices buffer with shape [L, 1, 1, 1].
    let indices_view = TensorView {
        buffer: indices.buffer,
        byte_offset: indices.offset,
        byte_size: indices.size,
        dims: [indices_len as u64, 1, 1, 1],
        byte_stride: [4, 4 * indices_len as u64, 4 * indices_len as u64, 4 * indices_len as u64],
        element_stride: [1, indices_len as u64, indices_len as u64, indices_len as u64],
        dtype: GgmlType::I32,
    };
    let push = binary_params_bytes(&src, &indices_view, &dst, 0.0, 0.0, 0);

    let (name, spirv) = match (src.dtype, dst.dtype) {
        (GgmlType::F32, GgmlType::F32) => ("get_rows_f32", shaders::GET_ROWS_F32_SPV.as_bytes()),
        (GgmlType::F16, GgmlType::F16) => ("get_rows_f16", shaders::GET_ROWS_F16_SPV.as_bytes()),
        (GgmlType::F16, GgmlType::F32) => {
            ("get_rows_f16_f32", shaders::GET_ROWS_F16_F32_SPV.as_bytes())
        }
        (GgmlType::BF16, GgmlType::F32) => {
            ("get_rows_bf16", shaders::GET_ROWS_BF16_SPV.as_bytes())
        }
        (GgmlType::Q6_K, GgmlType::F32) => {
            ("get_rows_q6_k", shaders::GET_ROWS_Q6_K_DEFAULT_SPV.as_bytes())
        }
        (GgmlType::I32, GgmlType::I32) => ("get_rows_i32", shaders::GET_ROWS_I32_SPV.as_bytes()),
        (s, d) => return Err(format!("get_rows: unsupported src/dst combo {s:?}/{d:?}").into()),
    };

    let key = PipelineKey::dense(name, 3, super::BINARY_PARAMS_BYTES, vec![0]);
    let pipeline = *ctx.pipelines.get(ctx.device, key, spirv)?;

    let ne00 = src.dims[0] as u32;
    let ne10 = indices_len;
    // get_rows_q6_k uses one workgroup per 256-element block (numthreads=64);
    // every other variant uses one workgroup per 512-element span (numthreads=512).
    let workgroups = if src.dtype == GgmlType::Q6_K {
        [ne00.div_ceil(256), ne10, 1]
    } else {
        [ne00.div_ceil(512), ne10, 1]
    };

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[src.range(), indices, dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}
