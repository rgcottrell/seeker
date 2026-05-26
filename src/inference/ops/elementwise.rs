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
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
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

    // Shader: 256 threads × num_iter=2, each workgroup owning a UNIQUE 512-wide
    // block (base = workgroup * 512). Non-overlapping, so every element is
    // written exactly once — required for in-place ops where dst aliases a
    // source (e.g. `residual += proj`); overlapping writes would double-apply
    // the op nondeterministically. Dispatch ceil(N/512) workgroups.
    let nelements: u64 = dst.dims.iter().product();
    let workgroups = [(nelements as u32).div_ceil(512), 1, 1];

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[a.range(), b.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
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
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

/// `softplus(x) = log(1 + exp(x))`. Same generic-unary dispatch shape
/// as `silu` / `sigmoid`. Used by the SSM block to map raw `α` projection
/// logits to a positive value before the `ssm_a` scaling.
pub fn record_softplus(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());
    let key = PipelineKey::dense("softplus_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SOFTPLUS_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
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

/// Per-row L2 normalize: `dst[r, c] = src[r, c] / max(sqrt(sum_c src^2), eps)`.
/// Dispatched with `src.dims[1..]` workgroups, each reducing over
/// `src.dims[0]` — so passing `src` shape `[head_dim, n_head, L, 1]`
/// gives per-(head, token) L2 normalization (which is what the SSM
/// block needs for Q and K before gated-delta-net).
pub fn record_l2_norm(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);
    let push = super::unary_params_bytes(&src, &dst, eps, 0.0);
    let key = PipelineKey::dense(
        "l2_norm_f32",
        2,
        super::UNARY_PARAMS_BYTES,
        Vec::new(),
    );
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::L2_NORM_F32_SPV.as_bytes())?;
    // `l2_norm.slang` decodes the row index as a packed
    // `z*262144 + y*512 + x` value — DIFFERENT convention from `rms_norm`,
    // which uses (x, y, z) → (row, channel, batch) directly. Pack total
    // rows (= dim1 × dim2 × dim3) along x, capped at 512 per stripe, with
    // overflow into y and (z × 262144).
    let total_rows = (src.dims[1].max(1) * src.dims[2].max(1) * src.dims[3].max(1)) as u32;
    let wg_z = total_rows / 262144;
    let wg_after_z = total_rows - wg_z * 262144;
    let wg_y = wg_after_z.div_ceil(512);
    let wg_x = (wg_after_z + wg_y.max(1) - 1).min(512);
    // Simpler when total_rows <= 512: just (total_rows, 1, 1).
    let workgroups = if total_rows <= 512 {
        [total_rows, 1, 1]
    } else if total_rows <= 512 * 512 {
        [512, total_rows.div_ceil(512), 1]
    } else {
        [512, 512, total_rows.div_ceil(262144)]
    };
    let _ = (wg_x, wg_y, wg_z);
    super::bind_and_dispatch(
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

/// Element-wise sigmoid (`σ(x) = 1 / (1 + exp(-x))`). Same dispatch shape
/// and push-constant layout as `record_silu` — `sigmoid.slang` is the
/// same generic-unary template, just a different SPV.
pub fn record_sigmoid(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    debug_assert_eq!(dst.dtype, GgmlType::F32);

    let nelements: u32 = src.dims.iter().product::<u64>() as u32;
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&nelements.to_ne_bytes());

    let key = PipelineKey::dense("sigmoid_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SIGMOID_F32_SPV.as_bytes())?;
    let workgroups = [nelements.div_ceil(512), 1, 1];
    super::bind_and_dispatch(
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
    let mut push = binary_params_bytes(&src, &indices_view, &dst, 0.0, 0.0, 0);

    // `elems_per_x` = elements covered per workgroup along the X (column)
    // axis, which sets the dispatch divisor:
    //   - plain get_rows (`get_rows.slang`): 512 threads × 1 elem  = 512
    //   - get_rows_quant (`get_rows_quant.slang`): 512 × 2 elems   = 1024
    //   - get_rows_q6_k (`get_rows_q6_k.slang`): 1 block / WG       = 256
    // All variants here declare only bindings [0,1,2] (slangc -O3 strips
    // the unused packed16 alias from the quant kernels' scalar path).
    let (name, spirv, elems_per_x) = match (src.dtype, dst.dtype) {
        (GgmlType::F32, GgmlType::F32) => ("get_rows_f32", shaders::GET_ROWS_F32_SPV.as_bytes(), 512),
        (GgmlType::F16, GgmlType::F16) => ("get_rows_f16", shaders::GET_ROWS_F16_SPV.as_bytes(), 512),
        (GgmlType::F16, GgmlType::F32) => {
            ("get_rows_f16_f32", shaders::GET_ROWS_F16_F32_SPV.as_bytes(), 512)
        }
        (GgmlType::BF16, GgmlType::F32) => {
            ("get_rows_bf16", shaders::GET_ROWS_BF16_SPV.as_bytes(), 512)
        }
        (GgmlType::I32, GgmlType::I32) => ("get_rows_i32", shaders::GET_ROWS_I32_SPV.as_bytes(), 512),
        (GgmlType::Q6_K, GgmlType::F32) => {
            ("get_rows_q6_k", shaders::GET_ROWS_Q6_K_DEFAULT_SPV.as_bytes(), 256)
        }
        (GgmlType::Q4_0, GgmlType::F32) => {
            ("get_rows_quant_q4_0", shaders::GET_ROWS_QUANT_Q4_0_SPV.as_bytes(), 1024)
        }
        (GgmlType::Q4_1, GgmlType::F32) => {
            ("get_rows_quant_q4_1", shaders::GET_ROWS_QUANT_Q4_1_SPV.as_bytes(), 1024)
        }
        (GgmlType::Q5_0, GgmlType::F32) => {
            ("get_rows_quant_q5_0", shaders::GET_ROWS_QUANT_Q5_0_SPV.as_bytes(), 1024)
        }
        (GgmlType::Q5_1, GgmlType::F32) => {
            ("get_rows_quant_q5_1", shaders::GET_ROWS_QUANT_Q5_1_SPV.as_bytes(), 1024)
        }
        (GgmlType::Q8_0, GgmlType::F32) => {
            ("get_rows_quant_q8_0", shaders::GET_ROWS_QUANT_Q8_0_SPV.as_bytes(), 1024)
        }
        (GgmlType::IQ4_NL, GgmlType::F32) => {
            ("get_rows_quant_iq4_nl", shaders::GET_ROWS_QUANT_IQ4_NL_SPV.as_bytes(), 1024)
        }
        (s, d) => return Err(format!("get_rows: unsupported src/dst combo {s:?}/{d:?}").into()),
    };

    let ne00 = src.dims[0] as u32;
    let ne10 = indices_len;

    // `get_rows_quant.slang` indexes `data_a[a_off + i00/QUANT_K]` in *block*
    // units, so it needs `nb01` = blocks-per-row. `binary_params_bytes`
    // fills `nb01` from `element_stride[1]`, which for a quant tensor is
    // `byte_stride[1] / rounded_elem_size` (e.g. Q8_0: (64·34)/2 = 1088) —
    // not the block count. Patch `nb01` (field index 6, byte offset 24) to
    // the true blocks-per-row. (`get_rows_q6_k` doesn't need this — it
    // derives the block index from `ne00/QUANT_K` directly.)
    if elems_per_x == 1024 {
        let (block_size, _) = src.dtype.block_layout();
        let blocks_per_row = ne00 / block_size as u32;
        push[24..28].copy_from_slice(&blocks_per_row.to_ne_bytes());
    }

    let key = PipelineKey::dense(name, 3, super::BINARY_PARAMS_BYTES, vec![0]);
    let pipeline = *ctx.pipelines.get(ctx.device, key, spirv)?;

    let workgroups = [ne00.div_ceil(elems_per_x), ne10, 1];

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[src.range(), indices, dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}
