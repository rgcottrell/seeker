//! Tensor casts between any pair of dtypes supported by the KV cache:
//! `{F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1}`.
//!
//! Dispatches the right shader based on (src.dtype, dst.dtype):
//! - `F32 ↔ F32`, `F16 ↔ F16`, `F32 ↔ F16` → `copy.slang` variants
//! - `F32 ↔ BF16`                          → `copy.slang` (bf16 variants)
//! - `F32 → quant`                         → `copy_to_quant.slang`
//! - `quant → F32`                         → `copy_from_quant.slang`
//!
//! All shaders share the same `UnaryParams` layout from
//! `shaders/include/generic_unary_head.slang` — 128 bytes, includes fastdiv
//! tables computed via the same magic-number algorithm llama.cpp uses.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

use super::{unary_params_bytes, UNARY_PARAMS_BYTES};

pub fn record_cast(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    let pick = pick_shader(src.dtype, dst.dtype)?;
    let push = unary_params_bytes(&src, &dst, 0.0, 0.0);

    // Quant shaders need an aliased data_a_packed16 view at an extra
    // binding slot — 2 for copy_from_quant, 3 for copy_to_quant.
    let (binding_indices, bindings): (Vec<u32>, Vec<crate::inference::buffer::BufferRange>) =
        match pick.kind {
            ShaderKind::Plain => (vec![0, 1], vec![src.range(), dst.range()]),
            ShaderKind::CopyFromQuant => {
                // src is the quant tensor; bind it again as packed16 at slot 2.
                (vec![0, 1, 2], vec![src.range(), dst.range(), src.range()])
            }
            ShaderKind::CopyToQuant => {
                // dst is the quant tensor; bind it again as packed16 at slot 3.
                (vec![0, 1, 3], vec![src.range(), dst.range(), dst.range()])
            }
        };

    let key = PipelineKey {
        name: pick.name.to_string(),
        binding_indices: binding_indices.clone(),
        push_size: UNARY_PARAMS_BYTES,
        spec_constants: Vec::new(),
        required_subgroup_size: None,
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, pick.spirv)?;

    // Workgroup count: each workgroup covers `elements_per_workgroup`
    // elements of work. For plain casts (copy variants) that's
    // `threads_per_workgroup * 1` element each. For copy_to_quant, 32
    // threads × QUANT_K=32 elements each = 1024. For copy_from_quant,
    // one workgroup processes one block = QUANT_K elements (LSX=1 for
    // non-IQ; LSX=16 with shmem init for IQ4_NL, but still one block).
    let nelements: u64 = src.dims.iter().product();
    let per_wg = pick.elements_per_workgroup;
    let workgroups = [
        ((nelements + per_wg - 1) / per_wg) as u32,
        1,
        1,
    ];

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &binding_indices,
        &bindings,
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

struct ShaderPick {
    name: &'static str,
    spirv: &'static [u8],
    elements_per_workgroup: u64,
    kind: ShaderKind,
}

enum ShaderKind {
    Plain,
    CopyToQuant,
    CopyFromQuant,
}

fn pick_shader(src: GgmlType, dst: GgmlType) -> Result<ShaderPick, Box<dyn Error>> {
    use GgmlType::*;
    fn plain(name: &'static str, spirv: &'static [u8]) -> ShaderPick {
        ShaderPick { name, spirv, elements_per_workgroup: 512, kind: ShaderKind::Plain }
    }
    fn to_quant(name: &'static str, spirv: &'static [u8]) -> ShaderPick {
        // copy_to_quant: 32 threads × QUANT_K=32 = 1024 elements / WG.
        ShaderPick { name, spirv, elements_per_workgroup: 1024, kind: ShaderKind::CopyToQuant }
    }
    fn from_quant(name: &'static str, spirv: &'static [u8]) -> ShaderPick {
        // copy_from_quant: one workgroup processes one block = QUANT_K=32 elements.
        ShaderPick { name, spirv, elements_per_workgroup: 32, kind: ShaderKind::CopyFromQuant }
    }
    Ok(match (src, dst) {
        (F32, F32) => plain("copy_f32", shaders::COPY_F32_SPV.as_bytes()),
        (F16, F16) => plain("copy_f16", shaders::COPY_F16_SPV.as_bytes()),
        (F32, F16) => plain("copy_f32_to_f16", shaders::COPY_F32_TO_F16_SPV.as_bytes()),
        (F16, F32) => plain("copy_f16_to_f32", shaders::COPY_F16_TO_F32_SPV.as_bytes()),
        (F32, BF16) => plain("copy_f32_to_bf16", shaders::COPY_F32_TO_BF16_SPV.as_bytes()),
        (BF16, F32) => plain("copy_bf16_to_f32", shaders::COPY_BF16_TO_F32_SPV.as_bytes()),
        (F32, I32) => plain("copy_f32_to_i32", shaders::COPY_F32_TO_I32_SPV.as_bytes()),

        (F32, Q4_0) => to_quant("copy_to_quant_q4_0", shaders::COPY_TO_QUANT_Q4_0_SPV.as_bytes()),
        (F32, Q4_1) => to_quant("copy_to_quant_q4_1", shaders::COPY_TO_QUANT_Q4_1_SPV.as_bytes()),
        (F32, Q5_0) => to_quant("copy_to_quant_q5_0", shaders::COPY_TO_QUANT_Q5_0_SPV.as_bytes()),
        (F32, Q5_1) => to_quant("copy_to_quant_q5_1", shaders::COPY_TO_QUANT_Q5_1_SPV.as_bytes()),
        (F32, Q8_0) => to_quant("copy_to_quant_q8_0", shaders::COPY_TO_QUANT_Q8_0_SPV.as_bytes()),
        (F32, IQ4_NL) => to_quant("copy_to_quant_iq4_nl", shaders::COPY_TO_QUANT_IQ4_NL_SPV.as_bytes()),

        (Q4_0, F32) => from_quant("copy_from_quant_q4_0", shaders::COPY_FROM_QUANT_Q4_0_SPV.as_bytes()),
        (Q4_1, F32) => from_quant("copy_from_quant_q4_1", shaders::COPY_FROM_QUANT_Q4_1_SPV.as_bytes()),
        (Q5_0, F32) => from_quant("copy_from_quant_q5_0", shaders::COPY_FROM_QUANT_Q5_0_SPV.as_bytes()),
        (Q5_1, F32) => from_quant("copy_from_quant_q5_1", shaders::COPY_FROM_QUANT_Q5_1_SPV.as_bytes()),
        (Q8_0, F32) => from_quant("copy_from_quant_q8_0", shaders::COPY_FROM_QUANT_Q8_0_SPV.as_bytes()),
        (IQ4_NL, F32) => from_quant("copy_from_quant_iq4_nl", shaders::COPY_FROM_QUANT_IQ4_NL_SPV.as_bytes()),

        _ => return Err(format!("cast {src:?} → {dst:?} not supported").into()),
    })
}

