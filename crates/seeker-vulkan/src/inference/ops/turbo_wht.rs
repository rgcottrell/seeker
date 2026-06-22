//! TurboQuant Walsh-Hadamard rotation dispatch (`turbo_wht.slang`).
//!
//! Rotates an F32 tensor in 128-element groups. TurboQuant stores K and V
//! WHT-rotated; for attention to stay correct the query is **forward**-rotated
//! before QK^T and the attention output is **inverse**-rotated afterward. The
//! transform is orthonormal (`<WHT(Q),WHT(K)> = <Q,K>`) and self-inverse up to
//! the per-direction 1/sqrt(128) scaling. `head_dim` must be a multiple of 128
//! so each head is a whole number of 128-element rotation groups (head_dim=256
//! → two groups per head; the dot product decomposes over the 128-chunks).

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

/// WHT direction. Discriminants match the shader's `direction` push field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WhtDir {
    /// pre-scale S1, post-scale S2 — rotate into the quantization basis.
    Forward = 0,
    /// pre-scale S2, post-scale S1 — undo the rotation.
    Inverse = 1,
}

/// `ne` (u32) + `direction` (u32).
const WHT_PUSH_BYTES: u32 = 2 * 4;

/// Record a WHT over `src` (F32, contiguous, element count a multiple of 128,
/// `head_dim` innermost) into `dst` (F32, same shape, a distinct buffer — must
/// not alias `src`). Emits a trailing compute barrier on `dst`.
pub fn record(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    direction: WhtDir,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32, "turbo_wht src must be F32");
    debug_assert_eq!(dst.dtype, GgmlType::F32, "turbo_wht dst must be F32");
    let ne: u64 = src.dims.iter().product();
    debug_assert_eq!(
        ne % 128,
        0,
        "turbo_wht requires the element count ({ne}) to be a multiple of 128 (head_dim % 128 == 0)"
    );

    let mut push = Vec::with_capacity(WHT_PUSH_BYTES as usize);
    push.extend_from_slice(&(ne as u32).to_ne_bytes());
    push.extend_from_slice(&(direction as u32).to_ne_bytes());

    let key = PipelineKey {
        name: "turbo_wht".to_string(),
        binding_indices: vec![0, 1],
        push_size: WHT_PUSH_BYTES,
        spec_constants: Vec::new(),
        required_subgroup_size: None,
    };
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::TURBO_WHT_SPV.as_bytes())?;

    // One workgroup (128 threads) per 128-element group. Pack the group count
    // into 3D the same way the shader decodes it (`z*262144 + y*512 + x`) so
    // large counts fit Vulkan's per-dimension workgroup-count limits; the
    // shader's `base + tid >= ne` guard discards any over-dispatched tail.
    let groups = (ne / 128) as u32;
    let gx = groups.clamp(1, 512);
    let gy = groups.div_ceil(512).clamp(1, 512);
    let gz = groups.div_ceil(512 * 512).max(1);

    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        [gx, gy, gz],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}
