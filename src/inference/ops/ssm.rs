//! State-space-model (Mamba-2-style) op dispatchers used by the Qwen35MoE
//! SSM blocks. Two shaders:
//!
//! * `ssm_conv.slang` — per-channel 1D convolution over the (post-padding)
//!   input. Mirrors llama.cpp's `ggml_ssm_conv`.
//! * `gated_delta_net.slang` — gated delta-net attention (the "linear
//!   attention" cousin of Mamba's selective scan). Sweeps tokens in order
//!   inside the workgroup, maintains a per-(head, batch) state matrix
//!   `S_V × S_V`, writes both the attention output AND the new state.
//!
//! These ops were ported into seeker's shader tree alongside the rest of
//! the llama.cpp port but never had Rust dispatchers wired — this file
//! provides them.

use std::error::Error;

use crate::inference::buffer::BufferRange;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

// ─── ssm_conv ───────────────────────────────────────────────────────────

const SSM_CONV_PUSH_BYTES: u32 = 11 * 4;

/// Per-channel 1D convolution. `src` is the pre-padded input with shape
/// `[n_padded_tokens, n_channels, n_seqs]` where
/// `n_padded_tokens = kernel_size - 1 + n_tokens`. `kernel` is the
/// learned weight `[kernel_size, n_channels]`. `dst` is the output
/// `[n_tokens, n_channels, n_seqs]`.
///
/// The caller is responsible for materializing the `kernel_size - 1`-
/// element conv state prefix in `src` before this dispatch (zeros on
/// the first forward; the trailing window of the previous forward in
/// streaming decode).
pub fn record_ssm_conv(
    ctx: &mut DispatchContext,
    src: TensorView,
    kernel: TensorView,
    dst: TensorView,
    n_channels: u32,
    n_padded_tokens: u32,
    n_tokens: u32,
    n_seqs: u32,
    kernel_size: u32,
) -> Result<(), Box<dyn Error>> {
    let mut push = [0u8; SSM_CONV_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }

    // src strides (in BYTES — shader divides by 4 internally):
    //   nb01 = bytes per channel (= n_padded_tokens * 4)
    //   nb02 = bytes per batch  (= n_channels * n_padded_tokens * 4)
    let src_nb01 = n_padded_tokens * 4;
    let src_nb02 = n_channels * n_padded_tokens * 4;
    // kernel:
    //   nb11 = bytes per channel (= kernel_size * 4)
    let kernel_nb11 = kernel_size * 4;
    // dst:
    //   nb0 = bytes per channel (innermost) = 4
    //   nb1 = bytes per token  = n_channels * 4
    //   nb2 = bytes per batch  = n_tokens * n_channels * 4
    let dst_nb0 = 4u32;
    let dst_nb1 = n_channels * 4;
    let dst_nb2 = n_tokens * n_channels * 4;

    put_u(&mut push, &mut w, src_nb01);
    put_u(&mut push, &mut w, src_nb02);
    put_u(&mut push, &mut w, kernel_nb11);
    put_u(&mut push, &mut w, dst_nb0);
    put_u(&mut push, &mut w, dst_nb1);
    put_u(&mut push, &mut w, dst_nb2);
    put_u(&mut push, &mut w, kernel_size);
    put_u(&mut push, &mut w, n_padded_tokens);
    put_u(&mut push, &mut w, n_channels);
    put_u(&mut push, &mut w, n_tokens);
    put_u(&mut push, &mut w, n_seqs);

    let key = PipelineKey::dense(
        "ssm_conv_f32",
        3,
        SSM_CONV_PUSH_BYTES,
        Vec::new(),
    );
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SSM_CONV_F32_SPV.as_bytes())?;
    let workgroups = [n_channels.div_ceil(32), n_tokens.div_ceil(16), n_seqs];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2],
        &[src.range(), kernel.range(), dst.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst.range());
    Ok(())
}

// ─── gated_delta_net ────────────────────────────────────────────────────

const GDN_PUSH_BYTES: u32 = 18 * 4;

/// Layout strides for a `[head_dim, n_heads, n_tokens, n_seqs]` tensor.
/// All strides in `element` units (the shader treats `data_q[off]` as a
/// flat F32 array). For contiguous F32 storage `[S, H, T, B]`:
///   `s1 = S`, `s2 = S*H`, `s3 = S*H*T`.
#[derive(Clone, Copy)]
pub struct GdnStrides {
    pub s1: u32, // head stride
    pub s2: u32, // token stride
    pub s3: u32, // seq stride
}

/// Gated delta-net dispatch. Output `dst` doubles as the state-out
/// buffer when `K == 1`: the first
/// `n_tokens * n_heads * S_V * n_seqs` floats hold the per-token
/// attention output, then starting at byte offset `s_off * 4` the
/// `n_heads * S_V * S_V * n_seqs` floats hold the updated state.
///
/// `state_in` is the previous-call state of the same shape.
pub fn record_gated_delta_net(
    ctx: &mut DispatchContext,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    g: TensorView,
    beta: TensorView,
    state_in: BufferRange,
    dst: BufferRange,
    // Number of value heads (= dispatched workgroups in X = `H` in shader).
    head_count_v: u32,
    // Number of key heads (= `neq1` in shader, used to wrap Q/K head index
    // when `head_count_v != head_count_k` for the SSM GQA-like repeat).
    head_count_k: u32,
    n_tokens: u32,
    n_seqs: u32,
    s_off_elem: u32, // offset (in F32 elements) within dst where state-out starts
    scale: f32,
    q_strides: GdnStrides,
    v_strides: GdnStrides,
    b_strides: GdnStrides,
    state_v: u32, // = S_V = head_v_dim
) -> Result<(), Box<dyn Error>> {
    let mut push = [0u8; GDN_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, head_count_v);   // H
    put_u(&mut push, &mut w, n_tokens);
    put_u(&mut push, &mut w, n_seqs);
    put_u(&mut push, &mut w, s_off_elem);     // s_off (in element units)
    put_u(&mut push, &mut w, q_strides.s1);   // sq1
    put_u(&mut push, &mut w, q_strides.s2);   // sq2
    put_u(&mut push, &mut w, q_strides.s3);   // sq3
    put_u(&mut push, &mut w, v_strides.s1);
    put_u(&mut push, &mut w, v_strides.s2);
    put_u(&mut push, &mut w, v_strides.s3);
    put_u(&mut push, &mut w, b_strides.s1);
    put_u(&mut push, &mut w, b_strides.s2);
    put_u(&mut push, &mut w, b_strides.s3);
    put_u(&mut push, &mut w, head_count_k);   // neq1: wrap Q/K head index for GQA-like repeat
    put_u(&mut push, &mut w, 1);              // rq3 = 1 for our text-only single-batch path
    put_f(&mut push, &mut w, scale);
    put_u(&mut push, &mut w, 1);              // K = 1 — single state snapshot
    put_u(&mut push, &mut w, 0);              // padding to round up to 18*4

    // Spec constants: S_V, KDA=0 (per-token scalar gate, not per-element),
    // SUBGROUP_SIZE=32, LANES_PER_COLUMN=32.
    let spec_constants = vec![state_v, 0, 32, 32];

    let key = PipelineKey {
        name: format!("gated_delta_net_f32_shmem_sv{state_v}"),
        binding_indices: vec![0, 1, 2, 3, 4, 5, 6],
        push_size: GDN_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(
        ctx.device,
        key,
        shaders::GATED_DELTA_NET_F32_SHMEM_SPV.as_bytes(),
    )?;

    let workgroups = [head_count_v, n_seqs, state_v];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3, 4, 5, 6],
        &[
            q.range(),
            k.range(),
            v.range(),
            g.range(),
            beta.range(),
            state_in,
            dst,
        ],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, dst);
    Ok(())
}
