//! GPU-resident sampler op recorders. Composed by
//! [`crate::inference::sample::Sampler::record_chain`].
//!
//! Each `record_*` function appends one or more compute dispatches to the
//! active command buffer and returns the working tensor (or, for the
//! terminal step, a 4-byte `BufferRange`) for the next stage to consume.
//! The pattern mirrors `cast.rs` / `elementwise.rs`: PipelineKey →
//! pipelines.get → descriptors.allocate_and_write → push constants →
//! record_dispatch → record_compute_barrier.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::buffer::BufferRange;
use crate::inference::command::{record_compute_barrier, record_dispatch};
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::{CachedPipeline, PipelineKey};
use crate::inference::sample::SamplerConfig;
use crate::inference::weights::TensorView;
use crate::shaders;

use super::{fastdiv_values, unary_params_bytes, UNARY_PARAMS_BYTES};

const GENERIC_PARAMS_BYTES: u32 = 6 * 4;
const SUM_ROWS_PARAMS_BYTES: u32 = 15 * 4;
const SOFT_MAX_PARAMS_BYTES: u32 = 17 * 4;
const TOPK_PARAMS_BYTES: u32 = 7 * 4;
const PENALTY_PARAMS_BYTES: u32 = 4 * 4;
const TOPK_BLOCK_SIZE: u32 = 1024;

/// Top-level dispatcher for the sampler chain. Returns a 4-byte
/// `BufferRange` holding the sampled token id as a `u32` (or `i32`, same
/// bit pattern for valid token indices). The engine reads exactly those
/// 4 bytes after the command buffer completes.
///
/// `penalty_pairs` is a snapshot of the recent-token `(token_id, count)`
/// list assembled on host by the [`Sampler`](crate::inference::sample::Sampler).
/// `uniform` is the per-step random draw for the categorical step;
/// ignored when the chain takes the greedy short-circuit.
pub fn record_chain(
    ctx: &mut DispatchContext,
    config: &SamplerConfig,
    logits: TensorView,
    penalty_pairs: &[(u32, u32)],
    uniform: f32,
) -> Result<BufferRange, Box<dyn Error>> {
    // Penalties run first because the repetition penalty's multiply/divide
    // is sign-conditional on the *raw* logit.
    if config.any_penalty() && !penalty_pairs.is_empty() {
        record_apply_penalties(
            ctx,
            logits,
            penalty_pairs,
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty,
        )?;
    }

    if config.is_greedy() {
        return record_greedy(ctx, logits);
    }

    // top_k → gather logits at those indices → (later: top_p / min_p) →
    // temp → categorical. Order matches llama.cpp's common-sampler
    // convention. Temperature is *last* because top_p does its own
    // softmax internally and applying 1/T ahead of that would shift its
    // cumulative cutoff.
    let (kept_logits, candidates) = if config.top_k > 0 && (config.top_k as u64) < logits.dims[0] {
        let (idx, gathered) = record_top_k(ctx, logits, config.top_k)?;
        (gathered, Some(idx))
    } else {
        (logits, None)
    };

    let kept_logits = if config.top_p < 1.0 && config.top_p > 0.0 {
        record_top_p(ctx, kept_logits, config.top_p)?
    } else {
        kept_logits
    };

    let kept_logits = if config.min_p > 0.0 {
        record_min_p(ctx, kept_logits, config.min_p)?
    } else {
        kept_logits
    };

    let temped = if (config.temperature - 1.0).abs() > 1e-9 {
        let kept_len = kept_logits.dims[0];
        let scaled = ctx.alloc_tensor([kept_len, 1, 1, 1], GgmlType::F32)?;
        record_scale(ctx, kept_logits, scaled, 1.0 / config.temperature, 0.0)?;
        scaled
    } else {
        kept_logits
    };

    record_categorical(ctx, temped, candidates, uniform)
}

/// Wrap `argmax.slang` for a single-row argmax over the vocab. Output is a
/// 4-byte slot holding the picked token id as `i32` (reinterpret as `u32`
/// on the host — valid token ids are non-negative).
pub fn record_greedy(
    ctx: &mut DispatchContext,
    logits: TensorView,
) -> Result<BufferRange, Box<dyn Error>> {
    debug_assert_eq!(logits.dtype, GgmlType::F32);

    let kx: u32 = logits.dims[0] as u32;
    let ky: u32 = 1;

    let out = ctx.alloc_scratch(4)?;

    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&kx.to_ne_bytes());
    push[4..8].copy_from_slice(&ky.to_ne_bytes());

    let key = PipelineKey::dense("argmax_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::ARGMAX_F32_SPV.as_bytes())?;
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[logits.range(), out],
        &push,
        [1, 1, 1],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(out)
}

/// `record_categorical`: inverse-CDF categorical sampling on a 1-D logits
/// tensor of length K. Mirrors `llama.cpp`'s `llama_sampler_dist_backend_apply`.
///
/// Ops sequence:
///   probs    = soft_max(logits)
///   cdf      = cumsum(probs)
///   diff     = sub(cdf, broadcast(uniform))
///   mask     = step(diff)                   // 1 once cdf >= rnd
///   count    = sum_rows(mask)               // # of i where cdf[i] >= rnd
///   idx_f    = scale(count, α=-1, β=K)      // K - count
///   idx      = cast(idx_f, F32→I32)
///   (if `candidates` set) token = get_rows(candidates, idx)
///
/// Returns the 4-byte slot holding the final token id.
fn record_categorical(
    ctx: &mut DispatchContext,
    logits: TensorView,
    candidates: Option<TensorView>,
    uniform: f32,
) -> Result<BufferRange, Box<dyn Error>> {
    debug_assert_eq!(logits.dtype, GgmlType::F32);
    let k = logits.dims[0];

    // Step 1: softmax.
    let probs = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_soft_max(ctx, logits, probs)?;

    // Step 2: cumulative sum.
    let cdf = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_cumsum(ctx, probs, cdf)?;

    // Step 3: upload uniform value into a 1-element scratch tensor.
    let uniform_t = ctx.alloc_tensor([1, 1, 1, 1], GgmlType::F32)?;
    // SAFETY: scratch is host-visible (Engine guarantees this on init).
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch not host-visible — categorical needs uniform upload")?;
    unsafe {
        let dst = host_ptr.add(uniform_t.byte_offset as usize) as *mut f32;
        std::ptr::write(dst, uniform);
    }

    // Step 4: diff = cdf - uniform (broadcast scalar).
    let diff = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    super::elementwise::record_sub(ctx, cdf, uniform_t, diff)?;

    // Step 5: mask = step(diff). One value per element.
    let mask = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_step(ctx, diff, mask)?;

    // Step 6: count = sum_rows(mask).
    let count = ctx.alloc_tensor([1, 1, 1, 1], GgmlType::F32)?;
    record_sum_rows(ctx, mask, count)?;

    // Step 7: idx_f = -count + K. scale.slang does α·x + β.
    // (`count` ∈ [1, K] when rnd ∈ [0, sum_cum), so idx_f ∈ [0, K-1].
    // If softmax/cumsum lose precision and the final cdf entry ends up
    // < rnd, count would be 0 → idx_f = K → OOB get_rows. We clamp to
    // [0, K-1] defensively, mirroring llama.cpp's fallback at
    // `llama_sampler_dist_apply`.)
    let idx_f_raw = ctx.alloc_tensor([1, 1, 1, 1], GgmlType::F32)?;
    record_scale(ctx, count, idx_f_raw, -1.0, k as f32)?;
    let idx_f = ctx.alloc_tensor([1, 1, 1, 1], GgmlType::F32)?;
    record_clamp(ctx, idx_f_raw, idx_f, 0.0, (k - 1) as f32)?;

    // Step 8: cast F32 → I32.
    let idx_i32 = ctx.alloc_tensor([1, 1, 1, 1], GgmlType::I32)?;
    super::cast::record_cast(ctx, idx_f, idx_i32)?;

    if let Some(cand) = candidates {
        // Map the position back to a vocab id via get_rows. `cand` holds
        // the K candidate token ids (i32) produced by top_k.
        let token = ctx.alloc_tensor([1, 1, 1, 1], GgmlType::I32)?;
        record_get_rows_i32(ctx, cand, idx_i32, token)?;
        return Ok(token.range());
    }

    Ok(idx_i32.range())
}

/// Apply repeat / frequency / presence penalties to the raw logits in place.
/// `pairs` is the deduplicated `(token_id, count)` list from the sampler's
/// recent-token ring; the host uploads it into a scratch SSBO.
///
/// The shader does one thread per pair, so we dispatch
/// `ceil(n_pairs / 256)` workgroups.
fn record_apply_penalties(
    ctx: &mut DispatchContext,
    logits: TensorView,
    pairs: &[(u32, u32)],
    repeat_p: f32,
    freq_p: f32,
    presence_p: f32,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(logits.dtype, GgmlType::F32);
    let n_pairs = pairs.len() as u32;
    if n_pairs == 0 {
        return Ok(());
    }

    // Pack the (token_id, count) pairs into a scratch SSBO. Each pair is
    // two u32s back-to-back, matching the shader's `StructuredBuffer<int2>`.
    let pairs_t = ctx.alloc_tensor([2 * n_pairs as u64, 1, 1, 1], GgmlType::I32)?;
    let host_ptr = ctx
        .scratch
        .host_ptr
        .ok_or("scratch not host-visible — penalties need pair upload")?;
    // SAFETY: `pairs_t` has byte_size = 2 * n_pairs * 4 = 8 * n_pairs bytes,
    // which is exactly the size of the flattened pair list.
    unsafe {
        let dst = host_ptr.add(pairs_t.byte_offset as usize) as *mut u32;
        for (i, &(tid, count)) in pairs.iter().enumerate() {
            std::ptr::write(dst.add(2 * i), tid);
            std::ptr::write(dst.add(2 * i + 1), count);
        }
    }

    let mut push = [0u8; PENALTY_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&n_pairs.to_ne_bytes());
    push[4..8].copy_from_slice(&repeat_p.to_ne_bytes());
    push[8..12].copy_from_slice(&freq_p.to_ne_bytes());
    push[12..16].copy_from_slice(&presence_p.to_ne_bytes());

    let key = PipelineKey::dense(
        "apply_penalties_f32",
        2,
        PENALTY_PARAMS_BYTES,
        Vec::new(),
    );
    let pipeline = *ctx.pipelines.get(
        ctx.device,
        key,
        shaders::APPLY_PENALTIES_F32_SPV.as_bytes(),
    )?;
    let workgroups = [n_pairs.div_ceil(256), 1, 1];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[logits.range(), pairs_t.range()],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}

/// Bitonic top-K via `topk_argsort.slang`. Returns:
///   - `indices`: `[K, 1, 1, 1]` I32 — the picked token ids, sorted DESC by logit.
///   - `gathered`: `[K, 1, 1, 1]` F32 — the matching logit values.
fn record_top_k(
    ctx: &mut DispatchContext,
    logits: TensorView,
    k: u32,
) -> Result<(TensorView, TensorView), Box<dyn Error>> {
    debug_assert_eq!(logits.dtype, GgmlType::F32);
    let n_vocab = logits.dims[0] as u32;

    // Output of the *first* pass: K candidates per workgroup, as int2 pairs.
    let mut current_count = n_vocab;
    let mut is_first = true;
    // Per-pass intermediates (`int2` pairs, packed as 2×I32). We always
    // bind a buffer for each of (data_s, data_t); for the first/last pass
    // the unused side just gets the existing buffer.
    let mut prev_intermediate: TensorView = logits; // placeholder, unused while is_first
    let final_indices = ctx.alloc_tensor([k as u64, 1, 1, 1], GgmlType::I32)?;

    loop {
        let num_wg = current_count.div_ceil(TOPK_BLOCK_SIZE);
        let is_last = num_wg == 1;
        let next_count = if is_last { k } else { num_wg * k };

        // Allocate the int2 output for non-last passes. For the last pass
        // we write directly into `final_indices`; data_t still needs a
        // valid binding (we'll bind whatever's at hand).
        let next_intermediate = if is_last {
            // Reuse final_indices as a dummy data_t slot — the shader writes
            // data_d in the last pass and ignores data_t.
            final_indices
        } else {
            // 2 × I32 per slot (the shader stores `int2(index, value_bits)`).
            ctx.alloc_tensor([2 * next_count as u64, 1, 1, 1], GgmlType::I32)?
        };

        let push = topk_params_bytes(
            n_vocab,
            current_count,
            next_count,
            k,
            1, // nrows
            is_first as u32,
            is_last as u32,
        );

        // 4 bindings: data_a (logits, used only on first), data_d (final),
        // data_s (int2 input), data_t (int2 output). Pipeline cache keys
        // off (name + binding count + push size), so different (first,
        // last) variants share the pipeline — good.
        let key = PipelineKey::dense("topk_argsort_f32", 4, TOPK_PARAMS_BYTES, Vec::new());
        let pipeline = *ctx.pipelines.get(
            ctx.device,
            key,
            shaders::TOPK_ARGSORT_F32_SPV.as_bytes(),
        )?;
        // For the first pass, prev_intermediate isn't initialized — but it's
        // not read in that case (first_pass=1 means the shader reads data_a
        // only). Bind logits to data_s slot too to keep the descriptor valid.
        let s_binding = if is_first { logits.range() } else { prev_intermediate.range() };
        let workgroups = [num_wg, 1, 1];
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &[0, 1, 2, 3],
            &[
                logits.range(),
                final_indices.range(),
                s_binding,
                next_intermediate.range(),
            ],
            &push,
            workgroups,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);

        if is_last {
            break;
        }
        prev_intermediate = next_intermediate;
        current_count = next_count;
        is_first = false;
    }

    // Gather the K logits at those indices: logits[final_indices[i]] for i in [0, K).
    let gathered = ctx.alloc_tensor([k as u64, 1, 1, 1], GgmlType::F32)?;
    record_gather_logits(ctx, logits, final_indices, gathered)?;
    Ok((final_indices, gathered))
}

fn topk_params_bytes(
    orig_ncols: u32,
    ncols_input: u32,
    ncols_output: u32,
    k: u32,
    nrows: u32,
    first_pass: u32,
    last_pass: u32,
) -> [u8; TOPK_PARAMS_BYTES as usize] {
    let mut out = [0u8; TOPK_PARAMS_BYTES as usize];
    let fields = [orig_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass];
    for (i, v) in fields.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
    }
    out
}

/// Gather K F32 logits at the K i32 indices produced by top_k. Uses
/// `get_rows.slang` after reshaping the 1-D logits as a `[hidden=1, vocab]`
/// table.
fn record_gather_logits(
    ctx: &mut DispatchContext,
    logits: TensorView,
    indices: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    // Reshape logits [n_vocab, 1, 1, 1] → [1, n_vocab, 1, 1] (hidden=1
    // wide rows of n_vocab "samples"). Element strides become [1, 1, n_vocab, n_vocab].
    let n_vocab = logits.dims[0];
    let elem = 4u64; // F32
    let src = TensorView {
        buffer: logits.buffer,
        byte_offset: logits.byte_offset,
        byte_size: logits.byte_size,
        dims: [1, n_vocab, 1, 1],
        byte_stride: [elem, elem, n_vocab * elem, n_vocab * elem],
        element_stride: [1, 1, n_vocab, n_vocab],
        dtype: GgmlType::F32,
    };
    let k = indices.dims[0];
    let dst_reshaped = TensorView {
        buffer: dst.buffer,
        byte_offset: dst.byte_offset,
        byte_size: dst.byte_size,
        dims: [1, k, 1, 1],
        byte_stride: [elem, elem, k * elem, k * elem],
        element_stride: [1, 1, k, k],
        dtype: GgmlType::F32,
    };
    super::elementwise::record_get_rows(ctx, src, indices.range(), k as u32, dst_reshaped)
}

/// I32 row gather: pick `dst[i] = table[indices[i]]`. Used to map the
/// final categorical index back to a vocab token id when top_k has
/// rewritten the candidate space. Uses the I32 variant of `get_rows.slang`.
fn record_get_rows_i32(
    ctx: &mut DispatchContext,
    table: TensorView,
    indices: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(table.dtype, GgmlType::I32);
    debug_assert_eq!(indices.dtype, GgmlType::I32);
    debug_assert_eq!(dst.dtype, GgmlType::I32);
    let k = table.dims[0];
    let l = indices.dims[0];
    let elem = 4u64;
    let table_reshaped = TensorView {
        buffer: table.buffer,
        byte_offset: table.byte_offset,
        byte_size: table.byte_size,
        dims: [1, k, 1, 1],
        byte_stride: [elem, elem, k * elem, k * elem],
        element_stride: [1, 1, k, k],
        dtype: GgmlType::I32,
    };
    let dst_reshaped = TensorView {
        buffer: dst.buffer,
        byte_offset: dst.byte_offset,
        byte_size: dst.byte_size,
        dims: [1, l, 1, 1],
        byte_stride: [elem, elem, l * elem, l * elem],
        element_stride: [1, 1, l, l],
        dtype: GgmlType::I32,
    };
    super::elementwise::record_get_rows(ctx, table_reshaped, indices.range(), l as u32, dst_reshaped)
}

/// `α·x + β` over a 1-D tensor via `scale.slang`.
fn record_scale(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    alpha: f32,
    beta: f32,
) -> Result<(), Box<dyn Error>> {
    let push = unary_params_bytes(&src, &dst, alpha, beta);
    let key = PipelineKey::dense("scale_f32", 2, UNARY_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SCALE_F32_SPV.as_bytes())?;
    let nelements: u64 = src.dims.iter().product();
    // scale.slang: 128 threads × num_iter=4 = 512 elements per workgroup.
    let workgroups = [(nelements as u32).div_ceil(512), 1, 1];
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

/// 1-D softmax over a single row (no attention mask, no sinks).
fn record_soft_max(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    let kx: u32 = src.dims[0] as u32;
    let push = soft_max_params_bytes(kx);
    // soft_max.slang binds 4 storage buffers: data_a (logits),
    // data_b (mask, unused → bind logits again), data_c (sinks, unused),
    // data_d (output). When KY=0 and has_sinks=0 the shader ignores b/c
    // bindings, but Vulkan still requires valid descriptors.
    let key = PipelineKey::dense("soft_max_f32", 4, SOFT_MAX_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SOFT_MAX_F32_SPV.as_bytes())?;
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3],
        &[src.range(), src.range(), src.range(), dst.range()],
        &push,
        [1, 1, 1],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}

/// Pack `SoftMaxParams` for the simple "softmax a single row" case.
fn soft_max_params_bytes(kx: u32) -> [u8; SOFT_MAX_PARAMS_BYTES as usize] {
    let mut out = [0u8; SOFT_MAX_PARAMS_BYTES as usize];
    let mut w = 0;
    let put_u = |out: &mut [u8], w: &mut usize, v: u32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };
    let put_f = |out: &mut [u8], w: &mut usize, v: f32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };
    put_u(&mut out, &mut w, kx);          // KX
    put_u(&mut out, &mut w, 0);           // KY (no mask)
    put_u(&mut out, &mut w, kx);          // ne00
    put_u(&mut out, &mut w, 1);           // ne01
    put_u(&mut out, &mut w, 1);           // ne02
    put_u(&mut out, &mut w, 0);           // ne12 (mask, unused)
    put_u(&mut out, &mut w, 0);           // ne13 (mask, unused)
    put_u(&mut out, &mut w, 0);           // nb11
    put_u(&mut out, &mut w, 0);           // nb12
    put_u(&mut out, &mut w, 0);           // nb13
    put_f(&mut out, &mut w, 1.0);         // scale
    put_f(&mut out, &mut w, 0.0);         // max_bias
    put_f(&mut out, &mut w, 0.0);         // m0
    put_f(&mut out, &mut w, 0.0);         // m1
    put_u(&mut out, &mut w, 0);           // n_head_log2
    put_u(&mut out, &mut w, 1);           // nrows_x
    put_u(&mut out, &mut w, 0);           // has_sinks
    out
}

/// Inclusive prefix sum over a single row via `cumsum.slang`.
fn record_cumsum(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    let n_cols: u32 = src.dims[0] as u32;
    let push = sum_rows_params_bytes(n_cols, &src, &dst, 1.0);
    let key = PipelineKey::dense("cumsum_f32", 2, SUM_ROWS_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::CUMSUM_F32_SPV.as_bytes())?;
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        [1, 1, 1],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}

/// Sum across the single row via `sum_rows.slang`. Output is a 1-element tensor.
fn record_sum_rows(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    let n_cols: u32 = src.dims[0] as u32;
    let push = sum_rows_params_bytes(n_cols, &src, &dst, 1.0);
    let key = PipelineKey::dense("sum_rows_f32", 2, SUM_ROWS_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::SUM_ROWS_F32_SPV.as_bytes())?;
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1],
        &[src.range(), dst.range()],
        &push,
        [1, 1, 1],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, ctx.scratch.buffer);
    Ok(())
}

/// Pack `SumRowsParams` (60 bytes) for the "single row, no broadcasting"
/// case shared by `cumsum.slang` and `sum_rows.slang`.
fn sum_rows_params_bytes(
    n_cols: u32,
    src: &TensorView,
    dst: &TensorView,
    weight: f32,
) -> [u8; SUM_ROWS_PARAMS_BYTES as usize] {
    let mut out = [0u8; SUM_ROWS_PARAMS_BYTES as usize];
    let mut w = 0;
    let put_u = |out: &mut [u8], w: &mut usize, v: u32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };
    let put_f = |out: &mut [u8], w: &mut usize, v: f32| {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    };

    put_u(&mut out, &mut w, n_cols);
    put_u(&mut out, &mut w, src.dims[1] as u32);     // ne01
    put_u(&mut out, &mut w, src.dims[2] as u32);     // ne02
    put_u(&mut out, &mut w, src.element_stride[1] as u32); // nb01
    put_u(&mut out, &mut w, src.element_stride[2] as u32); // nb02
    put_u(&mut out, &mut w, src.element_stride[3] as u32); // nb03
    put_u(&mut out, &mut w, dst.element_stride[1] as u32); // nb11
    put_u(&mut out, &mut w, dst.element_stride[2] as u32); // nb12
    put_u(&mut out, &mut w, dst.element_stride[3] as u32); // nb13
    put_f(&mut out, &mut w, weight);
    put_u(&mut out, &mut w, 0);                            // misalign_offsets
    // fastdiv(ne01 * ne02) and fastdiv(ne01)
    let (mp, l) = fastdiv_values((src.dims[1] * src.dims[2]) as u32);
    put_u(&mut out, &mut w, mp);
    put_u(&mut out, &mut w, l);
    let (mp, l) = fastdiv_values(src.dims[1] as u32);
    put_u(&mut out, &mut w, mp);
    put_u(&mut out, &mut w, l);
    out
}

/// Nucleus filtering: zero out the tail of `sorted_logits` whose
/// cumulative probability exceeds `p`. Returns a fresh K-element tensor
/// where dropped entries are `-inf`.
///
/// Composed entirely from existing primitives, matching llama.cpp's
/// `llama_sampler_top_p_backend_apply`:
///   probs   = soft_max(sorted)
///   cdf     = cumsum(probs)
///   margin  = scale(cdf, α=-1, β=p)            // p - cdf
///   mask    = step(margin)                     // 1 if cdf ≤ p, else 0
///   log_m   = log(mask)                        // 0 or -inf
///   kept    = add(sorted, log_m)
fn record_top_p(
    ctx: &mut DispatchContext,
    sorted_logits: TensorView,
    p: f32,
) -> Result<TensorView, Box<dyn Error>> {
    let k = sorted_logits.dims[0];
    let probs = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_soft_max(ctx, sorted_logits, probs)?;
    let cdf = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_cumsum(ctx, probs, cdf)?;
    let margin = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_scale(ctx, cdf, margin, -1.0, p)?;
    let mask = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_step(ctx, margin, mask)?;
    let log_mask = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_log(ctx, mask, log_mask)?;
    let kept = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    super::elementwise::record_add(ctx, sorted_logits, log_mask, kept)?;
    Ok(kept)
}

/// Min-P filtering: zero out anything below `min_p * max_prob`. Since the
/// K-element input from top_k is sorted DESC, `sorted_logits[0]` is the
/// max. We work in log space (matching llama.cpp's CPU and backend paths):
/// the cutoff is `max_logit + log(min_p)`, the mask is `step(logit - cutoff)`,
/// and we add `log(mask)` to push dropped entries to -inf.
fn record_min_p(
    ctx: &mut DispatchContext,
    sorted_logits: TensorView,
    min_p: f32,
) -> Result<TensorView, Box<dyn Error>> {
    let k = sorted_logits.dims[0];
    let elem = 4u64;

    // 1-element view onto sorted_logits[0] — the max logit since the
    // input came out of top_k sorted DESC.
    let max_view = TensorView {
        buffer: sorted_logits.buffer,
        byte_offset: sorted_logits.byte_offset,
        byte_size: elem,
        dims: [1, 1, 1, 1],
        byte_stride: [elem; 4],
        element_stride: [1; 4],
        dtype: GgmlType::F32,
    };

    // thresh = max + log(min_p). Computed on GPU via scale (α=1, β=log(min_p)).
    let log_min_p = min_p.ln();
    let thresh = ctx.alloc_tensor([1, 1, 1, 1], GgmlType::F32)?;
    record_scale(ctx, max_view, thresh, 1.0, log_min_p)?;

    // diff = sorted_logits - thresh (broadcast scalar). sub.slang handles
    // broadcast via norepeat=false when the b dim is 1.
    let diff = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    super::elementwise::record_sub(ctx, sorted_logits, thresh, diff)?;

    let mask = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_step(ctx, diff, mask)?;
    let log_mask = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    record_log(ctx, mask, log_mask)?;
    let kept = ctx.alloc_tensor([k, 1, 1, 1], GgmlType::F32)?;
    super::elementwise::record_add(ctx, sorted_logits, log_mask, kept)?;
    Ok(kept)
}

/// Clamp each element of `src` to `[lo, hi]` via `clamp.slang`. Push
/// constants share the unary layout with `param1=lo, param2=hi`.
fn record_clamp(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
    lo: f32,
    hi: f32,
) -> Result<(), Box<dyn Error>> {
    let push = unary_params_bytes(&src, &dst, lo, hi);
    let key = PipelineKey::dense("clamp_f32", 2, UNARY_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::CLAMP_F32_SPV.as_bytes())?;
    let nelements: u64 = src.dims.iter().product();
    let workgroups = [(nelements as u32).div_ceil(512), 1, 1];
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

/// Elementwise natural log via `log.slang`.
fn record_log(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    let push = unary_params_bytes(&src, &dst, 0.0, 0.0);
    let key = PipelineKey::dense("log_f32", 2, UNARY_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::LOG_F32_SPV.as_bytes())?;
    let nelements: u64 = src.dims.iter().product();
    // log.slang is 512 threads per WG, 1 element per thread.
    let workgroups = [(nelements as u32).div_ceil(512), 1, 1];
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

/// Heaviside step (`x >= 0 ? 1 : 0`) over a 1-D tensor via `step.slang`.
fn record_step(
    ctx: &mut DispatchContext,
    src: TensorView,
    dst: TensorView,
) -> Result<(), Box<dyn Error>> {
    let nelements: u64 = src.dims.iter().product();
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&(nelements as u32).to_ne_bytes());
    // KY, param1..4 all zero.
    let key = PipelineKey::dense("step_f32", 2, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::STEP_F32_SPV.as_bytes())?;
    let workgroups = [(nelements as u32).div_ceil(512), 1, 1];
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
