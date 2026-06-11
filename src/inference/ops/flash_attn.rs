//! `flash_attn` dispatch — scalar reference variant from
//! shaders/compute/flash_attn.slang.
//!
//! Spec constants: Bc, HSK, HSV, MASK_ENABLE, Clamp (5 in declaration order).
//! Push constants follow llama.cpp's `vk_flash_attn_push_constants`
//! (ggml-vulkan.cpp:9396), 128 bytes.
//!
//! Bindings are sparse: indices [0, 1, 2, 3, 5] (skipping 4, which the
//! reference shader reserves for split-K output).
//!
//! The Slang port writes its output in `[hidden = HSV * n_head, L]`
//! contiguous layout (not llama.cpp's `[HSV, L, n_head]` post-permute
//! layout), so the next matmul can read it directly.

use std::error::Error;

use crate::gguf::GgmlType;
use crate::inference::command::record_compute_barrier;
use crate::inference::context::DispatchContext;
use crate::inference::pipeline::PipelineKey;
use crate::inference::weights::TensorView;
use crate::shaders;

const FA_PUSH_BYTES: u32 = 32 * 4;

/// Combine-pass push (`FaSplitKParams`): D, ne1, ne2, ne3, k_num, sinks.
const FA_SPLIT_K_PUSH_BYTES: u32 = 6 * 4;

/// KV block width — must match the `Bc` spec constant in flash_attn.slang,
/// and the alignment llama.cpp snaps split-K KV chunks to.
const FA_BC: u32 = 32;

/// Placeholder compute-unit count when the device doesn't advertise one,
/// matching llama.cpp's flash-attention fallback.
const FA_SPLIT_CORE_COUNT_FALLBACK: u32 = 16;

/// Largest head_dim the coopmat `flash_attn_cm1` path accepts. cm1's LDS no
/// longer scales with head_dim on the K/V side (those are read coopmat-direct
/// from global); only the Q tile (Br × head_dim, in LDS) and the per-thread
/// register O accumulator scale with it. 512 covers gemma4's global layers
/// (key/value_length = 512), whose moderate-context prefill MUST use coopmat —
/// the scalar fallback trips the RADV watchdog past ~100-300 tokens. At 512 the
/// Q tile is 16×512 f16 = 16 KB LDS (~24 KB total, well under 64 KB) and the O
/// accumulator is Of[4][16] = 64 regs/thread. Validated byte-close to scalar at
/// 64/96/128/256; gemma4 256 (sliding) + 512 (global) verified vs llama.cpp.
const CM1_MAX_HEAD_DIM: u32 = 512;

/// Upper bound on the split-K workgroup count for any single flash-attn
/// dispatch. Caps `pick_k_num` and sizes the per-call partials buffer
/// so its (offset, size) descriptor binding stays stable across decode
/// tokens — required for the persistent-decode-cmdbuf optimization,
/// where the host changes only the indirect-dispatch wg count between
/// submits. On qwen35moe Strix Halo decode the heuristic naturally
/// saturates at ~8 (target=80 / base_wgs=10); 16 leaves headroom.
pub const FA_MAX_K_NUM: u32 = 16;

/// Max KV keys a single split-K workgroup may walk before the RADV/Strix-Halo
/// per-dispatch watchdog risks a device-lost (~14k empirically). `pick_k_num`
/// floors the split count at `ceil(kv / this)` so decode (and any low-base-wgs
/// path) stays under it at deep context. Note: `FA_MAX_K_NUM` caps the split, so
/// the walk is only guaranteed ≤ this up to `kv ≈ FA_MAX_K_NUM · this` (~128k);
/// beyond that the walk grows again (256k+ decode is not yet watchdog-safe).
const FA_SPLIT_MAX_WALK: u32 = 8192;

/// Max KV keys a single flash-attn workgroup may walk in the VISION
/// full-attention path before we force a KV-split. The vision tower attends
/// bidirectionally over all `n_pos` patches, so the single-pass kernel has each
/// workgroup walk all `n_pos` keys serially; past ~14k keys that one dispatch
/// trips the RADV/Strix-Halo per-dispatch watchdog (the device is lost). We
/// split the KV (split-K, direct dispatch, partials sized at the actual small
/// k_num) so each split walks ≤ this many keys. EMPIRICAL: on Strix Halo the
/// scalar F32 vision FA faults when a workgroup walks ≳ a few-thousand keys
/// (walk 8052 → device lost; 3220 and 2684 → fine), so 3000 keeps each split's
/// walk safely short. Lower = more splits = shorter walks = safer, but larger
/// partials (≈ (hd+2)·n·heads·k_num) — 3000 fits the 40k-float/token scratch
/// budget up to the 4096-token image cap. Override with `SEEKER_FA_VISION_WALK`
/// (also used to force the split at small n_pos to validate against single-pass).
/// Only read on the vision path (mask=None, n>1), never on the decode hot path.
fn vision_fa_kv_walk() -> u32 {
    std::env::var("SEEKER_FA_VISION_WALK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&w| w > 0)
        .unwrap_or(3_000)
}

/// cm1's single-pass KV ceiling. cm1 tolerates a far longer per-workgroup walk
/// than the scalar path (its coopmat QK^T/PV are ~5× faster per key, so the
/// dispatch stays under the watchdog), and single-pass is faster than split-K at
/// every measured size (no combine pass, no large partials buffer). Default high
/// so all default-cap images (n_pos ≤ 16104) run single-pass; KV beyond this
/// still split-Ks for safety. `SEEKER_FA_VISION_WALK` overrides (e.g. to force a
/// split for validation against the single-pass path).
fn cm1_fa_kv_walk() -> u32 {
    std::env::var("SEEKER_FA_VISION_WALK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&w| w > 0)
        .unwrap_or(16_384)
}

/// Max KV keys a single MASKED-PREFILL flash-attn workgroup may walk before the
/// RADV/Strix-Halo per-dispatch watchdog kills the device. The scalar F16 text
/// path tolerates ~14k keys (empirically: 8.7k OK, 16.9k device-lost), so we
/// force a KV-split sized to keep each split's walk ≤ this, with margin. Unlike
/// the decode/vision paths, multi-row text prefill saturates the GPU with
/// `L·heads` workgroups, so the base_wgs heuristic never splits it on its own.
/// `SEEKER_FA_PREFILL_WALK` overrides (force a split at small kv for validation,
/// or probe the ceiling). Only read on the masked path with n>1 (chunked or
/// standalone prefill); never on the decode hot path (n==1).
fn prefill_fa_kv_walk() -> u32 {
    std::env::var("SEEKER_FA_PREFILL_WALK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&w| w > 0)
        .unwrap_or(8_192)
}

/// Same heuristic as `pick_k_num`, clamped to `FA_MAX_K_NUM`. Used by
/// the Engine to decide whether a cached decode cmdbuf can be replayed
/// for the current `kv` — the wg count is baked into the cmdbuf via
/// `cmd_update_buffer`, so we have to re-record when the heuristic
/// would now pick a different value.
pub fn pick_k_num_clamped(shader_core_count: u32, base_wgs: u32, kv: u32) -> (u32, u32) {
    let (k, bps) = pick_k_num(shader_core_count, base_wgs, kv);
    if k <= FA_MAX_K_NUM {
        (k, bps)
    } else {
        let num_blocks = kv.div_ceil(FA_BC).max(1);
        (FA_MAX_K_NUM, num_blocks.div_ceil(FA_MAX_K_NUM))
    }
}

#[derive(Clone, Copy)]
pub struct FlashAttnParams {
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub gqa_ratio: u32, // n_head / n_head_kv
    pub scale: f32,     // 1 / sqrt(head_dim)
    /// Sliding-window attention span. `0` = no window (full causal, the
    /// default). When `> 0`, key column `kc` is masked for query row `r` unless
    /// `kc <= qpos && qpos - kc < swa_window`, where the absolute query
    /// position `qpos = kv_len - n_query + r` is derived per sequence from
    /// `DecodeDyn`. Applied analytically in-shader on top of any host mask,
    /// independent of `MASK_ENABLE`, so it covers prefill, chunked prefill,
    /// decode (split-K), and batched decode uniformly. Gemma4's sliding
    /// layers set this.
    pub swa_window: u32,
    /// Ring-buffer depth (tokens) of the K/V slab for a sliding-window layer,
    /// or `0` for a normal full-context slab. When `> 0` the slab wraps
    /// (logical position `p` at physical slot `p % ring_depth`): the kernel
    /// iterates `min(kv_len, ring_depth)` physical slots and recovers each
    /// slot's logical position for the causal + window mask. Pair with
    /// `swa_window > 0` and a full-slab (depth-`ring_depth`) K/V view.
    pub ring_depth: u32,
}

/// Whether the scalar flash-attn kernel has a compiled variant that can read
/// this `(K, V)` cache dtype pair directly (zero-copy). Callers use this to
/// decide between binding the cache layers straight into FA and materializing
/// to F32 first. All homogeneous (K==V) pairs are supported; only the exposed
/// heterogeneous pairs are.
pub fn supports_pair(k: GgmlType, v: GgmlType) -> bool {
    scalar_fa_variant(k, v).is_ok()
}

/// Pick the scalar `flash_attn.slang` SPV variant for a `(K, V)` cache dtype
/// pair. K and V are selected independently by the shader (`DATA_A_*`/`K_TYPE`
/// for K, `DATA_V_*`/`V_TYPE` for V), so the cache can be asymmetric. Homogeneous
/// (K==V) pairs keep their historical `flash_attn_f32_<dt>` names; heterogeneous
/// pairs are `flash_attn_f32_<K>_<V>` and only the exposed (compiled) combos are
/// accepted — extend the `//@variants` block in flash_attn.slang and add an arm
/// here to support a new pair.
fn scalar_fa_variant(
    k: GgmlType,
    v: GgmlType,
) -> Result<(&'static str, &'static [u8]), Box<dyn Error>> {
    use GgmlType::*;
    Ok(match (k, v) {
        // Homogeneous K == V.
        (F32, F32) => (
            "flash_attn_f32_f32",
            shaders::FLASH_ATTN_F32_F32_SPV.as_bytes(),
        ),
        (F16, F16) => (
            "flash_attn_f32_f16",
            shaders::FLASH_ATTN_F32_F16_SPV.as_bytes(),
        ),
        (BF16, BF16) => (
            "flash_attn_f32_bf16",
            shaders::FLASH_ATTN_F32_BF16_SPV.as_bytes(),
        ),
        (Q4_0, Q4_0) => (
            "flash_attn_f32_q4_0",
            shaders::FLASH_ATTN_F32_Q4_0_SPV.as_bytes(),
        ),
        (Q4_1, Q4_1) => (
            "flash_attn_f32_q4_1",
            shaders::FLASH_ATTN_F32_Q4_1_SPV.as_bytes(),
        ),
        (Q5_0, Q5_0) => (
            "flash_attn_f32_q5_0",
            shaders::FLASH_ATTN_F32_Q5_0_SPV.as_bytes(),
        ),
        (Q5_1, Q5_1) => (
            "flash_attn_f32_q5_1",
            shaders::FLASH_ATTN_F32_Q5_1_SPV.as_bytes(),
        ),
        (Q8_0, Q8_0) => (
            "flash_attn_f32_q8_0",
            shaders::FLASH_ATTN_F32_Q8_0_SPV.as_bytes(),
        ),
        (IQ4_NL, IQ4_NL) => (
            "flash_attn_f32_iq4_nl",
            shaders::FLASH_ATTN_F32_IQ4_NL_SPV.as_bytes(),
        ),
        // Heterogeneous K != V (precise K, compressed V).
        (F16, Q8_0) => (
            "flash_attn_f32_f16_q8_0",
            shaders::FLASH_ATTN_F32_F16_Q8_0_SPV.as_bytes(),
        ),
        (F16, Q4_0) => (
            "flash_attn_f32_f16_q4_0",
            shaders::FLASH_ATTN_F32_F16_Q4_0_SPV.as_bytes(),
        ),
        (Q8_0, Q4_0) => (
            "flash_attn_f32_q8_0_q4_0",
            shaders::FLASH_ATTN_F32_Q8_0_Q4_0_SPV.as_bytes(),
        ),
        (Q8_0, Q4_1) => (
            "flash_attn_f32_q8_0_q4_1",
            shaders::FLASH_ATTN_F32_Q8_0_Q4_1_SPV.as_bytes(),
        ),
        (Q8_0, Q5_0) => (
            "flash_attn_f32_q8_0_q5_0",
            shaders::FLASH_ATTN_F32_Q8_0_Q5_0_SPV.as_bytes(),
        ),
        (Q8_0, Q5_1) => (
            "flash_attn_f32_q8_0_q5_1",
            shaders::FLASH_ATTN_F32_Q8_0_Q5_1_SPV.as_bytes(),
        ),
        (Q8_0, IQ4_NL) => (
            "flash_attn_f32_q8_0_iq4_nl",
            shaders::FLASH_ATTN_F32_Q8_0_IQ4_NL_SPV.as_bytes(),
        ),
        (Q5_1, Q4_0) => (
            "flash_attn_f32_q5_1_q4_0",
            shaders::FLASH_ATTN_F32_Q5_1_Q4_0_SPV.as_bytes(),
        ),
        (Q5_0, Q4_0) => (
            "flash_attn_f32_q5_0_q4_0",
            shaders::FLASH_ATTN_F32_Q5_0_Q4_0_SPV.as_bytes(),
        ),
        // TurboQuant: homogeneous + the {q8_0, turbo2, turbo3, turbo4} cross
        // product (auto-asymmetric K=q8_0 + layer-adaptive Boundary-V mixes).
        (Turbo2_0, Turbo2_0) => (
            "flash_attn_f32_turbo2",
            shaders::FLASH_ATTN_F32_TURBO2_SPV.as_bytes(),
        ),
        (Turbo3_0, Turbo3_0) => (
            "flash_attn_f32_turbo3",
            shaders::FLASH_ATTN_F32_TURBO3_SPV.as_bytes(),
        ),
        (Turbo4_0, Turbo4_0) => (
            "flash_attn_f32_turbo4",
            shaders::FLASH_ATTN_F32_TURBO4_SPV.as_bytes(),
        ),
        (Q8_0, Turbo2_0) => (
            "flash_attn_f32_q8_0_turbo2",
            shaders::FLASH_ATTN_F32_Q8_0_TURBO2_SPV.as_bytes(),
        ),
        (Q8_0, Turbo3_0) => (
            "flash_attn_f32_q8_0_turbo3",
            shaders::FLASH_ATTN_F32_Q8_0_TURBO3_SPV.as_bytes(),
        ),
        (Q8_0, Turbo4_0) => (
            "flash_attn_f32_q8_0_turbo4",
            shaders::FLASH_ATTN_F32_Q8_0_TURBO4_SPV.as_bytes(),
        ),
        (Turbo2_0, Q8_0) => (
            "flash_attn_f32_turbo2_q8_0",
            shaders::FLASH_ATTN_F32_TURBO2_Q8_0_SPV.as_bytes(),
        ),
        (Turbo3_0, Q8_0) => (
            "flash_attn_f32_turbo3_q8_0",
            shaders::FLASH_ATTN_F32_TURBO3_Q8_0_SPV.as_bytes(),
        ),
        (Turbo4_0, Q8_0) => (
            "flash_attn_f32_turbo4_q8_0",
            shaders::FLASH_ATTN_F32_TURBO4_Q8_0_SPV.as_bytes(),
        ),
        (Turbo2_0, Turbo3_0) => (
            "flash_attn_f32_turbo2_turbo3",
            shaders::FLASH_ATTN_F32_TURBO2_TURBO3_SPV.as_bytes(),
        ),
        (Turbo2_0, Turbo4_0) => (
            "flash_attn_f32_turbo2_turbo4",
            shaders::FLASH_ATTN_F32_TURBO2_TURBO4_SPV.as_bytes(),
        ),
        (Turbo3_0, Turbo2_0) => (
            "flash_attn_f32_turbo3_turbo2",
            shaders::FLASH_ATTN_F32_TURBO3_TURBO2_SPV.as_bytes(),
        ),
        (Turbo3_0, Turbo4_0) => (
            "flash_attn_f32_turbo3_turbo4",
            shaders::FLASH_ATTN_F32_TURBO3_TURBO4_SPV.as_bytes(),
        ),
        (Turbo4_0, Turbo2_0) => (
            "flash_attn_f32_turbo4_turbo2",
            shaders::FLASH_ATTN_F32_TURBO4_TURBO2_SPV.as_bytes(),
        ),
        (Turbo4_0, Turbo3_0) => (
            "flash_attn_f32_turbo4_turbo3",
            shaders::FLASH_ATTN_F32_TURBO4_TURBO3_SPV.as_bytes(),
        ),
        // Precise f16 K + compressed turbo V.
        (F16, Turbo2_0) => (
            "flash_attn_f32_f16_turbo2",
            shaders::FLASH_ATTN_F32_F16_TURBO2_SPV.as_bytes(),
        ),
        (F16, Turbo3_0) => (
            "flash_attn_f32_f16_turbo3",
            shaders::FLASH_ATTN_F32_F16_TURBO3_SPV.as_bytes(),
        ),
        (F16, Turbo4_0) => (
            "flash_attn_f32_f16_turbo4",
            shaders::FLASH_ATTN_F32_F16_TURBO4_SPV.as_bytes(),
        ),
        (kk, vv) => {
            return Err(format!(
                "flash_attn: no shader variant for K/V dtype pair (K={kk:?}, V={vv:?}); \
                 add it to the //@variants block in flash_attn.slang and scalar_fa_variant"
            )
            .into());
        }
    })
}

/// Record flash attention.
///
/// `q` is `[head_dim, L, n_head]` (permuted view of the post-RoPE Q tensor),
/// `k` and `v` are `[head_dim, total_len, n_head_kv]`. `mask` is the F32
/// causal mask `[total_len, L]`, or `None` for single-token decode — there
/// every KV slot is causally visible, so the mask row is all-zeros and the
/// shader runs with `MASK_ENABLE=0`. `out` is `[hidden, L]` contiguous
/// (`[HSV, n_head, L]` post-permute).
///
/// For small `L` (decode) over a long KV cache, the KV dimension is split
/// across `k_num` workgroups per head and the partials merged by
/// `flash_attn_split_k_reduce`; otherwise a single workgroup per head walks
/// the whole cache serially, starving the GPU and making per-token latency
/// grow with context length.
#[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
pub fn record(
    ctx: &mut DispatchContext,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    mask: Option<TensorView>,
    out: TensorView,
    params: FlashAttnParams,
    // Actual KV length to iterate over. The caller passes this
    // explicitly instead of using `k.dims[1]` because the direct-cache
    // fast path binds the full cache layer (so `k.dims[1] = max_seq_len`)
    // and the shader bounds its iteration by `DecodeDyn::kv_len`.
    kv_actual: u32,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(q.dtype, GgmlType::F32);
    debug_assert_eq!(out.dtype, GgmlType::F32);
    if let Some(m) = mask {
        debug_assert_eq!(m.dtype, GgmlType::F32, "mask is always F32 now");
    }
    // K and V may now differ (asymmetric cache): the scalar shader reads each
    // side with its own dtype. The coopmat (cm1) fast paths below still require
    // K==V==F16, so asymmetric/quant combos fall through to the scalar variant.

    // Length of the always-visible cached prefix preceding this batch's
    // query tokens (= position_offset). The host builds only the within-chunk
    // [L × L] causal mask and the scalar shader treats columns < prefix_len as
    // visible; `kv_actual = prefix_len + L`.
    let prefix_len = kv_actual.saturating_sub(q.dims[1] as u32);

    // Cooperative-matrix flash-attention path (`flash_attn_cm1.slang`):
    // 4 subgroups, register-held O accumulator, coopmat QK^T + PV, F16 KV.
    // DEFAULT-ON for masked prefill (`SEEKER_FA_CM=0` opts out): the coopmat
    // QK^T/PV are ~5× faster per key than the scalar loop, so a deep-context
    // prefill ubatch (large kv) completes in one dispatch under the RADV
    // per-dispatch watchdog — the scalar loop is too slow and is lost past ~14k
    // keys. This is how llama.cpp does long-context prefill. cm1 reads the same
    // within-chunk `[L, L]` mask + `prefix_len` (mask_kv_offset) the scalar path
    // uses, so it serves chunked prefill at any offset (not just the first
    // ubatch). Single-token decode passes `mask = None` → scalar split-K path.
    let cm1_head_ok =
        params.head_dim_k <= CM1_MAX_HEAD_DIM && params.head_dim_v <= CM1_MAX_HEAD_DIM;

    // Ring (SWA window-capped) layers always take the scalar path: the ring
    // bounds the per-query walk to ≤ ring_depth keys (so it's watchdog-safe at
    // any context, removing cm1's reason to exist here), and the scalar kernel
    // is the one taught the physical-slot → logical-position ring mask. A ring
    // layer also passes `mask = None`, which would otherwise fall into the
    // maskless (bidirectional) cm1 vision path below — wrong for causal SWA.
    // (cm1 ring read is a perf follow-up.)
    if ctx.device.coop_matrix
        && k.dtype == GgmlType::F16
        && v.dtype == GgmlType::F16
        && !*crate::runtime_flags::FA_CM_DISABLED
        && cm1_head_ok
        && params.ring_depth == 0
        && let Some(m) = mask
    {
        return record_cm1(ctx, q, k, v, Some(m), out, params, kv_actual);
    }

    // Vision coopmat FA (DEFAULT-ON): maskless full bidirectional attention over
    // many query rows (n_pos). cm1 casts the F32 K/V to F16 (Bc-padded) and
    // split-Ks the KV internally so even full-res (n_pos=16104) stays under the
    // watchdog — ~5× faster than scalar on Strix Halo. `n == 1` (decode) is
    // excluded (scalar split-K path). `SEEKER_FA_CM_VISION=0` opts out.
    if ctx.device.coop_matrix
        && !*crate::runtime_flags::FA_CM_VISION_DISABLED
        && mask.is_none()
        && q.dims[1] > 1
        && cm1_head_ok
        && params.ring_depth == 0
    {
        return record_cm1(ctx, q, k, v, None, out, params, kv_actual);
    }

    let (variant_name, variant_spv) = scalar_fa_variant(k.dtype, v.dtype)?;

    let n = q.dims[1] as u32; // L (rows of Q per head)
    // kv_actual is the caller-provided KV length to iterate over.
    // k.dims[1] may be max_seq_len (direct-cache fast path) — the
    // shader uses DecodeDyn::kv_len to bound its loops.
    let kv = kv_actual;
    let ne1 = n; // output rows per head
    let ne2 = q.dims[2] as u32; // n_head
    let ne3 = q.dims[3].max(1) as u32; // batch
    let neq2 = q.dims[2] as u32;
    let neq3 = q.dims[3].max(1) as u32;
    let nek2 = k.dims[2] as u32;
    let nek3 = k.dims[3].max(1) as u32;
    let nev2 = v.dims[2] as u32;
    let nev3 = v.dims[3].max(1) as u32;
    // Mask dims only matter when MASK_ENABLE != 0; supply 1s when disabled.
    let (nem1, nem2, nem3) = match mask {
        Some(m) => (
            m.dims[1] as u32,
            m.dims[2].max(1) as u32,
            m.dims[3].max(1) as u32,
        ),
        None => (1u32, 1u32, 1u32),
    };
    let mask_enable: u32 = if mask.is_some() { 1 } else { 0 };
    let nb01 = q.element_stride[1] as u32;
    let nb02 = q.element_stride[2] as u32;
    let nb03 = q.element_stride[3] as u32;
    let nb11 = k.element_stride[1] as u32;
    let nb12 = k.element_stride[2] as u32;
    let nb13 = k.element_stride[3] as u32;
    let nb21 = v.element_stride[1] as u32;
    let nb22 = v.element_stride[2] as u32;
    let nb23 = v.element_stride[3] as u32;

    // ---- split-K decode heuristic (replicates llama.cpp, clamped to FA_MAX_K_NUM) ----
    let base_wgs = n * ne2 * ne3;
    // A single workgroup walking all `kv` keys serially trips the RADV/Strix-Halo
    // per-dispatch watchdog (device lost) once `kv` is large. The base_wgs split-K
    // heuristic only fires when parallelism is scarce — which never happens for a
    // multi-row forward (vision encode, or text PREFILL: `L·heads` already
    // saturate the GPU), so those would walk `kv` unbounded. Force a KV-split for
    // any n>1 forward whose per-workgroup walk would exceed the safe ceiling so
    // each split walks ≤ that many keys; the split-K branch below then uses a
    // DIRECT dispatch (no decode-replay) with partials sized at the actual k_num.
    // Single-token decode (n == 1) is unaffected — it split-Ks via the heuristic
    // and the indirect-replay path. The maskless (vision) and masked (text
    // prefill) scalar variants have different per-key cost, hence different walk
    // ceilings. The masked split-K result matches single-pass to fp-reduction
    // noise (validated: rel logit diff ~5e-4, identical greedy argmax).
    let walk = if mask.is_none() {
        vision_fa_kv_walk()
    } else {
        prefill_fa_kv_walk()
    };
    let long_walk_split = n > 1 && kv > walk;
    let (k_num, blocks_per_split) = if long_walk_split {
        let num_blocks = kv.div_ceil(FA_BC).max(1);
        let kf = kv.div_ceil(walk).clamp(2, num_blocks);
        (kf, num_blocks.div_ceil(kf))
    } else {
        pick_k_num_clamped(ctx.device.shader_core_count, base_wgs, kv)
    };
    // Mirror kv/k_num/blocks_per_split into the per-forward DecodeDyn
    // slot. The shader reads them from there so the recorded cmdbuf
    // is replay-stable (host overwrites these fields between submits
    // when the persistent-decode-cmdbuf path activates).
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 0, kv)?;
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 4, k_num)?;
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 8, blocks_per_split)?;

    // Pack push.
    let mut push = [0u8; FA_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, n);
    put_u(&mut push, &mut w, kv);
    put_u(&mut push, &mut w, ne1);
    put_u(&mut push, &mut w, ne2);
    put_u(&mut push, &mut w, ne3);
    put_u(&mut push, &mut w, neq2);
    put_u(&mut push, &mut w, neq3);
    put_u(&mut push, &mut w, nek2);
    put_u(&mut push, &mut w, nek3);
    put_u(&mut push, &mut w, nev2);
    put_u(&mut push, &mut w, nev3);
    put_u(&mut push, &mut w, nem1);
    put_u(&mut push, &mut w, nem2);
    put_u(&mut push, &mut w, nem3);
    put_u(&mut push, &mut w, nb01);
    put_u(&mut push, &mut w, nb02);
    put_u(&mut push, &mut w, nb03);
    put_u(&mut push, &mut w, nb11);
    put_u(&mut push, &mut w, nb12);
    put_u(&mut push, &mut w, nb13);
    put_u(&mut push, &mut w, nb21);
    put_u(&mut push, &mut w, nb22);
    put_u(&mut push, &mut w, nb23);
    put_f(&mut push, &mut w, params.scale);
    put_f(&mut push, &mut w, 0.0); // max_bias
    put_f(&mut push, &mut w, 0.0); // logit_softcap
    put_u(&mut push, &mut w, prefix_len); // mask_kv_offset (was mask_n_head_log2)
    put_u(&mut push, &mut w, params.ring_depth); // ring_depth (repurposed ALiBi m0 slot)
    put_u(&mut push, &mut w, params.swa_window); // swa_window (repurposed ALiBi m1 slot)
    put_u(&mut push, &mut w, params.gqa_ratio);
    put_u(&mut push, &mut w, blocks_per_split); // split_kv (blocks per split)
    put_u(&mut push, &mut w, k_num);

    let spec_constants = vec![
        FA_BC,             // Bc
        params.head_dim_k, // HSK
        params.head_dim_v, // HSV
        mask_enable,       // MASK_ENABLE
        0,                 // Clamp
    ];

    // `flash_attn.slang` runs one workgroup per (query-row, head, batch)
    // with WORKGROUP_SIZE=32 — pin to wave32 so the workgroup maps to
    // exactly one subgroup (rather than half a wave64).
    let key = PipelineKey {
        name: variant_name.to_string(),
        binding_indices: vec![0, 1, 2, 3, 4, 5, 6],
        push_size: FA_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant_spv)?;

    // Binding 3 (data_m) needs a valid descriptor even when MASK_ENABLE=0;
    // the shader guards every read on the spec constant, so bind any live
    // storage buffer (q) as a harmless stand-in.
    let mask_range = mask.map(|m| m.range()).unwrap_or_else(|| q.range());
    let dyn_range = ctx.decode_dyn;

    if k_num <= 1 {
        // Single pass: writes the final normalized output to data_o
        // (binding 5). data_o_split (binding 4) is unused here but still
        // needs a valid descriptor, so bind `out` to it too.
        //
        // Direct dispatch (no replay benefit on the single-pass branch
        // — grid is just `[n, ne2, ne3]`, all model-static, so a
        // recorded cmdbuf already replays correctly).
        let workgroups = [n, ne2, ne3];
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5, 6],
            &[
                q.range(),
                k.range(),
                v.range(),
                mask_range,
                out.range(),
                out.range(),
                dyn_range,
            ],
            &push,
            workgroups,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, out.range());
    } else {
        // Split-K: variable-size grid in y (= ne2 * k_num) — switch to
        // `vkCmdDispatchIndirect` so the recorded cmdbuf can serve any
        // runtime k_num via a host-side write to the 12-byte
        // `indirect_wg` slot. Partials are sized once at `FA_MAX_K_NUM`
        // so the binding (offset, size) is constant across calls.
        // Decode replay sizes partials at FA_MAX_K_NUM for a stable binding; the
        // direct-dispatch path (vision encode / text prefill) has no replay, so
        // size at the actual k_num (the combine indexes partials by runtime k_num
        // — see flash_attn_split_k_reduce.slang). At a large n this keeps the
        // partials bounded (k_num·(hd+2)·n·heads) instead of FA_MAX_K_NUM×.
        let knum_partials = if long_walk_split { k_num } else { FA_MAX_K_NUM };
        let partials_floats = (params.head_dim_v as u64 + 2)
            * n as u64
            * ne2 as u64
            * ne3 as u64
            * knum_partials as u64;
        let partials = ctx.alloc_tensor([partials_floats, 1, 1, 1], GgmlType::F32)?;

        if long_walk_split {
            // Direct dispatch — one workgroup per (query row, head, split):
            // grid [n, ne2 * k_num, ne3]. No indirect args (those exist only for
            // decode-cmdbuf replay where k_num varies between submits).
            super::bind_and_dispatch(
                ctx,
                &pipeline,
                &[0, 1, 2, 3, 4, 5, 6],
                &[
                    q.range(),
                    k.range(),
                    v.range(),
                    mask_range,
                    partials.range(),
                    out.range(),
                    dyn_range,
                ],
                &push,
                [n, ne2 * k_num, ne3],
            )?;
            record_compute_barrier(ctx.device, ctx.cmd, partials.range());
            record_split_k_combine(
                ctx,
                partials,
                out,
                params.head_dim_v,
                n,
                ne2,
                ne3,
                k_num,
                dyn_range,
            )?;
            return Ok(());
        }

        let indirect_wg = ctx.alloc_scratch(12)?;
        // Write the (wg_x, wg_y, wg_z) tuple via cmd_update_buffer —
        // recorded into the cmdbuf as a transfer op, naturally ordered
        // before the indirect dispatch by a transfer→draw_indirect
        // barrier. A direct host-mapped write empirically triggers
        // DEVICE_LOST on RADV STRIX_HALO when the indirect dispatch
        // fires past kv=33 — likely because indirect reads from
        // HOST_COHERENT memory aren't covered by the implicit
        // submit-time HOST→DRAW_INDIRECT dependency on this driver.
        //
        // NOTE: cmd_update_buffer bakes the value into the cmdbuf, so
        // this approach is incompatible with Phase 4 replay if k_num
        // varies between submits. Phase 4 will switch to a host→staging
        // → cmd_copy_buffer chain with the right barriers.
        let wg_data: [u32; 3] = [n, ne2 * k_num, ne3];
        let wg_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(wg_data.as_ptr() as *const u8, 12) };
        unsafe {
            ctx.device.device.cmd_update_buffer(
                ctx.cmd,
                indirect_wg.buffer,
                indirect_wg.offset,
                wg_bytes,
            );
            let bar = ash::vk::BufferMemoryBarrier::default()
                .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(ash::vk::AccessFlags::INDIRECT_COMMAND_READ)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .buffer(indirect_wg.buffer)
                .offset(indirect_wg.offset)
                .size(12);
            ctx.device.device.cmd_pipeline_barrier(
                ctx.cmd,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::PipelineStageFlags::DRAW_INDIRECT,
                ash::vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&bar),
                &[],
            );
        }
        super::bind_and_dispatch_indirect(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5, 6],
            &[
                q.range(),
                k.range(),
                v.range(),
                mask_range,
                partials.range(),
                out.range(),
                dyn_range,
            ],
            &push,
            indirect_wg,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, partials.range());
        // `dyn_range` (= ctx.decode_dyn) carries this call's k_num for the combine.
        record_split_k_combine(
            ctx,
            partials,
            out,
            params.head_dim_v,
            n,
            ne2,
            ne3,
            k_num,
            dyn_range,
        )?;
    }
    Ok(())
}

/// Batched **decode** flash attention: B independent sequences, one query
/// token each, each attending to its own KV slab over its own length.
///
/// `q` is `[head_dim, 1, n_head, B]`, `k`/`v` are `[head_dim, kv_pos, n_head_kv,
/// B]` where the batch (dim 3, stride `element_stride[3]`) selects a sequence's
/// slab. `kv_lens[b]` is sequence `b`'s cache length. `out` is `[hidden, B]`
/// (`[HSV, n_head, B]`). No mask (a decode query attends to its whole cache).
///
/// Split-K (M4): base parallelism is `n_head * B` workgroups (one per head per
/// sequence). When that under-fills the GPU — long context, few sequences — KV
/// is split across `k_num` workgroups per head and merged by
/// `flash_attn_split_k_reduce`, exactly as the single-sequence decode path. As
/// `B` grows the base already saturates, so the heuristic falls back to
/// `k_num = 1` (no split). `blocks_per_split` is sized from the longest
/// sequence so every sequence's full range is covered; a shorter sequence stops
/// early via the per-element `kv_base + c < p_KV` guard (`p_KV = kv_lens[seq]`),
/// and its over-long splits run zero iterations (the combine weights them to
/// ~0). The batched forward always re-records (no decode replay), so this uses
/// a plain direct dispatch — none of the single-seq indirect-dispatch machinery
/// that exists only to vary `k_num` across replayed submits. `SEEKER_FA_SPLIT=0`
/// disables it; `SEEKER_FA_SPLIT_KNUM=<n>` pins it.
#[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
pub fn record_batched(
    ctx: &mut DispatchContext,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    out: TensorView,
    params: FlashAttnParams,
    kv_lens: &[u32],
    dyn_range: crate::inference::buffer::BufferRange,
    // Per-sequence KV slab index (which `BatchKvCache` slab each sequence's K/V
    // lives in). `None` → identity (sequence `b` reads slab `b`), the
    // contiguous layout. `Some` lets a gathered batch read non-contiguous slabs
    // (continuous-batching with prefix-reuse parking). Must be `kv_lens.len()`.
    slots: Option<&[u32]>,
    // Per-sequence query-row counts (`L_s`). `None` → decode: each sequence
    // contributes one query row, split-K enabled, legacy Q/out layout
    // (`[head_dim, 1, n_head, B]`, batch stride nb03). `Some` → unified varlen:
    // sequence `s` contributes `query_lens[s]` rows packed flat in the token
    // dimension; the shader masks causally in-place (row r of seq s sees
    // `[0, kv_lens[s] - query_lens[s] + r]`). q/out are the flat
    // `[head_dim, N_total, n_head]` / `[hidden_v, N_total]` layouts (per-token
    // stride nb01, per-head nb02); split-K is disabled (the `L_s` rows already
    // fill the grid). Must be `kv_lens.len()`; each `query_lens[s] <= kv_lens[s]`.
    query_lens: Option<&[u32]>,
) -> Result<(), Box<dyn Error>> {
    debug_assert_eq!(q.dtype, GgmlType::F32);
    debug_assert_eq!(out.dtype, GgmlType::F32);
    // K and V may differ (asymmetric cache) — the scalar shader reads each side
    // with its own dtype.
    let b = kv_lens.len() as u32;
    debug_assert!(b >= 1, "record_batched needs at least one sequence");
    let varlen = query_lens.is_some();
    if let Some(ql) = query_lens {
        debug_assert_eq!(ql.len(), kv_lens.len(), "query_lens must match kv_lens");
        debug_assert!(
            ql.iter().zip(kv_lens).all(|(&q, &kv)| q >= 1 && q <= kv),
            "each query_lens[s] must be in 1..=kv_lens[s]"
        );
    } else {
        debug_assert_eq!(
            q.dims[3].max(1) as u32,
            b,
            "q batch dim must equal kv_lens.len()"
        );
    }
    // Flat per-sequence query offsets (prefix sum of query_lens) and the grid's
    // x-dim (max query rows). Decode: query_lens = all-1 → q_start[s]=s, 1 row.
    let query_lens_vec: Vec<u32> = query_lens
        .map(|q| q.to_vec())
        .unwrap_or_else(|| vec![1; b as usize]);
    let q_starts: Vec<u32> = query_lens_vec
        .iter()
        .scan(0u32, |acc, &l| {
            let s = *acc;
            *acc += l;
            Some(s)
        })
        .collect();
    let max_rows = query_lens_vec.iter().copied().max().unwrap_or(1);

    let (variant_name, variant_spv) = scalar_fa_variant(k.dtype, v.dtype)?;

    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);

    // Split-K decode heuristic for the batch (see the doc comment). `base_wgs`
    // counts one workgroup per (head, sequence); `k_num` splits each head's KV
    // only when that under-fills the GPU. When `k_num == 1`, `pick_k_num`
    // returns `blocks_per_split == ceil(max_kv / Bc)` — the whole range in one
    // split, identical to the pre-split single pass.
    let ne2 = q.dims[2] as u32; // n_head
    let ne3 = b;
    // Grid x-dim = query rows. Decode: 1 row/seq, split KV via the heuristic.
    // Varlen: max_rows rows/seq (the L_s rows already fill the grid → no split).
    let (n, k_num, blocks_per_split) = if varlen {
        (max_rows, 1u32, max_kv.div_ceil(FA_BC).max(1))
    } else {
        let base_wgs = ne2 * ne3; // one wg per (head, seq)
        let (k, bps) = pick_k_num_clamped(ctx.device.shader_core_count, base_wgs, max_kv);
        (1u32, k, bps)
    };

    // Per-sequence DecodeDyn array: entry b's kv_len bounds sequence b's KV
    // loop; entry 0's k_num / blocks_per_split are read uniformly by every
    // workgroup. Zero the rest so stale scratch never leaks into a field.
    //
    // CRITICAL: `dyn_range` is caller-provided and MUST live in scratch that
    // is NOT reclaimed by `scratch_restore` before the submit. The host writes
    // these fields now (record time) but the shader reads them at execute time
    // (after submit); if a later layer's host-write reuses this offset, the
    // shader reads a garbage `kv_len` → unbounded KV loop → GPU hang. (The
    // qwen hybrid hit exactly this: per-layer scratch reuse across SSM/attn
    // layers corrupted the length. Allocate once per forward, before the
    // layer checkpoint.)
    debug_assert!(
        dyn_range.size >= crate::inference::decode_dyn::DecodeDyn::SIZE * b as u64,
        "record_batched: dyn_range too small for {b} sequences"
    );
    {
        let host_ptr = ctx
            .scratch
            .host_ptr
            .ok_or("scratch not host-visible — record_batched requires mapped memory")?;
        let mut entries = vec![crate::inference::decode_dyn::DecodeDyn::default(); b as usize];
        for (i, e) in entries.iter_mut().enumerate() {
            e.kv_len = kv_lens[i];
            e.slot = slots.map_or(i as u32, |s| s[i]);
            e.n_query = query_lens_vec[i];
            e.q_start = q_starts[i];
        }
        entries[0].k_num = k_num;
        entries[0].blocks_per_split = blocks_per_split;
        unsafe {
            let dst = host_ptr.add(dyn_range.offset as usize)
                as *mut crate::inference::decode_dyn::DecodeDyn;
            std::ptr::copy_nonoverlapping(entries.as_ptr(), dst, entries.len());
        }
    }

    let ne1 = n; // n / ne2 / ne3 computed with the split-K heuristic above
    let push_strides = [
        n,
        max_kv,
        ne1,
        ne2,
        ne3,
        q.dims[2] as u32,           // neq2
        b,                          // neq3
        k.dims[2] as u32,           // nek2
        b,                          // nek3
        v.dims[2] as u32,           // nev2
        b,                          // nev3
        1,                          // nem1 (mask disabled)
        1,                          // nem2
        1,                          // nem3
        q.element_stride[1] as u32, // nb01
        q.element_stride[2] as u32, // nb02
        q.element_stride[3] as u32, // nb03
        k.element_stride[1] as u32, // nb11
        k.element_stride[2] as u32, // nb12
        k.element_stride[3] as u32, // nb13
        v.element_stride[1] as u32, // nb21
        v.element_stride[2] as u32, // nb22
        v.element_stride[3] as u32, // nb23
    ];

    let mut push = [0u8; FA_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    for s in push_strides {
        put_u(&mut push, &mut w, s);
    }
    put_f(&mut push, &mut w, params.scale);
    put_f(&mut push, &mut w, 0.0); // max_bias
    put_f(&mut push, &mut w, 0.0); // logit_softcap
    put_u(&mut push, &mut w, 0); // mask_kv_offset (no prefix mask on decode)
    put_f(&mut push, &mut w, 0.0); // m0
    put_u(&mut push, &mut w, params.swa_window); // swa_window (repurposed ALiBi m1 slot)
    put_u(&mut push, &mut w, params.gqa_ratio);
    put_u(&mut push, &mut w, blocks_per_split); // split_kv (blocks per split)
    put_u(&mut push, &mut w, k_num);

    let spec_constants = vec![
        FA_BC,
        params.head_dim_k,
        params.head_dim_v,
        0,                          // MASK_ENABLE (varlen masks causally in-shader)
        0,                          // Clamp
        if varlen { 1 } else { 0 }, // VARLEN
    ];
    let key = PipelineKey {
        name: variant_name.to_string(),
        binding_indices: vec![0, 1, 2, 3, 4, 5, 6],
        push_size: FA_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(ctx.device, key, variant_spv)?;

    if k_num <= 1 {
        // Single pass: one workgroup per (query-row, head, sequence) writes the
        // normalized output directly. data_o_split (binding 4) is unused but
        // still needs a live descriptor → bind `out`.
        let workgroups = [n, ne2, ne3];
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5, 6],
            &[
                q.range(),
                k.range(),
                v.range(),
                q.range(),   // data_m stand-in (MASK_ENABLE=0 → never read)
                out.range(), // data_o_split stand-in (k_num=1 → unused)
                out.range(),
                dyn_range,
            ],
            &push,
            workgroups,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, out.range());
    } else {
        // Split-K: grid.y packs (head, split) = ne2 * k_num; .z is the
        // sequence. Each workgroup writes its split's unnormalized O partial +
        // (L, M) into `partials`, then the combine kernel merges k_num partials
        // per (head, sequence). Direct dispatch — the batched forward always
        // re-records, so no indirect-dispatch / FA_MAX_K_NUM padding needed:
        // size partials at exactly k_num (the layout indexes by the runtime
        // k_num) and pin the grid to k_num here.
        let partials_floats =
            (params.head_dim_v as u64 + 2) * n as u64 * ne2 as u64 * ne3 as u64 * k_num as u64;
        let partials = ctx.alloc_tensor([partials_floats, 1, 1, 1], GgmlType::F32)?;
        let workgroups = [n, ne2 * k_num, ne3];
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5, 6],
            &[
                q.range(),
                k.range(),
                v.range(),
                q.range(), // data_m stand-in (MASK_ENABLE=0 → never read)
                partials.range(),
                out.range(),
                dyn_range,
            ],
            &push,
            workgroups,
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, partials.range());
        // The combine reads k_num from `data_dyn[0]` — pass the same batched
        // dyn_range whose entry 0 we set above (NOT ctx.decode_dyn).
        record_split_k_combine(
            ctx,
            partials,
            out,
            params.head_dim_v,
            n,
            ne2,
            ne3,
            k_num,
            dyn_range,
        )?;
    }
    Ok(())
}

/// Pick the KV-split factor `(k_num, blocks_per_split)` for the main
/// flash-attn dispatch, replicating llama.cpp's heuristic
/// (`ggml_vk_flash_attn` in ggml-vulkan.cpp).
///
/// llama.cpp aims for ~`2 × shader_core_count` total workgroups, then snaps
/// each split to a `Bc`-aligned KV chunk and re-derives the split count.
/// `base_wgs` is our no-split workgroup count (`n * n_head * batch`). Because
/// our kernel launches one workgroup per *full* query head (no GQA grouping),
/// `base_wgs` already equals llama.cpp's grouped total occupancy, so dividing
/// the target by it reproduces the same total workgroup count. (We omit
/// llama.cpp's Intel-Alchemist `×2` multiplier — irrelevant on this path.)
///
/// `SEEKER_FA_SPLIT=0` disables splitting; `SEEKER_FA_SPLIT_KNUM=<n>` pins it.
pub fn pick_k_num(shader_core_count: u32, base_wgs: u32, kv: u32) -> (u32, u32) {
    let num_blocks = kv.div_ceil(FA_BC).max(1);
    if *crate::runtime_flags::FA_SPLIT_DISABLED {
        return (1, num_blocks);
    }
    if let Some(k) = *crate::runtime_flags::FA_SPLIT_KNUM {
        let k = k.clamp(1, num_blocks);
        return (k, num_blocks.div_ceil(k));
    }

    // Placeholder core count when the device doesn't report one; target is
    // 2× that, the same as llama.cpp.
    let core_count = if shader_core_count != 0 {
        shader_core_count
    } else {
        FA_SPLIT_CORE_COUNT_FALLBACK
    };
    let target = core_count * 2;

    let mut split_k = 1u32;
    if base_wgs < target {
        split_k = target / base_wgs.max(1);
    }
    // Watchdog floor: force enough splits that no single workgroup walks more
    // than `FA_SPLIT_MAX_WALK` keys, even when base parallelism alone wouldn't
    // split (deep-context decode has base_wgs = n_head ≥ target, so the
    // heuristic leaves split_k = 1 and one workgroup walks all `kv` keys →
    // RADV device-lost past ~14k). A const (not the env helper) keeps the
    // decode hot path free of getenv.
    let walk_floor = kv.div_ceil(FA_SPLIT_MAX_WALK).max(1);
    split_k = split_k.max(walk_floor);
    if split_k <= 1 {
        return (1, num_blocks);
    }

    // Snap KV into `split_k` chunks rounded up to a multiple of Bc, then
    // re-derive split_k from the chunk size (matches llama.cpp's
    // `ROUNDUP_POW2(KV/split_k, alignment)` + `CEIL_DIV(KV, split_kv)`). This
    // self-corrects to never request more splits than there are blocks.
    let split_kv = roundup_mult((kv / split_k).max(1), FA_BC);
    let split_k = kv.div_ceil(split_kv);
    let blocks_per_split = split_kv / FA_BC; // exact: split_kv is a multiple of Bc
    (split_k, blocks_per_split)
}

/// Round `m` up to a multiple of `n` (n a power of two) — llama.cpp's
/// `ROUNDUP_POW2`.
fn roundup_mult(m: u32, n: u32) -> u32 {
    (m + n - 1) & !(n - 1)
}

/// Merge `k_num` per-split (O, L, M) partials into the final attention output
/// via `flash_attn_split_k_reduce`. One workgroup per (row, head, batch);
/// grid.y tiles HSV in BLOCK_SIZE(=32)-wide chunks.
#[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
fn record_split_k_combine(
    ctx: &mut DispatchContext,
    partials: TensorView,
    out: TensorView,
    head_dim_v: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
    k_num: u32,
    // DecodeDyn buffer the reduce kernel reads `k_num` from (`data_dyn[0]`).
    // Single-seq decode passes `ctx.decode_dyn` (k_num written via
    // `write_field_ctx`); batched decode passes its per-forward dyn_range whose
    // entry 0 carries the same k_num.
    dyn_range: crate::inference::buffer::BufferRange,
) -> Result<(), Box<dyn Error>> {
    let mut push = [0u8; FA_SPLIT_K_PUSH_BYTES as usize];
    let mut w = 0usize;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, head_dim_v); // D
    put_u(&mut push, &mut w, ne1);
    put_u(&mut push, &mut w, ne2);
    put_u(&mut push, &mut w, ne3);
    put_u(&mut push, &mut w, k_num);
    put_u(&mut push, &mut w, 0); // sinks (unused)

    let key = PipelineKey {
        name: "flash_attn_split_k_reduce_f32".to_string(),
        binding_indices: vec![0, 1, 2, 3],
        push_size: FA_SPLIT_K_PUSH_BYTES,
        spec_constants: vec![32], // BLOCK_SIZE
        // The reduce kernel is one 32-wide wave (bare WaveActiveMax/Sum,
        // no shared memory) — pin wave32 so workgroup == subgroup.
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(
        ctx.device,
        key,
        shaders::FLASH_ATTN_SPLIT_K_REDUCE_F32_SPV.as_bytes(),
    )?;
    // data_a = partials, data_s = sinks (unused → dummy `out`), data_d = out, data_dyn = DecodeDyn.
    let workgroups = [ne1, head_dim_v.div_ceil(32), ne2 * ne3];
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3],
        &[partials.range(), out.range(), out.range(), dyn_range],
        &push,
        workgroups,
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, out.range());
    Ok(())
}

/// Bc tile width baked into `flash_attn_cm1.slang`. The cm1 KV loop steps in
/// blocks of this many keys; K/V must have at least `ceil(KV/Bc)*Bc` rows so the
/// last block's direct coopmat loads stay in bounds.
const CM1_BC: u64 = 64;

/// Zero an F16 tensor's bytes via an F32-reinterpreted `fill_f32` dispatch
/// (there's no F16 fill shader; zero bytes read as 0.0 in either dtype).
fn fill_zero_f16(ctx: &mut DispatchContext, t: &TensorView) -> Result<(), Box<dyn Error>> {
    const GENERIC_PARAMS_BYTES: u32 = 6 * 4;
    debug_assert_eq!(t.byte_size % 4, 0, "F16 fill needs an even byte count");
    let n_f32 = (t.byte_size / 4) as u32;
    let view = TensorView {
        buffer: t.buffer,
        byte_offset: t.byte_offset,
        byte_size: t.byte_size,
        dims: [n_f32 as u64, 1, 1, 1],
        byte_stride: [4, 4, 4, 4],
        element_stride: [1, 1, 1, 1],
        dtype: GgmlType::F32,
    };
    let mut push = [0u8; GENERIC_PARAMS_BYTES as usize];
    push[0..4].copy_from_slice(&n_f32.to_ne_bytes());
    let key = PipelineKey::dense("fill_f32", 1, GENERIC_PARAMS_BYTES, Vec::new());
    let pipeline = *ctx
        .pipelines
        .get(ctx.device, key, shaders::FILL_F32_SPV.as_bytes())?;
    let workgroups = [n_f32.div_ceil(512), 1, 1];
    super::bind_and_dispatch(ctx, &pipeline, &[0], &[view.range()], &push, workgroups)?;
    record_compute_barrier(ctx.device, ctx.cmd, view.range());
    Ok(())
}

/// Cast an F32 `[hd, n_pos, n_head, 1]` (contiguous) K/V tensor to F16 in a
/// buffer padded with `CM1_BC` extra rows at the end, so cm1's direct coopmat
/// loads on the final KV block read in-bounds (the slack is zeroed; rows past
/// `p.KV` are excluded from the softmax via the kernel's CLAMP, and the PV step
/// multiplies them by P=0 — so the slack just needs to be finite). Returns a
/// logical `[hd, n_pos, n_head, 1]` view whose `range()` spans the padded buffer.
fn cast_kv_f16_padded(
    ctx: &mut DispatchContext,
    src: TensorView,
) -> Result<TensorView, Box<dyn Error>> {
    debug_assert_eq!(src.dtype, GgmlType::F32);
    let hd = src.dims[0];
    let np = src.dims[1];
    let nh = src.dims[2].max(1);
    let real_rows = np * nh;
    let buf = ctx.alloc_tensor([hd, real_rows + CM1_BC, 1, 1], GgmlType::F16)?;

    // Zero just the CM1_BC slack rows at the end (the cast overwrites the rest).
    let slack = TensorView {
        buffer: buf.buffer,
        byte_offset: buf.byte_offset + real_rows * hd * 2,
        byte_size: CM1_BC * hd * 2,
        dims: [hd, CM1_BC, 1, 1],
        byte_stride: [2, 2 * hd, 2 * hd * CM1_BC, 2 * hd * CM1_BC],
        element_stride: [1, hd, hd * CM1_BC, hd * CM1_BC],
        dtype: GgmlType::F16,
    };
    fill_zero_f16(ctx, &slack)?;

    // Contiguous F32→F16 cast of the real rows: src and dst both flattened to
    // [hd, n_pos*n_head, 1, 1] (src is contiguous, so this is the same memory).
    let src_flat = TensorView {
        dims: [hd, real_rows, 1, 1],
        byte_stride: [4, 4 * hd, 4 * hd * real_rows, 4 * hd * real_rows],
        element_stride: [1, hd, hd * real_rows, hd * real_rows],
        ..src
    };
    let dst_flat = TensorView {
        buffer: buf.buffer,
        byte_offset: buf.byte_offset,
        byte_size: real_rows * hd * 2,
        dims: [hd, real_rows, 1, 1],
        byte_stride: [2, 2 * hd, 2 * hd * real_rows, 2 * hd * real_rows],
        element_stride: [1, hd, hd * real_rows, hd * real_rows],
        dtype: GgmlType::F16,
    };
    crate::inference::ops::cast::record_cast(ctx, src_flat, dst_flat)?;

    // Logical [hd, n_pos, n_head, 1] view with contiguous head stride; byte_size
    // spans the whole padded buffer so the dispatch binding covers the slack.
    Ok(TensorView {
        buffer: buf.buffer,
        byte_offset: buf.byte_offset,
        byte_size: buf.byte_size,
        dims: [hd, np, nh, 1],
        byte_stride: [2, 2 * hd, 2 * hd * np, 2 * hd * np * nh],
        element_stride: [1, hd, hd * np, hd * np * nh],
        dtype: GgmlType::F16,
    })
}

/// Cooperative-matrix flash-attention path (`flash_attn_cm1.slang`).
///
/// 128-thread workgroup = 4 wave32 subgroups; spec constants HSK, HSV,
/// MASK_ENABLE, CLAMP. Each workgroup owns `Br=16` query rows (dispatch.x =
/// ceil(N/16)); grid.y packs (head, split-K index), grid.z is batch.
///   - K/V are read coopmat-direct from global (no LDS staging). The decoder
///     F16 cache is bound full so its last-block reads are in-bounds; vision F32
///     activations are cast to F16 in a `CM1_BC`-padded buffer (zeroed slack).
///   - `mask`: `Some` (masked prefill, cast F32→F16, `MASK_ENABLE=1`) or `None`
///     (maskless vision, `MASK_ENABLE=0`; K is bound as the unused slot-3 dummy).
///   - Long KV is split-K'd (each workgroup walks ≤ `vision_fa_kv_walk` keys),
///     partials merged by `flash_attn_split_k_reduce`.
#[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
fn record_cm1(
    ctx: &mut DispatchContext,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    mask: Option<TensorView>,
    out: TensorView,
    params: FlashAttnParams,
    kv_actual: u32,
) -> Result<(), Box<dyn Error>> {
    // cm1 reads K/V coopmat-direct from global. The decoder cache is already
    // F16 and bound as the full max_seq_len layer, so its last-block reads are
    // in-bounds — pass it through. The vision tower hands us F32 activations
    // sized exactly to n_pos, so cast to F16 into a Bc-row-padded buffer (zeroed
    // slack) so the last block's direct loads stay in bounds.
    let k = if k.dtype == GgmlType::F16 {
        k
    } else {
        cast_kv_f16_padded(ctx, k)?
    };
    let v = if v.dtype == GgmlType::F16 {
        v
    } else {
        cast_kv_f16_padded(ctx, v)?
    };

    // Cast F32 mask → F16 in scratch (decoder/prefill). The vision tower attends
    // bidirectionally with no mask (`MASK_ENABLE=0`); bind K as a harmless dummy
    // for binding 3 since the shader never reads it in that case.
    let mask_enable: u32 = if mask.is_some() { 1 } else { 0 };
    let mask_buf = match mask {
        Some(m) => {
            let mask_f16 = ctx.alloc_tensor(m.dims, GgmlType::F16)?;
            crate::inference::ops::cast::record_cast(ctx, m, mask_f16)?;
            mask_f16
        }
        None => k,
    };
    let n = q.dims[1] as u32;
    let kv = kv_actual;
    let ne1 = n;
    let ne2 = q.dims[2] as u32;
    let ne3 = q.dims[3].max(1) as u32;
    let neq2 = q.dims[2] as u32;
    let neq3 = q.dims[3].max(1) as u32;
    let nek2 = k.dims[2] as u32;
    let nek3 = k.dims[3].max(1) as u32;
    let nev2 = v.dims[2] as u32;
    let nev3 = v.dims[3].max(1) as u32;
    let (nem1, nem2, nem3) = match mask {
        Some(m) => (
            m.dims[1] as u32,
            m.dims[2].max(1) as u32,
            m.dims[3].max(1) as u32,
        ),
        None => (1u32, 1u32, 1u32),
    };
    let nb01 = q.element_stride[1] as u32;
    let nb02 = q.element_stride[2] as u32;
    let nb03 = q.element_stride[3] as u32;
    let nb11 = k.element_stride[1] as u32;
    let nb12 = k.element_stride[2] as u32;
    let nb13 = k.element_stride[3] as u32;
    let nb21 = v.element_stride[1] as u32;
    let nb22 = v.element_stride[2] as u32;
    let nb23 = v.element_stride[3] as u32;

    // ── cm1 split-K. UNLIKE the scalar path (which faults past ~3k keys/wg, so
    // splits at `vision_fa_kv_walk`=3000), cm1's coopmat throughput keeps the
    // per-workgroup walk well under the RADV watchdog: single-pass is clean AND
    // fastest (no combine pass) up to the full-res default cap (n_pos=16104,
    // measured 7.95 s vs 8.27 s split). So cm1 single-passes up to a high ceiling
    // and only split-Ks pathologically large KV beyond it. (More splits are also
    // a memory risk here: the partials buffer is k_num·(hd+2)·n·heads — ~1 GB at
    // k_num≈11, n_pos=16104.) Override the ceiling with `SEEKER_FA_VISION_WALK`.
    let cm1_walk = cm1_fa_kv_walk();
    let num_blocks = kv.div_ceil(CM1_BC as u32).max(1);
    let (k_num, blocks_per_split) = if kv > cm1_walk {
        let kf = kv.div_ceil(cm1_walk).clamp(2, num_blocks);
        (kf, num_blocks.div_ceil(kf))
    } else {
        (1u32, num_blocks)
    };
    // The reduce kernel reads k_num from DecodeDyn (binding 3); set it (+ kv and
    // blocks-per-split for parity with the scalar path).
    let dyn_range = ctx.decode_dyn;
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 0, kv)?;
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 4, k_num)?;
    crate::inference::decode_dyn::write_field_ctx(ctx, ctx.decode_dyn, 8, blocks_per_split)?;

    let mut push = [0u8; FA_PUSH_BYTES as usize];
    let mut w = 0;
    fn put_u(out: &mut [u8], w: &mut usize, v: u32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    fn put_f(out: &mut [u8], w: &mut usize, v: f32) {
        out[*w..*w + 4].copy_from_slice(&v.to_ne_bytes());
        *w += 4;
    }
    put_u(&mut push, &mut w, n);
    put_u(&mut push, &mut w, kv);
    put_u(&mut push, &mut w, ne1);
    put_u(&mut push, &mut w, ne2);
    put_u(&mut push, &mut w, ne3);
    put_u(&mut push, &mut w, neq2);
    put_u(&mut push, &mut w, neq3);
    put_u(&mut push, &mut w, nek2);
    put_u(&mut push, &mut w, nek3);
    put_u(&mut push, &mut w, nev2);
    put_u(&mut push, &mut w, nev3);
    put_u(&mut push, &mut w, nem1);
    put_u(&mut push, &mut w, nem2);
    put_u(&mut push, &mut w, nem3);
    put_u(&mut push, &mut w, nb01);
    put_u(&mut push, &mut w, nb02);
    put_u(&mut push, &mut w, nb03);
    put_u(&mut push, &mut w, nb11);
    put_u(&mut push, &mut w, nb12);
    put_u(&mut push, &mut w, nb13);
    put_u(&mut push, &mut w, nb21);
    put_u(&mut push, &mut w, nb22);
    put_u(&mut push, &mut w, nb23);
    let prefix_len = kv.saturating_sub(n);
    put_f(&mut push, &mut w, params.scale);
    put_f(&mut push, &mut w, 0.0);
    put_f(&mut push, &mut w, 0.0);
    put_u(&mut push, &mut w, prefix_len); // mask_kv_offset: cols < this are visible prefix
    put_u(&mut push, &mut w, params.ring_depth); // ring_depth (repurposed ALiBi m0 slot)
    put_u(&mut push, &mut w, params.swa_window); // swa_window (repurposed ALiBi m1 slot)
    put_u(&mut push, &mut w, params.gqa_ratio);
    put_u(&mut push, &mut w, blocks_per_split); // split_kv (blocks per split)
    put_u(&mut push, &mut w, k_num);

    let spec_constants = vec![
        params.head_dim_k, // HSK
        params.head_dim_v, // HSV
        mask_enable,       // MASK_ENABLE (0 for the maskless vision tower)
        1,                 // CLAMP — KV need not be a multiple of Bc=64
    ];

    let key = PipelineKey {
        name: "flash_attn_cm1_f32_f16".to_string(),
        binding_indices: vec![0, 1, 2, 3, 4, 5],
        push_size: FA_PUSH_BYTES,
        spec_constants,
        required_subgroup_size: Some(32),
    };
    let pipeline = *ctx.pipelines.get(
        ctx.device,
        key,
        shaders::FLASH_ATTN_CM1_F32_F16_SPV.as_bytes(),
    )?;

    // Workgroup.x covers Br=16 query rows; .y packs (head, split); .z is batch.
    if k_num <= 1 {
        // Single pass → final normalized output to data_o (binding 5). Bind out
        // to the unused partials slot (binding 4) too for a valid descriptor.
        super::bind_and_dispatch(
            ctx,
            &pipeline,
            &[0, 1, 2, 3, 4, 5],
            &[
                q.range(),
                k.range(),
                v.range(),
                mask_buf.range(),
                out.range(),
                out.range(),
            ],
            &push,
            [n.div_ceil(16), ne2, ne3],
        )?;
        record_compute_barrier(ctx.device, ctx.cmd, out.range());
        return Ok(());
    }

    // Split-K → unnormalized partials (binding 4), merged by the reduce kernel.
    // Sized at the actual k_num (vision is a fresh forward, never decode-replay).
    let partials_floats =
        (params.head_dim_v as u64 + 2) * n as u64 * ne2 as u64 * ne3 as u64 * k_num as u64;
    let partials = ctx.alloc_tensor([partials_floats, 1, 1, 1], GgmlType::F32)?;
    super::bind_and_dispatch(
        ctx,
        &pipeline,
        &[0, 1, 2, 3, 4, 5],
        &[
            q.range(),
            k.range(),
            v.range(),
            mask_buf.range(),
            partials.range(),
            out.range(),
        ],
        &push,
        [n.div_ceil(16), ne2 * k_num, ne3],
    )?;
    record_compute_barrier(ctx.device, ctx.cmd, partials.range());
    record_split_k_combine(
        ctx,
        partials,
        out,
        params.head_dim_v,
        n,
        ne2,
        ne3,
        k_num,
        dyn_range,
    )?;
    Ok(())
}
