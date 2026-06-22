// Fused stochastic-sampler finalizer for the post-top_k path.
//
// Replaces the chain `top_p → min_p → scale(1/T) → softmax → cumsum
// → sub(uniform) → step → sum_rows → scale → clamp → cast →
// get_rows_i32` (12 separate dispatches across `record_top_p`,
// `record_min_p`, `record_scale`, and `record_categorical`) with a
// single workgroup that walks `K ≤ BLOCK_SIZE` candidates through
// all four filters in shared memory and writes the picked token id.
//
// Phase 1: filters that operate on RAW (pre-temperature) softmax
//   - min_p: drop logits below `max + log(min_p)`
//   - top_p: compute raw softmax cumsum, drop everything past the
//     index where the *exclusive* cdf first exceeds `top_p`
//     (matches llama.cpp's `last_idx = i + 1` convention).
// Phase 2: apply temperature, recompute softmax, inverse-CDF sample.
//
// Spec constants `TOP_P_ON` / `MIN_P_ON` / `TEMP_ON` let the host
// strip the per-filter math (and its barriers) when a filter is
// disabled — slang folds the `if (… != 0 && …)` predicates at
// pipeline-build time.

//@variants
//  f32: A_TYPE=float
//@end-variants

[SpecializationConstant] const uint BLOCK_SIZE = 256;
[SpecializationConstant] const uint TOP_P_ON   = 1;
[SpecializationConstant] const uint MIN_P_ON   = 1;
[SpecializationConstant] const uint TEMP_ON    = 1;

struct CatParams {
    uint  k;                 // candidate count (≤ BLOCK_SIZE)
    float inv_temp;          // 1 / temperature
    float top_p;             // 1.0 → keep all
    float log_min_p;         // ln(min_p); 0.0 host-side disables via spec const
    float _uniform_unused;   // historical slot; live value reads from DecodeDyn
};

[[vk::push_constant]] ConstantBuffer<CatParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float> data_logits;     // [K], sorted DESC
[[vk::binding(1, 0)]] StructuredBuffer<int>   data_candidates; // [K], token ids
[[vk::binding(2, 0)]] RWStructuredBuffer<int> data_out;        // [1], picked token id

// Per-forward dynamic params. The RNG uniform draw changes every
// decode token; reading it here lets the recorded cmdbuf replay with
// only a host-side write to DecodeDyn::uniform_rng between submits.
#include "decode_dyn.slang"
[[vk::binding(3, 0)]] StructuredBuffer<DecodeDyn> data_dyn;

groupshared float scan[BLOCK_SIZE]; // reused for prefix-scan (Hillis-Steele)
groupshared float red[BLOCK_SIZE];  // reused for sum reductions

static const float NEG_FLT_MAX = -3.402823466e+38f;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid    = gl_LocalInvocationID.x;
    const bool active = tid < p.k;

    // ── Load logit. Inputs are sorted DESC, so `max_logit = logits[0]`. ──
    const float logit     = active ? data_logits[tid] : NEG_FLT_MAX;
    const float max_logit = data_logits[0];
    const float centered  = logit - max_logit;

    // ── Filter state. `kept` starts True for valid tids and gets ANDed
    //    with each enabled filter's predicate. ─────────────────────────
    bool kept = active;

    // ── min_p (in log space, relative to max) ────────────────────────
    if (MIN_P_ON != 0) {
        kept = kept && (centered >= p.log_min_p);
    }

    // ── top_p ────────────────────────────────────────────────────────
    // Compute raw softmax (no temperature) and its prefix-cumsum, then
    // drop everything where the *exclusive* cdf already crosses top_p.
    if (TOP_P_ON != 0 && p.top_p < 1.0) {
        const float raw_exp = kept ? exp(centered) : 0.0;

        // sum over the workgroup (destroys `red`)
        red[tid] = raw_exp;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
            if (int(tid) < s) red[tid] += red[tid + uint(s)];
            GroupMemoryBarrierWithGroupSync();
        }
        const float sum_raw = red[0];
        GroupMemoryBarrierWithGroupSync();

        // Hillis-Steele inclusive prefix scan over `raw_exp / sum_raw`.
        scan[tid] = (sum_raw > 0.0) ? (raw_exp / sum_raw) : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (int d = 1; d < int(BLOCK_SIZE); d <<= 1) {
            const float v = (int(tid) >= d) ? scan[tid - uint(d)] : 0.0;
            GroupMemoryBarrierWithGroupSync();
            scan[tid] += v;
            GroupMemoryBarrierWithGroupSync();
        }

        const float incl_cdf = scan[tid];
        const float my_prob  = (sum_raw > 0.0) ? (raw_exp / sum_raw) : 0.0;
        const float excl_cdf = incl_cdf - my_prob;
        // `<=`: keep the token that *crosses* top_p, matching the host
        // `sample_one` / `record_top_p` contract and llama.cpp (drop only
        // tokens strictly past the cutoff).
        kept = kept && (excl_cdf <= p.top_p);
    }

    // ── Apply temperature. The argmax is preserved (sort order intact
    //    because temp > 0 scales linearly), so we softmax around
    //    `centered * inv_temp` directly. ───────────────────────────────
    const float x_t = (TEMP_ON != 0) ? (centered * p.inv_temp) : centered;
    const float my_exp = kept ? exp(x_t) : 0.0;

    // sum after temperature + all filters
    red[tid] = my_exp;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (int(tid) < s) red[tid] += red[tid + uint(s)];
        GroupMemoryBarrierWithGroupSync();
    }
    const float sum_kept = red[0];
    GroupMemoryBarrierWithGroupSync();

    // inclusive prefix-scan of `my_exp`
    scan[tid] = my_exp;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int d = 1; d < int(BLOCK_SIZE); d <<= 1) {
        const float v = (int(tid) >= d) ? scan[tid - uint(d)] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        scan[tid] += v;
        GroupMemoryBarrierWithGroupSync();
    }
    const float my_incl = scan[tid];
    const float my_excl = my_incl - my_exp;

    // Default: the very last kept candidate wins (defensive in case
    // floating-point drift puts target exactly at sum_kept). Lane 0
    // writes a stable fallback before any other thread tries to win;
    // it's strictly overwritten when the winner predicate fires.
    if (tid == 0) {
        const uint last_idx = (p.k > 0) ? (p.k - 1) : 0;
        data_out[0] = data_candidates[last_idx];
    }
    GroupMemoryBarrierWithGroupSync();

    // Inverse-CDF: find the smallest tid where `incl > target`, with
    // `excl <= target` confirming it's the bucket that owns `target`.
    const float target = data_dyn[0].uniform_rng * sum_kept;
    if (active && kept && my_incl > target && my_excl <= target) {
        data_out[0] = data_candidates[tid];
    }
}
