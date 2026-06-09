// Wave-cooperative workgroup reductions.
//
// Replaces the log2(BLOCK_SIZE)-barrier shared-memory tree reduce with one
// WaveActiveSum/Max per wave, a single barrier, and a short serial fold of
// the per-wave partials (every thread folds, so the result is uniform across
// the workgroup with no extra broadcast barrier).
//
// CONTRACT: the dispatching pipeline must pin the subgroup width to 32
// (`PipelineKey::with_subgroup_size(32)` / `required_subgroup_size = Some(32)`)
// — the wave count is derived as block_size / 32. block_size must be a
// multiple of 32 and at most REDUCE_MAX_WAVES * 32 (= 512).
//
// Each call issues two barriers: the leading one protects the shared
// partials from a previous call's readers (back-to-back reductions in one
// kernel, e.g. soft_max's max-then-sum), the trailing one publishes them.

#ifndef REDUCE_FUNCS_SLANG
#define REDUCE_FUNCS_SLANG

static const uint REDUCE_MAX_WAVES = 16;

groupshared float  reduce_partials[REDUCE_MAX_WAVES];
groupshared float2 reduce_partials2[REDUCE_MAX_WAVES];

float workgroup_sum(float v, uint tid, uint block_size) {
    v = WaveActiveSum(v);
    GroupMemoryBarrierWithGroupSync();
    if ((tid & 31u) == 0u) { reduce_partials[tid / 32u] = v; }
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0f;
    for (uint i = 0; i < block_size / 32u; ++i) { total += reduce_partials[i]; }
    return total;
}

float2 workgroup_sum2(float2 v, uint tid, uint block_size) {
    v = float2(WaveActiveSum(v.x), WaveActiveSum(v.y));
    GroupMemoryBarrierWithGroupSync();
    if ((tid & 31u) == 0u) { reduce_partials2[tid / 32u] = v; }
    GroupMemoryBarrierWithGroupSync();
    float2 total = float2(0.0f, 0.0f);
    for (uint i = 0; i < block_size / 32u; ++i) { total += reduce_partials2[i]; }
    return total;
}

float workgroup_max(float v, uint tid, uint block_size) {
    v = WaveActiveMax(v);
    GroupMemoryBarrierWithGroupSync();
    if ((tid & 31u) == 0u) { reduce_partials[tid / 32u] = v; }
    GroupMemoryBarrierWithGroupSync();
    float total = reduce_partials[0];
    for (uint i = 1; i < block_size / 32u; ++i) { total = max(total, reduce_partials[i]); }
    return total;
}

#endif
