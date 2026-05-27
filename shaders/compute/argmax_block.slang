// First pass of a two-pass argmax: each workgroup handles a slice of the
// input row and emits one `int2(global_index, asint(value))` pair to the
// partials buffer.
//
// The old `argmax.slang` ran a single 32-thread workgroup over the whole
// vocab — for n_vocab=248 320 that's 7,750 sequential strides on 32
// lanes, ~1/40th of a CU. This pass uses 256 threads × 4 elements per
// thread = 1024 elements per workgroup, so n_vocab=248k → ~243 WGs in
// flight on a 40-CU chip. Saturates the GPU; the reduction tail is one
// log₂(256) shared-memory tree walk per WG.
//
// Output layout matches `topk_argsort_f32`'s int2 pair convention so
// the reduce pass can be written with the same dequant-and-compare
// pattern.

//@variants
//  f32: A_TYPE=float
//@end-variants

#include "generic_head.slang"

[SpecializationConstant] const uint BLOCK_SIZE        = 256;
[SpecializationConstant] const uint ELEMS_PER_THREAD  = 4;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;    // [KX] logits (KX = n_vocab)
[[vk::binding(1, 0)]] RWStructuredBuffer<int>  data_d;    // [num_wg, 2] int2 pairs

groupshared float tmpmax[BLOCK_SIZE];
groupshared uint  tmpidx[BLOCK_SIZE];

static const float NEG_FLT_MAX = -3.402823466e+38f;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid = gl_LocalInvocationID.x;
    const uint wg  = gl_WorkGroupID.x;
    const uint elems_per_wg = BLOCK_SIZE * ELEMS_PER_THREAD;
    const uint base = wg * elems_per_wg;

    // Phase 1: each thread folds ELEMS_PER_THREAD values into a local max.
    float lmax = NEG_FLT_MAX;
    uint  lidx = base + tid;
    [unroll] for (uint k = 0; k < ELEMS_PER_THREAD; ++k) {
        const uint i = base + tid + k * BLOCK_SIZE;
        if (i < p.KX) {
            const float v = float(data_a[i]);
            if (v > lmax) {
                lmax = v;
                lidx = i;
            }
        }
    }
    tmpmax[tid] = lmax;
    tmpidx[tid] = lidx;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: shared-memory tree reduction.
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (int(tid) < s) {
            if (tmpmax[tid] < tmpmax[tid + uint(s)]) {
                tmpmax[tid] = tmpmax[tid + uint(s)];
                tmpidx[tid] = tmpidx[tid + uint(s)];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Lane 0 writes the (idx, value_bits) pair for this WG.
    if (tid == 0) {
        data_d[2 * wg + 0] = int(tmpidx[0]);
        data_d[2 * wg + 1] = asint(tmpmax[0]);
    }
}
