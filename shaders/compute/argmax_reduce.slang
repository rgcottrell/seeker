// Second pass of two-pass argmax: reduce the per-workgroup partials
// emitted by `argmax_block_f32` into a single token id.
//
// Input: `[N, 2]` int2 pairs of `(index, asint(value))`.
// Output: 4 bytes = `int` (the winning index).
//
// One workgroup of `BLOCK_SIZE` threads strides through the partials
// (N is bounded — ~243 for n_vocab=248k with the BLOCK_SIZE=256,
// ELEMS_PER_THREAD=4 first pass), folds into a local (idx, value) max
// per thread, then runs a shared-memory tree reduction. Single
// dispatch; trivial cost (≪ 10 µs).

//@variants
//  f32: A_TYPE=float
//@end-variants

#include "generic_head.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 256;

[[vk::binding(0, 0)]] StructuredBuffer<int>      data_a;  // [N, 2] int2 pairs
[[vk::binding(1, 0)]] RWStructuredBuffer<int>    data_d;  // [1] final index

groupshared float tmpmax[BLOCK_SIZE];
groupshared uint  tmpidx[BLOCK_SIZE];

static const float NEG_FLT_MAX = -3.402823466e+38f;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid = gl_LocalInvocationID.x;
    // `p.KX` carries the partial count (N) — set by the host dispatcher.
    const uint n = p.KX;

    float lmax = NEG_FLT_MAX;
    uint  lidx = 0;
    for (uint i = tid; i < n; i += BLOCK_SIZE) {
        const uint  idx = uint(data_a[2 * i + 0]);
        const float val = asfloat(data_a[2 * i + 1]);
        if (val > lmax) {
            lmax = val;
            lidx = idx;
        }
    }
    tmpmax[tid] = lmax;
    tmpidx[tid] = lidx;
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (int(tid) < s) {
            if (tmpmax[tid] < tmpmax[tid + uint(s)]) {
                tmpmax[tid] = tmpmax[tid + uint(s)];
                tmpidx[tid] = tmpidx[tid + uint(s)];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        data_d[0] = int(tmpidx[0]);
    }
}
