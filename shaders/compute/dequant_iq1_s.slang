// Ported from llama.cpp ggml-vulkan dequant_iq1_s.comp, MIT license — see LICENSE.llama-cpp
//
// IQ1_S dequant. The qh field packs 4 × 3-bit grid hi-bits in bits 0..11, a 3-bit scale in
// bits 12..14, and a sign at bit 15. Each grid entry is two 2-bit-coded values per nibble.

//@variants
//  iq1_s: DATA_A_IQ1_S=1   D_TYPE=float   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<block_iq1_s>  data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE>     data_b;

int bitfield_extract_s2(int v, int offset) {
    // Sign-extend a 2-bit field starting at `offset`.
    return ((v << (30 - offset)) >> 30);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID) {
    const uint ib = gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x / 8u;
    init_iq_shmem(gl_WorkGroupSize);
    if (ib >= p.nel / 256u) { return; }

    const uint ib32  = gl_LocalInvocationID.x % 8u;
    const uint b_idx = 256u * ib + 32u * ib32;

    const uint  qh    = uint(data_a[ib].qh[ib32]);
    const float d     = float(data_a[ib].d);
    const float dl    = d * float(2u * ((qh >> 12) & 7u) + 1u);
    const float delta = (qh & 0x8000u) != 0u ? -IQ1S_DELTA : IQ1S_DELTA;

    [unroll] for (uint l = 0; l < 4; ++l) {
        const uint qs    = uint(data_a[ib].qs[4u * ib32 + l]);
        const uint hi    = (qh >> (3u * l)) & 7u;
        const int  grid  = int(iq1s_grid[qs | (hi << 8)]);
        [unroll] for (int j = 0; j < 8; ++j) {
            const int v = bitfield_extract_s2(grid, 2 * j);
            data_b[b_idx + 8u * l + uint(j)] = D_TYPE(dl * (float(v) + delta));
        }
    }
}
