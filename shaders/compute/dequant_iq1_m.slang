// Ported from llama.cpp ggml-vulkan dequant_iq1_m.comp, MIT license — see LICENSE.llama-cpp
//
// IQ1_M dequant. Block-level `d` is reconstructed from the high nibbles of the four 16-bit
// scales (unpacked as a half2x16 low half). Per-sub-block scales sit in the low 12 bits.

//@variants
//  iq1_m: DATA_A_IQ1_M=1   D_TYPE=float   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<block_iq1_m>  data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE>     data_b;

int bitfield_extract_s2(int v, int offset) {
    return (v << (30 - offset)) >> 30;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID) {
    const uint ib = gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x / 8u;
    init_iq_shmem(gl_WorkGroupSize);
    if (ib >= p.nel / 256u) { return; }

    const uint ib32  = gl_LocalInvocationID.x % 8u;
    const uint ib64  = ib32 / 2u;
    const uint b_idx = 256u * ib + 32u * ib32;

    // Reconstruct the f16 scale from the high nibbles of scales[0..3].
    const uint s0 = uint(data_a[ib].scales[0]) >> 12;
    const uint s1 = uint(data_a[ib].scales[1]) >> 12;
    const uint s2 = uint(data_a[ib].scales[2]) >> 12;
    const uint s3 = uint(data_a[ib].scales[3]) >> 12;
    const uint packed = s0 | (s1 << 4) | (s2 << 8) | (s3 << 12);
    const float d = float(asfloat16(uint16_t(packed)));

    const uint sc = uint(data_a[ib].scales[ib64]);
    [unroll] for (uint l = 0; l < 4; ++l) {
        const uint ib16 = 2u * ib32 + l / 2u;
        const float dl  = d * float(2u * ((sc >> (3u * (ib16 & 3u))) & 7u) + 1u);
        const uint qh   = uint(data_a[ib].qh[ib16]) >> (4u * (l & 1u));
        const uint qs   = uint(data_a[ib].qs[4u * ib32 + l]);
        const float delta = (qh & 8u) != 0u ? -IQ1M_DELTA : IQ1M_DELTA;
        const int grid    = int(iq1s_grid[qs | ((qh & 7u) << 8)]);
        [unroll] for (int j = 0; j < 8; ++j) {
            const int v = bitfield_extract_s2(grid, 2 * j);
            data_b[b_idx + 8u * l + uint(j)] = D_TYPE(dl * (float(v) + delta));
        }
    }
}
