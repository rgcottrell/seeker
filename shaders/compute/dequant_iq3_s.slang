// Ported from llama.cpp ggml-vulkan dequant_iq3_s.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_IQ3_S=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_IQ3_S=1 D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_LocalInvocationIndex

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    // 8 threads per superblock; each handles 1 scale (1 nibble) × 32 weights.
    const uint ib = gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x / 8;

    init_iq_shmem(uint3(256, 1, 1));

    if (ib >= p.nel / 256) { return; }

    const uint is    = gl_LocalInvocationID.x % 8;
    const uint b_idx = 256 * ib + 32 * is;

    const float d  = float(data_a[ib].d);
    const float db = d * (1.0f + 2.0f * float((data_a[ib].scales[is / 2] >> (4 * (is % 2))) & 0xFu));

    // 4 sign bytes, 1 qh byte, 8 qs bytes → 32 output values.
    const uint qh = data_a[ib].qh[is];
    [unroll] for (uint l = 0; l < 8; ++l) {
        const uint iqs   = 8 * is + l;
        const uint qs    = data_a[ib].qs[iqs];
        // 9th-bit-of-grid-index from qh, shifted into bit 8.
        const uint gidx  = qs | ((qh << (8 - l)) & 256u);
        const uint signs = uint(data_a[ib].signs[iqs / 2]) >> (4 * (l & 1));
        const uint4 grid = unpack8_u(iq3s_grid[gidx]);

        data_b[b_idx + 4 * l + 0] = D_TYPE(db * float(grid.x) * ((signs & 1) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 4 * l + 1] = D_TYPE(db * float(grid.y) * ((signs & 2) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 4 * l + 2] = D_TYPE(db * float(grid.z) * ((signs & 4) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 4 * l + 3] = D_TYPE(db * float(grid.w) * ((signs & 8) != 0 ? -1.0f : 1.0f));
    }
}
