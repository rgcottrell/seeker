// Ported from llama.cpp ggml-vulkan dequant_iq2_s.comp, MIT license — see LICENSE.llama-cpp
//
// IQ2_S dequantize. 256 elements per block; each of 32 sub-blocks (8 values) is decoded by
// one thread. The grid index is `qh<<8 | qs` (qh contributes 2 high bits per sub-block,
// shifted by 8-2*l). The sign byte comes from the second half of qs[].

//@variants
//  iq2_s: DATA_A_IQ2_S=1   D_TYPE=float   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<block_iq2_s>  data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE>     data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID) {
    const uint ib = gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x / 8u;
    init_iq_shmem(gl_WorkGroupSize);
    if (ib >= p.nel / 256u) { return; }

    const uint ib32  = gl_LocalInvocationID.x % 8u;
    const uint b_idx = 256u * ib + 32u * ib32;

    const float d      = float(data_a[ib].d);
    const uint  sc_raw = uint(data_a[ib].scales[ib32]);
    const float2 scale = float2(float(sc_raw & 0xFu), float(sc_raw >> 4));
    const float2 db    = d * (0.5f + scale) * 0.25f;

    const uint qh = uint(data_a[ib].qh[ib32]);
    [unroll] for (uint l = 0; l < 4; ++l) {
        uint qs = uint(data_a[ib].qs[4u * ib32 + l]);
        const uint sign = uint(data_a[ib].qs[QUANT_K_IQ2_S / 8u + 4u * ib32 + l]);
        qs |= (qh << (8u - 2u * l)) & 0x300u;
        const uint2 grid = iq2s_grid[qs];
        const float4 g0  = float4(unpack8_u(grid.x));
        const float4 g1  = float4(unpack8_u(grid.y));
        const float dbl  = db[l / 2];
        data_b[b_idx + 8u * l + 0] = D_TYPE(dbl * g0.x * ((sign &   1u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 1] = D_TYPE(dbl * g0.y * ((sign &   2u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 2] = D_TYPE(dbl * g0.z * ((sign &   4u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 3] = D_TYPE(dbl * g0.w * ((sign &   8u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 4] = D_TYPE(dbl * g1.x * ((sign &  16u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 5] = D_TYPE(dbl * g1.y * ((sign &  32u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 6] = D_TYPE(dbl * g1.z * ((sign &  64u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 7] = D_TYPE(dbl * g1.w * ((sign & 128u) != 0u ? -1.0f : 1.0f));
    }
}
