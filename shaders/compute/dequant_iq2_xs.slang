// Ported from llama.cpp ggml-vulkan dequant_iq2_xs.comp, MIT license — see LICENSE.llama-cpp
//
// IQ2_XS dequantize. 256 elements per block, 32 sub-blocks of 8 values each. Each thread
// handles one sub-block. The 16-bit qs entry encodes a 9-bit grid index and a 7-bit sign
// pattern; the 8th sign bit is the parity of the lower 7 (so the byte has even popcount).

//@variants
//  iq2_xs: DATA_A_IQ2_XS=1   D_TYPE=float   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<block_iq2_xs>  data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE>      data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID) {
    const uint ib = gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x / 8u;

    init_iq_shmem(gl_WorkGroupSize);

    if (ib >= p.nel / 256u) { return; }

    const uint ib32 = gl_LocalInvocationID.x % 8u;
    const uint b_idx = 256u * ib + 32u * ib32;

    const float d = float(data_a[ib].d);
    const uint sc_raw = uint(data_a[ib].scales[ib32]);
    const float2 scale = float2(float(sc_raw & 0xFu), float(sc_raw >> 4));
    const float2 db    = d * (0.5f + scale) * 0.25f;

    [unroll] for (uint l = 0; l < 4; ++l) {
        const uint qs    = uint(data_a[ib].qs[4u * ib32 + l]);
        const uint sign7 = qs >> 9;
        const uint sign8 = sign7 | (countbits(sign7) << 7);  // parity bit
        const uint2 grid = iq2xs_grid[qs & 511u];
        const float4 g0  = float4(unpack8_u(grid.x));
        const float4 g1  = float4(unpack8_u(grid.y));
        const float dbl  = db[l / 2];
        data_b[b_idx + 8u * l + 0] = D_TYPE(dbl * g0.x * ((sign8 &   1u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 1] = D_TYPE(dbl * g0.y * ((sign8 &   2u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 2] = D_TYPE(dbl * g0.z * ((sign8 &   4u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 3] = D_TYPE(dbl * g0.w * ((sign8 &   8u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 4] = D_TYPE(dbl * g1.x * ((sign8 &  16u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 5] = D_TYPE(dbl * g1.y * ((sign8 &  32u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 6] = D_TYPE(dbl * g1.z * ((sign8 &  64u) != 0u ? -1.0f : 1.0f));
        data_b[b_idx + 8u * l + 7] = D_TYPE(dbl * g1.w * ((sign8 & 128u) != 0u ? -1.0f : 1.0f));
    }
}
