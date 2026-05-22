// Ported from llama.cpp ggml-vulkan dequant_iq3_xxs.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_IQ3_XXS=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_IQ3_XXS=1 D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_LocalInvocationIndex

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    // 8 threads handle 1 superblock; each handles 1 scale block (32 weights).
    const uint ib = gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x / 8;

    init_iq_shmem(uint3(256, 1, 1));

    if (ib >= p.nel / 256) { return; }

    const uint is    = gl_LocalInvocationID.x % 8;
    const uint b_idx = 256 * ib + 32 * is;
    const uint s_idx = QUANT_K / 4 + 4 * is;  // sign/scale bytes begin after the 64 quant bytes

    const float d = float(data_a[ib].d);
    const uint signscale =
          uint(data_a[ib].qs[s_idx + 0])
        | (uint(data_a[ib].qs[s_idx + 1]) <<  8)
        | (uint(data_a[ib].qs[s_idx + 2]) << 16)
        | (uint(data_a[ib].qs[s_idx + 3]) << 24);
    const float db = d * (0.5f + float(signscale >> 28)) * 0.5f;

    [unroll] for (uint l = 0; l < 4; ++l) {
        const uint sign7 = (signscale >> (7u * l)) & 0x7Fu;
        const uint sign8 = sign7 | (countbits(sign7) << 7);
        const uint qs0 = data_a[ib].qs[8 * is + 2 * l];
        const uint qs1 = data_a[ib].qs[8 * is + 2 * l + 1];
        const uint4 grid0 = unpack8_u(iq3xxs_grid[qs0]);
        const uint4 grid1 = unpack8_u(iq3xxs_grid[qs1]);

        data_b[b_idx + 8 * l + 0] = D_TYPE(db * float(grid0.x) * ((sign8 &   1) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 8 * l + 1] = D_TYPE(db * float(grid0.y) * ((sign8 &   2) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 8 * l + 2] = D_TYPE(db * float(grid0.z) * ((sign8 &   4) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 8 * l + 3] = D_TYPE(db * float(grid0.w) * ((sign8 &   8) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 8 * l + 4] = D_TYPE(db * float(grid1.x) * ((sign8 &  16) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 8 * l + 5] = D_TYPE(db * float(grid1.y) * ((sign8 &  32) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 8 * l + 6] = D_TYPE(db * float(grid1.z) * ((sign8 &  64) != 0 ? -1.0f : 1.0f));
        data_b[b_idx + 8 * l + 7] = D_TYPE(db * float(grid1.w) * ((sign8 & 128) != 0 ? -1.0f : 1.0f));
    }
}
