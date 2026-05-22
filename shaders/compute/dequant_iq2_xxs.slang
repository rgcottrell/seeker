// Ported from llama.cpp ggml-vulkan dequant_iq2_xxs.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_IQ2_XXS=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_IQ2_XXS=1 D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_LocalInvocationIndex for init_iq_shmem

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    // Each thread handles 1 scale-block (32 weights). A whole block_iq2_xxs
    // carries 8 scale-blocks, so 8 threads per block_iq2_xxs.
    const uint ib = gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x / 8;

    init_iq_shmem(uint3(256, 1, 1));

    if (ib >= p.nel / 256) { return; }

    const uint is    = gl_LocalInvocationID.x % 8;
    const uint b_idx = 256 * ib + 32 * is;

    const float d = float(data_a[ib].d);

    // signscale packs 4 sign-bytes + a 4-bit scale: bytes [4..7] of the
    // per-scale-block sub-window of qs, packed into a uint32 little-endian.
    const uint signscale =
          uint(data_a[ib].qs[8 * is + 4])
        | (uint(data_a[ib].qs[8 * is + 5]) <<  8)
        | (uint(data_a[ib].qs[8 * is + 6]) << 16)
        | (uint(data_a[ib].qs[8 * is + 7]) << 24);
    const float db = d * (0.5f + float(signscale >> 28)) * 0.25f;

    [unroll] for (uint l = 0; l < 4; ++l) {
        // GLSL: bitfieldExtract(signscale, 7 * int(l), 7) — extract 7 bits.
        const uint sign7 = (signscale >> (7u * l)) & 0x7Fu;
        // The 8th sign bit is the parity of the low 7 bits.
        const uint sign8 = sign7 | (countbits(sign7) << 7);

        const uint qs   = data_a[ib].qs[8 * is + l];
        const uint2 grid = iq2xxs_grid[qs];
        const uint4 grid0 = unpack8_u(grid.x);
        const uint4 grid1 = unpack8_u(grid.y);

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
