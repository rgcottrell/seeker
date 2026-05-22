// Ported from llama.cpp ggml-vulkan dequant_iq4_xs.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_IQ4_XS=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_IQ4_XS=1 D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_LocalInvocationIndex for init_iq_shmem

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    // Each thread handles 1 subblock (1 scale + 32 quantized values).
    const uint ib = gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x / 8;

    init_iq_shmem(uint3(256, 1, 1));

    if (ib >= p.nel / 256) { return; }

    const uint ib32 = gl_LocalInvocationID.x % 8;

    const float d = float(data_a[ib].d);
    // Scales are 6 bits: 4 from scales_l (nibble-packed) + 2 from scales_h (pairs).
    const uint scale = ((data_a[ib].scales_l[ib32 / 2] >> (4 * (ib32 & 1))) & 0xF)
                     | (((data_a[ib].scales_h >> (2 * ib32)) & 3) << 4);
    const float dl = d * (int(scale) - 32);

    const uint b_idx = 256 * ib + 32 * ib32;
    const uint q_idx = 16 * ib32;
    [unroll] for (uint l = 0; l < 16; ++l) {
        data_b[b_idx + l +  0] = D_TYPE(dl * float(kvalues_iq4nl[data_a[ib].qs[q_idx + l] & 0xF]));
        data_b[b_idx + l + 16] = D_TYPE(dl * float(kvalues_iq4nl[data_a[ib].qs[q_idx + l] >>  4]));
    }
}
