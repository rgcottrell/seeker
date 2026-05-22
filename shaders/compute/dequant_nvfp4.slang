// Ported from llama.cpp ggml-vulkan dequant_nvfp4.comp, MIT license — see LICENSE.llama-cpp
//
// NVFP4 dequantization. 64-element blocks split into four 16-element sub-blocks; each
// sub-block has its own UE4M3 7-bit scale. 4-bit nibble weights dereference the shared
// kvalues_mxfp4 lattice. 64 threads cooperate per block.

//@variants
//  nvfp4: DATA_A_NVFP4=1   D_TYPE=float    FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<block_nvfp4> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE>    data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID) {
    const uint i = gl_WorkGroupID.x * 4u + gl_LocalInvocationID.x / 64u;

    init_iq_shmem(gl_WorkGroupSize);

    const uint tid = gl_LocalInvocationID.x % 64u;
    const uint sub = tid / 16u;
    const uint ir  = tid % 16u;
    const uint ib  = 16u * i + ir;
    if (ib >= p.nel / 64u) { return; }

    const uint q_idx = 8u * sub;
    const uint b_idx = 1024u * i + 64u * ir + 16u * sub;

    const float d = ue4m3_to_fp32(data_a[ib].d[sub]);

    [unroll] for (uint l = 0; l < 8; ++l) {
        const uint qs = uint(data_a[ib].qs[q_idx + l]);
        data_b[b_idx + l + 0] = D_TYPE(d * 0.5f * float(kvalues_mxfp4[ qs        & 0xF]));
        data_b[b_idx + l + 8] = D_TYPE(d * 0.5f * float(kvalues_mxfp4[(qs >> 4)       ]));
    }
}
