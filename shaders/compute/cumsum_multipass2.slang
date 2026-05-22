// Ported from llama.cpp ggml-vulkan cumsum_multipass2.comp, MIT license — see LICENSE.llama-cpp
//
// Second pass: each workgroup adds the sum of all preceding-block partials into its block's
// output. Reads data_t (per-block partials) and accumulates into data_d.

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_SubgroupInvocationID, gl_NumWorkGroups, gl_SubgroupID

#include "types.slang"
#include "sum_rows_head.slang"

[SpecializationConstant] const uint BLOCK_SIZE    = 128;
[SpecializationConstant] const uint SUBGROUP_SIZE = 32;

[[vk::binding(2, 0)]] StructuredBuffer<D_TYPE> data_t;

groupshared FLOAT_TYPE temp[BLOCK_SIZE / SUBGROUP_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID,
          uint3 gl_GlobalInvocationID  : SV_DispatchThreadID) {
    const uint row = gl_WorkGroupID.y;
    const uint tid = gl_LocalInvocationID.x;

    const uint i03        = fastdiv(row, p.ne0_12mp, p.ne0_12L);
    const uint i03_offset = i03 * p.ne01 * p.ne02;
    const uint i02        = fastdiv(row - i03_offset, p.ne0_1mp, p.ne0_1L);
    const uint i01        = row - i03_offset - i02 * p.ne01;

    const uint src_idx = get_aoffset() + i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03;
    const uint dst_idx = get_doffset() + i01 * p.nb11 + i02 * p.nb12 + i03 * p.nb13;

    const uint col = gl_GlobalInvocationID.x;

    float v = 0.0f;
    if (col < p.n_cols) {
        v = float(data_d[dst_idx + col]);
    }

    // Each thread sums one stride of preceding-block partials, then subgroup-reduces.
    uint c = tid;
    float sum = 0.0f;
    while (c < gl_WorkGroupID.x) {
        sum += float(data_t[c + gl_NumWorkGroups.x * row]);
        c += BLOCK_SIZE;
    }

    sum = WaveActiveSum(sum);
    if (gl_SubgroupInvocationID == 0) {
        temp[gl_SubgroupID] = FLOAT_TYPE(sum);
    }
    GroupMemoryBarrierWithGroupSync();
    sum = 0.0f;
    [unroll] for (uint s = 0; s < BLOCK_SIZE / SUBGROUP_SIZE; ++s) {
        sum += float(temp[s]);
    }

    if (col < p.n_cols) {
        data_d[dst_idx + col] = D_TYPE(v + sum);
    }
}
