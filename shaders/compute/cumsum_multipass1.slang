// Ported from llama.cpp ggml-vulkan cumsum_multipass1.comp, MIT license — see LICENSE.llama-cpp
//
// Per-workgroup row scan that emits per-block partials to data_t for a second pass.

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_SubgroupInvocationID, gl_NumWorkGroups

#include "types.slang"
#include "sum_rows_head.slang"

[SpecializationConstant] const uint BLOCK_SIZE    = 128;
[SpecializationConstant] const uint SUBGROUP_SIZE = 32;

[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_t;

groupshared FLOAT_TYPE partial[BLOCK_SIZE / SUBGROUP_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID,
          uint3 gl_GlobalInvocationID  : SV_DispatchThreadID) {
    const uint row = gl_WorkGroupID.y;
    const uint tid = gl_LocalInvocationID.x;
    const uint col = gl_GlobalInvocationID.x;

    const uint i03        = fastdiv(row, p.ne0_12mp, p.ne0_12L);
    const uint i03_offset = i03 * p.ne01 * p.ne02;
    const uint i02        = fastdiv(row - i03_offset, p.ne0_1mp, p.ne0_1L);
    const uint i01        = row - i03_offset - i02 * p.ne01;

    const uint src_idx = get_aoffset() + i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03;
    const uint dst_idx = get_doffset() + i01 * p.nb11 + i02 * p.nb12 + i03 * p.nb13;

    const uint subgroup_id = tid / SUBGROUP_SIZE;

    FLOAT_TYPE v = FLOAT_TYPE(0);
    if (col < p.n_cols) {
        v = FLOAT_TYPE(data_a[src_idx + col]);
    }
    v = WavePrefixSum(v) + v;  // inclusive scan

    if (gl_SubgroupInvocationID == SUBGROUP_SIZE - 1) {
        partial[subgroup_id] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = 0; j < subgroup_id; ++j) {
        v += partial[j];
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == BLOCK_SIZE - 1) {
        data_t[gl_WorkGroupID.x + gl_NumWorkGroups.x * row] = D_TYPE(v);
    }
    if (col < p.n_cols) {
        data_d[dst_idx + col] = D_TYPE(v);
    }
}
