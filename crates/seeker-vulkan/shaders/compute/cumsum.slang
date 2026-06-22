// Ported from llama.cpp ggml-vulkan cumsum.comp, MIT license — see LICENSE.llama-cpp
//
// Row-wise inclusive prefix sum using subgroup exclusive-add + workgroup-level shared partials.

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_SubgroupInvocationID, gl_NumWorkGroups

#include "types.slang"
#include "sum_rows_head.slang"

[SpecializationConstant] const uint BLOCK_SIZE      = 128;
[SpecializationConstant] const uint SUBGROUP_SIZE   = 32;
[SpecializationConstant] const uint ELEM_PER_THREAD = 4;

groupshared FLOAT_TYPE partial[BLOCK_SIZE / SUBGROUP_SIZE];
groupshared FLOAT_TYPE last_sum;

uint ceil_div(uint a, uint b) { return (a + b - 1) / b; }

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint row = gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;

    const uint i03        = fastdiv(row, p.ne0_12mp, p.ne0_12L);
    const uint i03_offset = i03 * p.ne01 * p.ne02;
    const uint i02        = fastdiv(row - i03_offset, p.ne0_1mp, p.ne0_1L);
    const uint i01        = row - i03_offset - i02 * p.ne01;

    const uint src_idx = get_aoffset() + i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03;
    const uint dst_idx = get_doffset() + i01 * p.nb11 + i02 * p.nb12 + i03 * p.nb13;

    const uint subgroup_id = tid / SUBGROUP_SIZE;

    if (tid == 0) { last_sum = FLOAT_TYPE(0); }

    uint col = tid * ELEM_PER_THREAD;
    const uint num_iter = ceil_div(p.n_cols, BLOCK_SIZE * ELEM_PER_THREAD);
    for (uint i = 0; i < num_iter; ++i) {
        FLOAT_TYPE v[ELEM_PER_THREAD];
        FLOAT_TYPE thread_sum = FLOAT_TYPE(0);
        [unroll] for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
            if (col + j < p.n_cols) {
                thread_sum += FLOAT_TYPE(data_a[src_idx + col + j]);
            }
            v[j] = thread_sum;
        }

        const FLOAT_TYPE excl = WavePrefixSum(thread_sum);
        [unroll] for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
            v[j] += excl;
        }
        // Last lane in each subgroup publishes the subgroup's partial total.
        if (gl_SubgroupInvocationID == SUBGROUP_SIZE - 1) {
            partial[subgroup_id] = v[ELEM_PER_THREAD - 1];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 0; s < subgroup_id; ++s) {
            [unroll] for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
                v[j] += partial[s];
            }
        }
        [unroll] for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
            v[j] += last_sum;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == BLOCK_SIZE - 1) {
            last_sum = v[ELEM_PER_THREAD - 1];
        }
        [unroll] for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
            if (col + j < p.n_cols) {
                data_d[dst_idx + col + j] = D_TYPE(v[j]);
            }
        }
        col += BLOCK_SIZE * ELEM_PER_THREAD;
    }
}
