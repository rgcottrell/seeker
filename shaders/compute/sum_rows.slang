// Ported from llama.cpp ggml-vulkan sum_rows.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "sum_rows_head.slang"
#include "reduce_funcs.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 32;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint row = gl_WorkGroupID.z * 262144 + gl_WorkGroupID.y * 512 + gl_WorkGroupID.x;
    const uint col = gl_LocalInvocationID.x;
    const float weight = p.weight;

    const uint i03 = fastdiv(row, p.ne0_12mp, p.ne0_12L);
    const uint i03_offset = i03 * p.ne01 * p.ne02;
    const uint i02 = fastdiv(row - i03_offset, p.ne0_1mp, p.ne0_1L);
    const uint i01 = row - i03_offset - i02 * p.ne01;

    const uint src_idx = get_aoffset() + i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03;
    const uint dst_idx = get_doffset() + i01 * p.nb11 + i02 * p.nb12 + i03 * p.nb13;

    FLOAT_TYPE partial = FLOAT_TYPE(0.0);

    for (uint i = col; i < p.n_cols; i += BLOCK_SIZE) {
        partial += FLOAT_TYPE(data_a[src_idx + i]);
    }

    const FLOAT_TYPE total = workgroup_sum(partial, col, BLOCK_SIZE);

    if (col == 0) {
        data_d[dst_idx] = D_TYPE(total * weight);
    }
}
