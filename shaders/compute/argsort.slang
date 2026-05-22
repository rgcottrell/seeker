// Ported from llama.cpp ggml-vulkan argsort.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=int         FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize, gl_NumWorkGroups

#include "types.slang"

[SpecializationConstant] const int BLOCK_SIZE         = 1024;
[SpecializationConstant] const int NCOLS_PADDED_LOG2  = 10;

struct ArgsortParams {
    uint ncols;
    uint ncols_padded;
    uint ncols_padded_log2;
    uint nrows;
    uint order;            // 0 = ascending, 1 = descending
    uint outer_start;
    uint outer_end;
    uint inner_start;
    uint inner_end;
};

[[vk::push_constant]] ConstantBuffer<ArgsortParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>    data_a;
[[vk::binding(2, 0)]] RWStructuredBuffer<int>    data_d;

groupshared int2 dst_row[BLOCK_SIZE];

void argsort(bool needs_bounds_check, uint row, uint tid_x) {
    const int  col        = int(tid_x);
    const uint row_offset = row * p.ncols;

    // (column index, float bits as int) so we sort by value, keep index.
    dst_row[col] = int2(col, asint(data_a[row_offset + col]));
    GroupMemoryBarrierWithGroupSync();

    const uint num_outer_loop_iters = uint(NCOLS_PADDED_LOG2);
    [unroll] for (uint k = 2, outer_idx = 0; outer_idx < num_outer_loop_iters; k *= 2, outer_idx++) {
        const uint num_inner_loop_iters = outer_idx + 1;
        [unroll] for (uint j = k / 2, inner_idx = 0; inner_idx < num_inner_loop_iters; j /= 2, inner_idx++) {
            const int ixj   = int(uint(col) ^ j);

            const int idx_0 = (uint(col) & k) == 0 ? col : ixj;
            const int idx_1 = (uint(col) & k) == 0 ? ixj : col;

            const int2 sh0 = dst_row[idx_0];
            const int2 sh1 = dst_row[idx_1];
            const bool idx_0_oob = needs_bounds_check ? uint(sh0.x) >= p.ncols : false;
            const bool idx_1_oob = needs_bounds_check ? uint(sh1.x) >= p.ncols : false;

            if ((idx_0_oob ||
                (!idx_1_oob && asfloat(sh0.y) > asfloat(sh1.y))) && (ixj > col)) {
                dst_row[idx_0] = sh1;
                dst_row[idx_1] = sh0;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (uint(col) < p.ncols) {
        if (p.order == 0) {  // ASC
            data_d[row_offset + uint(col)] = dst_row[col].x;
        } else {
            data_d[row_offset + p.ncols - uint(col) - 1u] = dst_row[col].x;
        }
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    uint row = gl_WorkGroupID.y;
    const bool needs_bounds_check = uint(BLOCK_SIZE) != p.ncols;
    while (row < p.nrows) {
        argsort(needs_bounds_check, row, gl_LocalInvocationID.x);
        row += gl_WorkGroupSize.y * gl_NumWorkGroups.y;
    }
}
