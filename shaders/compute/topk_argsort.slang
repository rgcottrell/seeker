// Ported from llama.cpp ggml-vulkan topk_argsort.comp, MIT license — see LICENSE.llama-cpp
//
// Multi-pass top-k via bitonic sort. Pass-aliased bindings (a/s share binding 0; d/t share
// binding 1 in GLSL) become distinct (set, binding) slots here because Slang's strict
// capability check rejects overlap — host code binds the same VkBuffer to both slots.

//@variants
//  f32: A_TYPE=float       FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_NumWorkGroups, gl_WorkGroupSize

#include "types.slang"

[SpecializationConstant] const int BLOCK_SIZE        = 1024;
[SpecializationConstant] const int NCOLS_PADDED_LOG2 = 10;

struct TopkParams {
    uint orig_ncols;
    uint ncols_input;
    uint ncols_output;
    uint k;
    uint nrows;
    uint first_pass;
    uint last_pass;
};

[[vk::push_constant]] ConstantBuffer<TopkParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;     // input on first pass
[[vk::binding(1, 0)]] RWStructuredBuffer<int>    data_d;     // final dst on last pass
[[vk::binding(2, 0)]] StructuredBuffer<int2>     data_s;     // intermediate input
[[vk::binding(3, 0)]] RWStructuredBuffer<int2>   data_t;     // intermediate output

groupshared int2 dst_row[BLOCK_SIZE];

void topk(bool needs_bounds_check, uint row, uint3 lid, uint3 wgid, uint3 gid) {
    const int col = int(lid.x);

    if (gid.x < p.ncols_input) {
        if (p.first_pass != 0) {
            const uint row_offset = row * p.ncols_input;
            dst_row[col] = int2(int(gid.x), asint(data_a[row_offset + gid.x]));
        } else {
            const uint row_offset = row * p.ncols_input;
            dst_row[col] = data_s[row_offset + gid.x];
        }
    } else {
        dst_row[col] = int2(int(p.orig_ncols), 0);
    }
    GroupMemoryBarrierWithGroupSync();

    if (p.k == 1) {
        // Pure max-reduction shortcut when only one element is needed.
        [unroll] for (int s = BLOCK_SIZE / 2; s >= 1; s /= 2) {
            if (col < s) {
                int2 a = dst_row[col];
                int2 b = dst_row[col + s];
                if (uint(a.x) >= p.orig_ncols ||
                    (uint(b.x) < p.orig_ncols && b.y > a.y)) {
                    dst_row[col] = b;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    } else {
        const uint num_outer_loop_iters = uint(NCOLS_PADDED_LOG2);
        for (uint k = 2, outer_idx = 0; outer_idx < num_outer_loop_iters; k *= 2, outer_idx++) {
            const uint num_inner_loop_iters = outer_idx + 1;
            for (uint j = k / 2, inner_idx = 0; inner_idx < num_inner_loop_iters; j /= 2, inner_idx++) {
                const int ixj = int(uint(col) ^ j);

                const int idx_0 = (uint(col) & k) == 0 ? col : ixj;
                const int idx_1 = (uint(col) & k) == 0 ? ixj : col;

                const int2 sh0 = dst_row[idx_0];
                const int2 sh1 = dst_row[idx_1];
                const bool idx_0_oob = needs_bounds_check ? uint(sh0.x) >= p.orig_ncols : false;
                const bool idx_1_oob = needs_bounds_check ? uint(sh1.x) >= p.orig_ncols : false;

                if ((idx_0_oob ||
                    (!idx_1_oob && asfloat(sh0.y) < asfloat(sh1.y))) && (ixj > col)) {
                    dst_row[idx_0] = sh1;
                    dst_row[idx_1] = sh0;
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
    }

    if (uint(col) < p.k) {
        if (p.last_pass != 0) {
            if (gid.x < p.ncols_input) {
                const uint row_offset = row * p.k;
                data_d[row_offset + uint(col)] = dst_row[col].x;
            }
        } else {
            if (wgid.x * p.k + uint(col) < p.ncols_output) {
                const uint row_offset = row * p.ncols_output + wgid.x * p.k;
                data_t[row_offset + uint(col)] = dst_row[col];
            }
        }
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID,
          uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const bool need_bc = (p.ncols_input % BLOCK_SIZE) != 0;
    uint row = gl_WorkGroupID.y;
    while (row < p.nrows) {
        topk(need_bc, row, gl_LocalInvocationID, gl_WorkGroupID, gl_GlobalInvocationID);
        row += gl_WorkGroupSize.y * gl_NumWorkGroups.y;
    }
}
