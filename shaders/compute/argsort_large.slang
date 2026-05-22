// Ported from llama.cpp ggml-vulkan argsort_large.comp, MIT license — see LICENSE.llama-cpp
//
// Out-of-shmem bitonic argsort for rows wider than a single workgroup can hold. The
// per-row (index, value-bits) pairs live in a workgroup-coherent buffer (tmp_idx) which
// stays resident across multiple kernel dispatches — each invocation runs one chunk of
// the outer/inner sort phases controlled by push constants.
//
// GLSL uses `controlBarrier(workgroup, workgroup, buffer, acquireRelease)` for the
// inter-invocation visibility; we use `DeviceMemoryBarrierWithGroupSync` which is
// equivalent in strength (device-scoped → workgroup-scoped is implied).

//@variants
//  f32: A_TYPE=float       FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_NumWorkGroups, gl_WorkGroupSize

#include "types.slang"

[SpecializationConstant] const int BLOCK_SIZE       = 1024;
[SpecializationConstant] const int WG_UNROLL_FACTOR = 2;
static const uint ASC = 0;

struct ArgsortLargeParams {
    uint ncols;
    uint ncols_padded;
    uint ncols_padded_log2;
    uint nrows;
    uint order;
    uint outer_start;
    uint outer_end;
    uint inner_start;
    uint inner_end;
};

[[vk::push_constant]] ConstantBuffer<ArgsortLargeParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<int2>   tmp_idx;
[[vk::binding(2, 0)]] RWStructuredBuffer<int>    data_d;

void argsort(bool needs_bounds_check, uint row, uint3 gid, uint3 lid) {
    int col0 = int(gid.x);
    col0 = (col0 % BLOCK_SIZE) + (col0 / BLOCK_SIZE) * BLOCK_SIZE * WG_UNROLL_FACTOR;

    const uint row_offset = row * p.ncols;
    const uint idx_offset = row * p.ncols_padded;

    bool need_barrier = false;

    if (p.outer_start == 0 && p.inner_start == 0) {
        [unroll] for (int u = 0; u < WG_UNROLL_FACTOR; ++u) {
            uint c = uint(u) * BLOCK_SIZE + uint(col0);
            if (c < p.ncols_padded) {
                int2 v = int2(int(c), asint(data_a[row_offset + c]));
                tmp_idx[idx_offset + c] = v;
            }
        }
        need_barrier = true;
    }

    [unroll] for (uint outer_idx = p.outer_start, k = (2u << p.outer_start); outer_idx < p.outer_end; k *= 2u, outer_idx++) {
        const uint inner_end = min(p.inner_end, outer_idx + 1u);
        for (uint j = k >> (p.inner_start + 1u), inner_idx = p.inner_start; inner_idx < inner_end; j /= 2u, inner_idx++) {
            if (need_barrier) {
                DeviceMemoryBarrierWithGroupSync();
            }
            need_barrier = true;
            [unroll] for (int u = 0; u < WG_UNROLL_FACTOR; ++u) {
                int c = u * BLOCK_SIZE + col0;
                const int ixj = int(uint(c) ^ j);
                if (ixj < c) { continue; }

                const int idx_0 = (uint(c) & k) == 0u ? c : ixj;
                const int idx_1 = (uint(c) & k) == 0u ? ixj : c;

                const int2 sh0 = tmp_idx[idx_offset + uint(idx_0)];
                const int2 sh1 = tmp_idx[idx_offset + uint(idx_1)];
                const bool idx_0_oob = needs_bounds_check ? uint(sh0.x) >= p.ncols : false;
                const bool idx_1_oob = needs_bounds_check ? uint(sh1.x) >= p.ncols : false;

                if (idx_0_oob ||
                    (!idx_1_oob && asfloat(sh0.y) > asfloat(sh1.y))) {
                    tmp_idx[idx_offset + uint(idx_0)] = sh1;
                    tmp_idx[idx_offset + uint(idx_1)] = sh0;
                }
            }
        }
    }

    if (p.outer_end == p.ncols_padded_log2 &&
        p.inner_end >= p.ncols_padded_log2 + 1u) {
        DeviceMemoryBarrierWithGroupSync();
        [unroll] for (int u = 0; u < WG_UNROLL_FACTOR; ++u) {
            uint c = uint(u) * BLOCK_SIZE + uint(col0);
            if (c < p.ncols) {
                if (p.order == ASC) {
                    data_d[row_offset + c] = tmp_idx[idx_offset + c].x;
                } else {
                    data_d[row_offset + p.ncols - c - 1u] = tmp_idx[idx_offset + c].x;
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_GlobalInvocationID  : SV_DispatchThreadID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID) {
    const bool needs_bc = p.ncols != p.ncols_padded;
    uint row = gl_WorkGroupID.y;
    while (row < p.nrows) {
        argsort(needs_bc, row, gl_GlobalInvocationID, gl_LocalInvocationID);
        row += gl_WorkGroupSize.y * gl_NumWorkGroups.y;
    }
}
