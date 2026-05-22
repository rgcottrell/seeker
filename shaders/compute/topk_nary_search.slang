// Ported from llama.cpp ggml-vulkan topk_nary_search.comp, MIT license — see LICENSE.llama-cpp
//
// Two-pass top-K using an N-ary search over float bits remapped to comparable uints. The
// inner loop counts elements per bucket, finds the bucket containing the K-th value, then
// recurses. Subgroup ballots collapse the per-lane decisions to compact stores.
//
// Aliased GLSL bindings A/S on slot 0 and D/T on slot 1 become distinct slots (0/2, 1/3)
// here — host wires the same VkBuffer to both slots per role.

//@variants
//  f32: A_TYPE=float       FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_NumWorkGroups, gl_WorkGroupSize

#include "types.slang"

[SpecializationConstant] const int BLOCK_SIZE        = 1024;
[SpecializationConstant] const int SUBGROUP_SIZE     = 32;
[SpecializationConstant] const int SUBGROUP_SIZE_LOG2 = 5;

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

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<int>    data_d;
[[vk::binding(2, 0)]] StructuredBuffer<int2>     data_s;
[[vk::binding(3, 0)]] RWStructuredBuffer<int2>   data_t;

groupshared int2 dst_row[BLOCK_SIZE];
groupshared int  counts[SUBGROUP_SIZE];
groupshared int  sh_min_idx;
groupshared uint sh_total;
groupshared uint offset_partials[BLOCK_SIZE / SUBGROUP_SIZE];
groupshared uint eq_min_partials[BLOCK_SIZE / SUBGROUP_SIZE];

// Remap a float to a uint such that unsigned comparison matches float order.
// +0.0 → 0x80000000, large positives → larger, negatives → inverted bit pattern.
uint f2ui(float x) {
    uint y = asuint(x);
    if ((y & 0x80000000u) != 0u) { y ^= 0xFFFFFFFFu; }
    else                         { y |= 0x80000000u; }
    return y;
}

void topk(uint row, uint3 lid, uint3 wgid, uint3 gid) {
    const int tid = int(lid.x);

    if (gid.x < p.ncols_input) {
        if (p.first_pass != 0) {
            const uint row_offset = row * p.ncols_input;
            dst_row[tid] = int2(int(gid.x), asint(data_a[row_offset + gid.x]));
        } else {
            const uint row_offset = row * p.ncols_input;
            dst_row[tid] = data_s[row_offset + gid.x];
        }
    } else {
        dst_row[tid] = int2(int(p.orig_ncols), int(0xFF800000u));  // -inf
    }
    GroupMemoryBarrierWithGroupSync();

    if (p.k == 1) {
        // Single output → straight max reduction.
        [unroll] for (int s = BLOCK_SIZE / 2; s >= 1; s /= 2) {
            if (tid < s) {
                int2 a = dst_row[tid];
                int2 b = dst_row[tid + s];
                if (uint(a.x) >= p.orig_ncols ||
                    (uint(b.x) < p.orig_ncols && b.y > a.y)) {
                    dst_row[tid] = b;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    } else {
        int shift = 32 - SUBGROUP_SIZE_LOG2;
        uint mask = ((1u << SUBGROUP_SIZE_LOG2) - 1u) << shift;

        uint range_min = 0u;
        uint range_max = 0xFF800000u;
        uint total = 0u;
        const uint limit = min(p.k, p.ncols_input - wgid.x * BLOCK_SIZE);

        while (mask != 0u) {
            GroupMemoryBarrierWithGroupSync();
            if (tid < SUBGROUP_SIZE) { counts[tid] = 0; }
            GroupMemoryBarrierWithGroupSync();

            if (uint(tid) < p.ncols_input) {
                const float y = asfloat(dst_row[tid].y);
                const uint fy = f2ui(y);
                if (fy >= range_min && fy < range_max) {
                    const uint bucket = (fy & mask) >> shift;
                    int dummy;
                    InterlockedAdd(counts[bucket], 1, dummy);
                }
            }
            GroupMemoryBarrierWithGroupSync();

            // Top-down inclusive scan within the first subgroup; find the leftmost
            // top-N prefix whose running total reaches `limit`.
            if (tid < SUBGROUP_SIZE) {
                uint partial_sum = uint(counts[SUBGROUP_SIZE - 1 - tid]);
                partial_sum = WavePrefixSum(partial_sum) + partial_sum + total;
                const uint4 ballot = WaveActiveBallot(partial_sum >= limit);
                const uint t = firstbitlow(ballot.x);
                if (uint(tid) == t) {
                    sh_min_idx = int(SUBGROUP_SIZE - 1 - int(t));
                    sh_total   = partial_sum;
                }
            }
            GroupMemoryBarrierWithGroupSync();
            const int min_idx = sh_min_idx;
            total = sh_total;

            range_max = range_min + (uint(min_idx + 1) << shift);
            range_min = range_min + (uint(min_idx) << shift);

            if (total == p.k) { break; }
            total -= uint(counts[min_idx]);
            mask >>= SUBGROUP_SIZE_LOG2;
            shift -= SUBGROUP_SIZE_LOG2;
            if (shift < 0) { shift = 0; }
        }

        const int2 v = dst_row[tid];

        if (total == p.k) {
            const bool top = f2ui(asfloat(v.y)) >= range_min;
            const uint4 b = WaveActiveBallot(top);
            const uint bit_count = countbits(b.x) + countbits(b.y) + countbits(b.z) + countbits(b.w);
            if ((tid % SUBGROUP_SIZE) == 0) {
                offset_partials[tid / SUBGROUP_SIZE] = bit_count;
            }
            GroupMemoryBarrierWithGroupSync();

            uint out_idx = 0u;
            [unroll] for (int i = 0; i < BLOCK_SIZE / SUBGROUP_SIZE; ++i) {
                if (i < tid / SUBGROUP_SIZE) { out_idx += offset_partials[i]; }
            }

            const uint bit_count_ex = WavePrefixCountBits(top);
            if (top) { dst_row[out_idx + bit_count_ex] = v; }
        } else {
            const bool top    = f2ui(asfloat(v.y)) >  range_min;
            const bool eq_min = f2ui(asfloat(v.y)) == range_min;
            const uint4 b_top    = WaveActiveBallot(top);
            const uint4 b_eq_min = WaveActiveBallot(eq_min);
            const uint bit_count_top    = countbits(b_top.x)    + countbits(b_top.y)    + countbits(b_top.z)    + countbits(b_top.w);
            const uint bit_count_eq_min = countbits(b_eq_min.x) + countbits(b_eq_min.y) + countbits(b_eq_min.z) + countbits(b_eq_min.w);
            if ((tid % SUBGROUP_SIZE) == 0) {
                offset_partials[tid / SUBGROUP_SIZE] = bit_count_top;
                eq_min_partials[tid / SUBGROUP_SIZE] = bit_count_eq_min;
            }
            GroupMemoryBarrierWithGroupSync();

            uint out_idx     = 0u;
            uint eq_min_base = 0u;
            uint eq_min_idx  = 0u;
            [unroll] for (int i = 0; i < BLOCK_SIZE / SUBGROUP_SIZE; ++i) {
                if (i < tid / SUBGROUP_SIZE) {
                    out_idx    += offset_partials[i];
                    eq_min_idx += eq_min_partials[i];
                }
                eq_min_base += offset_partials[i];
            }
            eq_min_idx += eq_min_base;

            const uint bit_count_ex_top    = WavePrefixCountBits(top);
            const uint bit_count_ex_eq_min = WavePrefixCountBits(eq_min);
            if (top) {
                dst_row[out_idx + bit_count_ex_top] = v;
            }
            if (eq_min && eq_min_idx + bit_count_ex_eq_min < p.k) {
                dst_row[eq_min_idx + bit_count_ex_eq_min] = v;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (uint(tid) < p.k) {
        if (p.last_pass != 0) {
            if (gid.x < p.ncols_input) {
                const uint row_offset = row * p.k;
                data_d[row_offset + uint(tid)] = dst_row[tid].x;
            }
        } else {
            if (wgid.x * p.k + uint(tid) < p.ncols_output) {
                const uint row_offset = row * p.ncols_output + wgid.x * p.k;
                data_t[row_offset + uint(tid)] = dst_row[tid];
            }
        }
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID        : SV_GroupID,
          uint3 gl_LocalInvocationID  : SV_GroupThreadID,
          uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint row = gl_WorkGroupID.y;
    while (row < p.nrows) {
        topk(row, gl_LocalInvocationID, gl_WorkGroupID, gl_GlobalInvocationID);
        row += gl_WorkGroupSize.y * gl_NumWorkGroups.y;
    }
}
