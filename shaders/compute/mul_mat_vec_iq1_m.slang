// Ported from llama.cpp ggml-vulkan mul_mat_vec_iq1_m.comp, MIT license — see LICENSE.llama-cpp
//
// IQ1_M row-times-vector. Block scale reconstructs from the high nibbles of the four 16-bit
// scales (unpacked as half2x16 low half); per-sub-block scales sit in the low 12 bits.

//@variants
//  default: DATA_A_IQ1_M=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"
#include "mul_mat_vec_head.slang"

int bitfield_extract_s2(int v, int offset) { return (v << (30 - offset)) >> 30; }

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset, uint ib32,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows) {
    const uint y_idx     = i * QUANT_K + 32 * ib32;
    const uint qh_base   = 2u * ib32;
    const uint qs_base   = 4u * ib32;
    const uint sc_index  = ib32 / 2u;
    const uint sc_shift  = 6u * (ib32 & 1u);

    uint ibi = a_offset + first_row * num_blocks_per_row + i;
    [unroll] for (uint n = 0; n < num_rows; ++n) {
        // Reconstruct f16 d from the four high nibbles.
        const uint s0 = uint(data_a[ibi].scales[0]) >> 12;
        const uint s1 = uint(data_a[ibi].scales[1]) >> 12;
        const uint s2 = uint(data_a[ibi].scales[2]) >> 12;
        const uint s3 = uint(data_a[ibi].scales[3]) >> 12;
        const float d = float(asfloat16(uint16_t(s0 | (s1 << 4) | (s2 << 8) | (s3 << 12))));
        const uint  sc = uint(data_a[ibi].scales[sc_index]) >> sc_shift;

        FLOAT_TYPE dl_cache[4];
        uint       gvf_cache[4];
        float      delta_cache[4];
        [unroll] for (uint l = 0; l < 4; ++l) {
            dl_cache[l] = FLOAT_TYPE(d * float(2u * ((sc >> (3u * (l / 2u))) & 7u) + 1u));
            const uint qh = uint(data_a[ibi].qh[qh_base + l / 2u]) >> (4u * (l & 1u));
            const uint qs = uint(data_a[ibi].qs[qs_base + l]);
            gvf_cache[l] = uint(iq1s_grid[qs | ((qh & 7u) << 8)]);
            delta_cache[l] = (qh & 8u) != 0u ? -IQ1M_DELTA : IQ1M_DELTA;
        }

        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            const uint base_b_idx = (j * p.batch_stride_b + b_offset + y_idx) / 4u;
            float4 b_vals[8];
            [unroll] for (uint idx = 0; idx < 8; ++idx) {
                b_vals[idx] = float4(data_b_v4[base_b_idx + idx]);
            }

            FLOAT_TYPE col_sum = FLOAT_TYPE(0);
            [unroll] for (uint l = 0; l < 4; ++l) {
                const int grid = int(gvf_cache[l]);
                const float dl = float(dl_cache[l]);
                const float4 fbits_v0 = float4(
                    float(bitfield_extract_s2(grid, 0)),
                    float(bitfield_extract_s2(grid, 2)),
                    float(bitfield_extract_s2(grid, 4)),
                    float(bitfield_extract_s2(grid, 6)));
                const float4 fbits_v1 = float4(
                    float(bitfield_extract_s2(grid,  8)),
                    float(bitfield_extract_s2(grid, 10)),
                    float(bitfield_extract_s2(grid, 12)),
                    float(bitfield_extract_s2(grid, 14)));
                const float4 delta_v = float4(delta_cache[l]);

                float4 sum_v = mad(b_vals[2u * l + 0u], fbits_v0 + delta_v, float4(0));
                       sum_v = mad(b_vals[2u * l + 1u], fbits_v1 + delta_v, sum_v);
                const FLOAT_TYPE sum = FLOAT_TYPE(sum_v.x + sum_v.y + sum_v.z + sum_v.w);
                col_sum = mad(FLOAT_TYPE(dl), sum, col_sum);
            }
            temp[j][n] += col_sum;
        }
        ibi += num_blocks_per_row;
    }
}

void compute_outputs(uint3 wg_id, uint tid_x, uint first_row, uint num_rows) {
    uint a_offset, b_offset, d_offset;
    get_offsets(wg_id, a_offset, b_offset, d_offset);

    const uint num_blocks_per_row = p.ncols / QUANT_K;
    const uint blocks_per_wg = BLOCK_SIZE / 8;
    const uint itid = tid_x % 8;
    const uint ix   = tid_x / 8;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint i = 0; i < NUM_ROWS; ++i) { temp[j][i] = FLOAT_TYPE(0); }
    }

    [unroll] for (uint i = ix; i < num_blocks_per_row; i += blocks_per_wg) {
        calc_superblock(temp, a_offset, b_offset, itid, i, num_blocks_per_row, first_row, num_rows);
    }

    reduce_result(temp, d_offset, first_row, num_rows, tid_x);
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);
    init_iq_shmem(uint3(BLOCK_SIZE, 1, 1));

    if (first_row + NUM_ROWS <= p.stride_d) {
        compute_outputs(gl_WorkGroupID, gl_LocalInvocationID.x, first_row, NUM_ROWS);
    } else {
        if (first_row >= p.stride_d) { return; }
        compute_outputs(gl_WorkGroupID, gl_LocalInvocationID.x, first_row, p.stride_d - first_row);
    }
}
