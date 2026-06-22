// Ported from llama.cpp ggml-vulkan mul_mat_vec_iq1_s.comp, MIT license — see LICENSE.llama-cpp
//
// IQ1_S row-times-vector. 8 threads cooperate per 256-element block. Each thread owns one
// sub-block of 32 weights; the grid is 2-bit packed signed values + IQ1S_DELTA bias.

//@variants
//  default: DATA_A_IQ1_S=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"
#include "mul_mat_vec_head.slang"

int bitfield_extract_s2(int v, int offset) { return (v << (30 - offset)) >> 30; }

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset, uint ib32,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows) {
    const uint y_idx_base = i * QUANT_K + 32 * ib32;
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        const uint base_b_idx = (j * p.batch_stride_b + b_offset + y_idx_base) / 4u;
        [unroll] for (uint l = 0; l < 4; ++l) {
            const float4 b_val_0 = float4(data_b_v4[base_b_idx + 2u * l       ]);
            const float4 b_val_1 = float4(data_b_v4[base_b_idx + 2u * l + 1u  ]);

            uint ibi = a_offset + first_row * num_blocks_per_row + i;
            [unroll] for (uint n = 0; n < num_rows; ++n) {
                const float d  = float(data_a[ibi].d);
                const uint  qh = uint(data_a[ibi].qh[ib32]);
                const float dl = d * float(2u * ((qh >> 12) & 7u) + 1u);

                const uint qs    = uint(data_a[ibi].qs[4u * ib32 + l]);
                const uint idxhi = (qh >> (3u * l)) & 7u;
                const int  grid  = int(iq1s_grid[qs | (idxhi << 8)]);

                const float delta_val = (qh & 0x8000u) != 0u ? -IQ1S_DELTA : IQ1S_DELTA;
                const float4 fbits0 = float4(
                    float(bitfield_extract_s2(grid, 0)),
                    float(bitfield_extract_s2(grid, 2)),
                    float(bitfield_extract_s2(grid, 4)),
                    float(bitfield_extract_s2(grid, 6)));
                const float4 fbits1 = float4(
                    float(bitfield_extract_s2(grid,  8)),
                    float(bitfield_extract_s2(grid, 10)),
                    float(bitfield_extract_s2(grid, 12)),
                    float(bitfield_extract_s2(grid, 14)));

                float4 sum_v = mad(b_val_0, fbits0 + float4(delta_val), float4(0));
                       sum_v = mad(b_val_1, fbits1 + float4(delta_val), sum_v);
                const FLOAT_TYPE sum = FLOAT_TYPE(dot(sum_v, float4(1.0f)));

                temp[j][n] = mad(FLOAT_TYPE(dl), sum, temp[j][n]);
                ibi += num_blocks_per_row;
            }
        }
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
