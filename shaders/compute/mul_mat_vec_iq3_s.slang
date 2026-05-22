// Ported from llama.cpp ggml-vulkan mul_mat_vec_iq3_s.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  default: DATA_A_IQ3_S=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_NumWorkGroups, gl_LocalInvocationIndex

#include "types.slang"
#include "mul_mat_vec_head.slang"

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset, uint ib32,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows) {
    const uint y_idx = i * QUANT_K + 32 * ib32;

    uint ibi = a_offset + first_row * num_blocks_per_row + i;
    [unroll] for (uint n = 0; n < num_rows; ++n) {
        const float d      = float(data_a[ibi].d);
        const uint  scale  = (data_a[ibi].scales[ib32 / 2] >> (4 * (ib32 & 1))) & 0xFu;
        const float dscale = d * (1.0f + 2.0f * float(scale));
        const uint  qh     = data_a[ibi].qh[ib32];

        FLOAT_TYPE sum[NUM_COLS];
        [unroll] for (uint j = 0; j < NUM_COLS; ++j) { sum[j] = FLOAT_TYPE(0); }

        [unroll] for (uint l = 0; l < 4; ++l) {
            // qs is two bytes from packed16; split with bit-shift (Slang has no unpack8 intrinsic).
            const uint qs_u = uint(data_a_packed16[ibi].qs[4 * ib32 + l]);
            const uint qs_x = qs_u & 0xFFu;
            const uint qs_y = (qs_u >> 8) & 0xFFu;
            const uint sign = data_a[ibi].signs[4 * ib32 + l];

            const float4 grid0 = float4(unpack8_u(iq3s_grid[qs_x | ((qh << (8 - 2 * l)) & 0x100u)]));
            const float4 grid1 = float4(unpack8_u(iq3s_grid[qs_y | ((qh << (7 - 2 * l)) & 0x100u)]));

            [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
                const float4 b0 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y_idx) / 4 + 2 * l + 0]);
                const float4 b4 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y_idx) / 4 + 2 * l + 1]);

                sum[j] =
                      mad(FLOAT_TYPE(b0.x), FLOAT_TYPE((sign &   1) != 0 ? -grid0.x : grid0.x),
                      mad(FLOAT_TYPE(b0.y), FLOAT_TYPE((sign &   2) != 0 ? -grid0.y : grid0.y),
                      mad(FLOAT_TYPE(b0.z), FLOAT_TYPE((sign &   4) != 0 ? -grid0.z : grid0.z),
                      mad(FLOAT_TYPE(b0.w), FLOAT_TYPE((sign &   8) != 0 ? -grid0.w : grid0.w),
                      mad(FLOAT_TYPE(b4.x), FLOAT_TYPE((sign &  16) != 0 ? -grid1.x : grid1.x),
                      mad(FLOAT_TYPE(b4.y), FLOAT_TYPE((sign &  32) != 0 ? -grid1.y : grid1.y),
                      mad(FLOAT_TYPE(b4.z), FLOAT_TYPE((sign &  64) != 0 ? -grid1.z : grid1.z),
                      mad(FLOAT_TYPE(b4.w), FLOAT_TYPE((sign & 128) != 0 ? -grid1.w : grid1.w),
                      sum[j]))))))));
            }
        }
        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            temp[j][n] = mad(dscale, sum[j], temp[j][n]);
        }
        ibi += num_blocks_per_row;
    }
}

void compute_outputs(uint3 wg_id, uint tid_x, uint first_row, uint num_rows) {
    uint a_offset, b_offset, d_offset;
    get_offsets(wg_id, a_offset, b_offset, d_offset);

    const uint num_blocks_per_row = p.ncols / QUANT_K;

    // 8 threads per block (vs 16 for most K-quants / IQ2-IQ4).
    const uint blocks_per_wg = BLOCK_SIZE / 8;
    const uint itid          = tid_x % 8;
    const uint ix            = tid_x / 8;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint i = 0; i < NUM_ROWS; ++i) {
            temp[j][i] = FLOAT_TYPE(0);
        }
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
