// Ported from llama.cpp ggml-vulkan mul_mat_vec_iq3_xxs.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  default: DATA_A_IQ3_XXS=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_NumWorkGroups, gl_LocalInvocationIndex

#include "types.slang"
#include "mul_mat_vec_head.slang"

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset, uint itid,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows) {
    const uint y_idx = i * QUANT_K + 16 * itid;
    const uint ib32  = itid / 2;  // 0..7

    uint ibi = a_offset + first_row * num_blocks_per_row + i;
    [unroll] for (uint n = 0; n < num_rows; ++n) {
        const float d = float(data_a[ibi].d);
        const uint signscale =
              uint(data_a_packed16[ibi].qs[QUANT_K / 8 + 2 * ib32    ])
            | (uint(data_a_packed16[ibi].qs[QUANT_K / 8 + 2 * ib32 + 1]) << 16);
        const float db = d * 0.5f * (0.5f + float(signscale >> 28));

        [unroll] for (uint l = 0; l < 2; ++l) {
            const uint qs0   = data_a[ibi].qs[8 * ib32 + 4 * (itid & 1) + 2 * l];
            const uint qs1   = data_a[ibi].qs[8 * ib32 + 4 * (itid & 1) + 2 * l + 1];
            const uint sign  = (signscale >> (7u * (2u * (itid & 1u) + l))) & 0x7Fu;
            const uint sign7 = countbits(sign);

            const float4 grid0 = float4(unpack8_u(iq3xxs_grid[qs0]));
            const float4 grid1 = float4(unpack8_u(iq3xxs_grid[qs1]));

            [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
                const float4 b0 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y_idx) / 4 + 2 * l + 0]);
                const float4 b4 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y_idx) / 4 + 2 * l + 1]);

                FLOAT_TYPE sum =
                      mad(FLOAT_TYPE(b0.x), FLOAT_TYPE((sign  &   1) != 0 ? -grid0.x : grid0.x),
                      mad(FLOAT_TYPE(b0.y), FLOAT_TYPE((sign  &   2) != 0 ? -grid0.y : grid0.y),
                      mad(FLOAT_TYPE(b0.z), FLOAT_TYPE((sign  &   4) != 0 ? -grid0.z : grid0.z),
                      mad(FLOAT_TYPE(b0.w), FLOAT_TYPE((sign  &   8) != 0 ? -grid0.w : grid0.w),
                      mad(FLOAT_TYPE(b4.x), FLOAT_TYPE((sign  &  16) != 0 ? -grid1.x : grid1.x),
                      mad(FLOAT_TYPE(b4.y), FLOAT_TYPE((sign  &  32) != 0 ? -grid1.y : grid1.y),
                      mad(FLOAT_TYPE(b4.z), FLOAT_TYPE((sign  &  64) != 0 ? -grid1.z : grid1.z),
                      mad(FLOAT_TYPE(b4.w), FLOAT_TYPE((sign7 &   1) != 0 ? -grid1.w : grid1.w),
                      FLOAT_TYPE(0.0)))))))));
                temp[j][n] = mad(db, sum, temp[j][n]);
            }
        }
        ibi += num_blocks_per_row;
    }
}

void compute_outputs(uint3 wg_id, uint tid_x, uint first_row, uint num_rows) {
    uint a_offset, b_offset, d_offset;
    get_offsets(wg_id, a_offset, b_offset, d_offset);

    const uint num_blocks_per_row = p.ncols / QUANT_K;

    const uint blocks_per_wg = BLOCK_SIZE / 16;
    const uint itid          = tid_x % 16;
    const uint ix            = tid_x / 16;

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
