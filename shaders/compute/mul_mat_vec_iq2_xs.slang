// Ported from llama.cpp ggml-vulkan mul_mat_vec_iq2_xs.comp, MIT license — see LICENSE.llama-cpp
//
// IQ2_XS row-times-vector. Per block of 256 elements, 16 threads coop on the dot product.
// Each thread fetches one uint16 qs (9-bit grid index + 7-bit signs, parity bit appended
// at runtime) and one nibble of the per-32-element scale, weighting two grid tetrads from
// `b`.

//@variants
//  default: DATA_A_IQ2_XS=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"
#include "mul_mat_vec_head.slang"

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset, uint itid,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows) {
    const uint y_idx         = i * QUANT_K + 16 * itid;
    const uint nibble_shift  = 4u * (itid & 1u);
    const uint ib32          = itid / 2u;  // 0..7

    uint ibi = a_offset + first_row * num_blocks_per_row + i;

    // db scale per (block, row) — depends only on the high/low nibble of scales[ib32].
    float db_vals[NUM_ROWS];
    [unroll] for (uint n = 0; n < num_rows; ++n) {
        const float d   = float(data_a[ibi].d);
        const uint  sc  = uint(data_a[ibi].scales[ib32]);
        const uint  scn = (sc >> nibble_shift) & 0xFu;
        db_vals[n] = d * (0.125f + float(scn) * 0.25f);
        ibi += num_blocks_per_row;
    }

    ibi = a_offset + first_row * num_blocks_per_row + i;
    [unroll] for (uint n = 0; n < num_rows; ++n) {
        float4 grid0_vals[2], grid1_vals[2];
        uint   sign_vals[2], sign7_vals[2];
        [unroll] for (uint l = 0; l < 2; ++l) {
            const uint qs     = uint(data_a[ibi].qs[2u * itid + l]);
            sign_vals[l]  = qs >> 9;
            sign7_vals[l] = countbits(sign_vals[l]);
            const uint2 g = iq2xs_grid[qs & 511u];
            grid0_vals[l] = float4(unpack8_u(g.x));
            grid1_vals[l] = float4(unpack8_u(g.y));
        }

        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            FLOAT_TYPE sum = FLOAT_TYPE(0);
            [unroll] for (uint l = 0; l < 2; ++l) {
                const uint sign  = sign_vals[l];
                const uint sign7 = sign7_vals[l];
                const float4 g0  = grid0_vals[l];
                const float4 g1  = grid1_vals[l];

                const uint b_idx = (j * p.batch_stride_b + b_offset + y_idx) / 4u + 2u * l;
                const float4 b0  = float4(data_b_v4[b_idx + 0u]);
                const float4 b4  = float4(data_b_v4[b_idx + 1u]);

                sum +=
                    mad(FLOAT_TYPE(b0.x), FLOAT_TYPE((sign  &   1u) != 0u ? -g0.x : g0.x),
                    mad(FLOAT_TYPE(b0.y), FLOAT_TYPE((sign  &   2u) != 0u ? -g0.y : g0.y),
                    mad(FLOAT_TYPE(b0.z), FLOAT_TYPE((sign  &   4u) != 0u ? -g0.z : g0.z),
                    mad(FLOAT_TYPE(b0.w), FLOAT_TYPE((sign  &   8u) != 0u ? -g0.w : g0.w),
                    mad(FLOAT_TYPE(b4.x), FLOAT_TYPE((sign  &  16u) != 0u ? -g1.x : g1.x),
                    mad(FLOAT_TYPE(b4.y), FLOAT_TYPE((sign  &  32u) != 0u ? -g1.y : g1.y),
                    mad(FLOAT_TYPE(b4.z), FLOAT_TYPE((sign  &  64u) != 0u ? -g1.z : g1.z),
                    mad(FLOAT_TYPE(b4.w), FLOAT_TYPE((sign7 &   1u) != 0u ? -g1.w : g1.w),
                    FLOAT_TYPE(0.0)))))))));
            }
            temp[j][n] = mad(FLOAT_TYPE(db_vals[n]), sum, temp[j][n]);
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
