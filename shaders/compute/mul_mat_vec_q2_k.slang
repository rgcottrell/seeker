// Ported from llama.cpp ggml-vulkan mul_mat_vec_q2_k.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  default: DATA_A_Q2_K=1 B_TYPE=float B_TYPEV2=float2 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize / gl_NumWorkGroups

#include "types.slang"
#include "mul_mat_vec_head.slang"

groupshared FLOAT_TYPE sccache1[2][BLOCK_SIZE / 16][16];
groupshared FLOAT_TYPE sccache2[2][BLOCK_SIZE / 16][16];

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset,
                     uint itid, uint v_im, uint ix,
                     uint q_offset, uint y_offset,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows,
                     bool all_threads,
                     inout uint csel) {
    const uint y_idx = i * QUANT_K + y_offset;

    [unroll] for (uint n = 0; n < num_rows; ++n) {
        const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
        csel ^= 1;

        if (!all_threads) {
            if (i < num_blocks_per_row) {
                const uint scale = uint(data_a[ib0 + i].scales[itid]);
                sccache1[csel][ix][itid] = FLOAT_TYPE(scale & 0xF);
                sccache2[csel][ix][itid] = FLOAT_TYPE((scale >> 4) & 0xF);
            }
            GroupMemoryBarrierWithGroupSync();
            if (i >= num_blocks_per_row) { continue; }
        } else {
            const uint scale = uint(data_a[ib0 + i].scales[itid]);
            sccache1[csel][ix][itid] = FLOAT_TYPE(scale & 0xF);
            sccache2[csel][ix][itid] = FLOAT_TYPE((scale >> 4) & 0xF);
            GroupMemoryBarrierWithGroupSync();
        }

        const uint qs_u32 = uint(data_a_packed16[ib0 + i].qs[q_offset / 2    ])
                          | (uint(data_a_packed16[ib0 + i].qs[q_offset / 2 + 8]) << 16);
        const float4 qs_u32_0 = float4(unpack8_u( qs_u32       & 0x03030303u));
        const float4 qs_u32_2 = float4(unpack8_u((qs_u32 >> 2) & 0x03030303u));
        const float4 qs_u32_4 = float4(unpack8_u((qs_u32 >> 4) & 0x03030303u));
        const float4 qs_u32_6 = float4(unpack8_u((qs_u32 >> 6) & 0x03030303u));

        const FLOAT_TYPEV2 dm = FLOAT_TYPEV2(data_a[ib0 + i].dm);

        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            float2 b0   = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 +  0]);
            float2 b16  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 +  8]);
            float2 b32  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 16]);
            float2 b48  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 24]);
            float2 b64  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 32]);
            float2 b80  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 40]);
            float2 b96  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 48]);
            float2 b112 = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 56]);

            FLOAT_TYPE sum1 = FLOAT_TYPE(0);
            FLOAT_TYPE sum2 = FLOAT_TYPE(0);
            [unroll] for (int l = 0; l < 2; ++l) {
                sum1 = mad(FLOAT_TYPE(b0[l]),   sccache1[csel][ix][    8 * v_im] * qs_u32_0[l    ],
                       mad(FLOAT_TYPE(b16[l]),  sccache1[csel][ix][1 + 8 * v_im] * qs_u32_0[l + 2],
                       mad(FLOAT_TYPE(b32[l]),  sccache1[csel][ix][2 + 8 * v_im] * qs_u32_2[l    ],
                       mad(FLOAT_TYPE(b48[l]),  sccache1[csel][ix][3 + 8 * v_im] * qs_u32_2[l + 2],
                       mad(FLOAT_TYPE(b64[l]),  sccache1[csel][ix][4 + 8 * v_im] * qs_u32_4[l    ],
                       mad(FLOAT_TYPE(b80[l]),  sccache1[csel][ix][5 + 8 * v_im] * qs_u32_4[l + 2],
                       mad(FLOAT_TYPE(b96[l]),  sccache1[csel][ix][6 + 8 * v_im] * qs_u32_6[l    ],
                       mad(FLOAT_TYPE(b112[l]), sccache1[csel][ix][7 + 8 * v_im] * qs_u32_6[l + 2], sum1))))))));
                sum2 = mad(FLOAT_TYPE(b0[l]),   sccache2[csel][ix][    8 * v_im],
                       mad(FLOAT_TYPE(b16[l]),  sccache2[csel][ix][1 + 8 * v_im],
                       mad(FLOAT_TYPE(b32[l]),  sccache2[csel][ix][2 + 8 * v_im],
                       mad(FLOAT_TYPE(b48[l]),  sccache2[csel][ix][3 + 8 * v_im],
                       mad(FLOAT_TYPE(b64[l]),  sccache2[csel][ix][4 + 8 * v_im],
                       mad(FLOAT_TYPE(b80[l]),  sccache2[csel][ix][5 + 8 * v_im],
                       mad(FLOAT_TYPE(b96[l]),  sccache2[csel][ix][6 + 8 * v_im],
                       mad(FLOAT_TYPE(b112[l]), sccache2[csel][ix][7 + 8 * v_im], sum2))))))));
            }
            temp[j][n] = mad(dm.x, sum1, mad(-dm.y, sum2, temp[j][n]));
        }
    }
}

void compute_outputs(uint3 wg_id, uint tid_x, uint first_row, uint num_rows) {
    uint a_offset, b_offset, d_offset;
    get_offsets(wg_id, a_offset, b_offset, d_offset);

    const uint num_blocks_per_row = p.ncols / QUANT_K;

    const uint it_size = BLOCK_SIZE / 16;
    const uint itid    = tid_x % 16;
    const uint ix      = tid_x / 16;

    const uint v_im = itid / 8;
    const uint v_in = itid - 8 * v_im;

    const uint l0       = 2 * v_in;
    const uint q_offset = 32 * v_im + l0;
    const uint y_offset = 128 * v_im + l0;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint i = 0; i < NUM_ROWS; ++i) {
            temp[j][i] = FLOAT_TYPE(0);
        }
    }

    uint csel = 0;
    const uint nbr_par_th = num_blocks_per_row % it_size;
    const uint nbr_all_th = num_blocks_per_row - nbr_par_th;
    uint i0 = 0;
    [unroll] for (; i0 < nbr_all_th; i0 += it_size) {
        calc_superblock(temp, a_offset, b_offset, itid, v_im, ix, q_offset, y_offset,
                        i0 + ix, num_blocks_per_row, first_row, num_rows, true, csel);
    }
    calc_superblock(temp, a_offset, b_offset, itid, v_im, ix, q_offset, y_offset,
                    i0 + ix, num_blocks_per_row, first_row, num_rows, false, csel);

    reduce_result(temp, d_offset, first_row, num_rows, tid_x);
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);

    if (first_row + NUM_ROWS <= p.stride_d) {
        compute_outputs(gl_WorkGroupID, gl_LocalInvocationID.x, first_row, NUM_ROWS);
    } else {
        if (first_row >= p.stride_d) { return; }
        compute_outputs(gl_WorkGroupID, gl_LocalInvocationID.x, first_row, p.stride_d - first_row);
    }
}
