// Ported from llama.cpp ggml-vulkan mul_mat_vec_q3_k.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  default: DATA_A_Q3_K=1 B_TYPE=float B_TYPEV2=float2 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize / gl_NumWorkGroups

#include "types.slang"
#include "mul_mat_vec_head.slang"

groupshared FLOAT_TYPE sccache[2][BLOCK_SIZE / 16][2][8];

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset,
                     uint ix, uint itid8, uint v_im, uint v_im4, uint v_in,
                     uint hm_m[4],
                     uint q_offset, uint y_offset,
                     uint s_shift,
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
                sccache[csel][ix][v_im][itid8] = FLOAT_TYPE(int8_t(
                    ((data_a[ib0 + i].scales[itid8        ] >> v_im4)  & 0xF) |
                    (((data_a[ib0 + i].scales[itid8 % 4 + 8] >> s_shift) & 3) << 4)
                ) - 32);
            }
            GroupMemoryBarrierWithGroupSync();
            if (i >= num_blocks_per_row) { continue; }
        }

        const uint hmk = ~(uint(data_a_packed16[ib0 + i].hmask[v_in    ])
                         | (uint(data_a_packed16[ib0 + i].hmask[v_in + 8]) << 16));
        const float4 hmk_0 = float4(unpack8_u(((hmk & hm_m[0]) >> (    v_im4)) << 2));
        const float4 hmk_1 = float4(unpack8_u(((hmk & hm_m[1]) >> (1 + v_im4)) << 2));
        const float4 hmk_2 = float4(unpack8_u(((hmk & hm_m[2]) >> (2 + v_im4)) << 2));
        const float4 hmk_3 = float4(unpack8_u(((hmk & hm_m[3]) >> (3 + v_im4)) << 2));

        uint qs_u32 = uint(data_a[ib0 + i].qs[q_offset    ])
                    | (uint(data_a[ib0 + i].qs[q_offset + 1]) << 8);
        qs_u32 |= (uint(data_a[ib0 + i].qs[q_offset + 16])
                | (uint(data_a[ib0 + i].qs[q_offset + 17]) << 8)) << 16;
        const float4 qs_u32_0 = float4(unpack8_u( qs_u32       & 0x03030303u));
        const float4 qs_u32_2 = float4(unpack8_u((qs_u32 >> 2) & 0x03030303u));
        const float4 qs_u32_4 = float4(unpack8_u((qs_u32 >> 4) & 0x03030303u));
        const float4 qs_u32_6 = float4(unpack8_u((qs_u32 >> 6) & 0x03030303u));

        if (all_threads) {
            sccache[csel][ix][v_im][itid8] = FLOAT_TYPE(int8_t(
                ((data_a[ib0 + i].scales[itid8        ] >> v_im4)  & 0xF) |
                (((data_a[ib0 + i].scales[itid8 % 4 + 8] >> s_shift) & 3) << 4)
            ) - 32);
            GroupMemoryBarrierWithGroupSync();
        }

        const FLOAT_TYPE d = FLOAT_TYPE(data_a[ib0 + i].d);

        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            float2 b0   = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 +  0]);
            float2 b16  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 +  8]);
            float2 b32  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 16]);
            float2 b48  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 24]);
            float2 b64  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 32]);
            float2 b80  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 40]);
            float2 b96  = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 48]);
            float2 b112 = float2(data_b_v2[(j * p.batch_stride_b + b_offset + y_idx) / 2 + 56]);

            FLOAT_TYPE sum = FLOAT_TYPE(0);
            [unroll] for (int l = 0; l < 2; ++l) {
                sum = mad(FLOAT_TYPE(b0[l])   * sccache[csel][ix][v_im][0], qs_u32_0[l    ] - hmk_0[l    ],
                      mad(FLOAT_TYPE(b16[l])  * sccache[csel][ix][v_im][1], qs_u32_0[l + 2] - hmk_0[l + 2],
                      mad(FLOAT_TYPE(b32[l])  * sccache[csel][ix][v_im][2], qs_u32_2[l    ] - hmk_1[l    ],
                      mad(FLOAT_TYPE(b48[l])  * sccache[csel][ix][v_im][3], qs_u32_2[l + 2] - hmk_1[l + 2],
                      mad(FLOAT_TYPE(b64[l])  * sccache[csel][ix][v_im][4], qs_u32_4[l    ] - hmk_2[l    ],
                      mad(FLOAT_TYPE(b80[l])  * sccache[csel][ix][v_im][5], qs_u32_4[l + 2] - hmk_2[l + 2],
                      mad(FLOAT_TYPE(b96[l])  * sccache[csel][ix][v_im][6], qs_u32_6[l    ] - hmk_3[l    ],
                      mad(FLOAT_TYPE(b112[l]) * sccache[csel][ix][v_im][7], qs_u32_6[l + 2] - hmk_3[l + 2], sum))))))));
            }
            temp[j][n] = mad(d, sum, temp[j][n]);
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
    const uint itid8   = itid % 8;

    const uint v_im  = itid / 8;
    const uint v_im4 = v_im * 4;
    const uint v_in  = itid - 8 * v_im;

    const uint m = 0x01010101u << (4 * v_im);
    uint hm_m[4];
    [unroll] for (uint j = 0; j < 4; ++j) { hm_m[j] = m << j; }

    const uint l0       = 2 * v_in;
    const uint q_offset = 32 * v_im + l0;
    const uint y_offset = 128 * v_im + l0;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint i = 0; i < NUM_ROWS; ++i) {
            temp[j][i] = FLOAT_TYPE(0);
        }
    }

    const uint s_shift = v_im4 + 2 * (itid8 / 4);

    uint csel = 0;
    const uint nbr_par_th = num_blocks_per_row % it_size;
    const uint nbr_all_th = num_blocks_per_row - nbr_par_th;
    uint i0 = 0;
    [unroll] for (; i0 < nbr_all_th; i0 += it_size) {
        calc_superblock(temp, a_offset, b_offset, ix, itid8, v_im, v_im4, v_in, hm_m,
                        q_offset, y_offset, s_shift,
                        i0 + ix, num_blocks_per_row, first_row, num_rows, true, csel);
    }
    calc_superblock(temp, a_offset, b_offset, ix, itid8, v_im, v_im4, v_in, hm_m,
                    q_offset, y_offset, s_shift,
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
