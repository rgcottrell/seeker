// Ported from llama.cpp ggml-vulkan mul_mat_vec_q4_k.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  default: DATA_A_Q4_K=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize / gl_NumWorkGroups

#include "types.slang"
#include "mul_mat_vec_head.slang"

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset, uint v_im,
                     uint q_offset, uint y_offset,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows) {
    const uint y1_idx = i * QUANT_K + y_offset;
    const uint y2_idx = y1_idx + 128;

    [unroll] for (uint n = 0; n < num_rows; ++n) {
        const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
        const FLOAT_TYPEV2 dm = FLOAT_TYPEV2(data_a[ib0 + i].dm);

        const uint scale0_u32 = data_a_packed16[ib0 + i].scales[v_im    ];
        const uint scale4_u32 = data_a_packed16[ib0 + i].scales[v_im + 2];
        const uint scale8_u32 = data_a_packed16[ib0 + i].scales[v_im + 4];

        const uint scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
        const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2;
        const float4 scale_0_4_l_f = float4(unpack8_u(scale_0_4_l & 0x3F3F3F3Fu));
        const float4 scale8_f      = float4(unpack8_u((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h));

        const FLOAT_TYPE sc0 = scale_0_4_l_f.x;
        const FLOAT_TYPE sc1 = scale_0_4_l_f.y;
        const FLOAT_TYPE sc2 = scale_0_4_l_f.z;
        const FLOAT_TYPE sc3 = scale_0_4_l_f.w;
        const FLOAT_TYPE sc4 = scale8_f.x;
        const FLOAT_TYPE sc5 = scale8_f.y;
        const FLOAT_TYPE sc6 = scale8_f.z;
        const FLOAT_TYPE sc7 = scale8_f.w;

        const uint qs0_u32  = data_a_packed32[ib0 + i].qs[q_offset / 4];
        const uint qs64_u32 = data_a_packed32[ib0 + i].qs[q_offset / 4 + 16];

        const uint qs0_u32_lo4  =  qs0_u32       & 0x0F0F0F0Fu;
        const uint qs0_u32_hi4  = (qs0_u32 >> 4) & 0x0F0F0F0Fu;
        const uint qs64_u32_lo4 =  qs64_u32       & 0x0F0F0F0Fu;
        const uint qs64_u32_hi4 = (qs64_u32 >> 4) & 0x0F0F0F0Fu;

        const float4 qs0_lo4  = float4(unpack8_u(qs0_u32_lo4));
        const float4 qs64_lo4 = float4(unpack8_u(qs64_u32_lo4));
        const float4 qs0_hi4  = float4(unpack8_u(qs0_u32_hi4));
        const float4 qs64_hi4 = float4(unpack8_u(qs64_u32_hi4));

        const FLOAT_TYPE q4_0  = qs0_lo4.x;
        const FLOAT_TYPE q4_1  = qs0_lo4.y;
        const FLOAT_TYPE q4_2  = qs0_lo4.z;
        const FLOAT_TYPE q4_3  = qs0_lo4.w;
        const FLOAT_TYPE q4_4  = qs0_hi4.x;
        const FLOAT_TYPE q4_5  = qs0_hi4.y;
        const FLOAT_TYPE q4_6  = qs0_hi4.z;
        const FLOAT_TYPE q4_7  = qs0_hi4.w;
        const FLOAT_TYPE q4_8  = qs64_lo4.x;
        const FLOAT_TYPE q4_9  = qs64_lo4.y;
        const FLOAT_TYPE q4_10 = qs64_lo4.z;
        const FLOAT_TYPE q4_11 = qs64_lo4.w;
        const FLOAT_TYPE q4_12 = qs64_hi4.x;
        const FLOAT_TYPE q4_13 = qs64_hi4.y;
        const FLOAT_TYPE q4_14 = qs64_hi4.z;
        const FLOAT_TYPE q4_15 = qs64_hi4.w;

        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            float4 by10  = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y1_idx) / 4    ]);
            float4 by132 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y1_idx) / 4 + 8]);
            float4 by20  = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y2_idx) / 4    ]);
            float4 by232 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y2_idx) / 4 + 8]);

            const FLOAT_TYPE sx = mad(FLOAT_TYPE(by10.x),  q4_0,
                                  mad(FLOAT_TYPE(by10.y),  q4_1,
                                  mad(FLOAT_TYPE(by10.z),  q4_2,
                                      FLOAT_TYPE(by10.w) * q4_3)));
            const FLOAT_TYPE sy = mad(FLOAT_TYPE(by132.x), q4_4,
                                  mad(FLOAT_TYPE(by132.y), q4_5,
                                  mad(FLOAT_TYPE(by132.z), q4_6,
                                      FLOAT_TYPE(by132.w) * q4_7)));
            const FLOAT_TYPE sz = mad(FLOAT_TYPE(by20.x),  q4_8,
                                  mad(FLOAT_TYPE(by20.y),  q4_9,
                                  mad(FLOAT_TYPE(by20.z),  q4_10,
                                      FLOAT_TYPE(by20.w) * q4_11)));
            const FLOAT_TYPE sw = mad(FLOAT_TYPE(by232.x), q4_12,
                                  mad(FLOAT_TYPE(by232.y), q4_13,
                                  mad(FLOAT_TYPE(by232.z), q4_14,
                                      FLOAT_TYPE(by232.w) * q4_15)));
            const FLOAT_TYPE smin =
                mad(FLOAT_TYPE(by10.x), sc2, mad(FLOAT_TYPE(by132.x), sc3, mad(FLOAT_TYPE(by20.x), sc6, mad(FLOAT_TYPE(by232.x), sc7,
                mad(FLOAT_TYPE(by10.y), sc2, mad(FLOAT_TYPE(by132.y), sc3, mad(FLOAT_TYPE(by20.y), sc6, mad(FLOAT_TYPE(by232.y), sc7,
                mad(FLOAT_TYPE(by10.z), sc2, mad(FLOAT_TYPE(by132.z), sc3, mad(FLOAT_TYPE(by20.z), sc6, mad(FLOAT_TYPE(by232.z), sc7,
                mad(FLOAT_TYPE(by10.w), sc2, mad(FLOAT_TYPE(by132.w), sc3, mad(FLOAT_TYPE(by20.w), sc6,     FLOAT_TYPE(by232.w) * sc7)))))))))))))));
            temp[j][n] = mad(dm.x, mad(sx, sc0, mad(sy, sc1, mad(sz, sc4, sw * sc5))), mad(-dm.y, smin, temp[j][n]));
        }
    }
}

void compute_outputs(uint3 wg_id, uint tid_x, uint first_row, uint num_rows) {
    uint a_offset, b_offset, d_offset;
    get_offsets(wg_id, a_offset, b_offset, d_offset);

    const uint num_blocks_per_row = p.ncols / QUANT_K;

    // 16 threads are used to process each block.
    const uint it_size = BLOCK_SIZE / 16;
    const uint itid    = tid_x % 16;
    const uint ix      = tid_x / 16;

    const uint il = itid / 4;                  // 0..3
    const uint ir = itid - 4 * il;              // 0..3
    const uint n  = 4;

    const uint v_im = il / 2;                  // 0 or 1
    const uint v_in = il % 2;

    const uint l0       = n * (2 * ir + v_in); // 0..15
    const uint q_offset = 32 * v_im + l0;
    const uint y_offset = 64 * v_im + l0;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint i = 0; i < NUM_ROWS; ++i) {
            temp[j][i] = FLOAT_TYPE(0);
        }
    }

    [unroll] for (uint i = ix; i < num_blocks_per_row; i += it_size) {
        calc_superblock(temp, a_offset, b_offset, v_im, q_offset, y_offset,
                        i, num_blocks_per_row, first_row, num_rows);
    }

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
