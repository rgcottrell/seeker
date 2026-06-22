// Ported from llama.cpp ggml-vulkan mul_mat_vec_q6_k.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  default: DATA_A_Q6_K=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  id:      DATA_A_Q6_K=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float MUL_MAT_ID=1
//@end-variants

import glsl;  // gl_WorkGroupSize / gl_NumWorkGroups

#include "types.slang"
#include "mul_mat_vec_head.slang"

// Scale cache shared across the workgroup: two banks (ping-pong via `csel`) to
// hide barrier cost. BLOCK_SIZE/16 entries (one per 16-thread block-group),
// each holding 16 i8 scales.
groupshared FLOAT_TYPE sccache[2][BLOCK_SIZE / 16][16];

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset,
                     uint itid, uint ix,
                     uint ql_offset, uint qh_offset, uint s_offset, uint y_offset,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows,
                     bool all_threads,
                     inout uint csel) {
    const uint y_idx = i * QUANT_K + y_offset;

    [unroll] for (uint n = 0; n < num_rows; ++n) {
        const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
        csel ^= 1;

        if (!all_threads) {  // tail iteration where not every thread has a block
            if (i < num_blocks_per_row) {
                sccache[csel][ix][itid] = FLOAT_TYPE(data_a[ib0 + i].scales[itid]);
            }
            GroupMemoryBarrierWithGroupSync();

            if (i >= num_blocks_per_row) { continue; }
        }

        const uint ql0_u32  = uint(data_a_packed16[ib0 + i].ql[ql_offset / 2     ])
                            | (uint(data_a_packed16[ib0 + i].ql[ql_offset / 2 +  1]) << 16);
        const uint ql32_u32 = uint(data_a_packed16[ib0 + i].ql[ql_offset / 2 + 16])
                            | (uint(data_a_packed16[ib0 + i].ql[ql_offset / 2 + 17]) << 16);

        const uint ql0_u32_lo4  =  ql0_u32       & 0x0F0F0F0Fu;
        const uint ql0_u32_hi4  = (ql0_u32 >> 4) & 0x0F0F0F0Fu;
        const uint ql32_u32_lo4 =  ql32_u32       & 0x0F0F0F0Fu;
        const uint ql32_u32_hi4 = (ql32_u32 >> 4) & 0x0F0F0F0Fu;

        const uint qh_u32  = uint(data_a_packed16[ib0 + i].qh[qh_offset / 2    ])
                           | (uint(data_a_packed16[ib0 + i].qh[qh_offset / 2 + 1]) << 16);
        const uint qh0_u32 = (qh_u32 & 0x03030303u) << 4;
        const uint qh2_u32 = (qh_u32 & 0x0C0C0C0Cu) << 2;
        const uint qh4_u32 = (qh_u32 & 0x30303030u);
        const uint qh6_u32 = (qh_u32 & 0xC0C0C0C0u) >> 2;

        const uint q0_u32 = ql0_u32_lo4  | qh0_u32;
        const uint q1_u32 = ql32_u32_lo4 | qh2_u32;
        const uint q2_u32 = ql0_u32_hi4  | qh4_u32;
        const uint q3_u32 = ql32_u32_hi4 | qh6_u32;

        const float4 q0 = float4(unpack8_u(q0_u32)) - 32.0f;
        const float4 q1 = float4(unpack8_u(q1_u32)) - 32.0f;
        const float4 q2 = float4(unpack8_u(q2_u32)) - 32.0f;
        const float4 q3 = float4(unpack8_u(q3_u32)) - 32.0f;

        if (all_threads) {
            sccache[csel][ix][itid] = FLOAT_TYPE(data_a[ib0 + i].scales[itid]);
            GroupMemoryBarrierWithGroupSync();
        }

        const FLOAT_TYPE d = FLOAT_TYPE(data_a[ib0 + i].d);

        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            float4 by0  = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y_idx) / 4     ]);
            float4 by32 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y_idx) / 4 +  8]);
            float4 by64 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y_idx) / 4 + 16]);
            float4 by96 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + y_idx) / 4 + 24]);

            FLOAT_TYPE sum[4] = { FLOAT_TYPE(0), FLOAT_TYPE(0), FLOAT_TYPE(0), FLOAT_TYPE(0) };
            [unroll] for (uint l = 0; l < 4; ++l) {
                sum[0] = mad(FLOAT_TYPE(by0[l]),  q0[l], sum[0]);
                sum[1] = mad(FLOAT_TYPE(by32[l]), q1[l], sum[1]);
                sum[2] = mad(FLOAT_TYPE(by64[l]), q2[l], sum[2]);
                sum[3] = mad(FLOAT_TYPE(by96[l]), q3[l], sum[3]);
            }
            temp[j][n] = mad(
                mad(sum[0], sccache[csel][ix][s_offset    ],
                mad(sum[1], sccache[csel][ix][s_offset + 2],
                mad(sum[2], sccache[csel][ix][s_offset + 4],
                    sum[3] * sccache[csel][ix][s_offset + 6]))),
                d, temp[j][n]);
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

    const uint v_im = itid / 8;                 // 0 or 1
    const uint v_in = itid - 8 * v_im;          // 0..7

    const uint l0 = 4 * v_in;                    // 0,4,...,28
    const uint is = v_in / 4;

    const uint ql_offset =  64 * v_im + l0;
    const uint qh_offset =  32 * v_im + l0;
    const uint s_offset  =   8 * v_im + is;
    const uint y_offset  = 128 * v_im + l0;

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
        calc_superblock(temp, a_offset, b_offset, itid, ix, ql_offset, qh_offset, s_offset, y_offset,
                        i0 + ix, num_blocks_per_row, first_row, num_rows, true, csel);
    }
    calc_superblock(temp, a_offset, b_offset, itid, ix, ql_offset, qh_offset, s_offset, y_offset,
                    i0 + ix, num_blocks_per_row, first_row, num_rows, false, csel);

    reduce_result(temp, d_offset, first_row, num_rows, tid_x);
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
#ifdef MUL_MAT_ID
    // Expert routing puts the expert index on wg_id.y (resolved in
    // get_offsets); rows tile on wg_id.x alone.
    const uint first_row = NUM_ROWS * gl_WorkGroupID.x;
#else
    const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);
#endif

    if (first_row + NUM_ROWS <= p.stride_d) {
        compute_outputs(gl_WorkGroupID, gl_LocalInvocationID.x, first_row, NUM_ROWS);
    } else {
        if (first_row >= p.stride_d) { return; }
        compute_outputs(gl_WorkGroupID, gl_LocalInvocationID.x, first_row, p.stride_d - first_row);
    }
}
