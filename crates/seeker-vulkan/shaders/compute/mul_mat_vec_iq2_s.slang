// Ported from llama.cpp ggml-vulkan mul_mat_vec_iq2_s.comp, MIT license — see LICENSE.llama-cpp
//
// IQ2_S row-times-vector. Grid index is qs (low 8 bits) plus 2 hi-bits unpacked from qh
// shifted by (8 - nibble_shift - 2*l). Sign byte lives at offset QUANT_K/8 in qs[].

//@variants
//  default: DATA_A_IQ2_S=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"
#include "mul_mat_vec_head.slang"

void calc_superblock(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                     uint a_offset, uint b_offset, uint itid,
                     uint i, uint num_blocks_per_row,
                     uint first_row, uint num_rows) {
    const uint y_idx        = i * QUANT_K + 16 * itid;
    const uint nibble_shift = 4u * (itid & 1u);
    const uint ib32         = itid / 2u;

    uint ibi = a_offset + first_row * num_blocks_per_row + i;
    [unroll] for (uint n = 0; n < num_rows; ++n) {
        const float d     = float(data_a[ibi].d);
        const uint  sc    = uint(data_a[ibi].scales[ib32]);
        const uint  scn   = (sc >> nibble_shift) & 0xFu;
        const float db    = d * (0.5f + float(scn)) * 0.25f;

        const uint qh = uint(data_a[ibi].qh[ib32]);
        // Two qs bytes (one per "l") packed into a uint16 at index `itid`.
        const uint qs_pair    = uint(data_a_packed16[ibi].qs[itid]);
        const uint sign_pair  = uint(data_a_packed16[ibi].qs[QUANT_K / 16u + itid]);
        const uint qs_lo_l0   = qs_pair & 0xFFu;
        const uint qs_lo_l1   = (qs_pair >> 8) & 0xFFu;
        const uint sign_l0    = sign_pair & 0xFFu;
        const uint sign_l1    = (sign_pair >> 8) & 0xFFu;

        [unroll] for (uint l = 0; l < 2; ++l) {
            const uint sign = l == 0 ? sign_l0 : sign_l1;
            const uint qs_lo = l == 0 ? qs_lo_l0 : qs_lo_l1;
            const uint qs   = qs_lo | ((qh << (8u - nibble_shift - 2u * l)) & 0x300u);
            const uint2 g   = iq2s_grid[qs];
            const float4 g0 = float4(unpack8_u(g.x));
            const float4 g1 = float4(unpack8_u(g.y));

            [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
                const uint b_idx = (j * p.batch_stride_b + b_offset + y_idx) / 4u + 2u * l;
                const float4 b0 = float4(data_b_v4[b_idx + 0u]);
                const float4 b4 = float4(data_b_v4[b_idx + 1u]);

                FLOAT_TYPE sum =
                      mad(FLOAT_TYPE(b0.x), FLOAT_TYPE((sign &   1u) != 0u ? -g0.x : g0.x),
                      mad(FLOAT_TYPE(b0.y), FLOAT_TYPE((sign &   2u) != 0u ? -g0.y : g0.y),
                      mad(FLOAT_TYPE(b0.z), FLOAT_TYPE((sign &   4u) != 0u ? -g0.z : g0.z),
                      mad(FLOAT_TYPE(b0.w), FLOAT_TYPE((sign &   8u) != 0u ? -g0.w : g0.w),
                      mad(FLOAT_TYPE(b4.x), FLOAT_TYPE((sign &  16u) != 0u ? -g1.x : g1.x),
                      mad(FLOAT_TYPE(b4.y), FLOAT_TYPE((sign &  32u) != 0u ? -g1.y : g1.y),
                      mad(FLOAT_TYPE(b4.z), FLOAT_TYPE((sign &  64u) != 0u ? -g1.z : g1.z),
                      mad(FLOAT_TYPE(b4.w), FLOAT_TYPE((sign & 128u) != 0u ? -g1.w : g1.w),
                      FLOAT_TYPE(0.0)))))))));
                temp[j][n] = mad(FLOAT_TYPE(db), sum, temp[j][n]);
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
    const uint itid = tid_x % 16;
    const uint ix   = tid_x / 16;

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
