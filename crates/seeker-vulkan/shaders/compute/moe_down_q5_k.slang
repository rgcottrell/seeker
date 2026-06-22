// MoE down-projection matvec with routing-weight fusion (Q5_K weights).
//
// Computes the per-token contribution of the routed FFN:
//
//   output[m] = sum_{k=0..n_expert_used} (
//                 routing_weights[k] *
//                 sum_{d=0..ff} dequant(down_exps[ids[k], m, d]) * ffn_h[d, k]
//               )
//
// In one dispatch. The standard mul_mat_vec_q5_k computes per-(token, expert)
// matvecs that the host would then weight-and-sum on a second pass; this
// kernel fuses both the expert-id indirection AND the routing-weighted
// reduction into a single workgroup, eliminating the intermediate
// `[n_embd, n_expert_used]` tensor.
//
// Mirrors the Q5_K dequant block in `mul_mat_vec_q5_k.slang` exactly —
// only the per-expert offset math and the running accumulator are added.
//
// Dispatch shape: [M.div_ceil(NUM_ROWS), 1, 1]. Each workgroup produces
// `NUM_ROWS` output rows by iterating internally over all
// `n_expert_used` experts.

//@variants
//  default: DATA_A_Q5_K=1 B_TYPE=float B_TYPEV2=float2 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"

// ── Push constants ────────────────────────────────────────────────────────
//
// Tighter than the regular MulMatVecParams since this kernel only services
// the MoE down step: no batch broadcast, no fusion flags, no NUM_COLS axis.
struct MoeDownParams {
    uint ncols;          // K reduction dim (= ff, expert_feed_forward_length)
    uint stride_a;       // Q5_K row stride within an expert slab (in blocks)
    uint stride_b;       // ffn_h column stride (= ff, in F32 elements)
    uint stride_d;       // output row count bound (= n_embd, used as M cap)
    uint batch_stride_a; // expert slab stride in *elements* (= ncols * n_embd);
                         // shader divides by QUANT_K to get block-index units.
    uint n_expert_used;  // = nei0 (top-k routing depth)
};

[[vk::push_constant]] ConstantBuffer<MoeDownParams> p;

// ── Bindings ───────────────────────────────────────────────────────────────
[[vk::binding(0, 0)]] StructuredBuffer<block_q5_K>          data_a;
[[vk::binding(1, 0)]] StructuredBuffer<float>               data_b;     // ffn_h
[[vk::binding(2, 0)]] RWStructuredBuffer<float>             data_d;     // output
[[vk::binding(3, 0)]] StructuredBuffer<block_q5_K_packed16> data_a_packed16;
[[vk::binding(5, 0)]] StructuredBuffer<float2>              data_b_v2;
[[vk::binding(7, 0)]] StructuredBuffer<int>                 data_ids;
[[vk::binding(8, 0)]] StructuredBuffer<float>               data_weights;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint NUM_ROWS = 2;

// ── Q5_K dequant + per-row partial dot ─────────────────────────────────────
//
// One call computes (over the lane's stride of blocks) the partial dot
// product against rows `[first_row, first_row + num_rows)` of the expert
// slab starting at `a_offset`, using ffn_h column starting at `b_offset`.
// Returns the per-row partial sum array (still subgroup-distributed —
// caller must `WaveActiveSum` to collapse).
//
// Inner dequant logic is identical to `mul_mat_vec_q5_k.slang::calc_superblock`.
void per_expert_partial(inout float partial[NUM_ROWS],
                        uint a_offset, uint b_offset, uint first_row, uint num_rows,
                        uint v_im, uint l0, uint q_offset, uint y_offset,
                        uint num_blocks_per_row, uint tid_x) {
    const uint it_size = BLOCK_SIZE / 16;
    const uint ix      = tid_x / 16;

    [unroll] for (uint i = ix; i < num_blocks_per_row; i += it_size) {
        const uint y1_idx = i * QUANT_K_Q5_K + y_offset;
        const uint y2_idx = y1_idx + 128;

        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
            const float2 dm = float2(data_a[ib0 + i].dm);

            const uint scale0_u32 = data_a_packed16[ib0 + i].scales[v_im    ];
            const uint scale4_u32 = data_a_packed16[ib0 + i].scales[v_im + 2];
            const uint scale8_u32 = data_a_packed16[ib0 + i].scales[v_im + 4];

            const uint scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
            const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2;
            const float4 scale_0_4_l_f = float4(unpack8_u(scale_0_4_l & 0x3F3F3F3Fu));
            const float4 scale8_f      = float4(unpack8_u((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0Fu) | scale_0_4_h));

            const float sc0 = scale_0_4_l_f.x;
            const float sc1 = scale_0_4_l_f.y;
            const float sc2 = scale_0_4_l_f.z;
            const float sc3 = scale_0_4_l_f.w;
            const float sc4 = scale8_f.x;
            const float sc5 = scale8_f.y;
            const float sc6 = scale8_f.z;
            const float sc7 = scale8_f.w;

            const uint qs0_16_u32  = uint(data_a_packed16[ib0 + i].qs[q_offset / 2     ])
                                   | (uint(data_a_packed16[ib0 + i].qs[q_offset / 2 +  8]) << 16);
            const uint qs64_80_u32 = uint(data_a_packed16[ib0 + i].qs[q_offset / 2 + 32])
                                   | (uint(data_a_packed16[ib0 + i].qs[q_offset / 2 + 40]) << 16);

            uint qs0_16_u32_lo4  =  qs0_16_u32       & 0x0F0F0F0Fu;
            uint qs0_16_u32_hi4  = (qs0_16_u32 >> 4) & 0x0F0F0F0Fu;
            uint qs64_80_u32_lo4 =  qs64_80_u32       & 0x0F0F0F0Fu;
            uint qs64_80_u32_hi4 = (qs64_80_u32 >> 4) & 0x0F0F0F0Fu;

            const uint qh = uint(data_a_packed16[ib0 + i].qh[l0 / 2    ])
                          | (uint(data_a_packed16[ib0 + i].qh[l0 / 2 + 8]) << 16);

            const uint qs0_16_lo4_offset16  = ((qh >> (2 * v_im)) & 0x01010101u) << 4;
            const uint qs0_16_hi4_offset16  = ((qh >> (2 * v_im)) & 0x02020202u) << 3;
            const uint qs64_80_lo4_offset16 =  (qh >> (2 * v_im)) & 0x10101010u;
            const uint qs64_80_hi4_offset16 = ((qh >> (2 * v_im)) & 0x20202020u) >> 1;

            qs0_16_u32_lo4  += qs0_16_lo4_offset16;
            qs0_16_u32_hi4  += qs0_16_hi4_offset16;
            qs64_80_u32_lo4 += qs64_80_lo4_offset16;
            qs64_80_u32_hi4 += qs64_80_hi4_offset16;

            const float4 qs0_16_lo4   = float4(unpack8_u(qs0_16_u32_lo4));
            const float4 qs64_80_lo4  = float4(unpack8_u(qs64_80_u32_lo4));
            const float4 qs0_16_hi4   = float4(unpack8_u(qs0_16_u32_hi4));
            const float4 qs64_80_hi4  = float4(unpack8_u(qs64_80_u32_hi4));

            const float q4_0  = qs0_16_lo4.x;
            const float q4_1  = qs0_16_lo4.y;
            const float q4_2  = qs0_16_lo4.z;
            const float q4_3  = qs0_16_lo4.w;
            const float q4_4  = qs0_16_hi4.x;
            const float q4_5  = qs0_16_hi4.y;
            const float q4_6  = qs0_16_hi4.z;
            const float q4_7  = qs0_16_hi4.w;
            const float q4_8  = qs64_80_lo4.x;
            const float q4_9  = qs64_80_lo4.y;
            const float q4_10 = qs64_80_lo4.z;
            const float q4_11 = qs64_80_lo4.w;
            const float q4_12 = qs64_80_hi4.x;
            const float q4_13 = qs64_80_hi4.y;
            const float q4_14 = qs64_80_hi4.z;
            const float q4_15 = qs64_80_hi4.w;

            float2 by10  = float2(data_b_v2[(b_offset + y1_idx) / 2     ]);
            float2 by116 = float2(data_b_v2[(b_offset + y1_idx) / 2 +  8]);
            float2 by132 = float2(data_b_v2[(b_offset + y1_idx) / 2 + 16]);
            float2 by148 = float2(data_b_v2[(b_offset + y1_idx) / 2 + 24]);
            float2 by20  = float2(data_b_v2[(b_offset + y2_idx) / 2     ]);
            float2 by216 = float2(data_b_v2[(b_offset + y2_idx) / 2 +  8]);
            float2 by232 = float2(data_b_v2[(b_offset + y2_idx) / 2 + 16]);
            float2 by248 = float2(data_b_v2[(b_offset + y2_idx) / 2 + 24]);

            const float sx = mad(by10.x,  q4_0,
                              mad(by10.y,  q4_1,
                              mad(by116.x, q4_2,  by116.y * q4_3)));
            const float sy = mad(by132.x, q4_4,
                              mad(by132.y, q4_5,
                              mad(by148.x, q4_6,  by148.y * q4_7)));
            const float sz = mad(by20.x,  q4_8,
                              mad(by20.y,  q4_9,
                              mad(by216.x, q4_10, by216.y * q4_11)));
            const float sw = mad(by232.x, q4_12,
                              mad(by232.y, q4_13,
                              mad(by248.x, q4_14, by248.y * q4_15)));
            const float smin =
                mad(by10.x  + by10.y  + by116.x + by116.y, sc2,
                mad(by132.x + by132.y + by148.x + by148.y, sc3,
                mad(by20.x  + by20.y  + by216.x + by216.y, sc6,
                   (by232.x + by232.y + by248.x + by248.y) * sc7)));
            partial[n] = mad(dm.x, mad(sx, sc0, mad(sy, sc1, mad(sz, sc4, sw * sc5))),
                              mad(-dm.y, smin, partial[n]));
        }
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid_x = gl_LocalInvocationID.x;
    const uint first_row = NUM_ROWS * gl_WorkGroupID.x;
    if (first_row >= p.stride_d) { return; }
    const uint num_rows = (first_row + NUM_ROWS <= p.stride_d)
        ? NUM_ROWS
        : (p.stride_d - first_row);

    // Per-row Q5_K layout constants — identical to mul_mat_vec_q5_k.slang.
    const uint itid    = tid_x % 16;
    const uint il      = itid / 4;
    const uint ir      = itid - 4 * il;
    const uint v_im    = il / 2;
    const uint v_in    = il % 2;
    const uint l0      = 4 * ir + 2 * v_in;
    const uint q_offset = 32 * v_im + l0;
    const uint y_offset = 64 * v_im + l0;
    const uint num_blocks_per_row = p.ncols / QUANT_K_Q5_K;

    // Accumulator for the routing-weighted sum across all experts.
    float acc[NUM_ROWS];
    [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { acc[n] = 0.0f; }

    // Loop over the `n_expert_used` routed experts. Each iteration:
    //   1. resolves expert_id and routing weight,
    //   2. computes lane-local partial dots against this expert's slab,
    //   3. collapses via WaveActiveSum (subgroup-width reduction),
    //   4. multiplies by the routing weight and accumulates into `acc`.
    for (uint k = 0; k < p.n_expert_used; ++k) {
        const uint expert_id = uint(data_ids[k]);
        const float routing  = data_weights[k];

        const uint a_offset = expert_id * (p.batch_stride_a / QUANT_K_Q5_K);
        const uint b_offset = k * p.stride_b;

        float partial[NUM_ROWS];
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { partial[n] = 0.0f; }

        per_expert_partial(partial, a_offset, b_offset, first_row, num_rows,
                           v_im, l0, q_offset, y_offset,
                           num_blocks_per_row, tid_x);

        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) {
            const float sum = WaveActiveSum(partial[n]);
            acc[n] = mad(routing, sum, acc[n]);
        }
    }

    // Single coherent write — only lane 0 stores; the accumulator is
    // identical across the subgroup after the WaveActiveSum broadcasts.
    if (tid_x == 0) {
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) {
            if (n < num_rows) {
                data_d[first_row + n] = D_TYPE(acc[n]);
            }
        }
    }
}
