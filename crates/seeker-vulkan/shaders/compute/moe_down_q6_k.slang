// MoE down-projection matvec with routing-weight fusion (Q6_K weights).
// Same kernel structure as `moe_down_q5_k.slang` — Q6_K dequant inner
// logic adapted from `mul_mat_vec_q6_k.slang`. Used for the down step
// of MoE FFN blocks whose `ffn_down_exps` are quantized as Q6_K
// instead of Q5_K (the Unsloth UD-Q4_K_XL checkpoint mixes both:
// blk.34, blk.38, blk.39 are Q6_K; the rest are Q5_K).
//
// We deliberately skip the scale-cache ping-pong from the standard
// Q6_K matvec — the savings amortize across many blocks per dispatch
// in that path, but our k-loop here issues a fresh barrier per expert
// anyway, and the inline scale reads are cheap enough at this dispatch
// shape (3 layers × ~250 workgroups).

//@variants
//  default: DATA_A_Q6_K=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"

struct MoeDownParams {
    uint ncols;          // = ff
    uint stride_a;       // Q6_K row stride within an expert slab (blocks)
    uint stride_b;       // ffn_h column stride (= ff, in F32 elements)
    uint stride_d;       // M bound (= n_embd)
    uint batch_stride_a; // expert slab stride in *elements* (= ncols * n_embd)
    uint n_expert_used;
};

[[vk::push_constant]] ConstantBuffer<MoeDownParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q6_K>          data_a;
[[vk::binding(1, 0)]] StructuredBuffer<float>               data_b;     // ffn_h
[[vk::binding(2, 0)]] RWStructuredBuffer<float>             data_d;     // output
[[vk::binding(3, 0)]] StructuredBuffer<block_q6_K_packed16> data_a_packed16;
[[vk::binding(4, 0)]] StructuredBuffer<float4>              data_b_v4;
[[vk::binding(7, 0)]] StructuredBuffer<int>                 data_ids;
[[vk::binding(8, 0)]] StructuredBuffer<float>               data_weights;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint NUM_ROWS = 2;

// Per-(row, expert) partial dot product. Each lane covers a strided
// subset of blocks across the row; the caller reduces via WaveActiveSum.
//
// Inline Q6_K dequant — no scale-cache optimization (see file-level
// comment). Mirrors the dequant math in `mul_mat_vec_q6_k.slang`.
void per_expert_partial(inout float partial[NUM_ROWS],
                        uint a_offset, uint b_offset, uint first_row, uint num_rows,
                        uint v_im, uint ql_offset, uint qh_offset,
                        uint s_offset, uint y_offset,
                        uint num_blocks_per_row, uint tid_x) {
    const uint it_size = BLOCK_SIZE / 16;
    const uint ix      = tid_x / 16;
    const uint itid    = tid_x % 16;

    [unroll] for (uint i = ix; i < num_blocks_per_row; i += it_size) {
        const uint y_idx = i * QUANT_K_Q6_K + y_offset;

        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;

            // 8 quant nibbles per thread (4 from ql0_*, 4 from ql32_*).
            const uint ql0_u32  = uint(data_a_packed16[ib0 + i].ql[ql_offset / 2     ])
                                | (uint(data_a_packed16[ib0 + i].ql[ql_offset / 2 +  1]) << 16);
            const uint ql32_u32 = uint(data_a_packed16[ib0 + i].ql[ql_offset / 2 + 16])
                                | (uint(data_a_packed16[ib0 + i].ql[ql_offset / 2 + 17]) << 16);

            const uint ql0_u32_lo4  =  ql0_u32       & 0x0F0F0F0Fu;
            const uint ql0_u32_hi4  = (ql0_u32 >> 4) & 0x0F0F0F0Fu;
            const uint ql32_u32_lo4 =  ql32_u32       & 0x0F0F0F0Fu;
            const uint ql32_u32_hi4 = (ql32_u32 >> 4) & 0x0F0F0F0Fu;

            // High 2 bits from qh, expanded to byte lanes.
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

            // 6-bit values are stored unsigned in [0, 64); subtract 32 to
            // recover the signed range [-32, 31].
            const float4 q0 = float4(unpack8_u(q0_u32)) - 32.0f;
            const float4 q1 = float4(unpack8_u(q1_u32)) - 32.0f;
            const float4 q2 = float4(unpack8_u(q2_u32)) - 32.0f;
            const float4 q3 = float4(unpack8_u(q3_u32)) - 32.0f;

            // 4 i8 sub-scales for this thread's 4 quartets of weights.
            const float sc0 = float(data_a[ib0 + i].scales[s_offset    ]);
            const float sc2 = float(data_a[ib0 + i].scales[s_offset + 2]);
            const float sc4 = float(data_a[ib0 + i].scales[s_offset + 4]);
            const float sc6 = float(data_a[ib0 + i].scales[s_offset + 6]);

            const float d = float(data_a[ib0 + i].d);

            // 16 input lanes per thread, aligned with the 16 nibbles above.
            float4 by0  = float4(data_b_v4[(b_offset + y_idx) / 4     ]);
            float4 by32 = float4(data_b_v4[(b_offset + y_idx) / 4 +  8]);
            float4 by64 = float4(data_b_v4[(b_offset + y_idx) / 4 + 16]);
            float4 by96 = float4(data_b_v4[(b_offset + y_idx) / 4 + 24]);

            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            [unroll] for (uint l = 0; l < 4; ++l) {
                sum0 = mad(by0[l],  q0[l], sum0);
                sum1 = mad(by32[l], q1[l], sum1);
                sum2 = mad(by64[l], q2[l], sum2);
                sum3 = mad(by96[l], q3[l], sum3);
            }
            partial[n] = mad(
                mad(sum0, sc0,
                mad(sum1, sc2,
                mad(sum2, sc4, sum3 * sc6))),
                d, partial[n]);
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

    // Q6_K per-thread layout (identical to mul_mat_vec_q6_k.slang).
    const uint itid    = tid_x % 16;
    const uint v_im    = itid / 8;
    const uint v_in    = itid - 8 * v_im;
    const uint l0      = 4 * v_in;
    const uint is      = v_in / 4;
    const uint ql_offset =  64 * v_im + l0;
    const uint qh_offset =  32 * v_im + l0;
    const uint s_offset  =   8 * v_im + is;
    const uint y_offset  = 128 * v_im + l0;

    const uint num_blocks_per_row = p.ncols / QUANT_K_Q6_K;

    float acc[NUM_ROWS];
    [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { acc[n] = 0.0f; }

    for (uint k = 0; k < p.n_expert_used; ++k) {
        const uint expert_id = uint(data_ids[k]);
        const float routing  = data_weights[k];

        const uint a_offset = expert_id * (p.batch_stride_a / QUANT_K_Q6_K);
        const uint b_offset = k * p.stride_b;

        float partial[NUM_ROWS];
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { partial[n] = 0.0f; }

        per_expert_partial(partial, a_offset, b_offset, first_row, num_rows,
                           v_im, ql_offset, qh_offset, s_offset, y_offset,
                           num_blocks_per_row, tid_x);

        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) {
            const float sum = WaveActiveSum(partial[n]);
            acc[n] = mad(routing, sum, acc[n]);
        }
    }

    if (tid_x == 0) {
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) {
            if (n < num_rows) {
                data_d[first_row + n] = D_TYPE(acc[n]);
            }
        }
    }
}
