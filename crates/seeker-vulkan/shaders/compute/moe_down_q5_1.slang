// MoE down-projection matvec with routing-weight fusion (Q5_1 weights).
// Same kernel structure as `moe_down_q8_0.slang` — only the dequant differs.
// DiffusionGemma's `ffn_down_exps` mix Q8_0 and Q5_1 per layer, so the down
// step needs both. Q5_1 is an AFFINE quant: value = q5·d + m (q5 ∈ [0,31]), so
// a block's contribution is `d·Σ(q·b) + m·Σ(b)` over the 32 block elements.

//@variants
//  default: B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"

struct MoeDownParams {
    uint ncols;          // = ff
    uint stride_a;       // unused (Q5_1 stride derived from blocks)
    uint stride_b;       // ffn_h column stride (= ff, in F32 elements)
    uint stride_d;       // M bound (= n_embd)
    uint batch_stride_a; // expert slab stride in *elements* (= ncols * n_embd)
    uint n_expert_used;
};

[[vk::push_constant]] ConstantBuffer<MoeDownParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q5_1>          data_a;
[[vk::binding(1, 0)]] StructuredBuffer<float>              data_b;     // ffn_h
[[vk::binding(2, 0)]] RWStructuredBuffer<float>            data_d;     // output
[[vk::binding(3, 0)]] StructuredBuffer<block_q5_1_packed16> data_a_packed16;
[[vk::binding(4, 0)]] StructuredBuffer<float4>             data_b_v4;  // unused alias
[[vk::binding(7, 0)]] StructuredBuffer<int>                data_ids;
[[vk::binding(8, 0)]] StructuredBuffer<float>              data_weights;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint NUM_ROWS = 2;
static const uint QUANT_K = 32;

// Per-(row, expert) partial dot product over the Q5_1 blocks of the row.
void per_expert_partial(inout float partial[NUM_ROWS],
                        uint a_offset, uint b_offset, uint first_row, uint num_rows,
                        uint num_blocks_per_row, uint tid_x) {
    for (uint ib = tid_x; ib < num_blocks_per_row; ib += BLOCK_SIZE) {
        const uint y_base = ib * QUANT_K;   // element index into ffn_h

        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint blk = a_offset + (first_row + n) * num_blocks_per_row + ib;
            const float d  = float(data_a[blk].d);
            const float m  = float(data_a[blk].m);
            const uint  qh = data_a[blk].qh;

            float s  = 0.0f; // Σ q·b
            float sb = 0.0f; // Σ b
            [unroll] for (uint j = 0; j < 16; ++j) {
                const uint vui = uint(data_a[blk].qs[j]);
                // 5-bit values: low nibble + qh bit j → element j; high nibble
                // + qh bit j+16 → element j+16 (ggml Q5_1 interleave).
                const float lo = float((vui & 0xFu)        | (((qh >> j) << 4) & 0x10u));
                const float hi = float(((vui >> 4) & 0xFu) | ((qh >> (j + 12)) & 0x10u));
                const float bl = data_b[b_offset + y_base + j];
                const float bh = data_b[b_offset + y_base + 16 + j];
                s  = mad(lo, bl, s);
                s  = mad(hi, bh, s);
                sb += bl + bh;
            }
            partial[n] += d * s + m * sb;
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

    const uint num_blocks_per_row = p.ncols / QUANT_K;

    float acc[NUM_ROWS];
    [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { acc[n] = 0.0f; }

    for (uint k = 0; k < p.n_expert_used; ++k) {
        const uint expert_id = uint(data_ids[k]);
        const float routing  = data_weights[k];

        const uint a_offset = expert_id * (p.batch_stride_a / QUANT_K);
        const uint b_offset = k * p.stride_b;

        float partial[NUM_ROWS];
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { partial[n] = 0.0f; }

        per_expert_partial(partial, a_offset, b_offset, first_row, num_rows,
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
