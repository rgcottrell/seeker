// MoE down-projection matvec with routing-weight fusion (Q8_0 weights).
// Same kernel structure as `moe_down_q5_k.slang` / `moe_down_q6_k.slang`
// — only the dequant inner logic differs. Used for the down step of MoE
// FFN blocks whose `ffn_down_exps` are quantized as Q8_0 (the Unsloth
// UD-Q5_K_XL checkpoint has 3 such blocks; the rest are Q6_K).
//
// Q8_0 is trivially decoded (one f16 block scale × 32 i8 quants), so
// there is no scale-cache to manage. Each lane owns a strided subset of
// the QUANT_K_Q8_0-sized blocks along the row and the caller reduces the
// per-lane partials with WaveActiveSum.

//@variants
//  default: DATA_A_Q8_0=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"

struct MoeDownParams {
    uint ncols;          // = ff
    uint stride_a;       // Q8_0 row stride within an expert slab (blocks)
    uint stride_b;       // ffn_h column stride (= ff, in F32 elements)
    uint stride_d;       // M bound (= n_embd)
    uint batch_stride_a; // expert slab stride in *elements* (= ncols * n_embd)
    uint n_expert_used;
};

[[vk::push_constant]] ConstantBuffer<MoeDownParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q8_0>          data_a;
[[vk::binding(1, 0)]] StructuredBuffer<float>               data_b;     // ffn_h
[[vk::binding(2, 0)]] RWStructuredBuffer<float>             data_d;     // output
[[vk::binding(3, 0)]] StructuredBuffer<block_q8_0_packed16> data_a_packed16;
[[vk::binding(4, 0)]] StructuredBuffer<float4>              data_b_v4;
[[vk::binding(7, 0)]] StructuredBuffer<int>                 data_ids;
[[vk::binding(8, 0)]] StructuredBuffer<float>               data_weights;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint NUM_ROWS = 2;

// Per-(row, expert) partial dot product. Each lane strides over the
// QUANT_K_Q8_0 (=32) element blocks of the row; within a block it walks
// the 32 quants in float4 chunks, sign-extending two i8 from each packed
// int16. The caller reduces lanes via WaveActiveSum.
void per_expert_partial(inout float partial[NUM_ROWS],
                        uint a_offset, uint b_offset, uint first_row, uint num_rows,
                        uint num_blocks_per_row, uint tid_x) {
    for (uint ib = tid_x; ib < num_blocks_per_row; ib += BLOCK_SIZE) {
        const uint y_base = ib * QUANT_K_Q8_0;   // element index into ffn_h

        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint blk = a_offset + (first_row + n) * num_blocks_per_row + ib;
            const float d = float(data_a[blk].d);

            float s = 0.0f;
            [unroll] for (uint g = 0; g < QUANT_K_Q8_0 / 4; ++g) {
                const uint iqs = g * 4;
                // Sign-extend 4 i8 packed into 2 int16 (low/high byte each).
                const int v0 = int(data_a_packed16[blk].qs[iqs / 2]);
                const int v1 = int(data_a_packed16[blk].qs[iqs / 2 + 1]);
                const float4 q = float4((v0 << 24) >> 24, (v0 << 16) >> 24,
                                        (v1 << 24) >> 24, (v1 << 16) >> 24);
                const float4 by = float4(data_b_v4[(b_offset + y_base + iqs) / 4]);
                s = mad(by.x, q.x, s);
                s = mad(by.y, q.y, s);
                s = mad(by.z, q.z, s);
                s = mad(by.w, q.w, s);
            }
            partial[n] = mad(d, s, partial[n]);
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

    const uint num_blocks_per_row = p.ncols / QUANT_K_Q8_0;

    float acc[NUM_ROWS];
    [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { acc[n] = 0.0f; }

    for (uint k = 0; k < p.n_expert_used; ++k) {
        const uint expert_id = uint(data_ids[k]);
        const float routing  = data_weights[k];

        const uint a_offset = expert_id * (p.batch_stride_a / QUANT_K_Q8_0);
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
