// Grouped MoE down matvec (Q5_1 weights): GROUPED counterpart of
// moe_down_q5_1.slang (see moe_down_grouped_q8_0.slang for the scheme). One
// workgroup per (row-tile, expert); reads the expert's down weight slab once and
// reuses it across the expert's tokens; writes the un-weighted per-(token, slot)
// partial to `inter[(tok*n_used+slot)*n_embd + m]`; moe_down_sum.slang folds the
// slots with the routing weights (byte-identical to the fused kernel).

import glsl;

#include "types.slang"

struct MoeDownGroupedParams {
    uint ncols;          // = ff
    uint stride_a;       // unused (Q5_1 stride derived from blocks)
    uint stride_b;       // ffn_h slot stride (= ff, F32 elements)
    uint stride_d;       // n_embd (output / inter row count)
    uint batch_stride_a; // expert slab stride in *elements* (= ncols * n_embd)
    uint n_expert_used;  // pairs-per-token (for the pair decode)
    uint batch_stride_b; // ffn_h token stride (= ff * n_expert_used)
};

[[vk::push_constant]] ConstantBuffer<MoeDownGroupedParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q5_1>          data_a;
[[vk::binding(1, 0)]] StructuredBuffer<float>              data_b;     // ffn_h
[[vk::binding(2, 0)]] RWStructuredBuffer<float>            inter;      // [n_embd × n_pairs]
[[vk::binding(3, 0)]] StructuredBuffer<block_q5_1_packed16> data_a_packed16;
[[vk::binding(9, 0)]] StructuredBuffer<uint>               group_pairs;
[[vk::binding(10, 0)]] StructuredBuffer<uint>              group_offsets;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint NUM_ROWS = 2;
static const uint QUANT_K = 32;

// Identical to moe_down_q5_1.slang's per-(row, expert) partial dot product.
void per_expert_partial(inout float partial[NUM_ROWS],
                        uint a_offset, uint b_offset, uint first_row, uint num_rows,
                        uint num_blocks_per_row, uint tid_x) {
    for (uint ib = tid_x; ib < num_blocks_per_row; ib += BLOCK_SIZE) {
        const uint y_base = ib * QUANT_K;

        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint blk = a_offset + (first_row + n) * num_blocks_per_row + ib;
            const float d  = float(data_a[blk].d);
            const float m  = float(data_a[blk].m);
            const uint  qh = data_a[blk].qh;

            float s  = 0.0f;
            float sb = 0.0f;
            [unroll] for (uint j = 0; j < 16; ++j) {
                const uint vui = uint(data_a[blk].qs[j]);
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
void main(uint3 gl_WorkGroupID: SV_GroupID, uint3 gl_LocalInvocationID: SV_GroupThreadID) {
    const uint tid_x = gl_LocalInvocationID.x;
    const uint expert = gl_WorkGroupID.z;
    const uint first_row = NUM_ROWS * gl_WorkGroupID.x;
    if (first_row >= p.stride_d) { return; }
    const uint num_rows = (first_row + NUM_ROWS <= p.stride_d)
        ? NUM_ROWS
        : (p.stride_d - first_row);

    const uint start = group_offsets[expert];
    const uint end   = group_offsets[expert + 1u];
    if (start == end) { return; }

    const uint num_blocks_per_row = p.ncols / QUANT_K;
    const uint a_offset = expert * (p.batch_stride_a / QUANT_K);

    for (uint g = start; g < end; ++g) {
        const uint pair = group_pairs[g];
        const uint tok  = pair / p.n_expert_used;
        const uint slot = pair - tok * p.n_expert_used;
        const uint b_offset = slot * p.stride_b + tok * p.batch_stride_b;

        float partial[NUM_ROWS];
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { partial[n] = 0.0f; }
        per_expert_partial(partial, a_offset, b_offset, first_row, num_rows,
                           num_blocks_per_row, tid_x);
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) {
            const float sum = WaveActiveSum(partial[n]);
            if (tid_x == 0u && n < num_rows) {
                inter[pair * p.stride_d + first_row + n] = sum;
            }
        }
    }
}
