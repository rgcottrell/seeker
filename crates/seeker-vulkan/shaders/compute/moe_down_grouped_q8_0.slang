// Grouped MoE down matvec (Q8_0 weights): the GROUPED counterpart of
// moe_down_q8_0.slang. Dispatched [n_embd/NUM_ROWS, 1, n_experts]; each
// workgroup owns one (row-tile, expert) and loops over the tokens routed to
// that expert (group_pairs[group_offsets[e] .. group_offsets[e+1]]), so the
// expert's down weight slab is read ONCE and reused across its tokens instead
// of re-read per token. It writes the *un-weighted* per-(token, slot) partial
// to `inter[(tok*n_used + slot)*n_embd + m]`; moe_down_sum.slang then folds the
// slots into the per-token output with the routing weights (so the float
// summation order matches the fused kernel — byte-identical result).

import glsl;

#include "types.slang"

struct MoeDownGroupedParams {
    uint ncols;          // = ff
    uint stride_a;       // Q8_0 row stride within an expert slab (blocks)
    uint stride_b;       // ffn_h slot stride (= ff, F32 elements)
    uint stride_d;       // n_embd (output / inter row count)
    uint batch_stride_a; // expert slab stride in *elements* (= ncols * n_embd)
    uint n_expert_used;  // pairs-per-token (for the pair decode)
    uint batch_stride_b; // ffn_h token stride (= ff * n_expert_used)
};

[[vk::push_constant]] ConstantBuffer<MoeDownGroupedParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q8_0>          data_a;
[[vk::binding(1, 0)]] StructuredBuffer<float>               data_b;     // ffn_h
[[vk::binding(2, 0)]] RWStructuredBuffer<float>             inter;      // [n_embd × n_pairs]
[[vk::binding(3, 0)]] StructuredBuffer<block_q8_0_packed16> data_a_packed16;
[[vk::binding(4, 0)]] StructuredBuffer<float4>              data_b_v4;
[[vk::binding(9, 0)]] StructuredBuffer<uint>                group_pairs;
[[vk::binding(10, 0)]] StructuredBuffer<uint>               group_offsets;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint NUM_ROWS = 2;

// Identical to moe_down_q8_0.slang's per-(row, expert) partial dot product.
void per_expert_partial(inout float partial[NUM_ROWS],
                        uint a_offset, uint b_offset, uint first_row, uint num_rows,
                        uint num_blocks_per_row, uint tid_x) {
    for (uint ib = tid_x; ib < num_blocks_per_row; ib += BLOCK_SIZE) {
        const uint y_base = ib * QUANT_K_Q8_0;

        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint blk = a_offset + (first_row + n) * num_blocks_per_row + ib;
            const float d = float(data_a[blk].d);

            float s = 0.0f;
            [unroll] for (uint g = 0; g < QUANT_K_Q8_0 / 4; ++g) {
                const uint iqs = g * 4;
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

    const uint num_blocks_per_row = p.ncols / QUANT_K_Q8_0;
    const uint a_offset = expert * (p.batch_stride_a / QUANT_K_Q8_0);

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
