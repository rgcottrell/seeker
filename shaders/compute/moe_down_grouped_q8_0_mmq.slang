// MMQ (integer-dot) variant of moe_down_grouped_q8_0.slang. Same grouping and
// output scatter — one workgroup per (row-tile, expert), loop the expert's
// tokens, write the un-weighted partial to inter[pair*stride_d + m] — but the
// per-block dot is a HW-accelerated packed int8 dot (dot4add_i8packed →
// OpSDot PackedVectorFormat4x8Bit) against a q8_1-quantized activation instead
// of a float MAC. Q8_0 has no min term, so result = d_w · d_act · Σ q_w·q_a.
// NOT bit-identical to the float kernel (the activation is quantized to int8),
// but coherence-preserving like ggml's MMQ.

import glsl;

#include "types.slang"

struct MoeDownGroupedParams {
    uint ncols;          // = ff
    uint stride_a;       // Q8_0 row stride within an expert slab (blocks)
    uint stride_b;       // q8_1 slot stride (in BLOCKS = ff/32)
    uint stride_d;       // n_embd (output / inter row count)
    uint batch_stride_a; // expert slab stride in *elements* (= ncols * n_embd)
    uint n_expert_used;  // pairs-per-token (for the pair decode)
    uint batch_stride_b; // q8_1 token stride (in BLOCKS = ff/32 * n_expert_used)
};

[[vk::push_constant]] ConstantBuffer<MoeDownGroupedParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q8_0>          data_a;          // weight (d)
[[vk::binding(1, 0)]] StructuredBuffer<block_q8_1_packed32> data_b;          // q8_1 activation (ds + packed qs)
[[vk::binding(2, 0)]] RWStructuredBuffer<float>            inter;            // [n_embd × n_pairs]
[[vk::binding(3, 0)]] StructuredBuffer<block_q8_0_packed16> data_a_packed16; // weight qs
[[vk::binding(9, 0)]] StructuredBuffer<uint>               group_pairs;
[[vk::binding(10, 0)]] StructuredBuffer<uint>              group_offsets;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint NUM_ROWS = 2;

void per_expert_partial(inout float partial[NUM_ROWS],
                        uint a_offset, uint b_blk_base, uint first_row, uint num_rows,
                        uint num_blocks_per_row, uint tid_x) {
    for (uint ib = tid_x; ib < num_blocks_per_row; ib += BLOCK_SIZE) {
        const uint ablk    = b_blk_base + ib;        // q8_1 activation block
        const float d_act  = float(data_b[ablk].ds.x);

        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint blk = a_offset + (first_row + n) * num_blocks_per_row + ib;
            const float d  = float(data_a[blk].d);
            int isum = 0;
            [unroll] for (uint g = 0; g < QUANT_K_Q8_0 / 4; ++g) {
                // weight: two int16 → 4 packed int8 (low 16 bits each)
                const int wv0 = int(data_a_packed16[blk].qs[2 * g]);
                const int wv1 = int(data_a_packed16[blk].qs[2 * g + 1]);
                const uint aw = (uint(wv0) & 0xFFFFu) | ((uint(wv1) & 0xFFFFu) << 16);
                const uint ba = uint(data_b[ablk].qs[g]); // packed int8×4
                isum = dot4add_i8packed(aw, ba, isum);
            }
            partial[n] = mad(d * d_act, float(isum), partial[n]);
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
        const uint b_blk_base = slot * p.stride_b + tok * p.batch_stride_b;

        float partial[NUM_ROWS];
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) { partial[n] = 0.0f; }
        per_expert_partial(partial, a_offset, b_blk_base, first_row, num_rows,
                           num_blocks_per_row, tid_x);
        [unroll] for (uint n = 0; n < NUM_ROWS; ++n) {
            const float sum = WaveActiveSum(partial[n]);
            if (tid_x == 0u && n < num_rows) {
                inter[pair * p.stride_d + first_row + n] = sum;
            }
        }
    }
}
