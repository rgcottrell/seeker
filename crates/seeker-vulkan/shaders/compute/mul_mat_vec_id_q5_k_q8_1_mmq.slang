// MMQ (integer-dot) grouped matvec_id for Q5_K expert weights × a q8_1-quantized
// activation. The compute-bound counterpart of mul_mat_vec_q5_k.slang's
// `id_grouped` variant: same grouping (group_pairs/group_offsets), same NUM_COLS
// dequant-reuse, same output positions — but each 32-element sub-block's dot is
// a HW-accelerated packed int8 dot (dot4add_i8packed → OpSDot
// PackedVectorFormat4x8Bit) instead of scalar float MACs.
//
// Q5_K dequant: w[k] = d·sc·q5[k] - dmin·mb. With a q8_1 activation x[k] =
// d_act·q8[k] (and block sum s_act = d_act·Σq8), one sub-block contributes
//   d·sc·d_act·Σ(q5·q8) - dmin·mb·s_act = d·sc·d_act·intdot - dmin·mb·s_act.
// The 5-bit weight quants (0..31) fit a signed int8, the activation is signed
// int8, so the packed dot is exact on the quantized operands. NOT bit-identical
// to the float kernel (the activation is quantized to int8) but coherence-
// preserving, like ggml's MMQ.

#include "types.slang"

struct Params {
    uint ncols;          // K = n_embd
    uint stride_d;       // output rows per (slot,token) = n_rows
    uint batch_stride_a; // expert weight slab stride in ELEMENTS (= ncols·n_rows)
    uint batch_stride_d; // output token stride (= n_rows·n_expert_used)
    uint nei0;           // n_expert_used (pair decode)
    uint blk_per_tok;    // q8_1 blocks per token = ncols/32
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q5_K>          data_a;       // weight (dm + scales)
[[vk::binding(6, 0)]] StructuredBuffer<block_q5_K_packed32> data_a_p32;   // weight (qs/qh, dword reads)
[[vk::binding(1, 0)]] StructuredBuffer<block_q8_1_packed32> data_b;       // q8_1 activation
[[vk::binding(2, 0)]] RWStructuredBuffer<float>            data_d;        // output
[[vk::binding(9, 0)]] StructuredBuffer<uint>              group_pairs;
[[vk::binding(10, 0)]] StructuredBuffer<uint>             group_offsets;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
[SpecializationConstant] const uint NUM_ROWS = 2;
[SpecializationConstant] const uint NUM_COLS = 4;

// One 32-element sub-block `sb` (superblock i = sb/8, sub-block s = sb%8): extract
// the 32 weight q5 quants into 8 packed-int8 words, then dot against NUM_COLS
// tokens' q8_1 activation blocks.
void calc_subblock(inout float temp[NUM_COLS][NUM_ROWS],
                   uint a_offset, uint b_blk_base[NUM_COLS], uint sb,
                   uint num_superblocks, uint first_row, uint num_rows) {
    const uint i = sb / 8u;       // superblock within the row
    const uint s = sb - i * 8u;   // sub-block within the superblock (0..7)
    const uint qs_uoff = (s >> 1) * 8u; // packed32 qs word offset (= (s/2)*32/4)
    const uint shift   = (s & 1u) * 4u;

    [unroll] for (uint n = 0; n < num_rows; ++n) {
        const uint blk = a_offset + (first_row + n) * num_superblocks + i;
        const float2 dm = float2(data_a[blk].dm);

        uint sc, mbyte;
        unpack_q4k_scale_min(data_a[blk].scales, s, sc, mbyte);

        // Extract the sub-block's 32 q5 quants → 8 packed-int8 words. Read the
        // 4-bit lows (qs) and 1-bit highs (qh) a dword (4 elements) at a time.
        uint wq[8];
        [unroll] for (uint g = 0; g < 8u; ++g) {
            const uint qsw = data_a_p32[blk].qs[qs_uoff + g];
            const uint qhw = data_a_p32[blk].qh[g];
            uint packed = 0u;
            [unroll] for (uint t = 0; t < 4u; ++t) {
                const uint nib  = (qsw >> (8u * t + shift)) & 0xFu;
                const uint hbit = (qhw >> (8u * t + s)) & 1u;
                packed |= (nib | (hbit << 4)) << (8u * t);
            }
            wq[g] = packed;
        }

        const float d_sc  = dm.x * float(sc);
        const float dm_mb = dm.y * float(mbyte);
        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            const uint ablk = b_blk_base[j] + sb;
            const float2 ds = float2(data_b[ablk].ds);
            int isum = 0;
            [unroll] for (uint g = 0; g < 8u; ++g) {
                isum = dot4add_i8packed(wq[g], uint(data_b[ablk].qs[g]), isum);
            }
            temp[j][n] += d_sc * ds.x * float(isum) - dm_mb * ds.y;
        }
    }
}

// Masked subgroup-reduce → gathered output offset (mirrors the float grouped path).
void reduce_nc(inout float temp[NUM_COLS][NUM_ROWS],
               uint d_offsets[NUM_COLS], uint first_row, uint num_rows,
               uint tid, uint nvalid) {
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const float sum = WaveActiveSum(temp[j][n]);
            if (tid == 0u && j < nvalid) {
                data_d[d_offsets[j] + first_row + n] = sum;
            }
        }
    }
}

void grouped_main(uint expert, uint first_row, uint num_rows, uint tid) {
    const uint start = group_offsets[expert];
    const uint end   = group_offsets[expert + 1u];
    if (start == end) { return; }

    const uint num_superblocks = p.ncols / QUANT_K_Q5_K; // 256-elem superblocks per row
    const uint num_blocks_32   = p.ncols / 32u;          // 32-elem sub-blocks per row
    const uint a_offset = expert * (p.batch_stride_a / QUANT_K_Q5_K);

    for (uint chunk = start; chunk < end; chunk += NUM_COLS) {
        const uint nvalid = min(NUM_COLS, end - chunk);
        uint b_blk_base[NUM_COLS];
        uint d_offsets[NUM_COLS];
        [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
            const uint idx  = (chunk + j < end) ? (chunk + j) : (end - 1u); // clamp pad
            const uint pair = group_pairs[idx];
            const uint tok  = pair / p.nei0;
            const uint slot = pair - tok * p.nei0;
            b_blk_base[j] = tok * p.blk_per_tok;
            d_offsets[j]  = slot * p.stride_d + tok * p.batch_stride_d;
        }

        float temp[NUM_COLS][NUM_ROWS];
        [unroll] for (uint jj = 0; jj < NUM_COLS; ++jj) {
            [unroll] for (uint ii = 0; ii < NUM_ROWS; ++ii) { temp[jj][ii] = 0.0f; }
        }
        for (uint sb = tid; sb < num_blocks_32; sb += BLOCK_SIZE) {
            calc_subblock(temp, a_offset, b_blk_base, sb, num_superblocks, first_row, num_rows);
        }
        reduce_nc(temp, d_offsets, first_row, num_rows, tid, nvalid);
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID: SV_GroupID, uint3 gl_LocalInvocationID: SV_GroupThreadID) {
    const uint expert    = gl_WorkGroupID.z;
    const uint first_row = NUM_ROWS * gl_WorkGroupID.x;
    if (first_row >= p.stride_d) { return; }
    const uint num_rows = min(NUM_ROWS, p.stride_d - first_row);
    grouped_main(expert, first_row, num_rows, gl_LocalInvocationID.x);
}
