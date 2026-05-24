// Gather rows from a Q6_K-quantized A by index list B, dequantize to D.
//
// Q6_K's inline dequant is too involved for the generic `dequantize(ib, iqs)`
// pattern in `get_rows_quant.slang` (no `dequantize` overload exists for
// Q6_K — only the inline path in `mul_mat_vec_q6_k.slang` and the full-tensor
// `dequant_q6_k.slang`). This shader mirrors `dequant_q6_k.slang`'s
// per-element formula but adds row indirection so we can target a single
// output row per workgroup.
//
// Dispatch shape (host): workgroups = [ne00 / QUANT_K, L, 1] where L is the
// number of indices in B and ne00 is the row size in elements. One workgroup
// dequants one 256-element Q6_K block into 256 F32 outputs.

//@variants
//  default: DATA_A_Q6_K=1 B_TYPE=int D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"
#include "generic_binary_head.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint block_in_row = gl_WorkGroupID.x;
    const uint out_row      = gl_WorkGroupID.y;

    const uint blocks_per_row = p.ne00 / QUANT_K;
    if (block_in_row >= blocks_per_row) { return; }
    if (out_row >= p.ne10)              { return; }

    // Indices buffer is `[L]` of u32; pick the source row id.
    const uint src_row = uint(data_b[out_row * p.nb10]);

    const uint a_block = src_row * blocks_per_row + block_in_row;

    const uint tid = gl_LocalInvocationID.x;
    const uint ip  = tid / 32;             // 0 or 1 (which 128-element half)
    const uint il  = tid - 32 * ip;        // 0..31
    const uint is  = 8 * ip + il / 16;     // 0..1 or 8..9

    const uint y_base = out_row * p.nb21 + block_in_row * QUANT_K + 128 * ip + il;
    const uint ql_idx = 64 * ip + il;
    const uint8_t qh  = data_a[a_block].qh[32 * ip + il];

    const FLOAT_TYPE d = FLOAT_TYPE(data_a[a_block].d);

    data_d[y_base +  0] = D_TYPE(d * FLOAT_TYPE(data_a[a_block].scales[is + 0] * (int8_t((data_a[a_block].ql[ql_idx +  0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32)));
    data_d[y_base + 32] = D_TYPE(d * FLOAT_TYPE(data_a[a_block].scales[is + 2] * (int8_t((data_a[a_block].ql[ql_idx + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32)));
    data_d[y_base + 64] = D_TYPE(d * FLOAT_TYPE(data_a[a_block].scales[is + 4] * (int8_t((data_a[a_block].ql[ql_idx +  0] >>  4) | (((qh >> 4) & 3) << 4)) - 32)));
    data_d[y_base + 96] = D_TYPE(d * FLOAT_TYPE(data_a[a_block].scales[is + 6] * (int8_t((data_a[a_block].ql[ql_idx + 32] >>  4) | (((qh >> 6) & 3) << 4)) - 32)));
}
