// Gather rows from a Q3_K-quantized A by index list B, dequantize to D.
//
// Q3_K's superblock packs a 2-bit `qs` plane + a 1-bit `hmask` high plane +
// 6-bit packed sub-scales (`scales[12]`), so it has no generic
// `dequantize(ib, iqs)` overload. Like `get_rows_q4_k.slang` / `get_rows_q5_k.slang`,
// this mirrors `dequant_q3_k.slang`'s per-element formula but adds row
// indirection so each workgroup targets a single output row.
//
// Dispatch shape (host): workgroups = [ne00 / QUANT_K, L, 1] where L is the
// number of indices in B and ne00 is the row size in elements. One workgroup
// dequants one 256-element Q3_K block into 256 F32 outputs (64 threads × 4).

//@variants
//  default: DATA_A_Q3_K=1 B_TYPE=int D_TYPE=float FLOAT_TYPE=float
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
    const uint ib = src_row * blocks_per_row + block_in_row;

    // Threading mirrors dequant_q3_k.slang: 64 threads, each writes 4 elems.
    const uint lid = gl_LocalInvocationID.x;
    const uint r   = lid / 4;          // 0..15
    const uint tid = r / 2;            // 0..7
    const uint is0 = r % 2;            // 0,1
    const uint l0  = 16 * is0 + 4 * (lid % 4);
    const uint n = tid / 4;            // 0,1
    const uint j = tid - 4 * n;        // 0..3

    const uint8_t m     = uint8_t(1 << (4 * n + j));
    const uint    is    = 8 * n + 2 * j + is0;
    const uint    shift = 2 * j;

    // Unpack the 6-bit sub-block scale `us` from the 12 packed scale bytes.
    const int8_t us = int8_t(
        is <  4 ? (data_a[ib].scales[is - 0] & 0xF) | (((data_a[ib].scales[is + 8] >> 0) & 3) << 4) :
        is <  8 ? (data_a[ib].scales[is - 0] & 0xF) | (((data_a[ib].scales[is + 4] >> 2) & 3) << 4) :
        is < 12 ? (data_a[ib].scales[is - 8] >>  4) | (((data_a[ib].scales[is + 0] >> 4) & 3) << 4) :
                  (data_a[ib].scales[is - 8] >>  4) | (((data_a[ib].scales[is - 4] >> 6) & 3) << 4)
    );
    const FLOAT_TYPE d_all = FLOAT_TYPE(data_a[ib].d);
    const FLOAT_TYPE dl    = d_all * FLOAT_TYPE(us - 32);

    const uint y_base  = out_row * p.nb21 + block_in_row * QUANT_K + 128 * n + 32 * j;
    const uint qs_idx  = 32 * n;

    for (uint l = l0; l < l0 + 4; ++l) {
        data_d[y_base + l] = D_TYPE(dl * FLOAT_TYPE(
            int8_t((data_a[ib].qs[qs_idx + l] >> shift) & 3) -
            (((data_a[ib].hmask[l] & m) != uint8_t(0)) ? 0 : 4)
        ));
    }
}
