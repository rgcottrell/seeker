// Gather rows from a Q4_K-quantized A by index list B, dequantize to D.
//
// Q4_K's superblock dequant (per-2-bit-packed sub-scales, no high-bit plane)
// has no generic `dequantize(ib, iqs)` overload, so — like `get_rows_q5_k.slang`
// for Q5_K — this mirrors `dequant_q4_k.slang`'s per-element formula but adds
// row indirection so each workgroup targets a single output row. It is exactly
// `get_rows_q5_k.slang` minus the 1-bit `qh` high plane (Q4_K has no `qh`); the
// `dm`/`scales` unpack into (d1,m1,d2,m2) is byte-for-byte identical.
//
// Dispatch shape (host): workgroups = [ne00 / QUANT_K, L, 1] where L is the
// number of indices in B and ne00 is the row size in elements. One workgroup
// dequants one 256-element Q4_K block into 256 F32 outputs (64 threads × 4).

//@variants
//  default: DATA_A_Q4_K=1 B_TYPE=int D_TYPE=float FLOAT_TYPE=float
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

    const uint tid = gl_LocalInvocationID.x;
    const uint il = tid / 16;   // 0..3
    const uint ir = tid % 16;   // 0..15
    const uint is = 2 * il;

    const FLOAT_TYPE dall = FLOAT_TYPE(data_a[ib].dm.x);
    const FLOAT_TYPE dmin = FLOAT_TYPE(data_a[ib].dm.y);

    const uint y_base = out_row * p.nb21 + block_in_row * QUANT_K + 64 * il + 2 * ir;
    const uint qs_idx = 32 * il + 2 * ir;

    uint scidx0      = (is < 4) ? is        : (is + 4);
    uint scidx1      = (is < 4) ? is        : (is - 4);
    uint scidxmask1  = (is < 4) ? 0x30      : 0xC0;
    uint scidxshift1 = (is < 4) ? 0         : 2;
    uint mbidx0      = is + 4;
    uint mbidx1      = (is < 4) ? is + 4    : is;
    uint mbidxmask0  = (is < 4) ? 0xF       : 0xF0;
    uint mbidxshift0 = (is < 4) ? 0         : 4;
    uint mbidxmask1  = (is < 4) ? 0x30      : 0xC0;
    uint mbidxshift1 = (is < 4) ? 0         : 2;

    uint8_t sc    = uint8_t((data_a[ib].scales[scidx0] & 0xF) | ((data_a[ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    uint8_t mbyte = uint8_t((data_a[ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | ((data_a[ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const FLOAT_TYPE d1 = dall * sc;
    const FLOAT_TYPE m1 = dmin * mbyte;

    scidx0      = (is < 4) ? is + 1 : (is + 5);
    scidx1      = (is < 4) ? is + 1 : (is - 3);
    scidxmask1  = (is < 4) ? 0x30   : 0xC0;
    scidxshift1 = (is < 4) ? 0      : 2;
    mbidx0      = is + 5;
    mbidx1      = (is < 4) ? is + 5 : is + 1;
    mbidxmask0  = (is < 4) ? 0xF    : 0xF0;
    mbidxshift0 = (is < 4) ? 0      : 4;
    mbidxmask1  = (is < 4) ? 0x30   : 0xC0;
    mbidxshift1 = (is < 4) ? 0      : 2;

    sc    = uint8_t((data_a[ib].scales[scidx0] & 0xF) | ((data_a[ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    mbyte = uint8_t((data_a[ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | ((data_a[ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const FLOAT_TYPE d2 = dall * sc;
    const FLOAT_TYPE m2 = dmin * mbyte;

    data_d[y_base     ] = D_TYPE(d1 * FLOAT_TYPE(data_a[ib].qs[qs_idx    ] & 0xF) - m1);
    data_d[y_base +  1] = D_TYPE(d1 * FLOAT_TYPE(data_a[ib].qs[qs_idx + 1] & 0xF) - m1);
    data_d[y_base + 32] = D_TYPE(d2 * FLOAT_TYPE(data_a[ib].qs[qs_idx    ]  >> 4) - m2);
    data_d[y_base + 33] = D_TYPE(d2 * FLOAT_TYPE(data_a[ib].qs[qs_idx + 1]  >> 4) - m2);
}
