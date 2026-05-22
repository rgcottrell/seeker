// Ported from llama.cpp ggml-vulkan dequant_q4_k.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_Q4_K=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_Q4_K=1 D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_b;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    [unroll] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint ib = gl_WorkGroupID.x * 256 + wgy;
        if (ib >= p.nel / QUANT_K) { return; }

        const uint tid = gl_LocalInvocationID.x;
        const uint il = tid / 8;
        const uint ir = tid % 8;
        const uint is = 2 * il;
        const uint n = 4;

        const FLOAT_TYPE dall = FLOAT_TYPE(data_a[ib].dm.x);
        const FLOAT_TYPE dmin = FLOAT_TYPE(data_a[ib].dm.y);

        const uint y_idx  = ib * QUANT_K + 64 * il + n * ir;
        const uint qs_idx = 32 * il + n * ir;

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

        [unroll] for (uint l = 0; l < n; ++l) {
            data_b[y_idx + l     ] = D_TYPE(d1 * FLOAT_TYPE(data_a[ib].qs[qs_idx + l] & 0xF) - m1);
            data_b[y_idx + l + 32] = D_TYPE(d2 * FLOAT_TYPE(data_a[ib].qs[qs_idx + l] >>  4) - m2);
        }
    }
}
