// Ported from llama.cpp ggml-vulkan dequant_q3_k.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_Q3_K=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_Q3_K=1 D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_b;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    [unroll] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint i = gl_WorkGroupID.x * 256 + wgy;
        if (i >= p.nel / QUANT_K) { return; }

        const uint r = gl_LocalInvocationID.x / 4;
        const uint tid = r / 2;
        const uint is0 = r % 2;
        const uint l0 = 16 * is0 + 4 * (gl_LocalInvocationID.x % 4);
        const uint n = tid / 4;
        const uint j = tid - 4 * n;

        const uint8_t m  = uint8_t(1 << (4 * n + j));
        const uint    is = 8 * n + 2 * j + is0;
        const uint    shift = 2 * j;

        const int8_t us = int8_t(
            is <  4 ? (data_a[i].scales[is - 0] & 0xF) | (((data_a[i].scales[is + 8] >> 0) & 3) << 4) :
            is <  8 ? (data_a[i].scales[is - 0] & 0xF) | (((data_a[i].scales[is + 4] >> 2) & 3) << 4) :
            is < 12 ? (data_a[i].scales[is - 8] >>  4) | (((data_a[i].scales[is + 0] >> 4) & 3) << 4) :
                      (data_a[i].scales[is - 8] >>  4) | (((data_a[i].scales[is - 4] >> 6) & 3) << 4)
        );
        const FLOAT_TYPE d_all = FLOAT_TYPE(data_a[i].d);
        const FLOAT_TYPE dl    = d_all * FLOAT_TYPE(us - 32);

        const uint y_idx  = i * QUANT_K + 128 * n + 32 * j;
        const uint qs_idx = 32 * n;

        for (uint l = l0; l < l0 + 4; ++l) {
            data_b[y_idx + l] = D_TYPE(dl * FLOAT_TYPE(
                int8_t((data_a[i].qs[qs_idx + l] >> shift) & 3) -
                (((data_a[i].hmask[l] & m) != uint8_t(0)) ? 0 : 4)
            ));
        }
    }
}
