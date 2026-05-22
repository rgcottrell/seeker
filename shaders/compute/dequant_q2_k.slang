// Ported from llama.cpp ggml-vulkan dequant_q2_k.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_Q2_K=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_Q2_K=1 D_TYPE=float16_t   FLOAT_TYPE=float
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

        const uint tid = gl_LocalInvocationID.x;
        const uint ip = tid / 32;
        const uint il = tid - 32 * ip;
        const uint is = 8 * ip + il / 16;

        const uint y_idx  = i * QUANT_K + 128 * ip + il;
        const uint ql_idx = 32 * ip + il;
        const uint8_t qs  = data_a[i].qs[32 * ip + il];

        const FLOAT_TYPE dall = FLOAT_TYPE(data_a[i].dm.x);
        const FLOAT_TYPE dmin = FLOAT_TYPE(data_a[i].dm.y);

        data_b[y_idx +  0] = D_TYPE(dall * FLOAT_TYPE((data_a[i].scales[is + 0] & 0xF) * ((qs >> 0) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is + 0] >> 4));
        data_b[y_idx + 32] = D_TYPE(dall * FLOAT_TYPE((data_a[i].scales[is + 2] & 0xF) * ((qs >> 2) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is + 2] >> 4));
        data_b[y_idx + 64] = D_TYPE(dall * FLOAT_TYPE((data_a[i].scales[is + 4] & 0xF) * ((qs >> 4) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is + 4] >> 4));
        data_b[y_idx + 96] = D_TYPE(dall * FLOAT_TYPE((data_a[i].scales[is + 6] & 0xF) * ((qs >> 6) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is + 6] >> 4));
    }
}
