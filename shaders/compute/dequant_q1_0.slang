// Ported from llama.cpp ggml-vulkan dequant_q1_0.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_Q1_0=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_Q1_0=1 D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint i = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;

    const uint tid = gl_LocalInvocationID.x % 64;
    const uint il  = tid / 4;       // 0..15 — which qs byte
    const uint ir  = tid % 4;       // 0..3
    const uint ib  = 4 * i + ir;
    if (ib >= p.nel / 128) { return; }

    const uint b_idx = 512 * i + 128 * ir + 8 * il;

    const float d    = float(data_a[ib].d);
    const uint  bits = uint(data_a[ib].qs[il]);

    [unroll] for (uint l = 0; l < 8; ++l) {
        // 1-bit-per-weight: bit set → +d, clear → -d.
        data_b[b_idx + l] = D_TYPE((bits & (1u << l)) != 0u ? d : -d);
    }
}
