// Ported from llama.cpp ggml-vulkan dequant_q8_0.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_Q8_0=1 D_TYPE=float       FLOAT_TYPE=float
//  f16: DATA_A_Q8_0=1 D_TYPE=float16_t   FLOAT_TYPE=float
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
    const uint il  = tid / 32;
    const uint ir  = tid % 32;
    const uint ib  = 32 * i + ir;
    if (ib >= p.nel / 32) { return; }

    const uint b_idx = 1024 * i + 32 * ir + 16 * il;
    const uint q_idx = 16 * il;
    const float d = float(data_a[ib].d);

    [unroll] for (uint l = 0; l < 16; l += 2) {
        data_b[b_idx + l    ] = D_TYPE(d * float(data_a[ib].qs[q_idx + l    ]));
        data_b[b_idx + l + 1] = D_TYPE(d * float(data_a[ib].qs[q_idx + l + 1]));
    }
}
