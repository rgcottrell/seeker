// Ported from llama.cpp ggml-vulkan softplus.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u + gl_GlobalInvocationID.y * 512u + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }
    const float x = float(data_a[i]);
    data_d[i] = D_TYPE((x > 20.0f) ? x : log(1.0f + exp(x)));
}
