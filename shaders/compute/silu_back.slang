// Ported from llama.cpp ggml-vulkan silu_back.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   B_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_g;  // gradient
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE> data_x;  // forward input
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u + gl_GlobalInvocationID.y * 512u + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }

    // d/dx[ x * sigmoid(x) ] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    const float xi = float(data_x[i]);
    const float s  = 1.0f / (1.0f + exp(-xi));
    data_d[i] = D_TYPE(float(data_g[i]) * (s + xi * s * (1.0f - s)));
}
