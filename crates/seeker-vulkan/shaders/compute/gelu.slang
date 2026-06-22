// Ported from llama.cpp ggml-vulkan gelu.comp, MIT license — see LICENSE.llama-cpp

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
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }

    const float xi = float(data_a[i]);
    const float val = SQRT_2_OVER_PI * xi * (1.0f + GELU_COEF_A * xi * xi);
    data_d[i] = D_TYPE(0.5f * xi * (2.0f - 2.0f / (exp(2.0f * val) + 1.0f)));
}
