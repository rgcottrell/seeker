// Ported from llama.cpp ggml-vulkan xielu.comp, MIT license — see LICENSE.llama-cpp
//
// Extended-ELU activation. param1=alpha_n, param2=alpha_p, param3=beta, param4=eps.

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u + gl_GlobalInvocationID.y * 512u + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }

    float x = float(data_a[i]);
    const float alpha_n = p.param1;
    const float alpha_p = p.param2;
    const float beta    = p.param3;
    const float eps     = p.param4;

    if (x > 0.0f) {
        x = alpha_p * x * x + beta * x;
    } else {
        const float min_x_eps = min(x, eps);
        x = (exp(min_x_eps) - 1.0f - x) * alpha_n + beta * x;
    }

    data_d[i] = D_TYPE(x);
}
