// Ported from llama.cpp ggml-vulkan opt_step_sgd.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<A_TYPE> data_x;
[[vk::binding(1, 0)]] StructuredBuffer<A_TYPE>   data_grad;
[[vk::binding(2, 0)]] StructuredBuffer<float>    data_params;  // [alpha, weight_decay]

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u + gl_GlobalInvocationID.y * 512u + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }

    const float alpha = data_params[0];
    const float keep  = 1.0f - alpha * data_params[1];

    data_x[i] = A_TYPE(float(data_x[i]) * keep - alpha * float(data_grad[i]));
}
