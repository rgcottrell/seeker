// Ported from llama.cpp ggml-vulkan opt_step_adamw.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<A_TYPE> data_x;
[[vk::binding(1, 0)]] StructuredBuffer<A_TYPE>   data_grad;
[[vk::binding(2, 0)]] RWStructuredBuffer<A_TYPE> data_gradm;
[[vk::binding(3, 0)]] RWStructuredBuffer<A_TYPE> data_gradv;
[[vk::binding(4, 0)]] StructuredBuffer<float>    data_params;  // [alpha, beta1, beta2, eps, wd, beta1h, beta2h]

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u + gl_GlobalInvocationID.y * 512u + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }

    const float alpha  = data_params[0];
    const float beta1  = data_params[1];
    const float beta2  = data_params[2];
    const float eps    = data_params[3];
    const float wd     = data_params[4];
    const float beta1h = data_params[5];
    const float beta2h = data_params[6];

    const float gi  = float(data_grad[i]);
    const float gmi = float(data_gradm[i]) * beta1 + gi      * (1.0f - beta1);
    const float gvi = float(data_gradv[i]) * beta2 + gi * gi * (1.0f - beta2);

    data_gradm[i] = A_TYPE(gmi);
    data_gradv[i] = A_TYPE(gvi);

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    data_x[i] = A_TYPE(float(data_x[i]) * (1.0f - alpha * wd) - alpha * mh / vh);
}
