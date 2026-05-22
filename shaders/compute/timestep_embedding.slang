// Ported from llama.cpp ggml-vulkan timestep_embedding.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"

struct TimestepParams {
    uint nb1;
    uint dim;
    uint max_period;
};

[[vk::push_constant]] ConstantBuffer<TimestepParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_WorkGroupID.y;
    const uint j = gl_GlobalInvocationID.x;
    const uint d_offset = i * p.nb1;
    const uint half_dim = p.dim / 2;

    if (p.dim % 2 != 0 && j == half_dim) {
        data_d[d_offset + 2 * half_dim] = D_TYPE(0.0f);
    }

    if (j >= half_dim) { return; }

    const float timestep = float(data_a[i]);
    const float freq = exp(-log(float(p.max_period)) * float(j) / float(half_dim));
    const float arg  = timestep * freq;
    data_d[d_offset + j]            = D_TYPE(cos(arg));
    data_d[d_offset + j + half_dim] = D_TYPE(sin(arg));
}
