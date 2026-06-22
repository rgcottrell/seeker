// Ported from llama.cpp ggml-vulkan norm.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"
#include "reduce_funcs.slang"

static const uint BLOCK_SIZE = 512;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint row = gl_WorkGroupID.z * 262144 + gl_WorkGroupID.y * 512 + gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;

    float2 partial = float2(0.0f, 0.0f);

    [unroll] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        const float xi = float(data_a[row * p.KX + col]);
        partial.x += xi;
        partial.y += xi * xi;
    }

    const float2 sum = workgroup_sum2(partial, tid, BLOCK_SIZE);

    const float mean = sum.x / p.KX;
    const float var  = sum.y / p.KX - mean * mean;
    const float inv_std = rsqrt(var + p.param1);

    [unroll] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        data_d[row * p.KX + col] = D_TYPE((float(data_a[row * p.KX + col]) - mean) * inv_std);
    }
}
