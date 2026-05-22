// Ported from llama.cpp ggml-vulkan soft_max_back.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   B_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 32;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_g;  // gradient
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE> data_y;  // forward softmax output
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

groupshared FLOAT_TYPE sum_yg[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint row = gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (row >= p.KY) { return; }

    const FLOAT_TYPE scale = FLOAT_TYPE(p.param1);

    sum_yg[tid] = FLOAT_TYPE(0.0f);
    [unroll] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        const FLOAT_TYPE gi = FLOAT_TYPE(data_g[row * p.KX + col]);
        const FLOAT_TYPE yi = FLOAT_TYPE(data_y[row * p.KX + col]);
        sum_yg[tid] += yi * gi;
    }

    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) { sum_yg[tid] += sum_yg[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }

    const FLOAT_TYPE dot_yg = sum_yg[0];

    [unroll] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        data_d[row * p.KX + col] = D_TYPE(scale
            * (FLOAT_TYPE(data_g[row * p.KX + col]) - dot_yg)
            * FLOAT_TYPE(data_y[row * p.KX + col]));
    }
}
