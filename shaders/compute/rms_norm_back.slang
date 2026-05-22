// Ported from llama.cpp ggml-vulkan rms_norm_back.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   B_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

static const uint BLOCK_SIZE = 512;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;  // upstream gradient
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE> data_b;  // forward input
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

groupshared FLOAT_TYPE sum_xx[BLOCK_SIZE];
groupshared FLOAT_TYPE sum_xg[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint row = gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;

    sum_xx[tid] = FLOAT_TYPE(0.0f);
    sum_xg[tid] = FLOAT_TYPE(0.0f);

    [unroll] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        const FLOAT_TYPE gi = FLOAT_TYPE(data_a[row * p.KX + col]);
        const FLOAT_TYPE xi = FLOAT_TYPE(data_b[row * p.KX + col]);
        sum_xx[tid] += xi * xi;
        sum_xg[tid] += xi * gi;
    }

    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (tid < uint(s)) {
            sum_xx[tid] += sum_xx[tid + s];
            sum_xg[tid] += sum_xg[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const FLOAT_TYPE eps     = FLOAT_TYPE(p.param1);
    const FLOAT_TYPE mean    = sum_xx[0] / FLOAT_TYPE(p.KX);
    const FLOAT_TYPE scale_g = rsqrt(mean + eps);
    const FLOAT_TYPE scale_x = -scale_g * sum_xg[0] / (sum_xx[0] + FLOAT_TYPE(p.KX) * eps);

    [unroll] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        data_d[row * p.KX + col] = D_TYPE(
            scale_g * FLOAT_TYPE(data_a[row * p.KX + col]) +
            scale_x * FLOAT_TYPE(data_b[row * p.KX + col])
        );
    }
}
