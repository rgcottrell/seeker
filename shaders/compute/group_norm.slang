// Ported from llama.cpp ggml-vulkan group_norm.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

static const uint BLOCK_SIZE = 512;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

groupshared float tmp[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint group_size = p.KX;
    const float eps = p.param1;

    const uint tid = gl_LocalInvocationID.x;
    const uint start = gl_WorkGroupID.x * group_size + tid;
    const uint end   = (gl_WorkGroupID.x + 1) * group_size;

    tmp[tid] = 0.0f;

    // Mean
    [unroll] for (uint col = start; col < end; col += BLOCK_SIZE) {
        tmp[tid] += float(data_a[col]);
    }

    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (tid < uint(s)) { tmp[tid] += tmp[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }

    const float mean = tmp[0] / group_size;
    GroupMemoryBarrierWithGroupSync();
    tmp[tid] = 0.0f;

    // Variance — write centered values and accumulate squares.
    [unroll] for (uint col = start; col < end; col += BLOCK_SIZE) {
        const float xi = float(data_a[col]) - mean;
        data_d[col] = D_TYPE(xi);
        tmp[tid] += xi * xi;
    }

    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (tid < uint(s)) { tmp[tid] += tmp[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }

    const float variance = tmp[0] / group_size;
    const float scale = rsqrt(variance + eps);

    [unroll] for (uint col = start; col < end; col += BLOCK_SIZE) {
        data_d[col] *= D_TYPE(scale);
    }
}
