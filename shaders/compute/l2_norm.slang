// Ported from llama.cpp ggml-vulkan l2_norm.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_unary_head.slang"
#include "types.slang"

static const uint BLOCK_SIZE = 512;

groupshared FLOAT_TYPE sum[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint row = gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;

    const uint i3        = row / (p.ne11 * p.ne12);
    const uint i3_offset = i3 * p.ne12 * p.ne11;
    const uint i2        = (row - i3_offset) / p.ne11;
    const uint i2_offset = i2 * p.ne11;
    const uint i1        = row - i3_offset - i2_offset;

    sum[tid] = FLOAT_TYPE(0.0f);

    [unroll] for (uint i0 = tid; i0 < p.ne00; i0 += BLOCK_SIZE) {
        const FLOAT_TYPE xi = FLOAT_TYPE(data_a[i3 * p.nb03 + i2 * p.nb02 + i1 * p.nb01 + i0]);
        sum[tid] += xi * xi;
    }

    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (tid < uint(s)) { sum[tid] += sum[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }

    const FLOAT_TYPE scale = 1.0f / max(sqrt(sum[0]), FLOAT_TYPE(p.param1));

    [unroll] for (uint i0 = tid; i0 < p.ne00; i0 += BLOCK_SIZE) {
        data_d[i3 * p.nb13 + i2 * p.nb12 + i1 * p.nb11 + i0] =
            D_TYPE(scale * FLOAT_TYPE(data_a[i3 * p.nb03 + i2 * p.nb02 + i1 * p.nb01 + i0]));
    }
}
