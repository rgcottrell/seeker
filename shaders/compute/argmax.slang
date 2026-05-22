// Ported from llama.cpp ggml-vulkan argmax.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=int         FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=int         FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const float FLT_MAX = 3.402823466e+38f;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

groupshared FLOAT_TYPE tmpmax[BLOCK_SIZE];
groupshared uint       tmp[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint row = gl_WorkGroupID.z * 262144 + gl_WorkGroupID.y * 512 + gl_WorkGroupID.x;
    const uint col = gl_LocalInvocationID.x;

    if (row >= p.KY) { return; }

    A_TYPE amax = A_TYPE(-FLT_MAX);
    uint acol = col;

    if (col < p.KX) {
        amax = data_a[row * p.KX + col];
    }

    for (uint i = col + BLOCK_SIZE; i < p.KX; i += BLOCK_SIZE) {
        A_TYPE val = data_a[row * p.KX + i];
        if (val > amax) {
            amax = val;
            acol = i;
        }
    }

    tmp[col] = acol;
    tmpmax[col] = amax;

    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (col < s && col + s < p.KX) {
            if (tmpmax[col] < tmpmax[col + s]) {
                tmpmax[col] = tmpmax[col + s];
                tmp[col]    = tmp[col + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (col == 0) {
        data_d[row] = D_TYPE(tmp[0]);
    }
}
