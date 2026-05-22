// Ported from llama.cpp ggml-vulkan add_id.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "types.slang"

struct AddIdParams {
    uint ne0;
    uint ne1;
    uint s01;
    uint s02;
    uint s11;
    uint s21;
};

[[vk::push_constant]] ConstantBuffer<AddIdParams> p;

static const uint BLOCK_SIZE = 512;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] StructuredBuffer<int>      data_c;
[[vk::binding(3, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint i1 = gl_WorkGroupID.x;
    const uint i2 = gl_WorkGroupID.y;

    const uint i11 = uint(data_c[i1 + i2 * p.s21]);

    const uint s1 = p.ne0;
    const uint s2 = p.ne0 * p.ne1;

    const uint d0 = i1 * s1 + i2 * s2;
    const uint a0 = i1 * p.s01 + i2 * p.s02;
    const uint b0 = i11 * p.s11;

    for (uint i0 = gl_LocalInvocationID.x; i0 < p.ne0; i0 += BLOCK_SIZE) {
        data_d[d0 + i0] = D_TYPE(FLOAT_TYPE(data_a[a0 + i0]) + FLOAT_TYPE(data_b[b0 + i0]));
    }
}
