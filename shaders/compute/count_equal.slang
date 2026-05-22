// Ported from llama.cpp ggml-vulkan count_equal.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  i32: A_TYPE=int       B_TYPE=int       D_TYPE=int       FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 64;
static const uint CHUNK_SIZE = 512;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint base = gl_WorkGroupID.x * CHUNK_SIZE;
    const uint col  = gl_LocalInvocationID.x;

    uint count = 0;
    [unroll] for (uint i = 0; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        const uint idx = base + i + col;
        if (idx >= p.KX) { break; }
        count += uint(data_a[idx] == data_b[idx]);
    }

    InterlockedAdd(data_d[0], D_TYPE(count));
}
