// Ported from llama.cpp ggml-vulkan count_experts.comp, MIT license — see LICENSE.llama-cpp
//
// Histograms expert-id assignments. One workgroup per expert; the workgroup sweeps its
// expert id over the full (ne00, ne01) and reduces the per-thread counts via shmem.

//@variants
//  u32:
//@end-variants

#include "types.slang"

struct CountExpertsParams {
    uint ne00;
    uint ne01;
    uint nb00;
    uint nb01;
    uint a_offset;
};

[[vk::push_constant]] ConstantBuffer<CountExpertsParams> p;

static const uint BLOCK_SIZE = 256;

[[vk::binding(0, 0)]] StructuredBuffer<uint>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> data_d;

groupshared uint vals[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint expert_id    = gl_WorkGroupID.x;
    const uint num_elements = p.ne00 * p.ne01;
    const uint tid          = gl_LocalInvocationID.x;

    uint count = 0;
    for (uint idx = tid; idx < num_elements; idx += BLOCK_SIZE) {
        const uint i01 = idx / p.ne00;
        const uint i00 = idx % p.ne00;
        const uint a = data_a[p.a_offset + i01 * p.nb01 + i00 * p.nb00];
        count += uint(a == expert_id);
    }

    vals[tid] = count;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) { vals[tid] += vals[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) { data_d[expert_id] = vals[0]; }
}
