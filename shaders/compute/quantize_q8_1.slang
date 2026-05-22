// Ported from llama.cpp ggml-vulkan quantize_q8_1.comp, MIT license — see LICENSE.llama-cpp
//
// On-the-fly quantization from f32 → Q8_1. Each block holds 32 i8 values plus an (d, sum*d)
// pair; 8 threads cooperate per block via either WaveClusteredAdd (USE_SUBGROUPS) or a tree
// reduction in shmem. The QBLOCK_X4 variant packs four blocks together for 128-bit stores.

//@variants
//  f32:           QBLOCK_X4=0
//  f32_x4:        QBLOCK_X4=1
//  f32_sg:        QBLOCK_X4=0  USE_SUBGROUPS=1
//  f32_x4_sg:     QBLOCK_X4=1  USE_SUBGROUPS=1
//@end-variants

import glsl;  // gl_SubgroupInvocationID, gl_NumWorkGroups

#include "types.slang"

struct QuantizeQ81Params {
    uint ne;
    uint num_blocks;
};

[[vk::push_constant]] ConstantBuffer<QuantizeQ81Params> p;

[SpecializationConstant] const uint GROUP_SIZE = 32;

[[vk::binding(0, 0)]] StructuredBuffer<float4>                       data_a;
#if QBLOCK_X4
[[vk::binding(1, 0)]] RWStructuredBuffer<block_q8_1_x4>              data_b;
#else
[[vk::binding(1, 0)]] RWStructuredBuffer<block_q8_1_packed32>        data_b;
#endif

#if !defined(USE_SUBGROUPS)
groupshared float shmem[GROUP_SIZE];
#endif

uint pack_i8x4(int4 v) {
    return (uint(v.x) & 0xFFu)
         | ((uint(v.y) & 0xFFu) << 8)
         | ((uint(v.z) & 0xFFu) << 16)
         | ((uint(v.w) & 0xFFu) << 24);
}

void quantize(uint wgid, uint tid) {
    // Each thread handles a float4; 8 threads → one Q8_1 block.
    const uint blocks_per_group = GROUP_SIZE / 8u;
    const uint block_in_wg = tid / 8u;
    const uint ib = wgid * blocks_per_group + block_in_wg;
    const uint iqs = tid % 8u;

#if QBLOCK_X4
    const uint ibx4_outer = ib / 4u;
    const uint ibx4_inner = ib % 4u;
    const uint required_x4_blocks = (p.ne + 127u) / 128u;
    if (ibx4_outer >= required_x4_blocks) { return; }
#endif

    const uint a_idx = ib * 8u + iqs;
    float4 vals = a_idx < p.ne / 4u ? data_a[a_idx] : float4(0, 0, 0, 0);
    const float4 abs_vals = abs(vals);
    const float thread_max = max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w));

#if defined(USE_SUBGROUPS)
    // 8-wide cluster max via butterfly shuffles (countbits trick).
    float amax = thread_max;
    [unroll] for (uint m = 1; m < 8; m *= 2) {
        amax = max(amax, WaveReadLaneAt(amax, WaveGetLaneIndex() ^ m));
    }
#else
    shmem[tid] = thread_max;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 4u; s > 0u; s >>= 1u) {
        if (iqs < s) { shmem[tid] = max(shmem[tid], shmem[tid + s]); }
        GroupMemoryBarrierWithGroupSync();
    }
    const float amax = shmem[block_in_wg * 8u];
#endif

    const float d     = amax / 127.0f;
    const float d_inv = d != 0.0f ? 1.0f / d : 0.0f;
    vals = round(vals * d_inv);

    const int4 ivals = int4(vals);
    const uint packed = pack_i8x4(ivals);
#if QBLOCK_X4
    data_b[ibx4_outer].qs[ibx4_inner * 8u + iqs] = int(packed);
#else
    data_b[ib].qs[iqs] = int(packed);
#endif

#if !defined(USE_SUBGROUPS)
    GroupMemoryBarrierWithGroupSync();
#endif

    const float thread_sum = vals.x + vals.y + vals.z + vals.w;
#if defined(USE_SUBGROUPS)
    float sum = thread_sum;
    [unroll] for (uint m = 1; m < 8; m *= 2) {
        sum += WaveReadLaneAt(sum, WaveGetLaneIndex() ^ m);
    }
#else
    shmem[tid] = thread_sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 4u; s > 0u; s >>= 1u) {
        if (iqs < s) { shmem[tid] += shmem[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }
    const float sum = shmem[tid];
#endif

    if (iqs == 0) {
#if QBLOCK_X4
        data_b[ibx4_outer].ds[ibx4_inner] = float16_t2(float16_t(d), float16_t(sum * d));
#else
        data_b[ib].ds = float16_t2(float16_t(d), float16_t(sum * d));
#endif
    }
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    uint wgid = gl_WorkGroupID.x;
    while (wgid < p.num_blocks) {
        quantize(wgid, gl_LocalInvocationID.x);
        wgid += gl_NumWorkGroups.x;
    }
}
