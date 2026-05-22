// Ported from llama.cpp ggml-vulkan multi_add.comp, MIT license — see LICENSE.llama-cpp
//
// Sum num_srcs source tensors into a single destination, optionally also writing the
// per-tile sum-of-squares for a fused RMS-norm pass. GLSL aliases 12 distinct buffer
// roles onto bindings 0..11 (each binding holds one VkBuffer at runtime); we keep the
// same shape but declare everything as `float` so the f32-only port works without an
// explicit reinterpret.

//@variants
//  f32:     A_TYPE=float   B_TYPE=float   D_TYPE=float   FLOAT_TYPE=float   ADD_RMS=0
//  rms_f32: A_TYPE=float   B_TYPE=float   D_TYPE=float   FLOAT_TYPE=float   ADD_RMS=1
//@end-variants

import glsl;  // gl_SubgroupID, gl_SubgroupInvocationID, gl_SubgroupSize

#include "types.slang"
#include "utils.slang"

struct MultiAddParams {
    uint ne20;
    uint ne21;
    uint ne22;
    uint ne23;
    uint nb[12][4];
    uint rms_partials;
};

[[vk::push_constant]] ConstantBuffer<MultiAddParams> p;

[SpecializationConstant] const uint num_srcs = 2;

// 12 buffer slots; for any given dispatch only slots [0, num_srcs] are touched as inputs,
// slot `num_srcs` as the output, and slot `num_srcs + 1` as the partials sink (RMS path).
[[vk::binding(0,  0)]] RWStructuredBuffer<float> buf0;
[[vk::binding(1,  0)]] RWStructuredBuffer<float> buf1;
[[vk::binding(2,  0)]] RWStructuredBuffer<float> buf2;
[[vk::binding(3,  0)]] RWStructuredBuffer<float> buf3;
[[vk::binding(4,  0)]] RWStructuredBuffer<float> buf4;
[[vk::binding(5,  0)]] RWStructuredBuffer<float> buf5;
[[vk::binding(6,  0)]] RWStructuredBuffer<float> buf6;
[[vk::binding(7,  0)]] RWStructuredBuffer<float> buf7;
[[vk::binding(8,  0)]] RWStructuredBuffer<float> buf8;
[[vk::binding(9,  0)]] RWStructuredBuffer<float> buf9;
[[vk::binding(10, 0)]] RWStructuredBuffer<float> buf10;
[[vk::binding(11, 0)]] RWStructuredBuffer<float> buf11;

float load_a(uint b, uint i) {
    switch (b) {
        case 0:  return buf0[i];
        case 1:  return buf1[i];
        case 2:  return buf2[i];
        case 3:  return buf3[i];
        case 4:  return buf4[i];
        case 5:  return buf5[i];
        case 6:  return buf6[i];
        case 7:  return buf7[i];
        case 8:  return buf8[i];
        case 9:  return buf9[i];
        case 10: return buf10[i];
        case 11: return buf11[i];
        default: return 0.0f;
    }
}

void store_d(uint b, uint i, float v) {
    switch (b) {
        case 0:  buf0[i]  = v; break;
        case 1:  buf1[i]  = v; break;
        case 2:  buf2[i]  = v; break;
        case 3:  buf3[i]  = v; break;
        case 4:  buf4[i]  = v; break;
        case 5:  buf5[i]  = v; break;
        case 6:  buf6[i]  = v; break;
        case 7:  buf7[i]  = v; break;
        case 8:  buf8[i]  = v; break;
        case 9:  buf9[i]  = v; break;
        case 10: buf10[i] = v; break;
        case 11: buf11[i] = v; break;
    }
}

void store_partial(uint b, uint i, float v) { store_d(b, i, v); }

uint src_idx(uint s, uint i00, uint i01, uint i02, uint i03) {
    return i03 * p.nb[s][3] + i02 * p.nb[s][2] + i01 * p.nb[s][1] + i00 * p.nb[s][0];
}

uint dst_idx(uint i00, uint i01, uint i02, uint i03) {
    const uint nb20 = p.nb[num_srcs][0];
    const uint nb21 = p.nb[num_srcs][1];
    const uint nb22 = p.nb[num_srcs][2];
    const uint nb23 = p.nb[num_srcs][3];
    return i03 * nb23 + i02 * nb22 + i01 * nb21 + i00 * nb20;
}

uint get_idx(uint3 gl_GlobalInvocationID) {
    return gl_GlobalInvocationID.z * 262144u + gl_GlobalInvocationID.y * 512u + gl_GlobalInvocationID.x;
}

static const uint num_threads = 256;

#if ADD_RMS
groupshared float sumsh[num_threads];
#endif

[shader("compute")]
[numthreads(num_threads, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID,
          uint3 gl_LocalInvocationID  : SV_GroupThreadID) {
    uint idx       = get_idx(gl_GlobalInvocationID);
    const uint orig_idx = idx;

    const uint ne       = p.ne20 * p.ne21 * p.ne22 * p.ne23;
    const uint num_iter = 2;

    float sum_sq = 0.0f;

    [unroll] for (uint i = 0; i < num_iter; ++i) {
        if (idx >= ne) {
            idx += num_threads;
            continue;
        }
        uint i00, i01, i02, i03;
        get_indices(idx, i00, i01, i02, i03, p.ne20, p.ne21, p.ne22, p.ne23);

        float sum = 0.0f;
        [unroll] for (uint s = 0; s < num_srcs; ++s) {
            sum += load_a(s, src_idx(s, i00, i01, i02, i03));
        }
        sum_sq += sum * sum;
        store_d(num_srcs, dst_idx(i00, i01, i02, i03), sum);

        idx += num_threads;
    }

#if ADD_RMS
    if (p.rms_partials != 0) {
        const uint NumSubgroups = num_threads / gl_SubgroupSize;
        sum_sq = WaveActiveSum(sum_sq);
        if (gl_SubgroupInvocationID == 0) {
            sumsh[gl_SubgroupID] = sum_sq;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint s = 8; s > 0; s >>= 1) {  // upper bound; NumSubgroups <= num_threads/gl_SubgroupSize, capped here for unroll
            if (gl_SubgroupID < s && gl_SubgroupInvocationID == 0 && s < NumSubgroups) {
                sum_sq += sumsh[gl_SubgroupID + s];
                sumsh[gl_SubgroupID] = sum_sq;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (gl_SubgroupID == 0 && gl_SubgroupInvocationID == 0) {
            store_partial(num_srcs + 1, orig_idx / (num_iter * num_threads), sum_sq);
        }
    }
#endif
}
