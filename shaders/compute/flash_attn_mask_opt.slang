// Ported from llama.cpp ggml-vulkan flash_attn_mask_opt.comp, MIT license — see LICENSE.llama-cpp
//
// Pre-scans the attention mask in Br×Bc tiles, packing two bits per tile into the output
// uint: bit 0 → "all -inf", bit 1 → "all zero". Lets the FA kernel cheap-skip whole tiles.
// Aliased data_a / data_av4 at GLSL binding 0 are split to bindings 0 and 2 here.

//@variants
//  f32:
//@end-variants

import glsl;  // gl_SubgroupID, gl_SubgroupInvocationID

#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE    = 128;
[SpecializationConstant] const uint NUM_SUBGROUPS = 4;
[SpecializationConstant] const uint Br            = 32;
[SpecializationConstant] const uint Bc            = 32;

struct FaMaskOptParams {
    uint nem0;
    uint nem1;
    uint nem2;
    uint nbm1;
    uint nbm2;
    uint nbm3;
    uint nbd1;
    uint nbd2;
    uint nbd3;
};

[[vk::push_constant]] ConstantBuffer<FaMaskOptParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>    data_a;
[[vk::binding(2, 0)]] StructuredBuffer<float16_t4>   data_av4;  // aliased view of data_a
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>       data_d;

groupshared float minsh[NUM_SUBGROUPS];
groupshared float maxsh[NUM_SUBGROUPS];

static const float FLT_MAX_OVER_2 = 1.7014117e+38f;  // 0x7EFFFFFF

void loadvec4(inout uint result, uint i0, uint i1, uint i2, uint i3, bool need_bc, uint tid) {
    [unroll] for (uint block_x = 0; block_x < 16; ++block_x) {
        float min_v =  FLT_MAX_OVER_2;
        float max_v = -FLT_MAX_OVER_2;

        [unroll] for (uint i = 0; i < Br * Bc / 4; i += BLOCK_SIZE) {
            uint j0 = (i + tid) % (Bc / 4);
            uint j1 = (i + tid) / (Bc / 4);

            j0 *= 4;
            j0 += (i0 * 16 + block_x) * Bc;
            j1 += i1 * Br;

            if (!need_bc || j0 + 3 < p.nem0) {
                const float4 f = float4(data_av4[(j0 + j1 * p.nbm1 + i2 * p.nbm2 + i3 * p.nbm3) / 4]);
                [unroll] for (int c = 0; c < 4; ++c) {
                    min_v = min(min_v, f[c]);
                    max_v = max(max_v, f[c]);
                }
            } else {
                [unroll] for (uint c = 0; c < 4; ++c) {
                    if (j0 + c < p.nem0) {
                        const float f = float(data_a[j0 + c + j1 * p.nbm1 + i2 * p.nbm2 + i3 * p.nbm3]);
                        min_v = min(min_v, f);
                        max_v = max(max_v, f);
                    }
                }
            }
        }
        min_v = WaveActiveMin(min_v);
        max_v = WaveActiveMax(max_v);
        if (gl_SubgroupInvocationID == 0) {
            minsh[gl_SubgroupID] = min_v;
            maxsh[gl_SubgroupID] = max_v;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0) {
            [unroll] for (uint i = 0; i < NUM_SUBGROUPS; ++i) {
                min_v = min(min_v, minsh[i]);
                max_v = max(max_v, maxsh[i]);
            }
            if (max_v <= -FLT_MAX_OVER_2) { result |= 1u << (2 * block_x); }
            if (min_v == 0.0f && max_v == 0.0f) { result |= 2u << (2 * block_x); }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint  gl_LocalInvocationIndex : SV_GroupIndex) {
    const uint tid = gl_LocalInvocationIndex;
    const uint i0  = gl_WorkGroupID.x;
    const uint i1  = gl_WorkGroupID.y;
    const uint i2  = gl_WorkGroupID.z % p.nem2;
    const uint i3  = gl_WorkGroupID.z / p.nem2;

    uint result = 0;

    if ((p.nem0 % Bc) == 0 && (p.nem1 % Br) == 0 &&
        ((Br * Bc) % (BLOCK_SIZE * 4)) == 0) {
        if ((i0 + 1) * 16 * Bc <= p.nem0) { loadvec4(result, i0, i1, i2, i3, false, tid); }
        else                              { loadvec4(result, i0, i1, i2, i3, true,  tid); }
    } else {
        [unroll] for (uint block_x = 0; block_x < 16; ++block_x) {
            float min_v =  FLT_MAX_OVER_2;
            float max_v = -FLT_MAX_OVER_2;

            [unroll] for (uint i = 0; i < Br * Bc; i += BLOCK_SIZE) {
                if ((Br * Bc % BLOCK_SIZE) != 0 && i + tid >= Br * Bc) { continue; }
                uint j0 = (i + tid) % Bc;
                uint j1 = (i + tid) / Bc;

                j0 += (i0 * 16 + block_x) * Bc;
                j1 += i1 * Br;

                if (j0 < p.nem0 && j1 < p.nem1) {
                    const float f = float(data_a[j0 + j1 * p.nbm1 + i2 * p.nbm2 + i3 * p.nbm3]);
                    min_v = min(min_v, f);
                    max_v = max(max_v, f);
                }
            }
            min_v = WaveActiveMin(min_v);
            max_v = WaveActiveMax(max_v);
            if (gl_SubgroupInvocationID == 0) {
                minsh[gl_SubgroupID] = min_v;
                maxsh[gl_SubgroupID] = max_v;
            }
            GroupMemoryBarrierWithGroupSync();
            if (tid == 0) {
                [unroll] for (uint i = 0; i < NUM_SUBGROUPS; ++i) {
                    min_v = min(min_v, minsh[i]);
                    max_v = max(max_v, maxsh[i]);
                }
                if (max_v <= -FLT_MAX_OVER_2) { result |= 1u << (2 * block_x); }
                if (min_v == 0.0f && max_v == 0.0f) { result |= 2u << (2 * block_x); }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (tid == 0) {
        data_d[i0 + i1 * p.nbd1 + i2 * p.nbd2 + i3 * p.nbd3] = result;
    }
}
