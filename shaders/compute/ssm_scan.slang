// Ported from llama.cpp ggml-vulkan ssm_scan.comp, MIT license — see LICENSE.llama-cpp
//
// Selective state-space scan (Mamba). Each subgroup owns one head-state slice; we run an
// outer token loop and per-iteration accumulate `state = state * dA + B * x_dt`, reducing
// the dot product with C across the subgroup. USE_SUBGROUP_ADD swaps the shmem reduce for
// WaveActiveSum.

//@variants
//  f32_subgroup: USE_SUBGROUP_ADD=1
//  f32_shared:
//@end-variants

import glsl;  // gl_SubgroupID, gl_SubgroupInvocationID

#include "types.slang"

[SpecializationConstant] const uint D_STATE       = 128;
[SpecializationConstant] const uint SUBGROUP_SIZE = 32;

static const uint c_factor = D_STATE / SUBGROUP_SIZE;

[[vk::binding(0, 0)]] StructuredBuffer<float>   s0;
[[vk::binding(1, 0)]] StructuredBuffer<float>   x;
[[vk::binding(2, 0)]] StructuredBuffer<float>   dt;
[[vk::binding(3, 0)]] StructuredBuffer<float>   A;
[[vk::binding(4, 0)]] StructuredBuffer<float>   B;
[[vk::binding(5, 0)]] StructuredBuffer<float>   C;
[[vk::binding(6, 0)]] StructuredBuffer<int>     ids;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> d;

struct SsmScanParams {
    uint nb02; uint nb03; uint nb12; uint nb13;
    uint nb21; uint nb22; uint nb31;
    uint nb42; uint nb43; uint nb52; uint nb53;
    uint s_off;
    uint n_head;
    uint d_head;
    uint n_group;
    uint n_tok;
};

[[vk::push_constant]] ConstantBuffer<SsmScanParams> p;

float softplus(float v) {
    return v <= 20.0f ? log(1.0f + exp(v)) : v;
}

#if !defined(USE_SUBGROUP_ADD)
groupshared float temp[D_STATE];
#endif

[shader("compute")]
[numthreads(D_STATE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint subgroup = gl_SubgroupID;
    const uint lane     = gl_SubgroupInvocationID;
    const uint tid      = subgroup * SUBGROUP_SIZE + lane;
    const uint subgroup_idx = gl_WorkGroupID.x * c_factor + subgroup;

    const uint head_idx = subgroup_idx / p.d_head;
    const uint head_off = (subgroup_idx % p.d_head) * 4u;
    const uint seq_idx  = gl_WorkGroupID.y;

    const uint group_off    = (head_idx / (p.n_head / p.n_group)) * D_STATE * 4u;
    const uint s0_base_idx  = (uint(ids[seq_idx]) * p.nb03 + head_idx * p.nb02 + head_off * D_STATE) / 4u;
    const uint x_base_idx   = (seq_idx * p.nb13 + subgroup_idx * 4u) / 4u;
    const uint dt_base_idx  = (seq_idx * p.nb22 + head_idx * 4u) / 4u;
    const uint A_base_idx   = (head_idx * p.nb31) / 4u;
    const uint B_base_idx   = (seq_idx * p.nb43 + group_off) / 4u;
    const uint C_base_idx   = (seq_idx * p.nb53 + group_off) / 4u;
    const uint y_base_idx   = seq_idx * p.n_tok * p.n_head * p.d_head + subgroup_idx;
    const uint s_base_idx   = (p.s_off + seq_idx * p.nb03 + head_idx * p.nb02 + head_off * D_STATE) / 4u;

    const uint stride_x  = p.nb12 / 4u;
    const uint stride_dt = p.nb21 / 4u;
    const uint stride_B  = p.nb42 / 4u;
    const uint stride_C  = p.nb52 / 4u;
    const uint stride_y  = p.n_head * p.d_head;

    float state[c_factor];
    [unroll] for (uint j = 0; j < c_factor; j++) {
        state[j] = s0[s0_base_idx + SUBGROUP_SIZE * j + lane];
    }

    const float a = A[A_base_idx];

    for (uint i = 0; i < p.n_tok; i++) {
        const float dt_soft_plus = softplus(dt[dt_base_idx + i * stride_dt]);
        const float dA   = exp(dt_soft_plus * a);
        const float x_dt = x[x_base_idx + i * stride_x] * dt_soft_plus;

        float state_sum = 0.0f;
        [unroll] for (uint j = 0; j < c_factor; j++) {
            const float Bv = B[B_base_idx + i * stride_B + SUBGROUP_SIZE * j + lane];
            const float Cv = C[C_base_idx + i * stride_C + SUBGROUP_SIZE * j + lane];
            state[j]   = state[j] * dA + Bv * x_dt;
            state_sum += state[j] * Cv;
        }

#if defined(USE_SUBGROUP_ADD)
        state_sum = WaveActiveSum(state_sum);
#else
        temp[tid] = state_sum;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint s = SUBGROUP_SIZE / 2; s > 0; s >>= 1) {
            if (lane < s) { temp[tid] += temp[tid + s]; }
            GroupMemoryBarrierWithGroupSync();
        }
        state_sum = temp[subgroup * SUBGROUP_SIZE];
        GroupMemoryBarrierWithGroupSync();
#endif

        if (lane == 0) {
            d[y_base_idx + i * stride_y] = state_sum;
        }
    }

    [unroll] for (uint j = 0; j < c_factor; j++) {
        d[s_base_idx + SUBGROUP_SIZE * j + lane] = state[j];
    }
}
