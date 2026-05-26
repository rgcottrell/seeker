// Ported from llama.cpp ggml-vulkan flash_attn_split_k_reduce.comp, MIT license — see LICENSE.llama-cpp
//
// Combines per-split (L, m, O) partials into the final attention output. Each workgroup
// handles one (row, head, batch) triple; D dimension is split across workgroup-y blocks.

//@variants
//  f32:
//@end-variants

#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 32;

struct FaSplitKParams {
    uint D;
    uint ne1;
    uint ne2;
    uint ne3;
    uint k_num;
    uint sinks;
};

[[vk::push_constant]] ConstantBuffer<FaSplitKParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<float>   data_s;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> data_d;

groupshared float tmpsh[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint n   = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    const uint i2  = gl_WorkGroupID.z % p.ne2;
    const uint i3  = gl_WorkGroupID.z / p.ne2;

    const uint D     = p.D;
    const uint k_num = p.k_num;

    const uint l_offset = D * p.ne1 * p.ne2 * p.ne3 * k_num + p.ne1 * 2u * (p.k_num * (i2 + p.ne2 * i3)) + n;
    const uint m_offset = D * p.ne1 * p.ne2 * p.ne3 * k_num + p.ne1 * 2u * (p.k_num * (i2 + p.ne2 * i3)) + p.ne1 + n;
    const uint lm_stride = p.ne1 * 2u;

    float m_max = -1.0f / 0.0f;
    for (uint k = 0; k + tid < k_num; k += BLOCK_SIZE) {
        const float m = data_a[m_offset + (k + tid) * lm_stride];
        m_max = max(m_max, m);
    }

    tmpsh[tid] = m_max;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            m_max = max(m_max, tmpsh[tid + s]);
            tmpsh[tid] = m_max;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    m_max = tmpsh[0];
    GroupMemoryBarrierWithGroupSync();

    float L = 0.0f;
    for (uint k = 0; k + tid < k_num; k += BLOCK_SIZE) {
        const float l = data_a[l_offset + (k + tid) * lm_stride];
        const float m = data_a[m_offset + (k + tid) * lm_stride];
        L += exp(m - m_max) * l;
    }

    tmpsh[tid] = L;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            L += tmpsh[tid + s];
            tmpsh[tid] = L;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    L = tmpsh[0];

    float sink = 0.0f;
    if (p.sinks != 0) {
        sink = data_s[n];
        float ms = 1.0f;
        float vs = 1.0f;
        if (sink > m_max) { ms = exp(m_max - sink); }
        else              { vs = exp(sink  - m_max); }
        L = L * ms + vs;
    }

    L = (L == 0.0f) ? 0.0f : 1.0f / L;

    const uint d = tid + gl_WorkGroupID.y * BLOCK_SIZE;
    if (d < D) {
        float O = 0.0f;
        [unroll] for (uint k = 0; k < k_num; ++k) {
            const uint o_offset = D * p.ne1 * (k + p.k_num * (i2 + p.ne2 * i3)) + D * n + d;
            const float m = data_a[m_offset + k * lm_stride];
            O += exp(m - m_max) * data_a[o_offset];
        }
        if (p.sinks != 0 && sink > m_max) {
            O *= exp(m_max - sink);
        }
        O *= L;

        const float FLT_MAX_ = asfloat(0x7F7FFFFFu);
        O = clamp(O, -FLT_MAX_, FLT_MAX_);

        // seeker layout: [HSV, n_head, L] (matches flash_attn.slang's single-
        // pass write), so the wo matmul reads attn_out directly. (llama.cpp's
        // original wrote [HSV, L, n_head] here.)
        data_d[i3 * p.ne2 * p.ne1 * D + n * p.ne2 * D + i2 * D + d] = O;
    }
}
