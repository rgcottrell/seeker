// Minimal flash-attention forward pass — F32 Q, F16 K/V, F32 O.
//
// Scope: dense K/V (no quants), MASK_ENABLE-only via the basic data_m binding (no
// MASK_OPT prefilter, no ALiBi slopes, no attention sinks, no split-K, no logit
// softcap, no MMQ integer-dot path). This is the simplest correctness reference;
// the production llama.cpp shader bolts the optimizations back on later.
//
// One workgroup per (query-row, head, batch). 32 threads = 1 subgroup cooperate
// across the head_size dim (HSK). The K/V iteration is sequential over Bc-wide
// blocks. Per KV block:
//   - Compute S[Bc] = Q · K^T[Bc]
//   - Apply mask if MASK_ENABLE
//   - Online softmax update (m, L)
//   - Accumulate O += P · V

//@variants
//  f32_f16: Q_TYPE=float    KV_TYPE=float16_t    D_TYPE=float
//@end-variants

import glsl;  // gl_SubgroupInvocationID

#include "types.slang"

[SpecializationConstant] const uint Bc           = 32;
[SpecializationConstant] const uint HSK          = 128;
[SpecializationConstant] const uint HSV          = 128;
[SpecializationConstant] const uint MASK_ENABLE  = 0;
[SpecializationConstant] const uint Clamp        = 0;

static const uint WORKGROUP_SIZE = 32;

struct FaParams {
    uint N;      // sequence length (number of query rows)
    uint KV;     // KV length

    uint ne1;
    uint ne2;
    uint ne3;

    uint neq2;
    uint neq3;
    uint nek2;
    uint nek3;
    uint nev2;
    uint nev3;
    uint nem1;
    uint nem2;
    uint nem3;

    uint nb01;
    uint nb02;
    uint nb03;
    uint nb11;
    uint nb12;
    uint nb13;
    uint nb21;
    uint nb22;
    uint nb23;

    float scale;
    float max_bias;          // unused in this stripped variant
    float logit_softcap;     // unused

    uint  mask_n_head_log2;  // unused
    float m0;                // unused
    float m1;                // unused

    uint gqa_ratio;
    uint split_kv;
    uint k_num;
};

[[vk::push_constant]] ConstantBuffer<FaParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<Q_TYPE>   data_q;
[[vk::binding(1, 0)]] StructuredBuffer<KV_TYPE>  data_k;
[[vk::binding(2, 0)]] StructuredBuffer<KV_TYPE>  data_v;
[[vk::binding(3, 0)]] StructuredBuffer<KV_TYPE>  data_m;
[[vk::binding(5, 0)]] RWStructuredBuffer<D_TYPE> data_o;

// Shared Q row (already premultiplied by scale).
groupshared float Q_sh[HSK];
// Per-block S values and probabilities.
groupshared float S_sh[Bc];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid    = gl_LocalInvocationID.x;
    const uint i_row  = gl_WorkGroupID.x;   // which Q row (within batch+head)
    const uint iq2    = gl_WorkGroupID.y;   // head index
    const uint iq3    = gl_WorkGroupID.z;   // batch index

    if (i_row >= p.N) { return; }

    // Broadcast factors so K/V can have fewer heads/batches than Q.
    const uint rk2 = p.neq2 / p.nek2;
    const uint rk3 = p.neq3 / p.nek3;
    const uint rv2 = p.neq2 / p.nev2;
    const uint rv3 = p.neq3 / p.nev3;
    const uint ik3 = iq3 / rk3;
    const uint ik2 = iq2 / rk2;
    const uint iv3 = iq3 / rv3;
    const uint iv2 = iq2 / rv2;

    // Q offset for this (row, head, batch). nb01 == row stride in elements.
    const uint q_off = iq3 * p.nb03 + iq2 * p.nb02 + i_row * p.nb01;

    // Load Q row into shmem, premultiplied by p.scale. 32 threads each load HSK/32 elems.
    for (uint d = tid; d < HSK; d += WORKGROUP_SIZE) {
        Q_sh[d] = float(data_q[q_off + d]) * p.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Per-thread state. Each thread owns HSV/WORKGROUP_SIZE = HSV/32 output elements.
    const uint HSV_PER_THREAD = HSV / WORKGROUP_SIZE;
    float Of[HSV_PER_THREAD];
    [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) { Of[d] = 0.0f; }
    float Lf = 0.0f;
    float Mf = -1.0e30f;

    // Mask base for this row (mask layout: [batch, head_kv, row, KV]).
    const uint m_off = (iq3 % p.nem3) * p.nem2 * p.nem1 * p.KV
                     + (iq2 % p.nem2) * p.nem1 * p.KV
                     + i_row * p.KV;

    // Iterate KV blocks.
    const uint num_blocks = (p.KV + Bc - 1) / Bc;
    for (uint j = 0; j < num_blocks; ++j) {
        const uint kv_base = j * Bc;

        // ── 1) Compute S[c] = dot(Q, K[kv_base+c]) for c in [0, Bc).
        //     32 threads × Bc outputs: each thread does one or more S values.
        for (uint c0 = 0; c0 < Bc; c0 += WORKGROUP_SIZE) {
            const uint c = c0 + tid;
            float s = 0.0f;
            if (c < Bc && kv_base + c < p.KV) {
                const uint k_off = iq3 * p.nb13 + ik2 * p.nb12 + (kv_base + c) * p.nb11;
                [unroll] for (uint d = 0; d < HSK; ++d) {
                    s += Q_sh[d] * float(data_k[k_off + d]);
                }
                if (MASK_ENABLE != 0) {
                    s += float(data_m[m_off + kv_base + c]);
                }
            } else {
                s = -1.0e30f;
            }
            if (c < Bc) { S_sh[c] = s; }
        }
        GroupMemoryBarrierWithGroupSync();

        // ── 2) Row max over the new S row + the running m.
        float M_new = Mf;
        [unroll] for (uint c = 0; c < Bc; ++c) {
            M_new = max(M_new, S_sh[c]);
        }

        // ── 3) Online softmax: rescale Of and Lf, then accumulate from this block.
        const float m_scale = exp(Mf - M_new);
        Lf *= m_scale;
        [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) {
            Of[d] *= m_scale;
        }
        // Compute P[c] = exp(S[c] - M_new), accumulate sum into Lf, and accumulate
        // Of[d] += P[c] * V[kv_base+c, d_global].
        [unroll] for (uint c = 0; c < Bc; ++c) {
            if (kv_base + c >= p.KV) { break; }
            const float p_c = exp(S_sh[c] - M_new);
            Lf += p_c;
            const uint v_off = iq3 * p.nb23 + iv2 * p.nb22 + (kv_base + c) * p.nb21;
            [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) {
                const uint d_global = d * WORKGROUP_SIZE + tid;
                Of[d] += p_c * float(data_v[v_off + d_global]);
            }
        }
        Mf = M_new;
        GroupMemoryBarrierWithGroupSync();
    }

    // ── 4) Final normalize + write back.
    const float inv_L = Lf == 0.0f ? 0.0f : 1.0f / Lf;
    const uint o_off = iq3 * p.ne2 * p.ne1 * HSV + iq2 * p.ne1 * HSV + i_row * HSV;
    [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) {
        const uint d_global = d * WORKGROUP_SIZE + tid;
        data_o[o_off + d_global] = D_TYPE(Of[d] * inv_L);
    }
}
