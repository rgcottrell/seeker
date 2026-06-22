// Ported from llama.cpp ggml-vulkan flash_attn_split_k_reduce.comp, MIT license — see LICENSE.llama-cpp
//
// Combines per-split (L, m, O) partials into the final attention output. Each workgroup
// handles one (row, head, batch) triple; D dimension is split across workgroup-y blocks.

//@variants
//  f32:
//@end-variants

#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 32;
// When 1, the merged output is written FLAT token-major to match the
// VARLEN single-pass FA write: sequence i3 contributes data_dyn[i3].n_query
// rows packed at data_dyn[i3].q_start (so row `n` of seq i3 → flat token
// `q_start + n`). Rows `n >= n_query` are not this sequence's (the main
// kernel skipped them, leaving garbage partials) — skip them so we neither
// read garbage nor write past / into another sequence's columns. When 0,
// the legacy [HSV, n_head, ne1, B] layout is written (n = row within seq i3).
[SpecializationConstant] const uint VARLEN = 0;

// The FA kernels run their online softmax in the log2 domain (Q pre-scaled
// by log2(e)), so the per-split (L, M) partials are log2-domain values and
// the merge uses exp2. Sinks arrive in the natural-log domain from the host
// and are converted on read.
static const float LOG2E = 1.44269504088896340736f;

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

// Per-forward dynamic params — see src/inference/decode_dyn.rs. The
// reduce kernel reads `k_num` (number of partials to merge) from this
// instead of the push struct so the recorded cmdbuf is replay-stable.
#include "decode_dyn.slang"
[[vk::binding(3, 0)]] StructuredBuffer<DecodeDyn> data_dyn;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint n   = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    const uint i2  = gl_WorkGroupID.z % p.ne2;
    const uint i3  = gl_WorkGroupID.z / p.ne2;

    // VARLEN: row `n` belongs to sequence i3 only if n < its n_query; the flat
    // output column is q_start + n. (Uniform across the wave, so the early-out
    // doesn't break the WaveActive reductions below.)
    uint q_start = 0u;
    if (VARLEN != 0u) {
        if (n >= data_dyn[i3].n_query) { return; }
        q_start = data_dyn[i3].q_start;
    }

    const uint D     = p.D;
    // k_num varies per decode token — read from DecodeDyn rather than push.
    const uint k_num = data_dyn[0].k_num;

    const uint l_offset = D * p.ne1 * p.ne2 * p.ne3 * k_num + p.ne1 * 2u * (k_num * (i2 + p.ne2 * i3)) + n;
    const uint m_offset = D * p.ne1 * p.ne2 * p.ne3 * k_num + p.ne1 * 2u * (k_num * (i2 + p.ne2 * i3)) + p.ne1 + n;
    const uint lm_stride = p.ne1 * 2u;

    float m_max = -1.0f / 0.0f;
    for (uint k = 0; k + tid < k_num; k += BLOCK_SIZE) {
        const float m = data_a[m_offset + (k + tid) * lm_stride];
        m_max = max(m_max, m);
    }

    // BLOCK_SIZE == 32 and the pipeline pins subgroup size 32, so the
    // workgroup is exactly one wave — reduce without shared memory.
    m_max = WaveActiveMax(m_max);

    float L = 0.0f;
    for (uint k = 0; k + tid < k_num; k += BLOCK_SIZE) {
        const float l = data_a[l_offset + (k + tid) * lm_stride];
        const float m = data_a[m_offset + (k + tid) * lm_stride];
        L += exp2(m - m_max) * l;
    }

    L = WaveActiveSum(L);

    float sink = 0.0f;
    if (p.sinks != 0) {
        sink = data_s[n] * LOG2E;  // host sinks are natural-log domain
        float ms = 1.0f;
        float vs = 1.0f;
        if (sink > m_max) { ms = exp2(m_max - sink); }
        else              { vs = exp2(sink  - m_max); }
        L = L * ms + vs;
    }

    L = (L == 0.0f) ? 0.0f : 1.0f / L;

    const uint d = tid + gl_WorkGroupID.y * BLOCK_SIZE;
    if (d < D) {
        float O = 0.0f;
        [unroll] for (uint k = 0; k < k_num; ++k) {
            const uint o_offset = D * p.ne1 * (k + k_num * (i2 + p.ne2 * i3)) + D * n + d;
            const float m = data_a[m_offset + k * lm_stride];
            O += exp2(m - m_max) * data_a[o_offset];
        }
        if (p.sinks != 0 && sink > m_max) {
            O *= exp2(m_max - sink);
        }
        O *= L;

        const float FLT_MAX_ = asfloat(0x7F7FFFFFu);
        O = clamp(O, -FLT_MAX_, FLT_MAX_);

        // seeker layout: [HSV, n_head, L] (matches flash_attn.slang's single-
        // pass write), so the wo matmul reads attn_out directly. (llama.cpp's
        // original wrote [HSV, L, n_head] here.) VARLEN writes flat token-major
        // at (q_start + n) — matching the VARLEN single-pass FA write — so the
        // unified forward's wo matmul reads the N_total-packed attn_out directly.
        if (VARLEN != 0u) {
            data_d[(q_start + n) * p.ne2 * D + i2 * D + d] = O;
        } else {
            data_d[i3 * p.ne2 * p.ne1 * D + n * p.ne2 * D + i2 * D + d] = O;
        }
    }
}
