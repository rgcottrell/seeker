// TurboQuant Walsh-Hadamard Transform — O(d log d) orthonormal rotation in
// 128-element groups. Ported from the llama-cpp-turboquant fork's
// `ggml/src/ggml-vulkan/vulkan-shaders/turbo_wht.comp`.
//
// TurboQuant stores K and V WHT-rotated (the rotation Gaussianizes each head's
// vector so the fixed PolarQuant codebook quantizes it well). For attention to
// stay correct the query Q must be forward-rotated before QK^T, and the
// attention output inverse-rotated afterward — both via this kernel. The
// transform is its own inverse up to the per-direction 1/sqrt(128) scaling (two
// passes compose to identity: H² = 128·I), and it is orthonormal so
// `<WHT(Q), WHT(K)> = <Q,K>`.
//
// One workgroup (128 threads) processes one 128-element group. A tensor of
// `ne` elements (= head_dim · n_head · n_tokens, head_dim a multiple of 128)
// is `ne/128` independent groups; head_dim=256 is simply two groups per head,
// each rotated independently — the dot product decomposes over the 128-chunks.
//
// direction 0 = forward (pre-scale by S1, post-scale by S2);
// direction 1 = inverse (pre-scale by S2, post-scale by S1).

import glsl; // gl_LocalInvocationID / SV_* builtins

#include "turbo_consts.slang" // TURBO_S1, TURBO_S2, TURBO_INV_SQRT_128

static const uint GROUP = 128;

struct WhtParams {
    uint ne;        // total element count (multiple of 128)
    uint direction; // 0 = forward, 1 = inverse
};
[[vk::push_constant]] ConstantBuffer<WhtParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>   data_a; // input  (F32, contiguous)
[[vk::binding(1, 0)]] RWStructuredBuffer<float> data_d; // output (F32, contiguous)

groupshared float x[GROUP];

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid  = gl_LocalInvocationID.x;
    // Pack the group index into 3D the same way the reference does, so a large
    // group count fits Vulkan's per-dimension workgroup-count limits.
    const uint group = gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x;
    const uint base  = group * GROUP;

    if (base + tid >= p.ne) { return; }

    const float fs = (p.direction == 0u) ? TURBO_S1[tid] : TURBO_S2[tid];
    const float ss = (p.direction == 0u) ? TURBO_S2[tid] : TURBO_S1[tid];

    x[tid] = data_a[base + tid];
    GroupMemoryBarrierWithGroupSync();

    x[tid] *= fs;
    GroupMemoryBarrierWithGroupSync();

    // Hadamard butterfly: stages h = 1, 2, 4, ..., 64.
    [unroll] for (uint h = 1u; h < 128u; h *= 2u) {
        if ((tid % (2u * h)) < h) {
            float a = x[tid];
            float b = x[tid + h];
            x[tid]     = a + b;
            x[tid + h] = a - b;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    data_d[base + tid] = x[tid] * TURBO_INV_SQRT_128 * ss;
}
