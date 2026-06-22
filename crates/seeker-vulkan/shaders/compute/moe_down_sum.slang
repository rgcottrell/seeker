// Routing-weighted cross-expert sum for the GROUPED MoE down path. The grouped
// down matvec (moe_down_grouped_*) writes a per-(token, expert-slot) partial
// `inter[(tok*n_used + slot)*n_embd + m]` (no routing weight). This kernel folds
// the n_used slots back into the per-token output, applying the (down-scale
// folded) routing weight in slot order — matching the fused kernel's k-loop so
// the result is byte-identical:
//
//   dst[tok*stride_d + m] = sum_{slot} routing[tok*n_used + slot]
//                                       * inter[(tok*n_used + slot)*n_embd + m]
//
// One workgroup per (m-tile, token); each thread owns one output row m.

import glsl;

struct MoeDownSumParams {
    uint n_embd;   // = stride_d (output row count, and inter row stride)
    uint n_used;   // n_expert_used (slots per token)
    uint n_tokens;
};

[[vk::push_constant]] ConstantBuffer<MoeDownSumParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>   inter;   // [n_embd × (n_used*n_tokens)]
[[vk::binding(1, 0)]] StructuredBuffer<float>   weights; // [n_used × n_tokens] (down-scale folded)
[[vk::binding(2, 0)]] RWStructuredBuffer<float> data_d;  // [n_embd × n_tokens]

[SpecializationConstant] const uint BLOCK_SIZE = 256;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_GlobalInvocationID: SV_DispatchThreadID, uint3 gl_WorkGroupID: SV_GroupID) {
    const uint m   = gl_GlobalInvocationID.x;
    const uint tok = gl_WorkGroupID.z;
    if (m >= p.n_embd || tok >= p.n_tokens) { return; }

    const uint pair_base = tok * p.n_used;
    float acc = 0.0f;
    for (uint slot = 0u; slot < p.n_used; ++slot) {
        const float w = weights[pair_base + slot];
        const float v = inter[(pair_base + slot) * p.n_embd + m];
        acc = mad(w, v, acc);
    }
    data_d[tok * p.n_embd + m] = acc;
}
