// Group MoE (token, expert-slot) pairs by their selected expert id, so a
// grouped expert matvec can read each expert's weight ONCE and reuse it across
// all the tokens routed to that expert (the weight re-read per token is the MoE
// prefill bandwidth bottleneck — diffusion-gemma: 128 experts, 8 used).
//
// Input `ids` is topk_moe's `[n_experts × n_tokens]` buffer; the selected
// expert ids for token `t` live at `ids[t*n_experts + slot]` for slot in
// `[0, n_used)`. Output is a counting-sort: `group_offsets[e]` is the start of
// expert `e`'s pairs in `group_pairs`, and each `group_pairs[g]` is the packed
// pair index `tok*n_used + slot`. `group_offsets[n_experts] = n_used*n_tokens`.
//
// Single-threaded by design: at most n_used*n_tokens (a few thousand) pairs and
// n_experts (≤512) buckets, run once per MoE layer — trivially cheap, and the
// sequential count/prefix/scatter avoids atomics entirely (correctness first).

import glsl;

struct MoeGroupParams {
    uint n_tokens;
    uint n_used;    // n_expert_used (pairs per token)
    uint n_experts; // also the ids row stride
};

[[vk::push_constant]] ConstantBuffer<MoeGroupParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<uint>   ids;           // [n_experts × n_tokens]
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> group_pairs;   // [n_used × n_tokens]
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> group_offsets; // [n_experts + 1]

static const uint MAX_EXPERTS = 512;
groupshared uint counts[MAX_EXPERTS];
groupshared uint wpos[MAX_EXPERTS];

[shader("compute")]
[numthreads(MAX_EXPERTS, 1, 1)]
void main(uint3 gl_LocalInvocationID: SV_GroupThreadID) {
    const uint t = gl_LocalInvocationID.x;
    const uint ne = p.n_experts;
    const uint nu = p.n_used;
    const uint npairs = nu * p.n_tokens;

    if (t < ne) {
        counts[t] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    // Thread 0 does the whole sequential counting sort (no atomics).
    if (t == 0u) {
        for (uint g = 0u; g < npairs; ++g) {
            const uint tok = g / nu;
            const uint slot = g - tok * nu;
            const uint e = ids[tok * ne + slot];
            counts[e] += 1u;
        }
        uint acc = 0u;
        for (uint e = 0u; e < ne; ++e) {
            group_offsets[e] = acc;
            wpos[e] = acc;
            acc += counts[e];
        }
        group_offsets[ne] = acc; // = npairs
        for (uint g = 0u; g < npairs; ++g) {
            const uint tok = g / nu;
            const uint slot = g - tok * nu;
            const uint e = ids[tok * ne + slot];
            group_pairs[wpos[e]] = g;
            wpos[e] += 1u;
        }
    }
}
