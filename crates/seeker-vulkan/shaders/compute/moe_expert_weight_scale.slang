// Fold a per-expert scale into the MoE routing weights (DiffusionGemma's
// `ffn_down_exps.scale`). The reference applies a per-expert scale to each
// expert's down output before the routing-weight multiply + cross-expert sum;
// because both are scalars per (used-slot, token), multiplying the routing
// weight `weights[slot,token]` by `scale[expert(slot,token)]` is identical and
// lets seeker reuse its fused `moe_down_*` kernel unchanged.
//
// `ids` is the top-k expert table `[n_experts, n_tokens]` from `topk_moe`
// (slot k of token t at `t*n_experts + k`); `weights` is `[n_used, n_tokens]`
// (slot k of token t at `t*n_used + k`), updated in place.

struct Params {
    uint total;      // n_used * n_tokens (invocations)
    uint n_used;     // experts kept per token
    uint n_experts;  // total experts the router scores (ids row stride)
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<uint>    ids;     // [n_experts, n_tokens]
[[vk::binding(1, 0)]] StructuredBuffer<float>   scale;   // [n_experts]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> weights; // [n_used, n_tokens]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = gid.x;
    if (i >= p.total) { return; }
    const uint token = i / p.n_used;
    const uint slot  = i % p.n_used;
    const uint expert = ids[token * p.n_experts + slot];
    weights[i] = weights[i] * scale[expert];
}
