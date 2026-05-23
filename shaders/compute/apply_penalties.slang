// In-place repeat / frequency / presence penalties applied at a sparse set
// of token positions. One thread per `(token_id, count)` pair — the host
// dedupes the recent-token ring buffer first, so two threads never target
// the same logit and no atomics are needed.
//
// Mirrors `llama_sampler_penalties_apply` in llama.cpp:
//   l = (l <= 0) ? l * repeat_p : l / repeat_p;
//   l -= count * freq_p + (count > 0 ? presence_p : 0)

//@variants
//  f32: A_TYPE=float
//@end-variants

#include "types.slang"

struct PenaltyParams {
    uint  n_pairs;
    float repeat_p;
    float freq_p;
    float presence_p;
};

[[vk::push_constant]] ConstantBuffer<PenaltyParams> p;

[[vk::binding(0, 0)]] RWStructuredBuffer<A_TYPE> data_logits;
[[vk::binding(1, 0)]] StructuredBuffer<int2>     data_pairs;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= p.n_pairs) { return; }

    const int2 pair = data_pairs[i];
    const uint token_id = uint(pair.x);
    const uint count    = uint(pair.y);

    A_TYPE l = data_logits[token_id];
    if (l <= A_TYPE(0)) {
        l = l * A_TYPE(p.repeat_p);
    } else {
        l = l / A_TYPE(p.repeat_p);
    }
    const A_TYPE bias = A_TYPE(float(count) * p.freq_p + (count > 0 ? p.presence_p : 0.0));
    data_logits[token_id] = l - bias;
}
