// Static per-token logit bias (`--logit-bias`). Adds a fixed bias to a sparse
// set of token logits before the rest of the sampler chain — used to boost,
// suppress, or hard-ban (`-inf`) specific tokens. One thread per entry; the
// host dedupes token ids, so two threads never target the same logit.
//
// Mirrors llama.cpp's `llama_sampler_logit_bias`. Unlike the penalty pass the
// data is constant for the session, so the count rides in a push constant and
// the (token_id, bias) table is uploaded once at record time (it persists
// across decode-cmdbuf replays).

//@variants
//  f32: A_TYPE=float
//@end-variants

#include "types.slang"

struct LogitBiasParams {
    uint n_entries;
};

[[vk::push_constant]] ConstantBuffer<LogitBiasParams> p;

[[vk::binding(0, 0)]] RWStructuredBuffer<A_TYPE> data_logits;
// (token_id, bias_bits): x = token id, y = the bias f32 reinterpreted as int.
[[vk::binding(1, 0)]] StructuredBuffer<int2>     data_bias;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= p.n_entries) { return; }

    const int2 entry = data_bias[i];
    const uint token_id = uint(entry.x);
    const A_TYPE bias = A_TYPE(asfloat(entry.y));
    data_logits[token_id] = data_logits[token_id] + bias;
}
