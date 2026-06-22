// Fused SSM `alpha = softplus(alpha_pre + bias) * ssm_a` step. Replaces
// three back-to-back elementwise dispatches (add, softplus, mul) — saves
// two barriers + two dispatch launches per SSM layer in qwen35moe.
//
// Inputs:
//   data_a (slot 0): alpha_pre, shape [KX] (= num_v * L)
//   data_b (slot 1): ssm_dt_bias, shape [KY] (= num_v) — broadcasts along L
//   data_c (slot 2): ssm_a, shape [KY] (= num_v) — broadcasts along L
// Output:
//   data_d (slot 3): alpha, shape [KX]
//
// `KX` is the total element count (num_v * L). `KY` is the broadcast
// period (num_v) used as `i % KY` to map a token's element into the
// per-head bias / ssm_a vectors. Mirrors the broadcast semantics of the
// unfused chain.

//@variants
//  f32: A_TYPE=float
//@end-variants

#include "generic_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;     // alpha_pre
[[vk::binding(1, 0)]] StructuredBuffer<A_TYPE>   data_b;     // ssm_dt_bias
[[vk::binding(2, 0)]] StructuredBuffer<A_TYPE>   data_c;     // ssm_a
[[vk::binding(3, 0)]] RWStructuredBuffer<A_TYPE> data_d;     // alpha (out)

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u
                 + gl_GlobalInvocationID.y * 512u
                 + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }
    const uint bcast = i % p.KY;
    const float x = float(data_a[i]) + float(data_b[bcast]);
    // softplus(x) — same numerically-stable form as `softplus.slang`.
    const float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    data_d[i] = A_TYPE(sp * float(data_c[bcast]));
}
