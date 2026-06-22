// Fused MoE residual update for qwen35moe — the entire tail of moe_ffn
// collapsed to a single dispatch.
//
// Math: `residual[i] = residual[i] + routed[i] + shared[i] * sigmoid(gate[token])`
// where `gate[token]` is a per-token scalar broadcast across the hidden
// dimension. Replaces four sequential dispatches (sigmoid → broadcast
// mul → routed add → shared add) and eliminates the `shared_gate_sig`
// and `shared_scaled` scratch slots. 40 MoE layers × 3 saved dispatches
// + 3 saved barriers = a real chunk of the decode dispatch overhead.

//@variants
//  f32: A_TYPE=float
//@end-variants

#include "generic_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   residual_in;
[[vk::binding(1, 0)]] StructuredBuffer<A_TYPE>   routed;
[[vk::binding(2, 0)]] StructuredBuffer<A_TYPE>   shared_v;
[[vk::binding(3, 0)]] StructuredBuffer<A_TYPE>   gate_scalar; // one float per token
[[vk::binding(4, 0)]] RWStructuredBuffer<A_TYPE> residual_out;

// `p.KX` is total element count (hidden * L). `p.KY` is the hidden
// dimension — the broadcast period used to map element i → token
// `i / hidden`.

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u
                 + gl_GlobalInvocationID.y * 512u
                 + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }

    const uint token = i / p.KY;
    const float g    = float(gate_scalar[token]);
    const float sig  = 1.0f / (1.0f + exp(-g));

    const float r = float(residual_in[i]);
    const float a = float(routed[i]);
    const float s = float(shared_v[i]);

    residual_out[i] = A_TYPE(r + a + s * sig);
}
