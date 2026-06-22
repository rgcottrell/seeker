// Fused 3-way element-wise sum: `dst[i] = a[i] + b[i] + c[i]`.
// Used by qwen35moe's MoE FFN to collapse the two sequential residual
// updates (`residual += routed`; `residual += shared_scaled`) into one
// dispatch. With `dst == a` the kernel is the in-place
// `residual += routed + shared_scaled` form; otherwise it's a generic
// 3-way add. 40 MoE layers × one saved dispatch + barrier = 80 ops/decode.

//@variants
//  f32: A_TYPE=float
//@end-variants

#include "generic_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<A_TYPE>   data_b;
[[vk::binding(2, 0)]] StructuredBuffer<A_TYPE>   data_c;
[[vk::binding(3, 0)]] RWStructuredBuffer<A_TYPE> data_d;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u
                 + gl_GlobalInvocationID.y * 512u
                 + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }
    data_d[i] = A_TYPE(float(data_a[i]) + float(data_b[i]) + float(data_c[i]));
}
