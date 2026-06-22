// Reduction pass for split-K mul_mat_vec — sums SPLIT_K partial outputs
// per row into the final output. Input layout `[M, SPLIT_K]` (split-major
// per row), output `[M]`. One thread per output row.

//@variants
//  f32: A_TYPE=float
//@end-variants

#include "generic_head.slang"

[SpecializationConstant] const uint SPLIT_K = 4;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   partials;
[[vk::binding(1, 0)]] RWStructuredBuffer<A_TYPE> out;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u
                 + gl_GlobalInvocationID.y * 512u
                 + gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }

    float sum = 0.0f;
    const uint base = i * SPLIT_K;
    [unroll] for (uint s = 0; s < SPLIT_K; ++s) {
        sum += float(partials[base + s]);
    }
    out[i] = A_TYPE(sum);
}
