// Ported from llama.cpp ggml-vulkan diag_mask_inf.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"

struct DiagMaskParams {
    uint ncols;
    uint rows_per_channel;
    uint n_past;
};

[[vk::push_constant]] ConstantBuffer<DiagMaskParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(1, 512, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint col = gl_GlobalInvocationID.y;
    const uint row = gl_GlobalInvocationID.x;
    if (col >= p.ncols) { return; }

    const uint i = row * p.ncols + col;
    if (col > p.n_past + row % p.rows_per_channel) {
        // Negative infinity in IEEE-754 single precision.
        data_d[i] = D_TYPE(asfloat(0xFF800000u));
    } else {
        data_d[i] = D_TYPE(data_a[i]);
    }
}
