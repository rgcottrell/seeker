// Ported from llama.cpp ggml-vulkan dequant_f32.comp, MIT license — see LICENSE.llama-cpp
//
// Pass-through copy from f32 source into a typed dst (D_TYPE = f32 / f16 / bf16-as-uint16).

//@variants
//  f32: A_TYPE=float    D_TYPE=float        FLOAT_TYPE=float
//  f16: A_TYPE=float    D_TYPE=float16_t    FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "dequant_head.slang"

[[vk::binding(0, 0)]] StructuredBuffer<float>    data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_b;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.x * 16u;
    if (i >= p.nel) { return; }

    [unroll] for (uint l = 0; l < 16; l++) {
        data_b[i + l] = D_TYPE(data_a[i + l]);
    }
}
