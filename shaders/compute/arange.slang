// Ported from llama.cpp ggml-vulkan arange.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: D_TYPE=float       FLOAT_TYPE=float
//  f16: D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "generic_head.slang"
#include "types.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= p.KX) { return; }
    // param1 = start, param2 = step
    data_d[i] = D_TYPE(p.param1 + p.param2 * float(i));
}
