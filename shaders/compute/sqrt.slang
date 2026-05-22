// Ported from llama.cpp ggml-vulkan sqrt.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = get_idx(gl_GlobalInvocationID);
    if (idx >= p.ne) { return; }
    const FLOAT_TYPE val = FLOAT_TYPE(data_a[get_aoffset() + src0_idx(idx)]);
    data_d[get_doffset() + dst_idx(idx)] = D_TYPE(sqrt(val));
}
