// Ported from llama.cpp ggml-vulkan copy.comp, MIT license — see LICENSE.llama-cpp
//
// Same-type copies (f32, f16) plus cross-type casts (f32↔f16). BF16 lives
// in cast_bf16.slang since it needs explicit bit-level conversion on MoltenVK.

//@variants
//  f32:        A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16:        A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//  f32_to_f16: A_TYPE=float       D_TYPE=float16_t   FLOAT_TYPE=float
//  f16_to_f32: A_TYPE=float16_t   D_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = get_idx(gl_GlobalInvocationID);
    if (idx >= p.ne) { return; }

    data_d[get_doffset() + dst_idx(idx)] = D_TYPE(data_a[get_aoffset() + src0_idx(idx)]);
}
