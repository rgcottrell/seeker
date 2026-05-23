// Ported from llama.cpp ggml-vulkan copy.comp, MIT license — see LICENSE.llama-cpp
//
// Same-type copies (f32, f16) plus cross-type casts (f32↔f16, f32↔bf16).
// BF16 is stored as raw `uint16_t` (MoltenVK on Apple Silicon does not expose
// VK_KHR_shader_bfloat16) and converted via the `bf16_to_fp32`/`fp32_to_bf16`
// helpers in types.slang.

//@variants
//  f32:         A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16:         A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//  f32_to_f16:  A_TYPE=float       D_TYPE=float16_t   FLOAT_TYPE=float
//  f16_to_f32:  A_TYPE=float16_t   D_TYPE=float       FLOAT_TYPE=float
//  f32_to_bf16: A_TYPE=float       D_TYPE=uint16_t    FLOAT_TYPE=float DATA_D_BF16=1
//  bf16_to_f32: DATA_A_BF16=1      D_TYPE=float       FLOAT_TYPE=float
//  f32_to_i32:  A_TYPE=float       D_TYPE=int         FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = get_idx(gl_GlobalInvocationID);
    if (idx >= p.ne) { return; }

#if defined(DATA_A_BF16)
    data_d[get_doffset() + dst_idx(idx)] = D_TYPE(bf16_to_fp32(uint32_t(data_a[get_aoffset() + src0_idx(idx)])));
#elif defined(DATA_D_BF16)
    float f = float(data_a[get_aoffset() + src0_idx(idx)]);
    data_d[get_doffset() + dst_idx(idx)] = D_TYPE(fp32_to_bf16(f));
#else
    data_d[get_doffset() + dst_idx(idx)] = D_TYPE(data_a[get_aoffset() + src0_idx(idx)]);
#endif
}
