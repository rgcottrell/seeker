// BF16 ↔ F32 bit-level conversion. MoltenVK on Apple Silicon does not expose
// VK_KHR_shader_bfloat16, so we store BF16 as `uint16_t` and convert manually.
//
// BF16 layout = top 16 bits of an IEEE 754 single (sign + 8-bit exp + 7-bit
// mantissa). Decode: `f32_bits = uint(bf16) << 16`. Encode (round-to-nearest-
// even): add the bias `0x7FFF + ((bits >> 16) & 1)` then truncate.

//@variants
//  bf16_to_f32: A_TYPE=uint16_t   D_TYPE=float       FLOAT_TYPE=float DIR_BF16_TO_F32=1
//  f32_to_bf16: A_TYPE=float       D_TYPE=uint16_t   FLOAT_TYPE=float DIR_F32_TO_BF16=1
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = get_idx(gl_GlobalInvocationID);
    if (idx >= p.ne) { return; }

    const uint a_off = get_aoffset() + src0_idx(idx);
    const uint d_off = get_doffset() + dst_idx(idx);

#ifdef DIR_BF16_TO_F32
    const uint bits = uint(data_a[a_off]) << 16;
    data_d[d_off] = asfloat(bits);
#endif
#ifdef DIR_F32_TO_BF16
    const float f = data_a[a_off];
    const uint bits = asuint(f);
    const uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    data_d[d_off] = uint16_t(rounded >> 16);
#endif
}
