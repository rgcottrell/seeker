// Ported from llama.cpp ggml-vulkan diag.comp, MIT license — see LICENSE.llama-cpp

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

    const uint i13        = fastdiv(idx, p.ne1_012mp, p.ne1_012L);
    const uint i13_offset = i13 * p.ne12 * p.ne11 * p.ne10;
    const uint i12        = fastdiv(idx - i13_offset, p.ne1_01mp, p.ne1_01L);
    const uint i12_offset = i12 * p.ne11 * p.ne10;
    const uint i11        = fastdiv(idx - i13_offset - i12_offset, p.ne1_0mp, p.ne1_0L);
    const uint i10        = idx - i13_offset - i12_offset - i11 * p.ne10;

    if (i10 == i11) {
        const float val = float(data_a[get_aoffset() + i13 * p.nb03 + i12 * p.nb02 + 0u * p.nb01 + i10 * p.nb00]);
        data_d[get_doffset() + dst_idx(idx)] = D_TYPE(val);
    } else {
        data_d[get_doffset() + dst_idx(idx)] = D_TYPE(0);
    }
}
