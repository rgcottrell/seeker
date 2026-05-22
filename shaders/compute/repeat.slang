// Ported from llama.cpp ggml-vulkan repeat.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

uint src0_idx_mod(uint idx) {
    const uint i13 = idx / (p.ne12 * p.ne11 * p.ne10);
    const uint i13_offset = i13 * p.ne12 * p.ne11 * p.ne10;
    const uint i12 = (idx - i13_offset) / (p.ne11 * p.ne10);
    const uint i12_offset = i12 * p.ne11 * p.ne10;
    const uint i11 = (idx - i13_offset - i12_offset) / p.ne10;
    const uint i10 = idx - i13_offset - i12_offset - i11 * p.ne10;
    return (i13 % p.ne03) * p.nb03 + (i12 % p.ne02) * p.nb02 + (i11 % p.ne01) * p.nb01 + (i10 % p.ne00) * p.nb00;
}

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = get_idx(gl_GlobalInvocationID);
    if (idx >= p.ne) { return; }

    data_d[get_doffset() + dst_idx(idx)] = D_TYPE(data_a[get_aoffset() + src0_idx_mod(idx)]);
}
