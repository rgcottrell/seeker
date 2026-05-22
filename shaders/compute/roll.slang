// Ported from llama.cpp ggml-vulkan roll.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

uint wrap_idx(int i, uint ne) {
    if (i < 0) { return uint(i + int(ne)); }
    if (uint(i) >= ne) { return uint(i) - ne; }
    return uint(i);
}

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = get_idx(gl_GlobalInvocationID);
    if (idx >= p.ne) { return; }

    const uint i3        = fastdiv(idx, p.ne1_012mp, p.ne1_012L);
    const uint i3_offset = i3 * p.ne12 * p.ne11 * p.ne10;
    const uint i2        = fastdiv(idx - i3_offset, p.ne1_01mp, p.ne1_01L);
    const uint i2_offset = i2 * p.ne11 * p.ne10;
    const uint i1        = fastdiv(idx - i3_offset - i2_offset, p.ne1_0mp, p.ne1_0L);
    const uint i0        = idx - i3_offset - i2_offset - i1 * p.ne10;

    // Roll-offsets are packed into two floats interpreted as bit patterns:
    // each 16-bit half is biased by 0x8000 to give a signed shift in [-32768, 32767].
    const uint p1 = asuint(p.param1);
    const uint p2 = asuint(p.param2);
    const int s0 = int(p1 >> 16)     - 0x8000;
    const int s1 = int(p1 & 0xFFFFu) - 0x8000;
    const int s2 = int(p2 >> 16)     - 0x8000;
    const int s3 = int(p2 & 0xFFFFu) - 0x8000;

    const uint i00 = wrap_idx(int(i0) - s0, p.ne10);
    const uint i01 = wrap_idx(int(i1) - s1, p.ne11);
    const uint i02 = wrap_idx(int(i2) - s2, p.ne12);
    const uint i03 = wrap_idx(int(i3) - s3, p.ne13);

    const uint a_idx = i03 * p.nb03 + i02 * p.nb02 + i01 * p.nb01 + i00 * p.nb00;
    const uint d_idx = i3  * p.nb13 + i2  * p.nb12 + i1  * p.nb11 + i0  * p.nb10;

    data_d[get_doffset() + d_idx] = D_TYPE(data_a[get_aoffset() + a_idx]);
}
