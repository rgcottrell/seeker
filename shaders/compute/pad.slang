// Ported from llama.cpp ggml-vulkan pad.comp, MIT license — see LICENSE.llama-cpp
//
// pad.comp has its own push-constant struct (lp/rp + circular flag), distinct
// from generic_unary_head — so we inline both bindings and helpers here rather
// than introduce a one-shot include.

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"

struct PadParams {
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint misalign_offsets;
    uint circular;
    uint lp0; uint rp0;
    uint lp1; uint rp1;
    uint lp2; uint rp2;
    uint lp3; uint rp3;
};

[[vk::push_constant]] ConstantBuffer<PadParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_doffset() { return p.misalign_offsets & 0xFFFF; }

uint wrap_around(int coord, uint size) {
    return uint(coord + int(size)) % size;
}

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
    if (idx >= p.ne) { return; }

    const uint i3 = idx / (p.ne12 * p.ne11 * p.ne10);
    const uint i3_offset = i3 * p.ne12 * p.ne11 * p.ne10;
    const uint i2 = (idx - i3_offset) / (p.ne11 * p.ne10);
    const uint i2_offset = i2 * p.ne11 * p.ne10;
    const uint i1 = (idx - i3_offset - i2_offset) / p.ne10;
    const uint i0 = idx - i3_offset - i2_offset - i1 * p.ne10;

    const uint s0_idx = (i3 - p.lp3) * p.nb03 + (i2 - p.lp2) * p.nb02 + (i1 - p.lp1) * p.nb01 + (i0 - p.lp0) * p.nb00;
    const uint d_idx  = i3 * p.nb13 + i2 * p.nb12 + i1 * p.nb11 + i0 * p.nb10;

    if (p.circular != 0u) {
        const uint ci0 = wrap_around(int(i0) - int(p.lp0), p.ne00);
        const uint ci1 = wrap_around(int(i1) - int(p.lp1), p.ne01);
        const uint ci2 = wrap_around(int(i2) - int(p.lp2), p.ne02);
        const uint ci3 = wrap_around(int(i3) - int(p.lp3), p.ne03);
        const uint c_src_idx = ci3 * p.nb03 + ci2 * p.nb02 + ci1 * p.nb01 + ci0 * p.nb00;
        data_d[get_doffset() + d_idx] = D_TYPE(data_a[get_aoffset() + c_src_idx]);
    } else {
        const bool is_src0 = i0 >= p.lp0 && i0 < p.ne10 - p.rp0
                          && i1 >= p.lp1 && i1 < p.ne11 - p.rp1
                          && i2 >= p.lp2 && i2 < p.ne12 - p.rp2
                          && i3 >= p.lp3 && i3 < p.ne13 - p.rp3;
        data_d[get_doffset() + d_idx] = D_TYPE(is_src0 ? data_a[get_aoffset() + s0_idx] : A_TYPE(0));
    }
}
