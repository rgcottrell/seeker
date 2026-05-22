// Ported from llama.cpp ggml-vulkan im2col_3d.comp, MIT license — see LICENSE.llama-cpp
//
// 3D im2col. Non-BDA only — host writes directly into data_d.

//@variants
//  f32:     A_TYPE=float    D_TYPE=float        D_SIZE=4
//  f32_f16: A_TYPE=float    D_TYPE=float16_t    D_SIZE=2
//@end-variants

import glsl;  // gl_NumWorkGroups

#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 32;

struct Im2col3dParams {
    uint64_t dst_addr_unused;
    uint nb10; uint nb11; uint nb12; uint nb13;
    uint s0; uint s1; uint s2;
    uint p0; uint p1; uint p2;
    uint d0; uint d1; uint d2;
    uint IW; uint IH; uint ID; uint IC;
    uint KW; uint OH;
    uint KD_KH_KW; uint KH_KW; uint IC_KD_KH_KW;
    uint N_OD_OH; uint OD_OH;
    uint OD_OH_OW_IC_KD_KH_KW; uint OH_OW_IC_KD_KH_KW; uint OW_IC_KD_KH_KW;
    uint misalign_offsets;
};

[[vk::push_constant]] ConstantBuffer<Im2col3dParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_doffset() { return p.misalign_offsets & 0xFFFFu; }

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= p.IC_KD_KH_KW) { return; }

    const uint iic = i / p.KD_KH_KW;
    const uint ikd = (i - iic * p.KD_KH_KW) / p.KH_KW;
    const uint ikh = (i - iic * p.KD_KH_KW - ikd * p.KH_KW) / p.KW;
    const uint ikw = i % p.KW;

    const uint iow = gl_GlobalInvocationID.y;
    for (uint iz = gl_GlobalInvocationID.z; iz < p.N_OD_OH; iz += gl_NumWorkGroups.z) {
        const uint in_ = iz / p.OD_OH;
        const uint iod = (iz - in_ * p.OD_OH) / p.OH;
        const uint ioh = iz % p.OH;

        const uint iiw = uint(int(iow) * int(p.s0) + int(ikw) * int(p.d0) - int(p.p0));
        const uint iih = uint(int(ioh) * int(p.s1) + int(ikh) * int(p.d1) - int(p.p1));
        const uint iid = uint(int(iod) * int(p.s2) + int(ikd) * int(p.d2) - int(p.p2));

        const uint offset_dst = in_  * p.OD_OH_OW_IC_KD_KH_KW
                              + iod  * p.OH_OW_IC_KD_KH_KW
                              + ioh  * p.OW_IC_KD_KH_KW
                              + iow  * p.IC_KD_KH_KW
                              + iic  * p.KD_KH_KW
                              + ikd  * p.KH_KW
                              + ikh  * p.KW
                              + ikw;
        const uint offset_src = (in_ * p.IC + iic) * p.nb13 + iid * p.nb12 + iih * p.nb11 + iiw * p.nb10;

        if (iih >= p.IH || iiw >= p.IW || iid >= p.ID) {
            data_d[offset_dst + get_doffset()] = D_TYPE(0);
        } else {
            data_d[offset_dst + get_doffset()] = D_TYPE(data_a[offset_src + get_aoffset()]);
        }
    }
}
