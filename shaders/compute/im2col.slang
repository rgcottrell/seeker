// Ported from llama.cpp ggml-vulkan im2col.comp, MIT license — see LICENSE.llama-cpp
//
// Im2col rearranges a 4D input into a column-major matmul layout. We only port the
// non-BDA (StructuredBuffer) path — buffer-device-address output isn't wired up here.

//@variants
//  f32:     A_TYPE=float    D_TYPE=float        D_SIZE=4
//  f32_f16: A_TYPE=float    D_TYPE=float16_t    D_SIZE=2
//@end-variants

import glsl;  // gl_NumWorkGroups, gl_WorkGroupID, gl_LocalInvocationID

#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint NUM_ITER = 512 / BLOCK_SIZE;

struct Im2colParams {
    // dst_addr (BDA) and batch_offset are kept but the BDA path is removed.
    uint64_t dst_addr_unused;
    uint batch_offset; uint offset_delta;
    uint IC;
    uint IW; uint IH;
    uint OW; uint OH;
    uint KW; uint KH;
    uint OH_batch;
    uint CHW;
    int s0; int s1;
    int p0; int p1;
    int d0; int d1;
    uint batch_IC;
};

[[vk::push_constant]] ConstantBuffer<Im2colParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

void im2col_one(uint ow, uint z_idx, uint3 lid, uint3 wgid) {
    const uint oh        = z_idx % p.OH;
    const uint batch_idx = z_idx / p.OH;

    const uint gidx       = lid.x;
    const uint src_batch  = batch_idx * p.batch_offset;
    const uint dst_row    = ((batch_idx * p.OH + oh) * p.OW + ow) * p.CHW;
    const uint KHKW       = p.KH * p.KW;

    uint wg_x = wgid.x;
    do {
        const uint wg_offset = wg_x * 512u;

        [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
            const uint chw_idx = wg_offset + gidx + i * BLOCK_SIZE;
            if (chw_idx >= p.CHW) { return; }

            const uint ic  = chw_idx / KHKW;
            const uint rem = chw_idx - ic * KHKW;
            const uint ky  = rem / p.KW;
            const uint kx  = rem - ky * p.KW;

            const uint iiw = uint(int(ow) * p.s0 + int(kx) * p.d0 - p.p0);
            const uint iih = uint(int(oh) * p.s1 + int(ky) * p.d1 - p.p1);

            A_TYPE val = A_TYPE(0);
            if (iih < p.IH && iiw < p.IW) {
                val = data_a[src_batch + ic * p.offset_delta + iih * p.IW + iiw];
            }
            data_d[dst_row + chw_idx] = D_TYPE(val);
        }

        wg_x += gl_NumWorkGroups.x;
    } while (wg_x * 512u < p.CHW);
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID,
          uint3 gl_LocalInvocationID  : SV_GroupThreadID,
          uint3 gl_WorkGroupID        : SV_GroupID) {
    uint ow = gl_GlobalInvocationID.y;
    while (ow < p.OW) {
        uint z = gl_GlobalInvocationID.z;
        while (z < p.OH_batch) {
            im2col_one(ow, z, gl_LocalInvocationID, gl_WorkGroupID);
            z += gl_NumWorkGroups.z;
        }
        ow += gl_NumWorkGroups.y;
    }
}
