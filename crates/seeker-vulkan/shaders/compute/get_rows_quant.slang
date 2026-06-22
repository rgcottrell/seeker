// Ported from llama.cpp ggml-vulkan get_rows_quant.comp, MIT license — see LICENSE.llama-cpp
//
// Gather rows from a quantized A by index list B, dequantize, write to D. Mirrors the
// dispatch shape of get_rows but uses dequantize / get_dm to materialize each pair.

//@variants
//  q4_0:   DATA_A_Q4_0=1     B_TYPE=uint    D_TYPE=float    FLOAT_TYPE=float
//  q4_1:   DATA_A_Q4_1=1     B_TYPE=uint    D_TYPE=float    FLOAT_TYPE=float
//  q5_0:   DATA_A_Q5_0=1     B_TYPE=uint    D_TYPE=float    FLOAT_TYPE=float
//  q5_1:   DATA_A_Q5_1=1     B_TYPE=uint    D_TYPE=float    FLOAT_TYPE=float
//  q8_0:   DATA_A_Q8_0=1     B_TYPE=uint    D_TYPE=float    FLOAT_TYPE=float
//  iq4_nl: DATA_A_IQ4_NL=1   B_TYPE=uint    D_TYPE=float    FLOAT_TYPE=float
//  q1_0:   DATA_A_Q1_0=1     B_TYPE=uint    D_TYPE=float    FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize, gl_NumWorkGroups

#include "types.slang"
#include "generic_binary_head.slang"
#if defined(A_TYPE_PACKED16)
[[vk::binding(3, 0)]] StructuredBuffer<A_TYPE_PACKED16> data_a_packed16;
#endif
#include "dequant_funcs.slang"

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i00 = gl_GlobalInvocationID.x * 2u;

#ifdef NEEDS_INIT_IQ_SHMEM
    init_iq_shmem(gl_WorkGroupSize);
#endif

    if (i00 >= p.ne00) { return; }

    uint gid_z = gl_GlobalInvocationID.z;
    while (gid_z < p.ne11 * p.ne12) {
        uint gid_y = gl_GlobalInvocationID.y;
        while (gid_y < p.ne10) {
            const uint i10 = gid_y;
            const uint i11 = gid_z / p.ne12;
            const uint i12 = gid_z % p.ne12;

            const uint i01 = uint(data_b[i10 * p.nb10 + i11 * p.nb11 + i12 * p.nb12]);

            const uint a_off = i01 * p.nb01 + i11 * p.nb02 + i12 * p.nb03;
            const uint d_off = i10 * p.nb21 + i11 * p.nb22 + i12 * p.nb23;

            const uint ib       = a_off + i00 / QUANT_K;
            const uint iqs      = (i00 % QUANT_K) / QUANT_R;
            const uint iybs     = i00 - i00 % QUANT_K;
            const uint y_offset = QUANT_R == 1 ? 1 : QUANT_K / 2;

            float2 v        = dequantize(ib, iqs, 0);
            const float2 dm = get_dm(ib, 0);
            v = v * dm.x + dm.y;

            data_d[d_off + iybs + iqs            ] = D_TYPE(v.x);
            data_d[d_off + iybs + iqs + y_offset ] = D_TYPE(v.y);

            gid_y += gl_NumWorkGroups.y;  // gl_WorkGroupSize.y == 1
        }
        gid_z += gl_NumWorkGroups.z;
    }
}
