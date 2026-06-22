// Ported from llama.cpp ggml-vulkan copy_from_quant.comp, MIT license — see LICENSE.llama-cpp
//
// Dequantize one source block per workgroup-invocation and stream into dst[idx..]. For
// I-quant types we run a 16-lane workgroup so init_iq_shmem(...) can populate the shared
// LUTs cooperatively, then only lane 0 actually writes.

//@variants
//  q1_0:   DATA_A_Q1_0=1     D_TYPE=float    FLOAT_TYPE=float
//  q4_0:   DATA_A_Q4_0=1     D_TYPE=float    FLOAT_TYPE=float
//  q4_1:   DATA_A_Q4_1=1     D_TYPE=float    FLOAT_TYPE=float
//  q5_0:   DATA_A_Q5_0=1     D_TYPE=float    FLOAT_TYPE=float
//  q5_1:   DATA_A_Q5_1=1     D_TYPE=float    FLOAT_TYPE=float
//  q8_0:   DATA_A_Q8_0=1     D_TYPE=float    FLOAT_TYPE=float
//  iq4_nl: DATA_A_IQ4_NL=1   D_TYPE=float    FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize, gl_LocalInvocationIndex

#include "types.slang"
#include "generic_unary_head.slang"

// dequant_funcs needs an aliased data_a_packed16 view for IQ4_NL and other types whose
// dequantize4 reads two bytes at a time. Slang's strict capability check rejects overlap
// at (set=0, binding=0), so we expose the same VkBuffer at binding 2 (host wires both).
#if defined(A_TYPE_PACKED16)
[[vk::binding(2, 0)]] StructuredBuffer<A_TYPE_PACKED16> data_a_packed16;
#endif

#include "dequant_funcs.slang"

#if defined(DATA_A_IQ4_NL) || defined(DATA_A_MXFP4) || defined(DATA_A_NVFP4)
static const uint LSX = 16;
#else
static const uint LSX = 1;
#endif

[shader("compute")]
[numthreads(LSX, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID,
          uint  gl_LocalInvocationIndex : SV_GroupIndex) {
#ifdef NEEDS_INIT_IQ_SHMEM
    init_iq_shmem(gl_WorkGroupSize);
    if (gl_LocalInvocationIndex != 0) { return; }
#endif

    const uint idx = gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x * QUANT_K;
    if (idx >= p.ne) { return; }

    const uint dst_base = get_doffset() + dst_idx(idx);
    const uint src_idx_a = src0_idx_quant(idx, QUANT_K);

    const uint a_offset = 0;
    const uint ib = src_idx_a;
    const float2 dm = get_dm(ib, a_offset);

    [unroll] for (int j = 0; j < QUANT_K; j += 4) {
        float4 v = dequantize4(ib, uint(j) / QUANT_R, a_offset);
        v = v * dm.x + float4(dm.y, dm.y, dm.y, dm.y);

#if QUANT_R == 2
        data_d[dst_base + uint(j)/2 +             0] = D_TYPE(v[0]);
        data_d[dst_base + uint(j)/2 + QUANT_K/2 + 0] = D_TYPE(v[1]);
        data_d[dst_base + uint(j)/2 +             1] = D_TYPE(v[2]);
        data_d[dst_base + uint(j)/2 + QUANT_K/2 + 1] = D_TYPE(v[3]);
#else
        data_d[dst_base + uint(j) + 0] = D_TYPE(v[0]);
        data_d[dst_base + uint(j) + 1] = D_TYPE(v[1]);
        data_d[dst_base + uint(j) + 2] = D_TYPE(v[2]);
        data_d[dst_base + uint(j) + 3] = D_TYPE(v[3]);
#endif
    }
}
