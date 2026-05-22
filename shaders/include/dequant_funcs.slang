// Ported from llama.cpp ggml-vulkan dequant_funcs.glsl (FP + legacy-quant subset),
// MIT license — see LICENSE.llama-cpp
//
// Inline dequantize / dequantize4 / get_dm helpers used by mul_mat_vec to avoid
// materializing the dequantized weight tensor. Each DATA_A_X enables exactly
// one set of overloads.
//
// Phase 4b: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0. BF16 and K/I-quants
// follow once their consumer kernels land.

#ifndef DEQUANT_FUNCS_SLANG
#define DEQUANT_FUNCS_SLANG

#include "types.slang"

// data_a / data_a_packed16 are declared by the shader that includes this file
// (see mul_mat_vec_head.slang). These helpers assume both are in scope when
// the corresponding DATA_A_X is set.

#if defined(DATA_A_F32) || defined(DATA_A_F16)
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    return float2(float(data_a[a_offset + ib]), float(data_a[a_offset + ib + 1]));
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(0.0f, 0.0f);
}
#endif

#if defined(DATA_A_BF16)
// BF16 is stored as uint16; bf16_to_fp32 reinterprets bits.
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    return float2(
        bf16_to_fp32(uint(data_a[a_offset + ib    ])),
        bf16_to_fp32(uint(data_a[a_offset + ib + 1]))
    );
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(0.0f, 0.0f);
}
#endif

#if defined(DATA_A_Q4_0)
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return float2(vui & 0xF, vui >> 4) - 8.0f;
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs / 2]);
    return float4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f;
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(float(data_a[a_offset + ib].d), 0.0f);
}
#endif

#if defined(DATA_A_Q4_1)
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return float2(vui & 0xF, vui >> 4);
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs / 2]);
    return float4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12);
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(float(data_a[a_offset + ib].d), float(data_a[a_offset + ib].m));
}
#endif

#if defined(DATA_A_Q5_0)
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = (uint(data_a[a_offset + ib].qh[1]) << 16) | uint(data_a[a_offset + ib].qh[0]);
    const int2 qh      = int2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui     = uint(data_a[a_offset + ib].qs[iqs]);
    return float2((vui & 0xF) | qh.x, (vui >> 4) | qh.y) - 16.0f;
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = (uint(data_a_packed16[a_offset + ib].qh[1]) << 16) | uint(data_a_packed16[a_offset + ib].qh[0]);
    const int2 qh0     = int2(((uint_qh >>  iqs     ) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const int2 qh1     = int2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui     = uint(data_a_packed16[a_offset + ib].qs[iqs / 2]);
    return float4(
        (vui & 0xF)        | qh0.x,
        ((vui >>  4) & 0xF) | qh0.y,
        ((vui >>  8) & 0xF) | qh1.x,
         (vui >> 12)        | qh1.y
    ) - 16.0f;
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(float(data_a[a_offset + ib].d), 0.0f);
}
#endif

#if defined(DATA_A_Q5_1)
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a[a_offset + ib].qh;
    const int2 qh      = int2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui     = uint(data_a[a_offset + ib].qs[iqs]);
    return float2((vui & 0xF) | qh.x, (vui >> 4) | qh.y);
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a_packed16[a_offset + ib].qh;
    const int2 qh0     = int2(((uint_qh >>  iqs     ) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const int2 qh1     = int2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui     = uint(data_a_packed16[a_offset + ib].qs[iqs / 2]);
    return float4(
        (vui & 0xF)        | qh0.x,
        ((vui >>  4) & 0xF) | qh0.y,
        ((vui >>  8) & 0xF) | qh1.x,
         (vui >> 12)        | qh1.y
    );
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(float(data_a[a_offset + ib].d), float(data_a[a_offset + ib].m));
}
#endif

#if defined(DATA_A_Q8_0)
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    return float2(int(data_a[a_offset + ib].qs[iqs]), int(data_a[a_offset + ib].qs[iqs + 1]));
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    // GLSL uses unpack8(int32_t(packed16.qs[iqs/2])).xy; Slang doesn't expose
    // GLSL's unpack8 directly. Manual sign-extending byte split via int16_t.
    const int v0 = int(data_a_packed16[a_offset + ib].qs[iqs / 2]);
    const int v1 = int(data_a_packed16[a_offset + ib].qs[iqs / 2 + 1]);
    // Low/high byte sign-extension: shift left 24 then arithmetic shift right.
    const int lo0 = (v0 << 24) >> 24;
    const int hi0 = (v0 << 16) >> 24;
    const int lo1 = (v1 << 24) >> 24;
    const int hi1 = (v1 << 16) >> 24;
    return float4(lo0, hi0, lo1, hi1);
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(float(data_a[a_offset + ib].d), 0.0f);
}
#endif

#if defined(DATA_A_IQ4_XS)
// QUANT_K=256; iqs is the within-superblock element index.
// Per-32-element subblock scale is split: low 4 bits in scales_l (nibble-packed),
// high 2 bits in scales_h (2-bit-packed) → 6-bit signed scale offset (range -32..31).
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32   = iqs / 32;
    const uint iq     = 16 * ib32 + (iqs % 16);
    const uint sl     = (data_a[a_offset + ib].scales_l[ib32 / 2] >> (4 * (ib32 & 1))) & 0xFu;
    const uint sh     = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3u;
    const uint qshift = (iqs & 16) >> 2;
    const uint qs0    = (uint(data_a[a_offset + ib].qs[iq    ]) >> qshift) & 0xFu;
    const uint qs1    = (uint(data_a[a_offset + ib].qs[iq + 1]) >> qshift) & 0xFu;
    const float dl    = float(int(sl | (sh << 4)) - 32);
    return dl * float2(float(kvalues_iq4nl[qs0]), float(kvalues_iq4nl[qs1]));
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32   = iqs / 32;
    const uint iq     = 16 * ib32 + (iqs % 16);
    const uint sl     = (data_a[a_offset + ib].scales_l[ib32 / 2] >> (4 * (ib32 & 1))) & 0xFu;
    const uint sh     = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3u;
    const uint qshift = (iqs & 16) >> 2;
    const uint4 qs    = unpack8_u((data_a_packed32[a_offset + ib].qs[iq / 4] >> qshift) & 0x0F0F0F0Fu);
    const float dl    = float(int(sl | (sh << 4)) - 32);
    return dl * float4(
        float(kvalues_iq4nl[qs.x]), float(kvalues_iq4nl[qs.y]),
        float(kvalues_iq4nl[qs.z]), float(kvalues_iq4nl[qs.w])
    );
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(float(data_a[a_offset + ib].d), 0.0f);
}
#endif

#if defined(DATA_A_MXFP4)
// 4-bit signed-lattice with E8M0 block scale. kvalues_mxfp4 is the 16-entry
// signed-byte LUT in groupshared (populated by init_iq_shmem).
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return float2(float(kvalues_mxfp4[vui & 0xF]), float(kvalues_mxfp4[vui >> 4])) * 0.5f;
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const float2 v0 = dequantize(ib, iqs,     a_offset);
    const float2 v1 = dequantize(ib, iqs + 1, a_offset);
    return float4(v0.x, v0.y, v1.x, v1.y);
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(e8m0_to_fp32(uint(data_a[a_offset + ib].e)), 0.0f);
}
#endif

#if defined(DATA_A_IQ4_NL)
// kvalues_iq4nl is a 16-entry FLOAT_TYPE table populated by init_iq_shmem(...).
// Consumers (mul_mat_vec etc.) MUST call init_iq_shmem before invoking these.
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return float2(float(kvalues_iq4nl[vui & 0xF]), float(kvalues_iq4nl[vui >> 4]));
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs / 2]);
    return float4(
        float(kvalues_iq4nl[ vui        & 0xF]),
        float(kvalues_iq4nl[(vui >>  4) & 0xF]),
        float(kvalues_iq4nl[(vui >>  8) & 0xF]),
        float(kvalues_iq4nl[ vui >> 12       ])
    );
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(float(data_a[a_offset + ib].d), 0.0f);
}
#endif

#if defined(DATA_A_Q1_0)
// Each qs byte packs 8 ternary trits as 1-bit signs (bit i → ±1.0).
float2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);
    return float2(
        (bits & 1u) != 0u ?  1.0f : -1.0f,
        (bits & 2u) != 0u ?  1.0f : -1.0f);
}
float4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);
    return float4(
        (bits & 1u) != 0u ?  1.0f : -1.0f,
        (bits & 2u) != 0u ?  1.0f : -1.0f,
        (bits & 4u) != 0u ?  1.0f : -1.0f,
        (bits & 8u) != 0u ?  1.0f : -1.0f);
}
float2 get_dm(uint ib, uint a_offset) {
    return float2(float(data_a[a_offset + ib].d), 0.0f);
}
#endif

#endif
