// Ported from llama.cpp ggml-vulkan copy_to_quant.comp, MIT license — see LICENSE.llama-cpp
//
// Quantize an f32 source into one of the legacy GGML block formats. Each block is owned by
// a single thread; 32 threads per workgroup. SET_ROWS adds an i32/i64 row-index gather.
//
// Aliased data_a_packed16 view is declared at binding 3 by the consuming shader before
// dequant_funcs.slang is pulled in (IQ4_NL path needs it).

//@variants
//  q4_0:        DATA_A_Q4_0=1     D_TYPE=float   FLOAT_TYPE=float
//  q4_1:        DATA_A_Q4_1=1     D_TYPE=float   FLOAT_TYPE=float
//  q5_0:        DATA_A_Q5_0=1     D_TYPE=float   FLOAT_TYPE=float
//  q5_1:        DATA_A_Q5_1=1     D_TYPE=float   FLOAT_TYPE=float
//  q8_0:        DATA_A_Q8_0=1     D_TYPE=float   FLOAT_TYPE=float
//  q1_0:        DATA_A_Q1_0=1     D_TYPE=float   FLOAT_TYPE=float
//  iq4_nl:      DATA_A_IQ4_NL=1   D_TYPE=float   FLOAT_TYPE=float
//  turbo2:      DATA_A_TURBO2_0=1 D_TYPE=float   FLOAT_TYPE=float
//  turbo3:      DATA_A_TURBO3_0=1 D_TYPE=float   FLOAT_TYPE=float
//  turbo4:      DATA_A_TURBO4_0=1 D_TYPE=float   FLOAT_TYPE=float
//  set_rows_q4_0_i32: SET_ROWS=1  DATA_A_Q4_0=1   B_TYPE=uint   B_SIZE=32   D_TYPE=float   FLOAT_TYPE=float
//  set_rows_q4_1_i32: SET_ROWS=1  DATA_A_Q4_1=1   B_TYPE=uint   B_SIZE=32   D_TYPE=float   FLOAT_TYPE=float
//  set_rows_q5_0_i32: SET_ROWS=1  DATA_A_Q5_0=1   B_TYPE=uint   B_SIZE=32   D_TYPE=float   FLOAT_TYPE=float
//  set_rows_q5_1_i32: SET_ROWS=1  DATA_A_Q5_1=1   B_TYPE=uint   B_SIZE=32   D_TYPE=float   FLOAT_TYPE=float
//  set_rows_q8_0_i32: SET_ROWS=1  DATA_A_Q8_0=1   B_TYPE=uint   B_SIZE=32   D_TYPE=float   FLOAT_TYPE=float
//  set_rows_q1_0_i32: SET_ROWS=1  DATA_A_Q1_0=1   B_TYPE=uint   B_SIZE=32   D_TYPE=float   FLOAT_TYPE=float
//  set_rows_iq4_nl_i32: SET_ROWS=1 DATA_A_IQ4_NL=1 B_TYPE=uint  B_SIZE=32   D_TYPE=float   FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize, gl_LocalInvocationIndex

#include "types.slang"
#include "turbo_consts.slang" // TURBO_S1/S2, TURBO_INV_SQRT_128, TURBO*_THRESH (turbo quantize)

#if defined(SET_ROWS) && QUANT_K == 1
static const uint BLOCK_SIZE = 512;
#else
static const uint BLOCK_SIZE = 32;
#endif

// generic_unary_head / generic_binary_head declare data_a / data_d at slots 0/1, so we
// can't include them here without colliding with our own data_s / data_q. Inline only the
// pieces we need: the push-constant struct + fastdiv + index helpers.

#include "utils.slang"

#if defined(SET_ROWS)
struct CopyParams {
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint ne20; uint ne21; uint ne22; uint ne23; uint nb20; uint nb21; uint nb22; uint nb23;
    uint misalign_offsets;
    float param1; float param2; int param3;
};
#else
struct CopyParams {
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint misalign_offsets;
    float param1; float param2;
    uint ne0_012mp; uint ne0_012L;
    uint ne0_01mp;  uint ne0_01L;
    uint ne0_0mp;   uint ne0_0L;
    uint ne1_012mp; uint ne1_012L;
    uint ne1_01mp;  uint ne1_01L;
    uint ne1_0mp;   uint ne1_0L;
};
#endif

[[vk::push_constant]] ConstantBuffer<CopyParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>    data_s;
#if defined(SET_ROWS)
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_i;
[[vk::binding(2, 0)]] RWStructuredBuffer<A_TYPE> data_q;
#else
[[vk::binding(1, 0)]] RWStructuredBuffer<A_TYPE> data_q;
#endif

#if defined(A_TYPE_PACKED16)
[[vk::binding(3, 0)]] StructuredBuffer<A_TYPE_PACKED16> data_a_packed16;
#endif

uint get_aoffset() { return p.misalign_offsets >> 16; }
#if defined(SET_ROWS)
uint get_boffset() { return (p.misalign_offsets >> 8) & 0xFFu; }
uint get_doffset() { return  p.misalign_offsets       & 0xFFu; }
#else
uint get_doffset() { return p.misalign_offsets & 0xFFFFu; }
#endif

uint mulhi_u32(uint a, uint b) { return uint((uint64_t(a) * uint64_t(b)) >> 32); }
uint fastdiv(uint n, uint mp, uint L) { return (mulhi_u32(n, mp) + n) >> L; }

#if !defined(SET_ROWS)
uint src0_idx(uint idx) {
    const uint i03 = fastdiv(idx, p.ne0_012mp, p.ne0_012L);
    const uint i03_offset = i03 * p.ne02 * p.ne01 * p.ne00;
    const uint i02 = fastdiv(idx - i03_offset, p.ne0_01mp, p.ne0_01L);
    const uint i02_offset = i02 * p.ne01 * p.ne00;
    const uint i01 = fastdiv(idx - i03_offset - i02_offset, p.ne0_0mp, p.ne0_0L);
    const uint i00 = idx - i03_offset - i02_offset - i01 * p.ne00;
    return i03 * p.nb03 + i02 * p.nb02 + i01 * p.nb01 + i00 * p.nb00;
}

uint dst_idx_quant(uint idx, uint qk) {
    const uint i13 = fastdiv(idx, p.ne1_012mp, p.ne1_012L);
    const uint i13_offset = i13 * p.ne12 * p.ne11 * p.ne10;
    const uint i12 = fastdiv(idx - i13_offset, p.ne1_01mp, p.ne1_01L);
    const uint i12_offset = i12 * p.ne11 * p.ne10;
    const uint i11 = fastdiv(idx - i13_offset - i12_offset, p.ne1_0mp, p.ne1_0L);
    const uint i10 = idx - i13_offset - i12_offset - i11 * p.ne10;
    return i13 * p.nb13 + i12 * p.nb12 + i11 * p.nb11 + (i10 / qk) * p.nb10;
}
#endif

#if defined(SET_ROWS)
void get_indices(uint idx, out uint i00, out uint i01, out uint i02, out uint i03) {
    get_indices(idx, i00, i01, i02, i03, p.ne00, p.ne01, p.ne02, p.ne03);
}
uint src0_idx(uint i00, uint i01, uint i02, uint i03) {
    return i03 * p.nb03 + i02 * p.nb02 + i01 * p.nb01 + i00 * p.nb00;
}
uint src1_idx(uint i00, uint i01, uint i02, uint i03) {
    return fastmod(i03, p.ne13) * p.nb13
         + fastmod(i02, p.ne12) * p.nb12
         + fastmod(i01, p.ne11) * p.nb11
         + fastmod(i00, p.ne10) * p.nb10;
}
uint dst_idx(uint i00, uint i01, uint i02, uint i03) {
    return i03 * p.nb23 + i02 * p.nb22 + i01 * p.nb21 + i00 * p.nb20;
}
#endif

#if defined(DATA_A_IQ4_NL)
// kvalues_iq4nl is defined in types.slang (groupshared, populated by init_iq_shmem).
uint best_index(float x) {
    if (x <= float(kvalues_iq4nl[0]))  { return 0; }
    if (x >= float(kvalues_iq4nl[15])) { return 15; }
    int ml = 0, mu = 15;
    while (mu - ml > 1) {
        int mav = (ml + mu) / 2;
        if (x < float(kvalues_iq4nl[mav])) { mu = mav; } else { ml = mav; }
    }
    return uint(x - float(kvalues_iq4nl[mu - 1]) < float(kvalues_iq4nl[mu]) - x ? mu - 1 : mu);
}
#endif

#if defined(DATA_A_Q4_0)
void quantize(uint dst_idx, uint src_idx) {
    float amax = 0.0f, vmax = 0.0f;
    [unroll] for (uint j = 0; j < QUANT_K_Q4_0; ++j) {
        const float v = data_s[src_idx + j];
        if (amax < abs(v)) { amax = abs(v); vmax = v; }
    }
    const float d  = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    data_q[dst_idx].d = float16_t(d);
    [unroll] for (uint j = 0; j < QUANT_K_Q4_0 / 2; ++j) {
        const float x0 = data_s[src_idx + 0                 + j] * id;
        const float x1 = data_s[src_idx + QUANT_K_Q4_0 / 2  + j] * id;
        const uint  xi0 = min(15u, uint(int(x0 + 8.5f)));
        const uint  xi1 = min(15u, uint(int(x1 + 8.5f)));
        data_q[dst_idx].qs[j] = uint8_t(xi0 | (xi1 << 4));
    }
}
#endif

#if defined(DATA_A_Q4_1)
void quantize(uint dst_idx, uint src_idx) {
    float vmin =  1.0f / 0.0f;
    float vmax = -vmin;
    [unroll] for (uint j = 0; j < QUANT_K_Q4_1; ++j) {
        const float v = data_s[src_idx + j];
        if (v < vmin) { vmin = v; }
        if (v > vmax) { vmax = v; }
    }
    const float d  = (vmax - vmin) / 15.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    data_q[dst_idx].d = float16_t(d);
    data_q[dst_idx].m = float16_t(vmin);
    [unroll] for (uint j = 0; j < QUANT_K_Q4_1 / 2; ++j) {
        const float x0 = (data_s[src_idx + 0                 + j] - vmin) * id;
        const float x1 = (data_s[src_idx + QUANT_K_Q4_1 / 2  + j] - vmin) * id;
        const uint xi0 = min(15u, uint(int(x0 + 0.5f)));
        const uint xi1 = min(15u, uint(int(x1 + 0.5f)));
        data_q[dst_idx].qs[j] = uint8_t(xi0 | (xi1 << 4));
    }
}
#endif

#if defined(DATA_A_Q5_0)
void quantize(uint dst_idx, uint src_idx) {
    float amax = 0.0f, vmax = 0.0f;
    [unroll] for (uint j = 0; j < QUANT_K_Q5_0; ++j) {
        const float v = data_s[src_idx + j];
        if (amax < abs(v)) { amax = abs(v); vmax = v; }
    }
    const float d  = vmax / -16.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    data_q[dst_idx].d = float16_t(d);
    uint qh = 0;
    [unroll] for (uint j = 0; j < QUANT_K_Q5_0 / 2; ++j) {
        const float x0 = data_s[src_idx + 0                 + j] * id;
        const float x1 = data_s[src_idx + QUANT_K_Q5_0 / 2  + j] * id;
        const uint xi0 = min(31u, uint(int(x0 + 16.5f)));
        const uint xi1 = min(31u, uint(int(x1 + 16.5f)));
        data_q[dst_idx].qs[j] = uint8_t((xi0 & 0xfu) | ((xi1 & 0xfu) << 4));
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QUANT_K_Q5_0 / 2);
    }
    data_q[dst_idx].qh[0] = uint16_t(qh & 0xFFFFu);
    data_q[dst_idx].qh[1] = uint16_t(qh >> 16);
}
#endif

#if defined(DATA_A_Q5_1)
void quantize(uint dst_idx, uint src_idx) {
    float mn = data_s[src_idx + 0];
    float mx = mn;
    [unroll] for (uint j = 1; j < QUANT_K_Q5_1; ++j) {
        const float v = data_s[src_idx + j];
        mn = v < mn ? v : mn;
        mx = v > mx ? v : mx;
    }
    const float d  = (mx - mn) / 31.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    data_q[dst_idx].d = float16_t(d);
    data_q[dst_idx].m = float16_t(mn);
    uint qh = 0;
    [unroll] for (uint j = 0; j < QUANT_K_Q5_1 / 2; ++j) {
        const float x0 = (data_s[src_idx + 0                + j] - mn) * id;
        const float x1 = (data_s[src_idx + QUANT_K_Q5_1 / 2 + j] - mn) * id;
        const uint xi0 = uint(int(x0 + 0.5f));
        const uint xi1 = uint(int(x1 + 0.5f));
        data_q[dst_idx].qs[j] = uint8_t((xi0 & 0xfu) | ((xi1 & 0xfu) << 4));
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QUANT_K_Q5_1 / 2);
    }
    data_q[dst_idx].qh = qh;
}
#endif

#if defined(DATA_A_Q8_0)
void quantize(uint dst_idx, uint src_idx) {
    float amax = 0.0f;
    [unroll] for (uint j = 0; j < QUANT_K_Q8_0; j++) {
        amax = max(amax, abs(data_s[src_idx + j]));
    }
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    data_q[dst_idx].d = float16_t(d);
    [unroll] for (uint j = 0; j < QUANT_K_Q8_0; ++j) {
        data_q[dst_idx].qs[j] = int8_t(round(data_s[src_idx + j] * id));
    }
}
#endif

#if defined(DATA_A_Q1_0)
void quantize(uint dst_idx, uint src_idx) {
    float sum_abs = 0.0f;
    [unroll] for (uint j = 0; j < QUANT_K_Q1_0; j++) {
        sum_abs += abs(data_s[src_idx + j]);
    }
    const float d = sum_abs / float(QUANT_K_Q1_0);
    data_q[dst_idx].d = float16_t(d);

    uint packed[QUANT_K_Q1_0 / 8];
    [unroll] for (uint j = 0; j < QUANT_K_Q1_0 / 8; ++j) { packed[j] = 0u; }
    [unroll] for (uint j = 0; j < QUANT_K_Q1_0; ++j) {
        if (data_s[src_idx + j] >= 0.0f) {
            packed[j / 8] |= 1u << (j % 8);
        }
    }
    [unroll] for (uint j = 0; j < QUANT_K_Q1_0 / 8; ++j) {
        data_q[dst_idx].qs[j] = uint8_t(packed[j]);
    }
}
#endif

#if defined(DATA_A_IQ4_NL)
void quantize(uint dst_idx, uint src_idx) {
    float amax = 0.0f, vmax = 0.0f;
    [unroll] for (uint j = 0; j < QUANT_K_IQ4_NL; ++j) {
        const float v = data_s[src_idx + j];
        if (amax < abs(v)) { amax = abs(v); vmax = v; }
    }

    float d  = vmax / float(kvalues_iq4nl[0]);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    float sumqx = 0.0f, sumq2 = 0.0f;
    [unroll] for (uint j = 0; j < QUANT_K_IQ4_NL / 2; ++j) {
        const float a0 = data_s[src_idx + 0                  + j];
        const float a1 = data_s[src_idx + QUANT_K_IQ4_NL / 2 + j];
        const float x0 = a0 * id;
        const float x1 = a1 * id;
        const uint xi0 = best_index(x0);
        const uint xi1 = best_index(x1);
        data_q[dst_idx].qs[j] = uint8_t(xi0 | (xi1 << 4));
        const float v0 = float(kvalues_iq4nl[xi0]);
        const float v1 = float(kvalues_iq4nl[xi1]);
        const float w0 = a0 * a0;
        const float w1 = a1 * a1;
        sumqx += w0 * v0 * a0 + w1 * v1 * a1;
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }
    data_q[dst_idx].d = float16_t(sumq2 > 0.0f ? sumqx / sumq2 : d);
}
#endif

#if defined(DATA_A_F32) || defined(DATA_A_F16)
void quantize(uint dst_idx, uint src_idx) {
    data_q[dst_idx] = A_TYPE(data_s[src_idx]);
}
#endif

#if defined(DATA_A_BF16)
void quantize(uint dst_idx, uint src_idx) {
    data_q[dst_idx] = A_TYPE(fp32_to_bf16(data_s[src_idx]));
}
#endif

// ── TurboQuant: WHT-rotated PolarQuant quantize-on-write (one thread/block) ──
// Each thread owns a whole 128-element block (= one head's WHT group): L2
// normalize → forward WHT (S1, butterfly, S2·1/√128) → nearest PolarQuant
// centroid → pack indices + store the corrected fp16 norm `gnrm/√Σcentroid²`.
// Matches the fork's copy_to_quant.comp SET_ROWS turbo path; the FA dequant in
// flash_attn.slang is the exact inverse of this packing.
#if defined(DATA_A_TURBO2_0) || defined(DATA_A_TURBO3_0) || defined(DATA_A_TURBO4_0)
float turbo_forward_rotate(uint src, out float x[128]) {
    float norm_sq = 0.0;
    [unroll] for (uint j = 0; j < 128u; ++j) {
        float v = data_s[src + j];
        x[j] = v;
        norm_sq += v * v;
    }
    float gnrm = sqrt(norm_sq);
    float inv = (gnrm > 1e-10) ? (1.0 / gnrm) : 0.0;
    [unroll] for (uint j = 0; j < 128u; ++j) { x[j] *= inv * TURBO_S1[j]; }
    [unroll] for (uint h = 1u; h < 128u; h *= 2u) {
        for (uint i = 0u; i < 128u; i += 2u * h) {
            for (uint k = i; k < i + h; ++k) {
                float a = x[k];
                float b = x[k + h];
                x[k]     = a + b;
                x[k + h] = a - b;
            }
        }
    }
    [unroll] for (uint j = 0; j < 128u; ++j) { x[j] = x[j] * TURBO_INV_SQRT_128 * TURBO_S2[j]; }
    return gnrm;
}
#endif

#if defined(DATA_A_TURBO2_0)
void quantize(uint dst_idx, uint src_idx) {
    float x[128];
    float gnrm = turbo_forward_rotate(src_idx, x);
    float rs = 0.0;
    [unroll] for (uint b = 0; b < 32u; ++b) {
        uint byte = 0u;
        [unroll] for (uint t = 0; t < 4u; ++t) {
            float rv = x[b * 4u + t];
            uint idx = rv < TURBO2_THRESH[0] ? 0u
                     : rv < TURBO2_THRESH[1] ? 1u
                     : rv < TURBO2_THRESH[2] ? 2u : 3u;
            rs += turbo2_centroids[idx] * turbo2_centroids[idx];
            byte |= idx << (2u * t);
        }
        data_q[dst_idx].qs[b] = uint8_t(byte);
    }
    float rn = sqrt(rs);
    data_q[dst_idx].norm = float16_t((rn > 1e-10) ? (gnrm / rn) : gnrm);
}
#endif

#if defined(DATA_A_TURBO3_0)
uint turbo3_index(float rv) {
    return rv < TURBO3_THRESH[0] ? 0u
         : rv < TURBO3_THRESH[1] ? 1u
         : rv < TURBO3_THRESH[2] ? 2u
         : rv < TURBO3_THRESH[3] ? 3u
         : rv < TURBO3_THRESH[4] ? 4u
         : rv < TURBO3_THRESH[5] ? 5u
         : rv < TURBO3_THRESH[6] ? 6u : 7u;
}
void quantize(uint dst_idx, uint src_idx) {
    float x[128];
    float gnrm = turbo_forward_rotate(src_idx, x);
    float rs = 0.0;
    // low 2 bits → qs (4 per byte), accumulate reconstruction norm
    [unroll] for (uint b = 0; b < 32u; ++b) {
        uint byte = 0u;
        [unroll] for (uint t = 0; t < 4u; ++t) {
            uint idx = turbo3_index(x[b * 4u + t]);
            rs += turbo3_centroids[idx] * turbo3_centroids[idx];
            byte |= (idx & 0x3u) << (2u * t);
        }
        data_q[dst_idx].qs[b] = uint8_t(byte);
    }
    // high bit → signs (8 per byte)
    [unroll] for (uint b = 0; b < 16u; ++b) {
        uint sbyte = 0u;
        [unroll] for (uint t = 0; t < 8u; ++t) {
            uint idx = turbo3_index(x[b * 8u + t]);
            sbyte |= ((idx >> 2) & 0x1u) << t;
        }
        data_q[dst_idx].signs[b] = uint8_t(sbyte);
    }
    float rn = sqrt(rs);
    data_q[dst_idx].norm = float16_t((rn > 1e-10) ? (gnrm / rn) : gnrm);
}
#endif

#if defined(DATA_A_TURBO4_0)
void quantize(uint dst_idx, uint src_idx) {
    float x[128];
    float gnrm = turbo_forward_rotate(src_idx, x);
    data_q[dst_idx].rnorm = float16_t(0.0); // reserved
    float rs = 0.0;
    [unroll] for (uint b = 0; b < 64u; ++b) {
        uint byte = 0u;
        [unroll] for (uint t = 0; t < 2u; ++t) {
            float rv = x[b * 2u + t];
            uint idx = 0u;
            [unroll] for (uint i = 0; i < 15u; ++i) { if (rv >= TURBO4_THRESH[i]) { idx = i + 1u; } }
            rs += turbo4_centroids[idx] * turbo4_centroids[idx];
            byte |= (idx & 0xFu) << (4u * t);
        }
        data_q[dst_idx].qs[b] = uint8_t(byte);
    }
    float rn = sqrt(rs);
    data_q[dst_idx].norm = float16_t((rn > 1e-10) ? (gnrm / rn) : gnrm);
}
#endif

#if defined(SET_ROWS)
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
#ifdef NEEDS_INIT_IQ_SHMEM
    init_iq_shmem(gl_WorkGroupSize);
#endif
    const uint idx = ((gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x) * BLOCK_SIZE + gl_LocalInvocationID.x) * QUANT_K;
    if (idx >= p.ne) { return; }

    uint i00, i01, i02, i03;
    get_indices(idx, i00, i01, i02, i03);

    const uint i12 = fastmod(i03, p.ne12);
    const uint i11 = fastmod(i02, p.ne11);
    const uint i10 = i01;

    // B_TYPE may be uint (i32 index) or uint2 (i64 index swizzled to .x); both reduce to a uint here.
#if B_SIZE == 64
    const uint i1 = data_i[src1_idx(i10, i11, i12, 0) + get_boffset()].x;
#else
    const uint i1 = uint(data_i[src1_idx(i10, i11, i12, 0) + get_boffset()]);
#endif

    const uint s0_idx = src0_idx(i00, i01, i02, i03) + get_aoffset();
    const uint d_idx  = dst_idx(i00 / QUANT_K, i1, i02, i03) + get_doffset();
    quantize(d_idx, s0_idx);
}
#else
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
#ifdef NEEDS_INIT_IQ_SHMEM
    init_iq_shmem(gl_WorkGroupSize);
#endif
    const uint idx = (gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x) * QUANT_K;
    if (idx >= p.ne) { return; }

    const uint d_idx = dst_idx_quant(idx, QUANT_K);
    const uint s_idx = get_aoffset() + src0_idx(idx);
    quantize(d_idx, s_idx);
}
#endif
