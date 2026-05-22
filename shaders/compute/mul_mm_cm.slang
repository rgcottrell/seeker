// Cooperative-matrix matmul (Slang `linalg::CoopMat` → SPV_KHR_cooperative_matrix).
//
// One shader, all A dtypes — the only divergence across variants is the
// `stage_a_tile()` body, which fans the workgroup out across one A-tile and
// writes f16 staging values into `buf_a`. Everything else (B staging, CoopMat
// inner loop, output store) is identical regardless of A type.
//
// Per workgroup:
//   - One BM × BN output tile, 4 subgroups cooperating
//   - K dim walked in BK-wide steps with both A and B staged through shmem
//   - Each subgroup owns one 16×16 CoopMat accumulator fragment
//   - Two CoopMat MulAdds per outer step (BK / TK == 2)

//@variants
//  f32:       DATA_A_F32=1     B_TYPE=float       D_TYPE=float
//  f16:       DATA_A_F16=1     B_TYPE=float16_t   D_TYPE=float
//  bf16:      DATA_A_BF16=1    B_TYPE=float       D_TYPE=float
//  f16_f32:   DATA_A_F16=1     B_TYPE=float       D_TYPE=float
//  f32_f16:   DATA_A_F32=1     B_TYPE=float16_t   D_TYPE=float
//  q4_0_f16:  DATA_A_Q4_0=1    B_TYPE=float16_t   D_TYPE=float
//  q4_0_f32:  DATA_A_Q4_0=1    B_TYPE=float       D_TYPE=float
//  q4_1_f16:  DATA_A_Q4_1=1    B_TYPE=float16_t   D_TYPE=float
//  q4_1_f32:  DATA_A_Q4_1=1    B_TYPE=float       D_TYPE=float
//  q5_0_f16:  DATA_A_Q5_0=1    B_TYPE=float16_t   D_TYPE=float
//  q5_0_f32:  DATA_A_Q5_0=1    B_TYPE=float       D_TYPE=float
//  q5_1_f16:  DATA_A_Q5_1=1    B_TYPE=float16_t   D_TYPE=float
//  q5_1_f32:  DATA_A_Q5_1=1    B_TYPE=float       D_TYPE=float
//  q8_0_f16:  DATA_A_Q8_0=1    B_TYPE=float16_t   D_TYPE=float
//  q8_0_f32:  DATA_A_Q8_0=1    B_TYPE=float       D_TYPE=float
//  iq4_nl_f16: DATA_A_IQ4_NL=1 B_TYPE=float16_t   D_TYPE=float
//  iq4_nl_f32: DATA_A_IQ4_NL=1 B_TYPE=float       D_TYPE=float
//  mxfp4_f16: DATA_A_MXFP4=1   B_TYPE=float16_t   D_TYPE=float
//  mxfp4_f32: DATA_A_MXFP4=1   B_TYPE=float       D_TYPE=float
//  q2_k_f16:  DATA_A_Q2_K=1    B_TYPE=float16_t   D_TYPE=float
//  q2_k_f32:  DATA_A_Q2_K=1    B_TYPE=float       D_TYPE=float
//  q4_k_f16:  DATA_A_Q4_K=1    B_TYPE=float16_t   D_TYPE=float
//  q4_k_f32:  DATA_A_Q4_K=1    B_TYPE=float       D_TYPE=float
//  q5_k_f16:  DATA_A_Q5_K=1    B_TYPE=float16_t   D_TYPE=float
//  q5_k_f32:  DATA_A_Q5_K=1    B_TYPE=float       D_TYPE=float
//  q6_k_f16:  DATA_A_Q6_K=1    B_TYPE=float16_t   D_TYPE=float
//  q6_k_f32:  DATA_A_Q6_K=1    B_TYPE=float       D_TYPE=float
//  iq4_xs_f16: DATA_A_IQ4_XS=1  B_TYPE=float16_t   D_TYPE=float
//  iq4_xs_f32: DATA_A_IQ4_XS=1  B_TYPE=float       D_TYPE=float
//  q3_k_f16:  DATA_A_Q3_K=1    B_TYPE=float16_t   D_TYPE=float
//  q3_k_f32:  DATA_A_Q3_K=1    B_TYPE=float       D_TYPE=float
//  iq2_xxs_f16: DATA_A_IQ2_XXS=1 B_TYPE=float16_t  D_TYPE=float
//  iq2_xxs_f32: DATA_A_IQ2_XXS=1 B_TYPE=float      D_TYPE=float
//  iq3_xxs_f16: DATA_A_IQ3_XXS=1 B_TYPE=float16_t  D_TYPE=float
//  iq3_xxs_f32: DATA_A_IQ3_XXS=1 B_TYPE=float      D_TYPE=float
//  nvfp4_f16:   DATA_A_NVFP4=1   B_TYPE=float16_t  D_TYPE=float
//  nvfp4_f32:   DATA_A_NVFP4=1   B_TYPE=float      D_TYPE=float
//  iq2_xs_f16:  DATA_A_IQ2_XS=1  B_TYPE=float16_t  D_TYPE=float
//  iq2_xs_f32:  DATA_A_IQ2_XS=1  B_TYPE=float      D_TYPE=float
//  iq2_s_f16:   DATA_A_IQ2_S=1   B_TYPE=float16_t  D_TYPE=float
//  iq2_s_f32:   DATA_A_IQ2_S=1   B_TYPE=float      D_TYPE=float
//  iq3_s_f16:   DATA_A_IQ3_S=1   B_TYPE=float16_t  D_TYPE=float
//  iq3_s_f32:   DATA_A_IQ3_S=1   B_TYPE=float      D_TYPE=float
//  iq1_s_f16:   DATA_A_IQ1_S=1   B_TYPE=float16_t  D_TYPE=float
//  iq1_s_f32:   DATA_A_IQ1_S=1   B_TYPE=float      D_TYPE=float
//  iq1_m_f16:   DATA_A_IQ1_M=1   B_TYPE=float16_t  D_TYPE=float
//  iq1_m_f32:   DATA_A_IQ1_M=1   B_TYPE=float      D_TYPE=float
//@end-variants

import glsl;

using namespace linalg;

#include "types.slang"

// CoopMat tile dims. 16×16×16 is the safe baseline supported across vendors.
static const uint TM = 16;
static const uint TN = 16;
static const uint TK = 16;

// BK = 32 universally. For legacy quants (Q4_0..MXFP4) this matches QUANT_K, so each
// outer step covers exactly one A block. For K-quants (256-element super-blocks) each
// outer step covers 1/8 of a block; the dequant code computes the right sub-block from
// gk % QUANT_K. Keeping BK fixed at 32 also caps the shmem footprint at ~5 KB.
static const uint BK = 32;

static const uint BM = 32;
static const uint BN = 32;

static const uint WORKGROUP_SIZE = 128;
static const uint SUBGROUP_SIZE  = 32;

// 2×2 warp layout over the 32×32 output. Each subgroup owns a 16×16 fragment.
static const uint WM = BM / 2;
static const uint WN = BN / 2;
static const uint WARPS_PER_ROW = BM / WM;

struct MulMmCmParams {
    uint M;
    uint N;
    uint K;
    uint stride_a;       // in elements for dense, in elements for quants (host divides by QUANT_K when indexing)
    uint stride_b;
    uint stride_d;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;
    uint num_batches;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
};

[[vk::push_constant]] ConstantBuffer<MulMmCmParams> p;

// A's storage type is picked by types.slang per DATA_A_X.
[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

// Padding columns kill bank conflicts on the K stride.
static const uint AS_STRIDE = BK + 4;
static const uint BS_STRIDE = BK + 4;

groupshared float16_t buf_a[BM * AS_STRIDE];
groupshared float16_t buf_b[BN * BS_STRIDE];

// Per-DATA_A_X: write a BM × BK tile of dequantized f16 into buf_a starting at
// (row0, k0) in the global A coordinate system. WORKGROUP_SIZE threads cooperate.
void stage_a_tile(uint row0, uint k0, uint tid, uint a_base_blocks) {
#if defined(DATA_A_F32) || defined(DATA_A_F16)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                v = float16_t(float(data_a[a_base_blocks + gr * p.stride_a + gk]));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_BF16)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                // A_TYPE is uint16_t for bf16; bf16_to_fp32 reinterprets bits.
                v = float16_t(bf16_to_fp32(uint(data_a[a_base_blocks + gr * p.stride_a + gk])));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_Q4_0)
    [unroll] for (uint i = 0; i < BM * (BK / 2); i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * (BK / 2)) {
            const uint r = idx / (BK / 2);
            const uint j = idx % (BK / 2);
            const uint gr = row0 + r;
            float16_t v_lo = float16_t(0);
            float16_t v_hi = float16_t(0);
            if (gr < p.M && k0 < p.K) {
                const uint bi = a_base_blocks + gr * (p.stride_a / QUANT_K) + k0 / QUANT_K;
                const float d  = float(data_a[bi].d);
                const uint  qb = uint(data_a[bi].qs[j]);
                v_lo = float16_t((float(int(qb & 0xFu)) - 8.0f) * d);
                v_hi = float16_t((float(int(qb >> 4))  - 8.0f) * d);
            }
            buf_a[r * AS_STRIDE + j        ] = v_lo;
            buf_a[r * AS_STRIDE + j + (BK / 2) ] = v_hi;
        }
    }
#elif defined(DATA_A_Q4_1)
    [unroll] for (uint i = 0; i < BM * (BK / 2); i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * (BK / 2)) {
            const uint r = idx / (BK / 2);
            const uint j = idx % (BK / 2);
            const uint gr = row0 + r;
            float16_t v_lo = float16_t(0);
            float16_t v_hi = float16_t(0);
            if (gr < p.M && k0 < p.K) {
                const uint bi = a_base_blocks + gr * (p.stride_a / QUANT_K) + k0 / QUANT_K;
                const float d  = float(data_a[bi].d);
                const float m  = float(data_a[bi].m);
                const uint  qb = uint(data_a[bi].qs[j]);
                v_lo = float16_t(float(qb & 0xFu) * d + m);
                v_hi = float16_t(float(qb >> 4)   * d + m);
            }
            buf_a[r * AS_STRIDE + j        ] = v_lo;
            buf_a[r * AS_STRIDE + j + (BK / 2) ] = v_hi;
        }
    }
#elif defined(DATA_A_Q5_0)
    [unroll] for (uint i = 0; i < BM * (BK / 2); i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * (BK / 2)) {
            const uint r = idx / (BK / 2);
            const uint j = idx % (BK / 2);
            const uint gr = row0 + r;
            float16_t v_lo = float16_t(0);
            float16_t v_hi = float16_t(0);
            if (gr < p.M && k0 < p.K) {
                const uint bi = a_base_blocks + gr * (p.stride_a / QUANT_K) + k0 / QUANT_K;
                const float d  = float(data_a[bi].d);
                const uint qb  = uint(data_a[bi].qs[j]);
                const uint qh  = uint(data_a[bi].qh[0]) | (uint(data_a[bi].qh[1]) << 16);
                const uint h_lo = (qh >> j)              & 1u;
                const uint h_hi = (qh >> (j + (BK / 2))) & 1u;
                v_lo = float16_t((float(int((qb & 0xFu) | (h_lo << 4))) - 16.0f) * d);
                v_hi = float16_t((float(int((qb >>  4)  | (h_hi << 4))) - 16.0f) * d);
            }
            buf_a[r * AS_STRIDE + j        ] = v_lo;
            buf_a[r * AS_STRIDE + j + (BK / 2) ] = v_hi;
        }
    }
#elif defined(DATA_A_Q5_1)
    [unroll] for (uint i = 0; i < BM * (BK / 2); i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * (BK / 2)) {
            const uint r = idx / (BK / 2);
            const uint j = idx % (BK / 2);
            const uint gr = row0 + r;
            float16_t v_lo = float16_t(0);
            float16_t v_hi = float16_t(0);
            if (gr < p.M && k0 < p.K) {
                const uint bi = a_base_blocks + gr * (p.stride_a / QUANT_K) + k0 / QUANT_K;
                const float d  = float(data_a[bi].d);
                const float m  = float(data_a[bi].m);
                const uint qb  = uint(data_a[bi].qs[j]);
                const uint qh  = uint(data_a[bi].qh);
                const uint h_lo = (qh >> j)              & 1u;
                const uint h_hi = (qh >> (j + (BK / 2))) & 1u;
                v_lo = float16_t(float((qb & 0xFu) | (h_lo << 4)) * d + m);
                v_hi = float16_t(float((qb >>  4)  | (h_hi << 4)) * d + m);
            }
            buf_a[r * AS_STRIDE + j        ] = v_lo;
            buf_a[r * AS_STRIDE + j + (BK / 2) ] = v_hi;
        }
    }
#elif defined(DATA_A_Q8_0)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            float16_t v = float16_t(0);
            if (gr < p.M && k0 + c < p.K) {
                const uint bi = a_base_blocks + gr * (p.stride_a / QUANT_K) + k0 / QUANT_K;
                const float d = float(data_a[bi].d);
                v = float16_t(float(int(data_a[bi].qs[c])) * d);
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ4_NL)
    [unroll] for (uint i = 0; i < BM * (BK / 2); i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * (BK / 2)) {
            const uint r = idx / (BK / 2);
            const uint j = idx % (BK / 2);
            const uint gr = row0 + r;
            float16_t v_lo = float16_t(0);
            float16_t v_hi = float16_t(0);
            if (gr < p.M && k0 < p.K) {
                const uint bi = a_base_blocks + gr * (p.stride_a / QUANT_K) + k0 / QUANT_K;
                const float d  = float(data_a[bi].d);
                const uint  qb = uint(data_a[bi].qs[j]);
                v_lo = float16_t(float(kvalues_iq4nl[qb & 0xFu]) * d);
                v_hi = float16_t(float(kvalues_iq4nl[qb >> 4 ]) * d);
            }
            buf_a[r * AS_STRIDE + j        ] = v_lo;
            buf_a[r * AS_STRIDE + j + (BK / 2) ] = v_hi;
        }
    }
#elif defined(DATA_A_MXFP4)
    [unroll] for (uint i = 0; i < BM * (BK / 2); i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * (BK / 2)) {
            const uint r = idx / (BK / 2);
            const uint j = idx % (BK / 2);
            const uint gr = row0 + r;
            float16_t v_lo = float16_t(0);
            float16_t v_hi = float16_t(0);
            if (gr < p.M && k0 < p.K) {
                const uint bi = a_base_blocks + gr * (p.stride_a / QUANT_K) + k0 / QUANT_K;
                const float d  = e8m0_to_fp32(data_a[bi].e);
                const uint  qb = uint(data_a[bi].qs[j]);
                v_lo = float16_t(float(kvalues_mxfp4[qb & 0xFu]) * d);
                v_hi = float16_t(float(kvalues_mxfp4[qb >> 4 ]) * d);
            }
            buf_a[r * AS_STRIDE + j        ] = v_lo;
            buf_a[r * AS_STRIDE + j + (BK / 2) ] = v_hi;
        }
    }
// ── K-quants (256-element super-blocks) ───────────────────────────────────────
// All K-quants iterate element-by-element. For position p inside a block, the
// sub-block layout is consistent across Q2_K / Q4_K / Q5_K (different from Q6_K):
//   sub_idx = p / 32 (0..7)     — selects scale/min pair
//   il      = sub_idx / 2       — il-pair index 0..3
//   half_hi = sub_idx & 1       — low (0) or high (1) of the pair
//   qs_off  = il * 32 + (p % 32)  (or il * 16 + (p % 32) / 2 for some)
//
// Q2_K layout uses a different partition (16 sub-blocks of 16 elements).
// Q6_K stripes scales across 4 sub-rows × 2 halves with il_high.
#elif defined(DATA_A_Q4_K)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi      = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_in_b  = gk % QUANT_K;             // 0..255
                const uint sub_idx = p_in_b >> 5;              // 0..7
                const uint half_hi = sub_idx & 1u;
                const uint il      = sub_idx >> 1;             // 0..3
                const uint qs_off  = il * 32u + (p_in_b & 31u);

                const float dall = float(data_a[bi].dm.x);
                const float dmin = float(data_a[bi].dm.y);

                uint sc_raw, mb_raw;
                if (sub_idx < 4u) {
                    sc_raw = uint(data_a[bi].scales[sub_idx])         & 0x3Fu;
                    mb_raw = uint(data_a[bi].scales[sub_idx + 4u])    & 0x3Fu;
                } else {
                    sc_raw = (uint(data_a[bi].scales[sub_idx + 4u]) & 0x0Fu)
                           | ((uint(data_a[bi].scales[sub_idx - 4u]) & 0xC0u) >> 2);
                    mb_raw = (uint(data_a[bi].scales[sub_idx + 4u]) >> 4)
                           | ((uint(data_a[bi].scales[sub_idx])     & 0xC0u) >> 2);
                }
                const uint qb = uint(data_a[bi].qs[qs_off]);
                const uint nibble = (half_hi == 0u) ? (qb & 0xFu) : (qb >> 4);
                v = float16_t(dall * float(sc_raw) * float(nibble) - dmin * float(mb_raw));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_Q5_K)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi      = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_in_b  = gk % QUANT_K;
                const uint sub_idx = p_in_b >> 5;
                const uint half_hi = sub_idx & 1u;
                const uint il      = sub_idx >> 1;
                const uint qs_off  = il * 32u + (p_in_b & 31u);

                const float dall = float(data_a[bi].dm.x);
                const float dmin = float(data_a[bi].dm.y);

                uint sc_raw, mb_raw;
                if (sub_idx < 4u) {
                    sc_raw = uint(data_a[bi].scales[sub_idx])      & 0x3Fu;
                    mb_raw = uint(data_a[bi].scales[sub_idx + 4u]) & 0x3Fu;
                } else {
                    sc_raw = (uint(data_a[bi].scales[sub_idx + 4u]) & 0x0Fu)
                           | ((uint(data_a[bi].scales[sub_idx - 4u]) & 0xC0u) >> 2);
                    mb_raw = (uint(data_a[bi].scales[sub_idx + 4u]) >> 4)
                           | ((uint(data_a[bi].scales[sub_idx])     & 0xC0u) >> 2);
                }
                // 5th bit lives in qh[]; bit position sub_idx within byte qh[p_in_b % 32].
                const uint qb     = uint(data_a[bi].qs[qs_off]);
                const uint qh_byt = uint(data_a[bi].qh[p_in_b & 31u]);
                const uint nibble = (half_hi == 0u) ? (qb & 0xFu) : (qb >> 4);
                const uint hbit   = (qh_byt >> sub_idx) & 1u;
                const uint val5   = nibble | (hbit << 4);
                v = float16_t(dall * float(sc_raw) * float(val5) - dmin * float(mb_raw));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_Q6_K)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;                  // 0..255
                const uint ip   = p_b >> 7;                      // 0 or 1 (which 128-half)
                const uint p_h  = p_b & 0x7Fu;                   // position in half (0..127)
                const uint sub_row = p_h >> 5;                   // 0..3
                const uint il_in_h = p_h & 0x1Fu;                // 0..31
                // ql layout: low nibble for sub_row < 2, high for sub_row >= 2; offset 0 or 32.
                const uint ql_off  = 64u * ip + il_in_h + ((sub_row & 1u) * 32u);
                const uint qb_raw  = uint(data_a[bi].ql[ql_off]);
                const uint nibble  = (sub_row < 2u) ? (qb_raw & 0xFu) : (qb_raw >> 4);
                // High 2 bits at qh[32*ip + il_in_h] bits (2*sub_row, 2*sub_row+1).
                const uint qh_raw  = uint(data_a[bi].qh[32u * ip + il_in_h]);
                const uint hi2     = (qh_raw >> (2u * sub_row)) & 3u;
                const int  six_bit = int(nibble | (hi2 << 4));
                const int  signed_v = six_bit - 32;
                const uint scale_idx = 8u * ip + (il_in_h >> 4) + 2u * sub_row;
                const int  sc       = int(data_a[bi].scales[scale_idx]);
                const float d       = float(data_a[bi].d);
                v = float16_t(d * float(sc) * float(signed_v));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_Q2_K)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;
                const uint ip   = p_b >> 7;                       // 0 or 1
                const uint p_h  = p_b & 0x7Fu;                    // 0..127
                const uint sub_row = p_h >> 5;                    // 0..3
                const uint il   = p_h & 0x1Fu;                    // 0..31
                const uint qs_byte = uint(data_a[bi].qs[32u * ip + il]);
                const uint two_bit = (qs_byte >> (2u * sub_row)) & 3u;
                const uint sc_idx  = 8u * ip + (il >> 4) + 2u * sub_row;
                const uint scale_b = uint(data_a[bi].scales[sc_idx]);
                const float dall = float(data_a[bi].dm.x);
                const float dmin = float(data_a[bi].dm.y);
                v = float16_t(dall * float(scale_b & 0xFu) * float(two_bit)
                            - dmin * float(scale_b >> 4));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_Q3_K)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;             // 0..255
                const uint n    = p_b >> 7;                 // 0 or 1
                const uint p_h  = p_b & 0x7Fu;              // 0..127
                const uint sub_row = p_h >> 5;              // 0..3
                const uint p_sr = p_b & 0x1Fu;              // 0..31

                const uint qs_byte = uint(data_a[bi].qs[32u * n + p_sr]);
                const uint two_bit = (qs_byte >> (2u * sub_row)) & 3u;
                const uint hmask_b = uint(data_a[bi].hmask[p_sr]);
                const uint hbit    = (hmask_b >> (p_b >> 5)) & 1u;
                const int  signed3 = int(two_bit) - (hbit != 0u ? 0 : 4);

                // Scale extraction: 6-bit value `us`, then dl = d * (us - 32).
                const uint is = p_b >> 4;                   // 0..15
                uint us;
                if (is < 4u) {
                    us = (uint(data_a[bi].scales[is])      & 0x0Fu)
                       | (((uint(data_a[bi].scales[is + 8u]) >> 0) & 3u) << 4);
                } else if (is < 8u) {
                    us = (uint(data_a[bi].scales[is])      & 0x0Fu)
                       | (((uint(data_a[bi].scales[is + 4u]) >> 2) & 3u) << 4);
                } else if (is < 12u) {
                    us = (uint(data_a[bi].scales[is - 8u]) >> 4)
                       | (((uint(data_a[bi].scales[is])      >> 4) & 3u) << 4);
                } else {
                    us = (uint(data_a[bi].scales[is - 8u]) >> 4)
                       | (((uint(data_a[bi].scales[is - 4u]) >> 6) & 3u) << 4);
                }
                const float d  = float(data_a[bi].d);
                const float dl = d * float(int(us) - 32);
                v = float16_t(dl * float(signed3));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_NVFP4)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi    = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b   = gk % QUANT_K;       // 0..63
                const uint sub   = p_b >> 4;           // 0..3 sub-block (16 elem)
                const uint p_s   = p_b & 0xFu;         // 0..15
                const uint l     = p_s & 7u;           // 0..7
                const uint half_high = p_s >> 3;       // 0 or 1
                const uint qb     = uint(data_a[bi].qs[8u * sub + l]);
                const uint nibble = (half_high == 0u) ? (qb & 0xFu) : (qb >> 4);
                const float d     = ue4m3_to_fp32(data_a[bi].d[sub]) * 0.5f;
                v = float16_t(d * float(kvalues_mxfp4[nibble]));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ2_XS)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;             // 0..255
                const uint ib32 = p_b >> 5;                 // 0..7
                const uint p_s  = p_b & 0x1Fu;              // 0..31
                const uint l    = p_s >> 3;                 // 0..3
                const uint k_in = p_s & 7u;                 // 0..7

                const uint qs_val   = uint(data_a[bi].qs[4u * ib32 + l]);
                const uint grid_idx = qs_val & 0x1FFu;
                const uint sign7    = qs_val >> 9;
                const uint sign8    = sign7 | (countbits(sign7) << 7);
                const uint sign_bit = (sign8 >> k_in) & 1u;

                const uint2 grid = iq2xs_grid[grid_idx];
                const uint4 g0 = unpack8_u(grid.x);
                const uint4 g1 = unpack8_u(grid.y);
                uint gv;
                switch (k_in) {
                    case 0: gv = g0.x; break;
                    case 1: gv = g0.y; break;
                    case 2: gv = g0.z; break;
                    case 3: gv = g0.w; break;
                    case 4: gv = g1.x; break;
                    case 5: gv = g1.y; break;
                    case 6: gv = g1.z; break;
                    default: gv = g1.w; break;
                }

                const uint sc_byte = uint(data_a[bi].scales[ib32]);
                const uint scale_nibble = (l < 2u) ? (sc_byte & 0xFu) : (sc_byte >> 4);
                const float d  = float(data_a[bi].d);
                const float db = d * (0.5f + float(scale_nibble)) * 0.25f;

                v = float16_t(db * float(gv) * (sign_bit != 0u ? -1.0f : 1.0f));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ2_S)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;
                const uint ib32 = p_b >> 5;
                const uint p_s  = p_b & 0x1Fu;
                const uint l    = p_s >> 3;
                const uint k_in = p_s & 7u;

                const uint qs_lo = uint(data_a[bi].qs[4u * ib32 + l]);
                const uint qh_b  = uint(data_a[bi].qh[ib32]);
                // Two extra high bits from qh per `l`, shifted to bits 8..9 of grid_idx.
                const uint grid_idx = qs_lo | ((qh_b << (8u - 2u * l)) & 0x300u);
                // Sign byte lives in the second half of qs[].
                const uint sign_byte = uint(data_a[bi].qs[QUANT_K / 8u + 4u * ib32 + l]);
                const uint sign_bit  = (sign_byte >> k_in) & 1u;

                const uint2 grid = iq2s_grid[grid_idx];
                const uint4 g0 = unpack8_u(grid.x);
                const uint4 g1 = unpack8_u(grid.y);
                uint gv;
                switch (k_in) {
                    case 0: gv = g0.x; break;
                    case 1: gv = g0.y; break;
                    case 2: gv = g0.z; break;
                    case 3: gv = g0.w; break;
                    case 4: gv = g1.x; break;
                    case 5: gv = g1.y; break;
                    case 6: gv = g1.z; break;
                    default: gv = g1.w; break;
                }

                const uint sc_byte = uint(data_a[bi].scales[ib32]);
                const uint scale_nibble = (l < 2u) ? (sc_byte & 0xFu) : (sc_byte >> 4);
                const float d  = float(data_a[bi].d);
                const float db = d * (0.5f + float(scale_nibble)) * 0.25f;

                v = float16_t(db * float(gv) * (sign_bit != 0u ? -1.0f : 1.0f));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ3_S)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;
                const uint is   = p_b >> 5;                 // 0..7
                const uint p_s  = p_b & 0x1Fu;              // 0..31
                const uint l    = p_s >> 2;                 // 0..7  (4-weight groups)
                const uint k_in = p_s & 3u;                 // 0..3

                const uint iqs   = 8u * is + l;
                const uint qs_b  = uint(data_a[bi].qs[iqs]);
                const uint qh_b  = uint(data_a[bi].qh[is]);
                const uint gidx  = qs_b | (((qh_b >> l) & 1u) << 8);
                const uint signs = uint(data_a[bi].signs[iqs / 2u]) >> (4u * (l & 1u));
                const uint sign_bit = (signs >> k_in) & 1u;

                const uint4 g = unpack8_u(iq3s_grid[gidx]);
                uint gv;
                switch (k_in) {
                    case 0: gv = g.x; break;
                    case 1: gv = g.y; break;
                    case 2: gv = g.z; break;
                    default: gv = g.w; break;
                }

                const uint sc_byte = uint(data_a[bi].scales[is >> 1]);
                const uint scale_nibble = (sc_byte >> (4u * (is & 1u))) & 0xFu;
                const float d  = float(data_a[bi].d);
                const float db = d * (1.0f + 2.0f * float(scale_nibble));

                v = float16_t(db * float(gv) * (sign_bit != 0u ? -1.0f : 1.0f));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ1_S)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;
                const uint ib32 = p_b >> 5;
                const uint p_s  = p_b & 0x1Fu;
                const uint l    = p_s >> 3;                 // 0..3
                const uint j    = p_s & 7u;                 // 0..7  (2-bit value within grid u16)

                const uint qh    = uint(data_a[bi].qh[ib32]);
                const uint hi    = (qh >> (3u * l)) & 7u;
                const uint qs_b  = uint(data_a[bi].qs[4u * ib32 + l]);
                const int  grid  = int(iq1s_grid[qs_b | (hi << 8)]);

                // Sign-extend the 2 bits at offset 2*j.
                const int signed_2bit = (grid << (30 - int(2u * j))) >> 30;

                const float d     = float(data_a[bi].d);
                const float dl    = d * float(2u * ((qh >> 12) & 7u) + 1u);
                const float delta = ((qh & 0x8000u) != 0u) ? -IQ1S_DELTA : IQ1S_DELTA;

                v = float16_t(dl * (float(signed_2bit) + delta));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ1_M)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;
                const uint ib32 = p_b >> 5;
                const uint p_s  = p_b & 0x1Fu;
                const uint l    = p_s >> 3;
                const uint j    = p_s & 7u;
                const uint ib16 = 2u * ib32 + (l >> 1);   // 0..15

                // Reconstruct f16 d from the high nibbles of scales[0..3].
                const uint s0 = uint(data_a[bi].scales[0]) >> 12;
                const uint s1 = uint(data_a[bi].scales[1]) >> 12;
                const uint s2 = uint(data_a[bi].scales[2]) >> 12;
                const uint s3 = uint(data_a[bi].scales[3]) >> 12;
                const float d = float(asfloat16(uint16_t(s0 | (s1 << 4) | (s2 << 8) | (s3 << 12))));

                // Sub-scale: 3 bits per sub-block, packed in scales[ib32/2] >> (6 * (ib32 & 1)).
                const uint sc = uint(data_a[bi].scales[ib32 >> 1]) >> (6u * (ib32 & 1u));
                const float dl = d * float(2u * ((sc >> (3u * (ib16 & 3u))) & 7u) + 1u);

                const uint qh_b  = uint(data_a[bi].qh[ib16]) >> (4u * (l & 1u));
                const uint qs_b  = uint(data_a[bi].qs[4u * ib32 + l]);
                const int  grid  = int(iq1s_grid[qs_b | ((qh_b & 7u) << 8)]);
                const int  signed_2bit = (grid << (30 - int(2u * j))) >> 30;
                const float delta = (qh_b & 8u) != 0u ? -IQ1M_DELTA : IQ1M_DELTA;

                v = float16_t(dl * (float(signed_2bit) + delta));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ2_XXS)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi    = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b   = gk % QUANT_K;             // 0..255
                const uint is    = p_b >> 5;                 // 0..7  scale-block
                const uint p_s   = p_b & 0x1Fu;              // 0..31 position in scale-block
                const uint l     = p_s >> 3;                 // 0..3  grid-index quartet
                const uint k_in  = p_s & 7u;                 // 0..7  position within 8-weight grid

                // signscale: bytes [8*is+4 .. 8*is+7] packed LE
                const uint ss = uint(data_a[bi].qs[8u * is + 4u])
                              | (uint(data_a[bi].qs[8u * is + 5u]) <<  8)
                              | (uint(data_a[bi].qs[8u * is + 6u]) << 16)
                              | (uint(data_a[bi].qs[8u * is + 7u]) << 24);
                const float d  = float(data_a[bi].d);
                const float db = d * (0.5f + float(ss >> 28)) * 0.25f;

                const uint sign7 = (ss >> (7u * l)) & 0x7Fu;
                const uint sign8 = sign7 | (countbits(sign7) << 7);
                const uint sign_bit = (sign8 >> k_in) & 1u;

                const uint grid_idx = uint(data_a[bi].qs[8u * is + l]);
                const uint2 grid    = iq2xxs_grid[grid_idx];
                const uint4 g0 = unpack8_u(grid.x);
                const uint4 g1 = unpack8_u(grid.y);
                uint g;
                switch (k_in) {
                    case 0: g = g0.x; break;
                    case 1: g = g0.y; break;
                    case 2: g = g0.z; break;
                    case 3: g = g0.w; break;
                    case 4: g = g1.x; break;
                    case 5: g = g1.y; break;
                    case 6: g = g1.z; break;
                    default: g = g1.w; break;
                }
                v = float16_t(db * float(g) * (sign_bit != 0u ? -1.0f : 1.0f));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ3_XXS)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;
                const uint is   = p_b >> 5;
                const uint p_s  = p_b & 0x1Fu;
                const uint l    = p_s >> 3;
                const uint k_in = p_s & 7u;

                // Sign-scale bytes start after the 64 grid bytes (= QUANT_K / 4).
                const uint ss_off = (QUANT_K / 4u) + 4u * is;
                const uint ss = uint(data_a[bi].qs[ss_off + 0u])
                              | (uint(data_a[bi].qs[ss_off + 1u]) <<  8)
                              | (uint(data_a[bi].qs[ss_off + 2u]) << 16)
                              | (uint(data_a[bi].qs[ss_off + 3u]) << 24);
                const float d  = float(data_a[bi].d);
                const float db = d * (0.5f + float(ss >> 28)) * 0.5f;

                const uint sign7 = (ss >> (7u * l)) & 0x7Fu;
                const uint sign8 = sign7 | (countbits(sign7) << 7);
                const uint sign_bit = (sign8 >> k_in) & 1u;

                // Pair of grid indices per "l".
                const uint qs_idx = (k_in < 4u)
                    ? uint(data_a[bi].qs[8u * is + 2u * l])
                    : uint(data_a[bi].qs[8u * is + 2u * l + 1u]);
                const uint4 g = unpack8_u(iq3xxs_grid[qs_idx]);
                uint gv;
                switch (k_in & 3u) {
                    case 0: gv = g.x; break;
                    case 1: gv = g.y; break;
                    case 2: gv = g.z; break;
                    default: gv = g.w; break;
                }
                v = float16_t(db * float(gv) * (sign_bit != 0u ? -1.0f : 1.0f));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#elif defined(DATA_A_IQ4_XS)
    [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
        const uint idx = i + tid;
        if (idx < BM * BK) {
            const uint r = idx / BK;
            const uint c = idx % BK;
            const uint gr = row0 + r;
            const uint gk = k0 + c;
            float16_t v = float16_t(0);
            if (gr < p.M && gk < p.K) {
                const uint bi   = a_base_blocks + gr * (p.stride_a / QUANT_K) + gk / QUANT_K;
                const uint p_b  = gk % QUANT_K;
                const uint ib32 = p_b >> 5;                  // 0..7
                const uint l    = p_b & 0x1Fu;               // 0..31 within sub-block
                const uint use_hi = (l >= 16u) ? 1u : 0u;
                const uint q_off  = 16u * ib32 + (l & 15u);
                const uint qb     = uint(data_a[bi].qs[q_off]);
                const uint nibble = (use_hi == 0u) ? (qb & 0xFu) : (qb >> 4);
                // 6-bit scale: 4 low bits from scales_l, 2 high bits from scales_h.
                const uint sc_lo = (uint(data_a[bi].scales_l[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0xFu;
                const uint sc_hi = (uint(data_a[bi].scales_h) >> (2u * ib32)) & 3u;
                const int  scale = int(sc_lo | (sc_hi << 4));
                const float d    = float(data_a[bi].d);
                const float dl   = d * float(scale - 32);
                v = float16_t(dl * float(kvalues_iq4nl[nibble]));
            }
            buf_a[r * AS_STRIDE + c] = v;
        }
    }
#endif
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_LocalInvocationID   : SV_GroupThreadID) {
    const uint batch_idx = gl_WorkGroupID.z;
    const uint i13 = batch_idx / p.ne12;
    const uint i12 = batch_idx % p.ne12;
    const uint i03 = i13 / p.broadcast3;
    const uint i02 = i12 / p.broadcast2;
    const uint batch_idx_a = i03 * p.ne02 + i02;

    // For dense, `a_base_blocks` is just the element offset; for quants it is the
    // block offset (host configures `batch_stride_a` to be in elements regardless,
    // we divide by QUANT_K to get blocks).
#if defined(DATA_A_F32) || defined(DATA_A_F16) || defined(DATA_A_BF16)
    const uint a_base = batch_idx_a * p.batch_stride_a;
#else
    const uint a_base = batch_idx_a * (p.batch_stride_a / QUANT_K);
#endif
    const uint b_base = batch_idx   * p.batch_stride_b;
    const uint d_base = batch_idx   * p.batch_stride_d;

    const uint tid     = gl_LocalInvocationID.x;
    const uint warp_id = tid / SUBGROUP_SIZE;
    const uint warp_r  = warp_id % WARPS_PER_ROW;
    const uint warp_c  = warp_id / WARPS_PER_ROW;

    const uint row0 = gl_WorkGroupID.x * BM;
    const uint col0 = gl_WorkGroupID.y * BN;
    if (row0 >= p.M || col0 >= p.N) { return; }

#ifdef NEEDS_INIT_IQ_SHMEM
    // IQ4_NL / MXFP4 need the kvalues table populated cooperatively.
    init_iq_shmem(uint3(WORKGROUP_SIZE, 1, 1));
#endif

    var sums = CoopMat<float, MemoryScope.Subgroup, TM, TN,
                       CoopMatMatrixUse::MatrixAccumulator>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        stage_a_tile(row0, k0, tid, a_base);

        // B is dense — col-major in (K, N), staged the same way regardless of A type.
        [unroll] for (uint i = 0; i < BN * BK; i += WORKGROUP_SIZE) {
            const uint idx = i + tid;
            if (idx < BN * BK) {
                const uint r = idx / BK;
                const uint c = idx % BK;
                const uint gn = col0 + r;
                const uint gk = k0 + c;
                float16_t v = float16_t(0);
                if (gn < p.N && gk < p.K) {
                    v = float16_t(float(data_b[b_base + gn * p.stride_b + gk]));
                }
                buf_b[r * BS_STRIDE + c] = v;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kk = 0; kk < BK; kk += TK) {
            var matA = CoopMat<float16_t, MemoryScope.Subgroup, TM, TK,
                               CoopMatMatrixUse::MatrixA>(float16_t(0));
            var matB = CoopMat<float16_t, MemoryScope.Subgroup, TK, TN,
                               CoopMatMatrixUse::MatrixB>(float16_t(0));
            matA.Load<CoopMatMatrixLayout::RowMajor>(buf_a,
                                                     warp_r * WM * AS_STRIDE + kk,
                                                     AS_STRIDE);
            matB.Load<CoopMatMatrixLayout::ColumnMajor>(buf_b,
                                                        warp_c * WN * BS_STRIDE + kk,
                                                        BS_STRIDE);
            sums = coopMatMulAdd<float, false>(matA, matB, sums);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const uint sub_row = row0 + warp_r * WM;
    const uint sub_col = col0 + warp_c * WN;
    if (sub_row < p.M && sub_col < p.N) {
        sums.Store<CoopMatMatrixLayout::RowMajor>(data_d,
                                                  d_base + sub_row * p.stride_d + sub_col,
                                                  p.stride_d);
    }
}
