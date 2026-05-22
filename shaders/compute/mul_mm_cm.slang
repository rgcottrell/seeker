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
//@end-variants

import glsl;

using namespace linalg;

#include "types.slang"

// CoopMat tile dims. 16×16×16 is the safe baseline supported across vendors.
static const uint TM = 16;
static const uint TN = 16;
static const uint TK = 16;

// Workgroup tile dims. BK is set per A type — quant blocks set their own QUANT_K,
// dense types use 32 so the inner loop still runs BK/TK = 2 CoopMat MulAdds.
#if defined(DATA_A_F32) || defined(DATA_A_F16) || defined(DATA_A_BF16)
static const uint BK = 32;
#else
static const uint BK = QUANT_K;
#endif

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
