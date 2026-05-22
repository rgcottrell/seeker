// Scalar blocked-GEMM matmul (no cooperative matrix). Mirrors llama.cpp's
// mul_mm.comp non-COOPMAT path: each workgroup owns a BM × BN output tile,
// split across NUM_WARPS subgroups. Each subgroup walks a WM × WN region
// using register tiles of TM × TN per CoopMat-equivalent fragment (here
// just `regC[TM][TN]` arrays). A and B are staged through groupshared in
// BK-wide blocks per outer K step.
//
// First cut: F16 weights only (DATA_A_F16). Quant variants would dequant
// into buf_a in the staging step — same template as mul_mm_cm.

//@variants
//  f16:       DATA_A_F16=1   B_TYPE=float16_t   D_TYPE=float
//  f16_f32:   DATA_A_F16=1   B_TYPE=float       D_TYPE=float
//@end-variants

import glsl;

#include "types.slang"

// Tile shape — kept modest so the register footprint fits typical GPUs.
static const uint TM = 4;
static const uint TN = 4;

static const uint BM = 32;
static const uint BN = 32;
static const uint BK = 16;

static const uint WORKGROUP_SIZE = 64;       // 2 subgroups @ 32
static const uint SUBGROUP_SIZE  = 32;
static const uint NUM_WARPS      = WORKGROUP_SIZE / SUBGROUP_SIZE;

static const uint WM = BM / 2;               // 16 — vertical tile per subgroup
static const uint WN = BN;                   // 32 — full width
static const uint WMITER = 1;
static const uint WNITER = (WM * WN) / (SUBGROUP_SIZE * TM * TN * WMITER);

static const uint WSUBM = WM / WMITER;
static const uint WSUBN = WN / WNITER;
static const uint WARPS_PER_ROW = BM / WM;   // 2

struct MulMmParams {
    uint M;
    uint N;
    uint K;
    uint stride_a;
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

[[vk::push_constant]] ConstantBuffer<MulMmParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

static const uint AS_STRIDE = BK + 4;
static const uint BS_STRIDE = BK + 4;

groupshared float buf_a[BM * AS_STRIDE];
groupshared float buf_b[BN * BS_STRIDE];

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

    const uint a_base = batch_idx_a * p.batch_stride_a;
    const uint b_base = batch_idx   * p.batch_stride_b;
    const uint d_base = batch_idx   * p.batch_stride_d;

    const uint tid     = gl_LocalInvocationID.x;
    const uint warp_id = tid / SUBGROUP_SIZE;
    const uint warp_r  = warp_id % WARPS_PER_ROW;
    const uint warp_c  = warp_id / WARPS_PER_ROW;
    const uint tiw     = tid % SUBGROUP_SIZE;
    const uint tiwr    = tiw % (WSUBM / TM);
    const uint tiwc    = tiw / (WSUBM / TM);

    const uint row0 = gl_WorkGroupID.x * BM;
    const uint col0 = gl_WorkGroupID.y * BN;
    if (row0 >= p.M || col0 >= p.N) { return; }

    float regC[TM][TN];
    [unroll] for (uint r = 0; r < TM; ++r) {
        [unroll] for (uint c = 0; c < TN; ++c) {
            regC[r][c] = 0.0f;
        }
    }

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        // Stage A (BM × BK) into buf_a.
        [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
            const uint idx = i + tid;
            if (idx < BM * BK) {
                const uint rr = idx / BK;
                const uint cc = idx % BK;
                const uint gr = row0 + rr;
                const uint gk = k0 + cc;
                float v = 0.0f;
                if (gr < p.M && gk < p.K) {
                    v = float(data_a[a_base + gr * p.stride_a + gk]);
                }
                buf_a[rr * AS_STRIDE + cc] = v;
            }
        }
        // Stage B (BN × BK), column-major in (K, N).
        [unroll] for (uint i = 0; i < BN * BK; i += WORKGROUP_SIZE) {
            const uint idx = i + tid;
            if (idx < BN * BK) {
                const uint rr = idx / BK;
                const uint cc = idx % BK;
                const uint gn = col0 + rr;
                const uint gk = k0 + cc;
                float v = 0.0f;
                if (gn < p.N && gk < p.K) {
                    v = float(data_b[b_base + gn * p.stride_b + gk]);
                }
                buf_b[rr * BS_STRIDE + cc] = v;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Inner FMA loop over the BK dim.
        [unroll] for (uint kk = 0; kk < BK; ++kk) {
            float regA[TM];
            float regB[TN];
            [unroll] for (uint r = 0; r < TM; ++r) {
                regA[r] = buf_a[(warp_r * WM + tiwr * TM + r) * AS_STRIDE + kk];
            }
            [unroll] for (uint c = 0; c < TN; ++c) {
                regB[c] = buf_b[(warp_c * WN + tiwc * TN + c) * BS_STRIDE + kk];
            }
            [unroll] for (uint r = 0; r < TM; ++r) {
                [unroll] for (uint c = 0; c < TN; ++c) {
                    regC[r][c] = mad(regA[r], regB[c], regC[r][c]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write C tile back to D.
    const uint sub_row = row0 + warp_r * WM + tiwr * TM;
    const uint sub_col = col0 + warp_c * WN + tiwc * TN;
    [unroll] for (uint r = 0; r < TM; ++r) {
        [unroll] for (uint c = 0; c < TN; ++c) {
            const uint gr = sub_row + r;
            const uint gc = sub_col + c;
            if (gr < p.M && gc < p.N) {
                data_d[d_base + gr * p.stride_d + gc] = D_TYPE(regC[r][c]);
            }
        }
    }
}
