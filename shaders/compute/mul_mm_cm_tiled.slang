// Multi-tile cooperative-matrix matmul. Mirrors llama.cpp's mul_mm.comp COOPMAT path:
// each workgroup owns one BM×BN output tile, split across NUM_WARPS subgroups; each
// subgroup walks cms_per_row × cms_per_col CoopMat fragments through the K dim. A and B
// are staged through shmem in BK-wide blocks so memory traffic is amortized.
//
// First-cut dense variants only (F16 / BF16 weights, F32 accumulator). Quant variants
// would dequantize into the shmem-staging step before CoopMatLoad — that's a separate
// follow-up keyed off DATA_A_<QUANT>.

//@variants
//  f16:        DATA_A_F16=1   B_TYPE=float16_t   D_TYPE=float
//  f16_f32:    DATA_A_F16=1   B_TYPE=float       D_TYPE=float
//@end-variants

import glsl;

using namespace linalg;

#include "types.slang"

// CoopMat tile dims — 16×16×16 is the safe baseline supported across vendors.
static const uint TM = 16;
static const uint TN = 16;
static const uint TK = 16;

// Workgroup tile dims. BM×BN output, BK along the K dim per shmem stage.
// 4 subgroups (128 threads at SubGroupSize=32) → each subgroup owns 1 fragment row,
// 2 fragment columns of the 32×32 output tile.
static const uint BM = 32;
static const uint BN = 32;
static const uint BK = 16;

static const uint WORKGROUP_SIZE = 128;
static const uint SUBGROUP_SIZE  = 32;
static const uint NUM_WARPS      = WORKGROUP_SIZE / SUBGROUP_SIZE;

// Each subgroup gets a WM × WN region of the output tile.
static const uint WM = BM / 2;   // 16 — fragments stacked vertically
static const uint WN = BN / 2;   // 16 — split across 2 warps per row

static const uint CMS_PER_ROW = WM / TM;   // 1
static const uint CMS_PER_COL = WN / TN;   // 1
static const uint WARPS_PER_ROW = BM / WM; // 2

struct MulMmCmTiledParams {
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

[[vk::push_constant]] ConstantBuffer<MulMmCmTiledParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

// Padding column kills shared-memory bank conflicts on the K stride.
static const uint AS_STRIDE = BK + 4;
static const uint BS_STRIDE = BK + 4;

groupshared A_TYPE buf_a[BM * AS_STRIDE];
groupshared B_TYPE buf_b[BN * BS_STRIDE];

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

    const uint tile_row = gl_WorkGroupID.x;
    const uint tile_col = gl_WorkGroupID.y;
    const uint row0     = tile_row * BM;
    const uint col0     = tile_col * BN;
    if (row0 >= p.M || col0 >= p.N) { return; }

    // Per-subgroup accumulator fragments. With CMS_PER_ROW == CMS_PER_COL == 1
    // each subgroup owns a single 16×16 tile of the 32×32 output.
    var sums = CoopMat<float, MemoryScope.Subgroup, TM, TN,
                       CoopMatMatrixUse::MatrixAccumulator>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        // Stage A tile (BM × BK).
        [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
            const uint idx = i + tid;
            if (idx < BM * BK) {
                const uint r = idx / BK;
                const uint c = idx % BK;
                const uint global_r = row0 + r;
                const uint global_c = k0 + c;
                A_TYPE v = A_TYPE(0);
                if (global_r < p.M && global_c < p.K) {
                    v = data_a[a_base + global_r * p.stride_a + global_c];
                }
                buf_a[r * AS_STRIDE + c] = v;
            }
        }
        // Stage B tile (BN × BK), column-major in (K, N).
        [unroll] for (uint i = 0; i < BN * BK; i += WORKGROUP_SIZE) {
            const uint idx = i + tid;
            if (idx < BN * BK) {
                const uint r = idx / BK;   // N
                const uint c = idx % BK;   // K
                const uint global_n = col0 + r;
                const uint global_k = k0 + c;
                B_TYPE v = B_TYPE(0);
                if (global_n < p.N && global_k < p.K) {
                    v = data_b[b_base + global_n * p.stride_b + global_k];
                }
                buf_b[r * BS_STRIDE + c] = v;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Load this subgroup's fragments from shmem and accumulate.
        var matA = CoopMat<A_TYPE, MemoryScope.Subgroup, TM, TK,
                           CoopMatMatrixUse::MatrixA>(A_TYPE(0));
        var matB = CoopMat<B_TYPE, MemoryScope.Subgroup, TK, TN,
                           CoopMatMatrixUse::MatrixB>(B_TYPE(0));

        matA.Load<CoopMatMatrixLayout::RowMajor>(buf_a,
                                                 warp_r * WM * AS_STRIDE,
                                                 AS_STRIDE);
        matB.Load<CoopMatMatrixLayout::ColumnMajor>(buf_b,
                                                    warp_c * WN * BS_STRIDE,
                                                    BS_STRIDE);
        sums = coopMatMulAdd<float, false>(matA, matB, sums);

        GroupMemoryBarrierWithGroupSync();
    }

    // Write this subgroup's fragment back to D.
    const uint sub_row = row0 + warp_r * WM;
    const uint sub_col = col0 + warp_c * WN;
    if (sub_row < p.M && sub_col < p.N) {
        sums.Store<CoopMatMatrixLayout::RowMajor>(data_d,
                                                  d_base + sub_row * p.stride_d + sub_col,
                                                  p.stride_d);
    }
}
