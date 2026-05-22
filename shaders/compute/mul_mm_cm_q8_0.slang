// Quantized cooperative-matrix matmul: Q8_0 weights × F16 (or F32) activations.
// Dequantizes each Q8_0 block into a groupshared F16 staging buffer, then runs a
// CoopMat tile through it. This is the basic pattern for any KHR-coopmat quant matmul —
// the same skeleton extends to Q4_0/Q4_1/Q5_*/IQ* by swapping the dequant inner loop.
//
// Layout assumptions:
//   - A is row-major (M, K), stored as block_q8_0 with QUANT_K_Q8_0 = 32 elements per
//     block. So `p.stride_a` is in elements; the host wires `data_a` as a
//     StructuredBuffer<block_q8_0> and divides offsets by QUANT_K_Q8_0.
//   - B is column-major (K, N) in F16 (or F32 promoted to F16 via cast).
//   - D is row-major (M, N), F32.

//@variants
//  q8_0_f16:   B_TYPE=float16_t   D_TYPE=float
//  q8_0_f32:   B_TYPE=float       D_TYPE=float
//@end-variants

import glsl;

using namespace linalg;

#include "types.slang"

static const uint TM = 16;
static const uint TN = 16;
static const uint TK = 16;

static const uint BM = 32;
static const uint BN = 32;
// Block-K must be a multiple of QUANT_K_Q8_0 (32). One Q8_0 block fully covers BK.
static const uint BK = QUANT_K_Q8_0;

static const uint WORKGROUP_SIZE = 128;
static const uint SUBGROUP_SIZE  = 32;
static const uint NUM_WARPS      = WORKGROUP_SIZE / SUBGROUP_SIZE;

static const uint WM = BM / 2;
static const uint WN = BN / 2;
static const uint WARPS_PER_ROW = BM / WM;

struct MulMmCmQ80Params {
    uint M;
    uint N;
    uint K;
    uint stride_a;   // in elements (not blocks)
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

[[vk::push_constant]] ConstantBuffer<MulMmCmQ80Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q8_0> data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>     data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE>   data_d;

// Staging buffers — pad K stride to dodge shared-memory bank conflicts.
static const uint AS_STRIDE = BK + 4;
static const uint BS_STRIDE = BK + 4;
groupshared float16_t buf_a[BM * AS_STRIDE];
groupshared float16_t buf_b[BN * BS_STRIDE];

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

    // batch_stride_a is in elements (matching the convention from the generic head);
    // the structured buffer indexes blocks, so divide by QUANT_K_Q8_0.
    const uint a_base_blocks = batch_idx_a * (p.batch_stride_a / QUANT_K_Q8_0);
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

    var sums = CoopMat<float, MemoryScope.Subgroup, TM, TN,
                       CoopMatMatrixUse::MatrixAccumulator>(0.0f);

    // K is in elements; one outer step covers BK = 32 elements = 1 Q8_0 block.
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        // Dequantize A tile (BM × BK) → buf_a. Each thread owns one (row, sub-quad)
        // for a balanced 128-way fan-out over 32 rows × 32 elements = 1024 entries.
        [unroll] for (uint i = 0; i < BM * BK; i += WORKGROUP_SIZE) {
            const uint idx = i + tid;
            if (idx < BM * BK) {
                const uint r = idx / BK;
                const uint c = idx % BK;
                const uint global_r = row0 + r;
                float16_t v = float16_t(0);
                if (global_r < p.M && k0 + c < p.K) {
                    const uint block_idx = a_base_blocks
                                         + global_r * (p.stride_a / QUANT_K_Q8_0)
                                         + k0 / QUANT_K_Q8_0;
                    const float d = float(data_a[block_idx].d);
                    v = float16_t(float(int(data_a[block_idx].qs[c])) * d);
                }
                buf_a[r * AS_STRIDE + c] = v;
            }
        }
        // Stage B tile (BN × BK), column-major in (K, N).
        [unroll] for (uint i = 0; i < BN * BK; i += WORKGROUP_SIZE) {
            const uint idx = i + tid;
            if (idx < BN * BK) {
                const uint r = idx / BK;
                const uint c = idx % BK;
                const uint global_n = col0 + r;
                const uint global_k = k0 + c;
                float16_t v = float16_t(0);
                if (global_n < p.N && global_k < p.K) {
                    v = float16_t(float(data_b[b_base + global_n * p.stride_b + global_k]));
                }
                buf_b[r * BS_STRIDE + c] = v;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // CoopMat tiles are 16×16, BK is 32 — run two MulAdds back-to-back over the
        // K dim, advancing the load offset by TK between them.
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
