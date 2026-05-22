// NV cooperative_matrix2 matmul. Uses SPV_NV_cooperative_matrix2's
// OpCooperativeMatrixLoadTensorNV + tensor-layout/tensor-view APIs to do
// a single coop-matrix tile per workgroup with hardware-assisted tensor
// addressing (clamping, slicing, transposed B view).
//
// Slang exposes the API via `linalg::TensorLayout<>`, `linalg::TensorView<>`,
// and `CoopMat<>.Load(buf, off, layout[, view])`. The `cooperative_matrix_2`
// capability + spvCooperativeMatrixTensorAddressingNV are enabled in build.rs.
//
// First cut: F16 weights × F16 activations → F32 output, no MUL_MAT_ID,
// no quant DECODEFUNC, no split-K (k_split == K). BM × BN output tile per
// workgroup with workgroup-scope CoopMats.

//@variants
//  f16:       DATA_A_F16=1   B_TYPE=float16_t   D_TYPE=float
//@end-variants

using namespace linalg;

#include "types.slang"

// CoopMat dims must be compile-time constants. BLOCK_SIZE also must be static
// (not a spec const) because workgroup-scope CoopMat types reference LocalSizeId,
// and spirv-val rejects forward refs from CoopMat type → spec-const local size.
static const uint BLOCK_SIZE = 256;
static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;

struct MulMmCm2Params {
    uint M;
    uint N;
    uint K;
    uint stride_a;
    uint stride_b;
    uint stride_d;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;
    uint base_work_group_z;
    uint num_batches;
    uint k_split;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
    uint padded_N;
};

[[vk::push_constant]] ConstantBuffer<MulMmCm2Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<float16_t> data_a;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float>   data_d;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID) {
    const uint batch_idx = gl_WorkGroupID.z + p.base_work_group_z;
    const uint i13 = batch_idx / p.ne12;
    const uint i12 = batch_idx % p.ne12;
    const uint i03 = i13 / p.broadcast3;
    const uint i02 = i12 / p.broadcast2;
    const uint batch_idx_a = i03 * p.ne02 + i02;

    const uint ir = gl_WorkGroupID.x;   // M tile
    const uint ic = gl_WorkGroupID.y;   // N tile
    if (ir * BM >= p.M || ic * BN >= p.padded_N) { return; }

    const uint pos_a = batch_idx_a * p.batch_stride_a;
    const uint pos_b = batch_idx   * p.batch_stride_b;
    const uint pos_d = batch_idx   * p.batch_stride_d;

    // ── Tensor layouts. A and B are row-major (M, K) and (padded_N, K). D is
    // written row-major (N, M) and then transposed at store time so the
    // host sees (M, N). All use ClampMode.Constant so out-of-range reads
    // return 0 / writes are no-ops at the M/N edges.
    TensorLayout<2, CoopMatClampMode.Constant> tlA;
    TensorLayout<2, CoopMatClampMode.Constant> tlB;
    TensorLayout<2, CoopMatClampMode.Constant> tlD;
    let tlA1 = tlA.Dimension(p.M, p.K).Stride(p.stride_a, 1);
    let tlB1 = tlB.Dimension(p.padded_N, p.K).Stride(p.stride_b, 1);
    let tlD1 = tlD.Dimension(p.N, p.M).Stride(p.stride_d, 1);

    // Swap dims 0,1 → store result row-major in (M, N) on the host side.
    TensorView<2, false, 1, 0> tvTranspose;

    var sum = CoopMat<float, MemoryScope.Workgroup, BM, BN,
                      CoopMatMatrixUse.MatrixAccumulator>(0.0f);

    const uint end_k = p.K;
    for (uint block_k = 0; block_k < end_k; block_k += BK) {
        let tlA_slice = tlA1.Slice(ir * BM, BM, block_k, BK);
        let tlB_slice = tlB1.Slice(ic * BN, BN, block_k, BK);

        var mat_a = CoopMat<float16_t, MemoryScope.Workgroup, BM, BK,
                            CoopMatMatrixUse.MatrixA>.Load(data_a, pos_a, tlA_slice);
        var mat_b = CoopMat<float16_t, MemoryScope.Workgroup, BK, BN,
                            CoopMatMatrixUse.MatrixB>.Load(data_b, pos_b, tlB_slice, tvTranspose);

        sum = coopMatMulAdd<float, false>(mat_a, mat_b, sum);
    }

    // Store back with transpose view → (M, N) row-major in user buffer.
    let tlD_slice = tlD1.Slice(ic * BN, BN, ir * BM, BM);
    sum.Store(data_d, pos_d, tlD_slice, tvTranspose);
}
