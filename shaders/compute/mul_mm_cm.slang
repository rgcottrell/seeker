// Cooperative-matrix matmul (Slang linalg::CoopMat / SPV_KHR_cooperative_matrix).
//
// This is a first-cut KHR-cm port — F16 weights, F16 activations, F32 accumulator,
// no MUL_MAT_ID. Each subgroup owns a single 16×16 tile of the output and streams the
// K dimension in 16-wide blocks. A and B are loaded directly from device memory; we
// don't stage through shmem in this first cut.
//
// Variants follow the same naming as mul_mm.comp's KHR-coopmat path but limited to
// the {DATA_A_F32 | DATA_A_F16} × B_TYPE={float, float16_t} grid.

//@variants
//  f32:        DATA_A_F32=1   B_TYPE=float       D_TYPE=float
//  f16:        DATA_A_F16=1   B_TYPE=float16_t   D_TYPE=float
//  f16_f32:    DATA_A_F16=1   B_TYPE=float       D_TYPE=float
//  f32_f16:    DATA_A_F32=1   B_TYPE=float16_t   D_TYPE=float
//@end-variants

using namespace linalg;

#include "types.slang"

// Tile sizes are KHR-coopmat's native 16×16×16 — most implementations support this
// for {f16,f16}→f32 and {bf16,bf16}→f32; larger tiles need device-feature queries.
static const uint TILE_M = 16;
static const uint TILE_N = 16;
static const uint TILE_K = 16;

struct MulMmCmParams {
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

[[vk::push_constant]] ConstantBuffer<MulMmCmParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

// One subgroup per output tile. SubgroupSize must match what the device reports —
// we hard-code 32 here for the SPIR-V; the host can override via spec constants
// later if needed.
[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID) {
    const uint batch_idx = gl_WorkGroupID.z;
    const uint i13 = batch_idx / p.ne12;
    const uint i12 = batch_idx % p.ne12;
    const uint i03 = i13 / p.broadcast3;
    const uint i02 = i12 / p.broadcast2;
    const uint batch_idx_a = i03 * p.ne02 + i02;

    const uint tile_row = gl_WorkGroupID.x;
    const uint tile_col = gl_WorkGroupID.y;
    const uint row0     = tile_row * TILE_M;
    const uint col0     = tile_col * TILE_N;
    if (row0 >= p.M || col0 >= p.N) { return; }

    const uint a_off = batch_idx_a * p.batch_stride_a;
    const uint b_off = batch_idx   * p.batch_stride_b;
    const uint d_off = batch_idx   * p.batch_stride_d;

    var matC = CoopMat<float, MemoryScope.Subgroup, TILE_M, TILE_N,
                       CoopMatMatrixUse::MatrixAccumulator>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += TILE_K) {
        var matA = CoopMat<A_TYPE, MemoryScope.Subgroup, TILE_M, TILE_K,
                           CoopMatMatrixUse::MatrixA>(A_TYPE(0));
        var matB = CoopMat<B_TYPE, MemoryScope.Subgroup, TILE_K, TILE_N,
                           CoopMatMatrixUse::MatrixB>(B_TYPE(0));

        // A is row-major in (M, K), B is column-major in (K, N) — match llama.cpp's
        // canonical layout where the host transposes B before dispatch.
        matA.Load<CoopMatMatrixLayout::RowMajor>(data_a,
                                                 a_off + row0 * p.stride_a + k0,
                                                 p.stride_a);
        matB.Load<CoopMatMatrixLayout::ColumnMajor>(data_b,
                                                    b_off + col0 * p.stride_b + k0,
                                                    p.stride_b);
        matC = coopMatMulAdd<float, false>(matA, matB, matC);
    }

    matC.Store<CoopMatMatrixLayout::RowMajor>(data_d,
                                              d_off + row0 * p.stride_d + col0,
                                              p.stride_d);
}
