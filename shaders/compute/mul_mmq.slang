// Integer-dot blocked matmul (Q8_0 × Q8_1). The matmul-shaped sibling of
// mul_mat_vecq — instead of a single output row, each workgroup produces a
// BM × BN tile, sharing A and B blocks across multiple output cells through
// shmem-staged caches.
//
// Layout assumptions (mirrors llama.cpp's mmq path):
//   - A: (M, K) row-major, Q8_0 blocks. stride_a is in elements; (stride_a / 32)
//     gives blocks per row.
//   - B: (N, K) row-major, quantized to Q8_1_x4 bundles. Each bundle holds 4
//     consecutive K-blocks for one N column.
//   - D: (M, N) row-major, F32.
//
// Per outer K step (one 32-element block):
//   1. Stage A tile (BM rows × 8 int32) and per-row d scales into shmem
//   2. Stage B tile (BN cols × 8 int32) and per-col (d, sum*d) into shmem
//   3. Each thread accumulates one or more (row, col) outputs via 8 dot4add
//      calls per block, scaled by d_a · d_b.x

//@variants
//  q8_0_f32: DATA_A_Q8_0=1   D_TYPE=float    FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"

// Output tile per workgroup, picked so 128 threads handle 4 outputs each (TM*TN=4).
static const uint BM = 16;
static const uint BN = 32;
static const uint WORKGROUP_SIZE = 128;
static const uint TM = 2;
static const uint TN = 2;
// BM * BN == WORKGROUP_SIZE * TM * TN (16 * 32 == 128 * 2 * 2 = 512) — sized so each
// thread owns one TM × TN block of the output tile.

// Each thread's output region: (TM rows × TN cols) at (tid_y*TM, tid_x*TN).
static const uint THREADS_X = BN / TN;  // 16
static const uint THREADS_Y = BM / TM;  // 8

struct MmqParams {
    uint M;
    uint N;
    uint K;
    uint stride_a;   // in elements
    uint stride_b;   // in elements (B is col-strided)
    uint stride_d;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;
    uint base_work_group_z;
    uint num_batches;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
};

[[vk::push_constant]] ConstantBuffer<MmqParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q8_0_packed16> data_a;
[[vk::binding(1, 0)]] StructuredBuffer<block_q8_1_x4>       data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE>            data_d;

// Per-block-step shared caches: 8 packed int32 quants + a scale per row/col.
groupshared int   a_qs[BM * 8];
groupshared float a_d[BM];
groupshared int   b_qs[BN * 8];
groupshared float b_d[BN];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid     = gl_LocalInvocationID.x;
    const uint tid_x   = tid % THREADS_X;
    const uint tid_y   = tid / THREADS_X;

    const uint tile_row = gl_WorkGroupID.x;
    const uint tile_col = gl_WorkGroupID.y;
    const uint batch_idx = gl_WorkGroupID.z + p.base_work_group_z;

    const uint i13 = batch_idx / p.ne12;
    const uint i12 = batch_idx % p.ne12;
    const uint i03 = i13 / p.broadcast3;
    const uint i02 = i12 / p.broadcast2;
    const uint batch_idx_a = i03 * p.ne02 + i02;

    const uint a_row_blocks  = p.stride_a / 32u;
    const uint a_base_blocks = batch_idx_a * (p.batch_stride_a / 32u);
    // B is column-major in (N, K) — col stride in blocks = stride_b / 32, packed as
    // x4 bundles so the x4 stride per column is stride_b / 128.
    const uint b_col_x4      = p.stride_b / 128u;
    const uint b_base_x4     = batch_idx * (p.batch_stride_b / 128u);

    // Per-thread output accumulator (TM × TN).
    float acc[TM][TN];
    [unroll] for (uint r = 0; r < TM; ++r) {
        [unroll] for (uint c = 0; c < TN; ++c) { acc[r][c] = 0.0f; }
    }

    const uint num_blocks = p.K / 32u;
    for (uint blk = 0; blk < num_blocks; ++blk) {
        // ── Stage A tile: BM rows × 8 packed i32 + 1 d per row.
        // BM * 8 = 128 i32 slots; one slot per thread.
        if (tid < BM * 8u) {
            const uint r       = tid / 8u;
            const uint i       = tid % 8u;
            const uint gr      = tile_row * BM + r;
            int packed = 0;
            float d_r  = 0.0f;
            if (gr < p.M) {
                const uint a_blk = a_base_blocks + gr * a_row_blocks + blk;
                packed = int((uint(data_a[a_blk].qs[i * 2u]) & 0xFFFFu)
                           | (uint(data_a[a_blk].qs[i * 2u + 1u]) << 16));
                if (i == 0u) { d_r = float(data_a[a_blk].d); }
            }
            a_qs[r * 8u + i] = packed;
            if (i == 0u) { a_d[r] = d_r; }
        }

        // ── Stage B tile: BN cols × 8 packed i32 + 1 d per col.
        // BN * 8 = 256 slots; 2 slots per thread.
        [unroll] for (uint s = 0; s < (BN * 8u) / WORKGROUP_SIZE; ++s) {
            const uint slot = s * WORKGROUP_SIZE + tid;
            if (slot < BN * 8u) {
                const uint c       = slot / 8u;
                const uint i       = slot % 8u;
                const uint gc      = tile_col * BN + c;
                int packed = 0;
                float d_c  = 0.0f;
                if (gc < p.N) {
                    const uint bx4   = b_base_x4 + gc * b_col_x4 + blk / 4u;
                    const uint binx4 = blk % 4u;
                    packed = data_b[bx4].qs[binx4 * 8u + i];
                    if (i == 0u) { d_c = float(data_b[bx4].ds[binx4].x); }
                }
                b_qs[c * 8u + i] = packed;
                if (i == 0u) { b_d[c] = d_c; }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Inner dot products per (r, c) in this thread's TM × TN tile.
        [unroll] for (uint r = 0; r < TM; ++r) {
            const uint a_r = tid_y * TM + r;
            const float d_a = a_d[a_r];
            [unroll] for (uint c = 0; c < TN; ++c) {
                const uint b_c = tid_x * TN + c;
                int idot = 0;
                [unroll] for (uint i = 0; i < 8; ++i) {
                    idot = dot4add_i8packed(uint(a_qs[a_r * 8u + i]),
                                            uint(b_qs[b_c * 8u + i]),
                                            idot);
                }
                acc[r][c] += float(idot) * d_a * b_d[b_c];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write the TM × TN tile back.
    const uint global_row0 = tile_row * BM + tid_y * TM;
    const uint global_col0 = tile_col * BN + tid_x * TN;
    [unroll] for (uint r = 0; r < TM; ++r) {
        const uint gr = global_row0 + r;
        if (gr >= p.M) { continue; }
        [unroll] for (uint c = 0; c < TN; ++c) {
            const uint gc = global_col0 + c;
            if (gc < p.N) {
                data_d[batch_idx * p.batch_stride_d + gr * p.stride_d + gc] = D_TYPE(acc[r][c]);
            }
        }
    }
}
