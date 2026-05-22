// Integer-dot mat-vec: A is Q8_0 weights, B is pre-quantized Q8_1 activations.
//
// Uses HLSL SM 6.4's `dot4add_i8packed(uint a, uint b, int acc)` which lowers to
// SPV_KHR_integer_dot_product via the `sm_6_4` capability already enabled in build.rs.
// Per A block (32 i8 weights) × matching B block (32 i8 activations, +d, +sum*d):
//   8 dot4add calls → int32 sum
//   acc += int_sum * d_a * d_b.x       (canonical Q8_0×Q8_1 dot — sum_d is unused
//                                       because Q8_0 has no min/offset)
//
// One workgroup per output row. 32 threads cooperate across the K dim via a shmem
// tree-reduce, mirroring mul_mat_vec's structure.

//@variants
//  q8_0_f32: DATA_A_Q8_0=1   D_TYPE=float    FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"

static const uint BLOCK_SIZE = 32;

struct MmvqParams {
    uint ncols;             // K dim, in elements
    uint stride_a;          // in elements
    uint stride_b;          // in elements (activation tensor)
    uint stride_d;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;
    uint fusion_flags;      // unused for now
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
};

[[vk::push_constant]] ConstantBuffer<MmvqParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<block_q8_0_packed16> data_a;
[[vk::binding(1, 0)]] StructuredBuffer<block_q8_1_x4>       data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE>            data_d;

groupshared float tmp[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid = gl_LocalInvocationID.x;
    const uint row = gl_WorkGroupID.x;

    // Batch + GQA broadcasting (mirrors mul_mat_vec.slang).
    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;
    const uint i13 = batch_idx / p.ne12;
    const uint i12 = batch_idx % p.ne12;
    const uint i03 = i13 / p.broadcast3;
    const uint i02 = i12 / p.broadcast2;
    const uint batch_idx_a = i03 * p.ne02 + i02;

    // A is indexed in blocks (QUANT_K_Q8_0 = 32 elements per block).
    // B is indexed in 4-block bundles (block_q8_1_x4 packs 4 blocks).
    const uint a_row_blocks   = p.stride_a / 32u;
    const uint a_base_blocks  = batch_idx_a * (p.batch_stride_a / 32u) + row * a_row_blocks;
    const uint b_base_x4      = batch_idx   * (p.batch_stride_b / 128u);

    float acc = 0.0f;
    const uint num_blocks = p.ncols / 32u;

    for (uint b = tid; b < num_blocks; b += BLOCK_SIZE) {
        const uint  a_blk = a_base_blocks + b;
        const uint  bx4   = b_base_x4 + b / 4u;
        const uint  binx4 = b % 4u;

        const float d_a = float(data_a[a_blk].d);
        const float d_b = float(data_b[bx4].ds[binx4].x);

        int idot = 0;
        [unroll] for (uint i = 0; i < 8; ++i) {
            // Combine two i16 quants from A into 4 packed i8s (uint32).
            const uint a_packed = (uint(data_a[a_blk].qs[i * 2u]) & 0xFFFFu)
                                | (uint(data_a[a_blk].qs[i * 2u + 1u]) << 16);
            // B is already packed (int32 per 4 i8s).
            const uint b_packed = uint(data_b[bx4].qs[binx4 * 8u + i]);
            idot = dot4add_i8packed(a_packed, b_packed, idot);
        }
        acc += float(idot) * d_a * d_b;
    }

    tmp[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2u; s > 0u; s >>= 1) {
        if (tid < s) { tmp[tid] += tmp[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        data_d[batch_idx * p.batch_stride_d + row] = D_TYPE(tmp[0]);
    }
}
