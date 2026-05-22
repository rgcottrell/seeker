// Ported from llama.cpp ggml-vulkan mul_mat_vec_nc.comp, MIT license — see LICENSE.llama-cpp
//
// Non-contiguous mat-vec: row/channel/batch all have arbitrary strides. Each workgroup
// processes one (i3, channel, row) triple and reduces a partial dot product via shmem.

//@variants
//  f32:  A_TYPE=float       B_TYPE=float       D_TYPE=float
//  f16:  A_TYPE=float16_t   B_TYPE=float       D_TYPE=float
//@end-variants

#include "types.slang"

static const uint BLOCK_SIZE                  = 32;
static const uint MAT_VEC_FUSION_FLAGS_BIAS0  = 0x1;
static const uint MAT_VEC_FUSION_FLAGS_BIAS1  = 0x2;

struct MulMatVecNcParams {
    uint ncols_x;
    uint nrows_x;
    uint row_stride_x;
    uint channel_stride_x;
    uint channel_stride_y;
    uint channel_x_divisor;
    uint ne12;
    uint b_offset;
    uint d_offset;
    uint nb03;
    uint nb13;
    uint nb23;
    uint fusion_flags;
};

[[vk::push_constant]] ConstantBuffer<MulMatVecNcParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;
[[vk::binding(3, 0)]] StructuredBuffer<D_TYPE>   data_fuse0;  // bias 0 (host wires same buffer if unused)
[[vk::binding(4, 0)]] StructuredBuffer<D_TYPE>   data_fuse1;  // bias 1
[[vk::binding(5, 0)]] StructuredBuffer<float4>   data_a_v4;   // vec4 alias of data_a
[[vk::binding(6, 0)]] StructuredBuffer<float4>   data_b_v4;   // vec4 alias of data_b

groupshared float tmp[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID        : SV_GroupID,
          uint3 gl_GlobalInvocationID : SV_DispatchThreadID,
          uint3 gl_LocalInvocationID  : SV_GroupThreadID) {
    const uint tid       = gl_LocalInvocationID.x;
    const uint row_x     = gl_GlobalInvocationID.y;
    const uint channel   = gl_GlobalInvocationID.z;
    const uint i3        = gl_WorkGroupID.x;
    const uint channel_x = channel / p.channel_x_divisor;
    const uint channel_y = channel % p.ne12;

    const uint nrows_y   = p.ncols_x;
    const uint nrows_dst = p.nrows_x;
    const uint row_dst   = row_x;

    const uint idst = i3 * p.nb23 + channel * nrows_dst + row_dst;

    float temp = 0.0f;

    const bool is_aligned = (p.ncols_x % 4) == 0 && (p.row_stride_x % 4) == 0 && (p.channel_stride_x % 4) == 0;

    for (uint col_x0 = 0; col_x0 < p.ncols_x;) {
        const uint unroll_count = 2;
        if (col_x0 + unroll_count * 4 * BLOCK_SIZE <= p.ncols_x && is_aligned) {
            [unroll] for (uint i = 0; i < unroll_count; ++i) {
                const uint col_x = col_x0 + 4 * tid;
                const uint row_y = col_x;
                const uint ix = i3 * p.nb03 + channel_x * p.channel_stride_x + row_x * p.row_stride_x + col_x;
                const uint iy = i3 * p.nb13 + channel_y * p.channel_stride_y + row_y;

                const float4 av4 = data_a_v4[ix / 4];
                const float4 bv4 = data_b_v4[iy / 4];
                temp += dot(av4, bv4);
                col_x0 += 4 * BLOCK_SIZE;
            }
        } else if (col_x0 + 4 * BLOCK_SIZE <= p.ncols_x && is_aligned) {
            const uint col_x = col_x0 + 4 * tid;
            const uint row_y = col_x;
            const uint ix = i3 * p.nb03 + channel_x * p.channel_stride_x + row_x * p.row_stride_x + col_x;
            const uint iy = i3 * p.nb13 + channel_y * p.channel_stride_y + row_y;

            const float4 av4 = data_a_v4[ix / 4];
            const float4 bv4 = data_b_v4[iy / 4];
            temp += dot(av4, bv4);
            col_x0 += 4 * BLOCK_SIZE;
        } else {
            const uint col_x = col_x0 + tid;
            if (col_x >= p.ncols_x) { break; }

            const uint row_y = col_x;
            const uint ix = i3 * p.nb03 + channel_x * p.channel_stride_x + row_x * p.row_stride_x + col_x;
            const uint iy = i3 * p.nb13 + channel_y * p.channel_stride_y + row_y;

            const float xi = float(data_a[ix]);
            temp = mad(xi, float(data_b[iy]), temp);
            col_x0 += BLOCK_SIZE;
        }
    }

    tmp[tid] = temp;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (tid < uint(s)) { tmp[tid] += tmp[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float r = tmp[0];
        if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) { r += float(data_fuse0[idst]); }
        if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) { r += float(data_fuse1[idst]); }
        data_d[idst] = D_TYPE(r);
    }
}
