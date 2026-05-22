// Ported from llama.cpp ggml-vulkan mul_mat_vec_p021.comp, MIT license — see LICENSE.llama-cpp
//
// Permuted (transposed) mat-vec. gqa_ratio in [1,8] lets one A row pair with multiple B
// channels. The shmem-reduction variant works on any subgroup size; subgroup-add path is
// gated on USE_SUBGROUP_ADD.

//@variants
//  f32:          A_TYPE=float       B_TYPE=float       D_TYPE=float
//  f16:          A_TYPE=float16_t   B_TYPE=float       D_TYPE=float
//  f32_sg:       A_TYPE=float       B_TYPE=float       D_TYPE=float     USE_SUBGROUP_ADD=1
//  f16_sg:       A_TYPE=float16_t   B_TYPE=float       D_TYPE=float     USE_SUBGROUP_ADD=1
//@end-variants

#include "types.slang"

static const uint MAT_VEC_FUSION_FLAGS_BIAS0 = 0x1;
static const uint MAT_VEC_FUSION_FLAGS_BIAS1 = 0x2;

[SpecializationConstant] const int  BLOCK_SIZE = 32;
[SpecializationConstant] const uint gqa_ratio  = 1;

struct MulMatVecP021Params {
    uint ncols_x;
    uint nrows_x;
    uint nchannels_x;
    uint nchannels_y;
    uint b_offset;
    uint d_offset;
    uint fusion_flags;
};

[[vk::push_constant]] ConstantBuffer<MulMatVecP021Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;
[[vk::binding(3, 0)]] StructuredBuffer<D_TYPE>   data_fuse0;
[[vk::binding(4, 0)]] StructuredBuffer<D_TYPE>   data_fuse1;
[[vk::binding(5, 0)]] StructuredBuffer<float4>   data_a_v4;
[[vk::binding(6, 0)]] StructuredBuffer<float4>   data_b_v4;

#if !defined(USE_SUBGROUP_ADD)
groupshared float tmp[8][BLOCK_SIZE];
#endif

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID,
          uint3 gl_LocalInvocationID  : SV_GroupThreadID) {
    const uint tid   = gl_LocalInvocationID.x;
    const uint row_x = gl_GlobalInvocationID.y;

    uint channel, channel_x;
    if (gqa_ratio > 1) {
        channel_x = gl_GlobalInvocationID.z;
        channel   = channel_x * gqa_ratio;
    } else {
        channel   = gl_GlobalInvocationID.z;
        channel_x = channel / (p.nchannels_y / p.nchannels_x);
    }

    const uint nrows_y   = p.ncols_x;
    const uint nrows_dst = p.nrows_x;
    const uint row_dst   = row_x;

    float temp[8];
    [unroll] for (uint i = 0; i < 8; ++i) { temp[i] = 0.0f; }

    const bool is_aligned = (p.ncols_x % 4) == 0 && (p.nchannels_x % 4) == 0 && (nrows_y % 4) == 0;

    for (uint col_x0 = 0; col_x0 < p.ncols_x; col_x0 += BLOCK_SIZE) {
        if (col_x0 + 4 * BLOCK_SIZE <= p.ncols_x && is_aligned) {
            uint col_x = col_x0 + 4 * tid;
            const uint row_y = col_x;

            const uint ix = row_x * p.nchannels_x * p.ncols_x + channel_x * p.ncols_x + col_x;
            const float4 av4 = data_a_v4[ix / 4];

            [unroll] for (uint c = 0; c < gqa_ratio; ++c) {
                const uint iy = (channel + c) * nrows_y + row_y;
                const float4 bv4 = data_b_v4[iy / 4];
                temp[c] += dot(av4, bv4);
            }
            col_x0 += 3 * BLOCK_SIZE;
        } else {
            const uint col_x = col_x0 + tid;
            if (col_x >= p.ncols_x) { break; }

            const uint ix = row_x * p.nchannels_x * p.ncols_x + channel_x * p.ncols_x + col_x;
            const float xi = float(data_a[ix]);
            const uint row_y = col_x;

            [unroll] for (uint c = 0; c < gqa_ratio; ++c) {
                const uint iy = (channel + c) * nrows_y + row_y;
                temp[c] = mad(xi, float(data_b[iy]), temp[c]);
            }
        }
    }

#if defined(USE_SUBGROUP_ADD)
    {
        float4 t = float4(temp[0], temp[1], temp[2], temp[3]);
        t = WaveActiveSum(t);
        temp[0] = t[0]; temp[1] = t[1]; temp[2] = t[2]; temp[3] = t[3];
        if (gqa_ratio > 4) {
            t = float4(temp[4], temp[5], temp[6], temp[7]);
            t = WaveActiveSum(t);
            temp[4] = t[0]; temp[5] = t[1]; temp[6] = t[2]; temp[7] = t[3];
        }
    }
#else
    [unroll] for (uint c = 0; c < gqa_ratio; ++c) { tmp[c][tid] = temp[c]; }
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < uint(s)) {
            [unroll] for (uint c = 0; c < gqa_ratio; ++c) {
                temp[c] += tmp[c][tid + s];
                tmp[c][tid] = temp[c];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint c = 0; c < gqa_ratio; ++c) { temp[c] = tmp[c][tid]; }
#endif

    if (tid == 0) {
        [unroll] for (uint c = 0; c < gqa_ratio; ++c) {
            const uint idst = (channel + c) * nrows_dst + row_dst;
            float r = temp[c];
            if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) { r += float(data_fuse0[idst]); }
            if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) { r += float(data_fuse1[idst]); }
            data_d[idst] = D_TYPE(r);
        }
    }
}
