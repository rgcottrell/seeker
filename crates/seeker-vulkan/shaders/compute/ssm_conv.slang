// Ported from llama.cpp ggml-vulkan ssm_conv.comp, MIT license — see LICENSE.llama-cpp
//
// Per-channel 1D convolution used by state-space models. Source data is f32 throughout.

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "types.slang"

[SpecializationConstant] const uint BLOCK_SIZE    = 32;
[SpecializationConstant] const uint TOKENS_PER_WG = 16;
// 1 → apply `silu(x) = x / (1 + exp(-x))` to the output in place,
// saving the trailing element-wise dispatch in qwen35moe's SSM block.
// 0 → keep the unfused contract (smoke test / future callers).
[SpecializationConstant] const uint FUSE_SILU      = 0;

[[vk::binding(0, 0)]] StructuredBuffer<float>    src0;
[[vk::binding(1, 0)]] StructuredBuffer<float>    src1;
[[vk::binding(2, 0)]] RWStructuredBuffer<float>  dst;

struct SsmConvParams {
    uint nb01; uint nb02;
    uint nb11;
    uint dst_nb0; uint dst_nb1; uint dst_nb2;
    uint nc; uint ncs; uint nr; uint n_t; uint n_s;
};

[[vk::push_constant]] ConstantBuffer<SsmConvParams> p;

[shader("compute")]
[numthreads(BLOCK_SIZE, TOKENS_PER_WG, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID,
          uint3 gl_WorkGroupID        : SV_GroupID,
          uint3 gl_LocalInvocationID  : SV_GroupThreadID) {
    const uint i1 = gl_GlobalInvocationID.x;
    const uint i2 = gl_WorkGroupID.y * TOKENS_PER_WG + gl_LocalInvocationID.y;
    const uint i3 = gl_WorkGroupID.z;

    if (i1 >= p.nr || i2 >= p.n_t || i3 >= p.n_s) { return; }

    const uint src0_base = i3 * (p.nb02 / 4u) + i2 + i1 * (p.nb01 / 4u);
    const uint src1_base = i1 * (p.nb11 / 4u);

    float sum = 0.0f;
    if (p.nc == 4) {
        sum = dot(
            float4(src0[src0_base], src0[src0_base + 1], src0[src0_base + 2], src0[src0_base + 3]),
            float4(src1[src1_base], src1[src1_base + 1], src1[src1_base + 2], src1[src1_base + 3])
        );
    } else {
        [unroll] for (uint i0 = 0; i0 < p.nc; i0++) {
            sum += src0[src0_base + i0] * src1[src1_base + i0];
        }
    }

    const uint dst_idx = i3 * (p.dst_nb2 / 4u) + i2 * (p.dst_nb1 / 4u) + i1;
    if (FUSE_SILU != 0) {
        dst[dst_idx] = sum / (1.0f + exp(-sum));
    } else {
        dst[dst_idx] = sum;
    }
}
