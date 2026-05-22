// Ported from llama.cpp ggml-vulkan soft_max_large_common.glsl, MIT license — see LICENSE.llama-cpp

#ifndef SOFT_MAX_LARGE_HEAD_SLANG
#define SOFT_MAX_LARGE_HEAD_SLANG

#include "types.slang"

struct SoftMaxLargeParams {
    uint KX;
    uint KY;
    uint ne00;
    uint ne01;
    uint ne02;
    uint ne12;
    uint ne13;
    uint nb11;
    uint nb12;
    uint nb13;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint n_head_log2;
    uint nrows_x;
    uint has_sinks;
};

[[vk::push_constant]] ConstantBuffer<SoftMaxLargeParams> p;

[SpecializationConstant] const uint BLOCK_SIZE = 128;
[SpecializationConstant] const uint num_iters  = 4;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
[[vk::binding(2, 0)]] StructuredBuffer<float>    data_c;
[[vk::binding(3, 0)]] RWStructuredBuffer<D_TYPE> data_d;
[[vk::binding(4, 0)]] RWStructuredBuffer<float>  data_m;
[[vk::binding(5, 0)]] RWStructuredBuffer<float>  data_s;

groupshared FLOAT_TYPE vals[BLOCK_SIZE];

float get_slope(uint rowx) {
    float slope = 1.0f;

    if (p.max_bias > 0.0f) {
        const uint h = (rowx / p.ne01) % p.ne02;  // head index
        const float base = h < p.n_head_log2 ? p.m0 : p.m1;
        const uint  expn = h < p.n_head_log2 ? h + 1u : 2u * (h - p.n_head_log2) + 1u;
        slope = pow(base, float(expn));
    }
    return slope;
}

#endif
