// Ported from llama.cpp ggml-vulkan glu_head.glsl, MIT license — see LICENSE.llama-cpp

#ifndef GLU_HEAD_SLANG
#define GLU_HEAD_SLANG

#include "types.slang"

struct GluParams {
    uint  N;
    uint  ne00;
    uint  ne20;
    uint  mode;     // 0 = default (a, a+offset), 1 = swapped, 2 = split (a, b)
    float alpha;
    float limit;
    uint  nb01;
    uint  nb02;
    uint  nb03;
    uint  ne01;
    uint  ne02;
    uint  nb11;
    uint  nb12;
    uint  nb13;
    uint  ne11;
    uint  ne12;
};

[[vk::push_constant]] ConstantBuffer<GluParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] StructuredBuffer<A_TYPE>   data_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

#endif
