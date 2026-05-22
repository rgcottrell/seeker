// Ported from llama.cpp ggml-vulkan dequant_head.glsl, MIT license — see LICENSE.llama-cpp

#ifndef DEQUANT_HEAD_SLANG
#define DEQUANT_HEAD_SLANG

struct DequantParams {
    uint M;
    uint K;
    uint stride_a;
    uint stride_b;
    uint nel;
};

[[vk::push_constant]] ConstantBuffer<DequantParams> p;

#endif
