// Ported from llama.cpp ggml-vulkan, MIT license — see LICENSE.llama-cpp

#ifndef GENERIC_HEAD_SLANG
#define GENERIC_HEAD_SLANG

struct GenericParams {
    uint KX;
    uint KY;
    float param1;
    float param2;
    float param3;
    float param4;
};

[[vk::push_constant]] ConstantBuffer<GenericParams> p;

#endif
