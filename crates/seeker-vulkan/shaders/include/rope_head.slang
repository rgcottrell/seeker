// Ported from llama.cpp ggml-vulkan rope_head.glsl, MIT license — see LICENSE.llama-cpp

#ifndef ROPE_HEAD_SLANG
#define ROPE_HEAD_SLANG

#include "types.slang"
#include "rope_params.slang"

#ifndef ROPE_D_TYPE
#define ROPE_D_TYPE float
#endif

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>      rope_data_a;
[[vk::binding(1, 0)]] StructuredBuffer<int>         rope_data_pos;
[[vk::binding(2, 0)]] StructuredBuffer<float>       rope_data_ff;
[[vk::binding(3, 0)]] RWStructuredBuffer<ROPE_D_TYPE> rope_data_d;
[[vk::binding(4, 0)]] StructuredBuffer<uint2>       rope_data_i;  // set_rows indices

[[vk::push_constant]] ConstantBuffer<rope_params> pc;

#endif
