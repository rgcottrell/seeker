// Ported from llama.cpp ggml-vulkan reglu.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "glu_head.slang"

float op(float a, float b) {
    return max(a, 0.0f) * b;
}

#include "glu_main.slang"
