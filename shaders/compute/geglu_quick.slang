// Ported from llama.cpp ggml-vulkan geglu_quick.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "glu_head.slang"

static const float GELU_QUICK_COEF = -1.702f;

float op(float a, float b) {
    return a * (1.0f / (1.0f + exp(GELU_QUICK_COEF * a))) * b;
}

#include "glu_main.slang"
