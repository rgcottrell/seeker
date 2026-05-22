// Ported from llama.cpp ggml-vulkan geglu.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "glu_head.slang"

static const float GELU_COEF_A    = 0.044715f;
static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

float op(float a, float b) {
    const float val = SQRT_2_OVER_PI * a * (1.0f + GELU_COEF_A * a * a);
    return 0.5f * a * (2.0f - 2.0f / (exp(2.0f * val) + 1.0f)) * b;
}

#include "glu_main.slang"
