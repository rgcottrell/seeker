// Ported from llama.cpp ggml-vulkan geglu_erf.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "glu_head.slang"

// Abramowitz & Stegun 7.1.26 / Hastings approximation of erf.
static const float p_erf  =  0.3275911f;
static const float a1_erf =  0.254829592f;
static const float a2_erf = -0.284496736f;
static const float a3_erf =  1.421413741f;
static const float a4_erf = -1.453152027f;
static const float a5_erf =  1.061405429f;
static const float SQRT_2_INV = 0.70710678118654752440084436210484f;

float op(float a, float b) {
    const float a_div_sqr2 = a * SQRT_2_INV;
    const float sign_x = sign(a_div_sqr2);
    const float x = abs(a_div_sqr2);
    const float t = 1.0f / (1.0f + p_erf * x);
    const float y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    const float erf_approx = sign_x * y;
    return 0.5f * a * (1.0f + erf_approx) * b;
}

#include "glu_main.slang"
