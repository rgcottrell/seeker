// GeGLU: `gelu(a) * b`. The tanh-approx gelu matches `gelu.slang` (and ggml's
// `ggml_gelu` / `ggml_geglu_split`), so DiffusionGemma's combined-`ffn_gate_up`
// MoE experts (mode 0: gate = first half, up = second half of one tensor) and
// any split GeGLU reuse the shared `glu_main` body.

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "glu_head.slang"

float op(float a, float b) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const float val  = SQRT_2_OVER_PI * a * (1.0f + GELU_COEF_A * a * a);
    const float gelu = 0.5f * a * (2.0f - 2.0f / (exp(2.0f * val) + 1.0f));
    return gelu * b;
}

#include "glu_main.slang"
