// Ported from llama.cpp ggml-vulkan swiglu_oai.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "glu_head.slang"

float op(float a, float b) {
    const float xi = min(a, p.limit);
    const float gi = max(min(b, p.limit), -p.limit);

    float out_glu = xi / (1.0f + exp(-xi * p.alpha));
    out_glu = out_glu * (1.0f + gi);
    return out_glu;
}

#include "glu_main.slang"
