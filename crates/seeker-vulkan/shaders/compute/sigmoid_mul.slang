// Fused `sigmoid(a) * b` element-wise op, dispatched via the same GLU
// framework as `swiglu.slang`. Used by qwen35moe's attention block to
// collapse the two `sigmoid(q_gate)` → `mul(attn_out, q_gate_sig)`
// dispatches into one.

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "glu_head.slang"

float op(float a, float b) {
    // sigmoid(a) * b = b / (1 + exp(-a))
    return b / (1.0f + exp(-a));
}

#include "glu_main.slang"
