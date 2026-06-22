// Ported from llama.cpp ggml-vulkan rope_multi.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: DATA_A_F32=1   D_TYPE=float       FLOAT_TYPE=float ROPE_D_TYPE=float
//  f16: DATA_A_F16=1   D_TYPE=float16_t   FLOAT_TYPE=float ROPE_D_TYPE=float16_t
//@end-variants

#include "rope_funcs.slang"

[shader("compute")]
[numthreads(1, 256, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i0  = 2 * gl_GlobalInvocationID.y;
    const uint row = gl_GlobalInvocationID.x + 32768u * gl_GlobalInvocationID.z;
    if (row >= pc.nrows) { return; }
    const uint i3 = row / (pc.ne01 * pc.ne02);
    const uint i2 = (row - i3 * pc.ne01 * pc.ne02) / pc.ne01;
    const uint i1 = row - i3 * pc.ne01 * pc.ne02 - i2 * pc.ne01;

    rope_multi(i0, i1, i2, i3, pc);
}
