// Ported from llama.cpp ggml-vulkan concat.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   B_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_binary_head.slang"

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
    const int dim = p.param3;

    if (idx >= p.ne) { return; }

    const uint i3 = idx / (p.ne22 * p.ne21 * p.ne20);
    const uint i3_offset = i3 * p.ne22 * p.ne21 * p.ne20;
    const uint i2 = (idx - i3_offset) / (p.ne21 * p.ne20);
    const uint i2_offset = i2 * p.ne21 * p.ne20;
    const uint i1 = (idx - i3_offset - i2_offset) / p.ne20;
    const uint i0 = idx - i3_offset - i2_offset - i1 * p.ne20;

    uint o[4] = { 0, 0, 0, 0 };
    o[dim] = dim == 0 ? p.ne00 : (dim == 1 ? p.ne01 : (dim == 2 ? p.ne02 : p.ne03));

    const uint s0_idx = i3 * p.nb03 + i2 * p.nb02 + i1 * p.nb01 + i0 * p.nb00;
    const uint s1_idx = (i3 - o[3]) * p.nb13 + (i2 - o[2]) * p.nb12 + (i1 - o[1]) * p.nb11 + (i0 - o[0]) * p.nb10;
    const uint d_idx  = i3 * p.nb23 + i2 * p.nb22 + i1 * p.nb21 + i0 * p.nb20;

    const bool is_src0 = i0 < p.ne00 && i1 < p.ne01 && i2 < p.ne02 && i3 < p.ne03;

    data_d[get_doffset() + d_idx] = D_TYPE(
        is_src0 ? data_a[get_aoffset() + s0_idx]
                : data_b[get_boffset() + s1_idx]
    );
}
