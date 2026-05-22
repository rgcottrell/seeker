// Ported from llama.cpp ggml-vulkan acc.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_binary_head.slang"

// false → SET (overwrite with src1), true → ACC (accumulate src1 into src0).
[SpecializationConstant] const bool ACC = true;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= p.ne) { return; }

    const uint offset = uint(p.param3);
    const uint src1_i = idx - offset;
    const uint i3   = src1_i / p.nb03;
    const uint rem2 = src1_i - i3 * p.nb03;
    const uint i2   = rem2 / p.nb02;
    const uint rem1 = rem2 - i2 * p.nb02;
    const uint i1   = rem1 / p.nb01;
    const uint i0   = rem1 % p.nb01;

    if (i0 < p.ne10 && i1 < p.ne11 && i2 < p.ne12 && i3 < p.ne13) {
        if (ACC) {
            data_d[get_doffset() + idx] = D_TYPE(
                FLOAT_TYPE(data_a[get_aoffset() + idx]) +
                FLOAT_TYPE(data_b[get_boffset() + src1_idx(i0, i1, i2, i3)])
            );
        } else {
            data_d[get_doffset() + idx] = D_TYPE(FLOAT_TYPE(data_b[get_boffset() + src1_idx(i0, i1, i2, i3)]));
        }
    } else {
        data_d[get_doffset() + idx] = D_TYPE(FLOAT_TYPE(data_a[get_aoffset() + idx]));
    }
}
