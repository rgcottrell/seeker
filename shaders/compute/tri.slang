// Ported from llama.cpp ggml-vulkan tri.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

static const int GGML_TRI_TYPE_UPPER_DIAG = 0;
static const int GGML_TRI_TYPE_UPPER      = 1;
static const int GGML_TRI_TYPE_LOWER_DIAG = 2;
static const int GGML_TRI_TYPE_LOWER      = 3;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = get_idx(gl_GlobalInvocationID);
    if (idx >= p.ne) { return; }

    const uint i03        = fastdiv(idx, p.ne0_012mp, p.ne0_012L);
    const uint i03_offset = i03 * p.ne02 * p.ne01 * p.ne00;
    const uint i02        = fastdiv(idx - i03_offset, p.ne0_01mp, p.ne0_01L);
    const uint i02_offset = i02 * p.ne01 * p.ne00;
    const uint i01        = fastdiv(idx - i03_offset - i02_offset, p.ne0_0mp, p.ne0_0L);
    const uint i00        = idx - i03_offset - i02_offset - i01 * p.ne00;

    const int param = asint(p.param1);
    bool pass = false;
    switch (param) {
        case GGML_TRI_TYPE_UPPER_DIAG: pass = i00 >= i01; break;
        case GGML_TRI_TYPE_UPPER:      pass = i00 >  i01; break;
        case GGML_TRI_TYPE_LOWER_DIAG: pass = i00 <= i01; break;
        case GGML_TRI_TYPE_LOWER:      pass = i00 <  i01; break;
    }

    if (pass) {
        const float val = float(data_a[get_aoffset() + src0_idx(idx)]);
        data_d[get_doffset() + dst_idx(idx)] = D_TYPE(val);
    } else {
        data_d[get_doffset() + dst_idx(idx)] = D_TYPE(0);
    }
}
