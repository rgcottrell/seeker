// Ported from llama.cpp ggml-vulkan div.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   B_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_binary_head.slang"

static const uint num_threads = 256;

[shader("compute")]
[numthreads(num_threads, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = get_idx(gl_GlobalInvocationID);
    const uint num_iter = 2;

    [unroll] for (uint i = 0; i < num_iter; ++i) {
        if (idx >= p.ne) { continue; }
        uint i00, i01, i02, i03;
        get_indices(idx, i00, i01, i02, i03);

        data_d[get_doffset() + dst_idx(i00, i01, i02, i03)] = D_TYPE(
            FLOAT_TYPE(data_a[get_aoffset() + src0_idx(i00, i01, i02, i03)]) /
            FLOAT_TYPE(data_b[get_boffset() + src1_idx(i00, i01, i02, i03)])
        );

        idx += num_threads;
    }
}
