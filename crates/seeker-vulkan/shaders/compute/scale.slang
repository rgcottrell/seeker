// Ported from llama.cpp ggml-vulkan scale.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

static const uint num_threads = 128;

[shader("compute")]
[numthreads(num_threads, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = get_idx(gl_GlobalInvocationID);

    // num_threads * num_iter must equal 512, to match the wg_denoms and get_idx calculation.
    const uint num_iter = 4;

    [unroll] for (uint i = 0; i < num_iter; ++i) {
        if (idx >= p.ne) { continue; }

        data_d[get_doffset() + idx] = D_TYPE(
            FLOAT_TYPE(data_a[get_aoffset() + idx]) * FLOAT_TYPE(p.param1) + FLOAT_TYPE(p.param2)
        );
        idx += num_threads;
    }
}
