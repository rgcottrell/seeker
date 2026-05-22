// Ported from llama.cpp ggml-vulkan contig_copy.comp, MIT license — see LICENSE.llama-cpp
//
// num_threads * num_iter must equal 512 to match the wg_denoms and get_idx calculation.

//@variants
//  f32_f32:  A_TYPE=float        D_TYPE=float
//  f32_i32:  A_TYPE=float        D_TYPE=int
//  i32_f32:  A_TYPE=int          D_TYPE=float
//  f32_f16:  A_TYPE=float        D_TYPE=float16_t
//  f16_f16:  A_TYPE=float16_t    D_TYPE=float16_t
//  f16_f32:  A_TYPE=float16_t    D_TYPE=float
//  f32_bf16: A_TYPE=float        D_TYPE=uint16_t   DATA_D_BF16=1
//  bf16_f32: DATA_A_BF16=1       D_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_unary_head.slang"

static const uint num_threads = 128;
static const uint num_iter    = 4;

void copy_one(uint idx) {
#if defined(DATA_A_BF16)
    data_d[get_doffset() + idx] = D_TYPE(bf16_to_fp32(uint32_t(data_a[get_aoffset() + idx])));
#elif defined(DATA_D_BF16)
    float f = float(data_a[get_aoffset() + idx]);
    data_d[get_doffset() + idx] = D_TYPE(fp32_to_bf16(f));
#else
    data_d[get_doffset() + idx] = D_TYPE(data_a[get_aoffset() + idx]);
#endif
}

[shader("compute")]
[numthreads(num_threads, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = get_idx(gl_GlobalInvocationID);

    if (idx + (num_iter - 1) * num_threads < p.ne) {
        [unroll] for (uint i = 0; i < num_iter; ++i) {
            copy_one(idx);
            idx += num_threads;
        }
    } else {
        [unroll] for (uint i = 0; i < num_iter; ++i) {
            if (idx < p.ne) { copy_one(idx); }
            idx += num_threads;
        }
    }
}
