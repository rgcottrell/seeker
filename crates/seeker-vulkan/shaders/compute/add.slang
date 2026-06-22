// Ported from llama.cpp ggml-vulkan add.comp, MIT license — see LICENSE.llama-cpp
//
// Phase 1: the ADD_RMS fusion branch (subgroup-reduce sum-of-squares for
// fused rms_norm) is omitted. Revisit when rms_norm.slang lands.

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   B_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"
#include "generic_binary_head.slang"

static const uint num_threads = 256;

[shader("compute")]
[numthreads(num_threads, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    // Each workgroup owns a UNIQUE 512-wide block: threads cover [base, base+256)
    // on iter 0 and [base+256, base+512) on iter 1. Non-overlapping, so each
    // output element is written exactly once — correct even when dst aliases a
    // source (in-place `residual += x`). Host dispatches ceil(ne/512) groups.
    uint idx = gl_WorkGroupID.x * 512 + gl_LocalInvocationID.x;

    // num_threads * num_iter must equal 512, to match the per-workgroup block.
    const uint num_iter = 2;

    [unroll] for (uint i = 0; i < num_iter; ++i) {
        if (idx >= p.ne) { continue; }
        uint i00, i01, i02, i03;
        get_indices(idx, i00, i01, i02, i03);

        FLOAT_TYPE sum =
            FLOAT_TYPE(data_a[get_aoffset() + src0_idx(i00, i01, i02, i03)]) +
            FLOAT_TYPE(data_b[get_boffset() + src1_idx(i00, i01, i02, i03)]);
        data_d[get_doffset() + dst_idx(i00, i01, i02, i03)] = D_TYPE(sum);

        idx += num_threads;
    }
}
