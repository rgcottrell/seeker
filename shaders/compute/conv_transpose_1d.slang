// Ported from llama.cpp ggml-vulkan conv_transpose_1d.comp, MIT license — see LICENSE.llama-cpp
//
// Sliding-window 1D transposed conv. Each workgroup owns one Cout channel and streams
// over L_block windows, accumulating partial sums into a single 4096-entry groupshared
// buffer. Block-size 128 threads.

//@variants
//  f32: A_TYPE=float    B_TYPE=float    D_TYPE=float    FLOAT_TYPE=float
//@end-variants

#include "types.slang"

static const uint BS = 128;

struct ConvT1dParams {
    uint Cout;
    uint Cin;
    uint K;
    uint L;
    uint KL;

    uint nb01;
    uint nb02;
    uint nb11;
    uint nb1;

    int s0;
};

[[vk::push_constant]] ConstantBuffer<ConvT1dParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;   // kernel:  [K, Cout, Cin]
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;   // input:   [L, Cin]
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;   // dst:     [KL, Cout]

groupshared D_TYPE tmp[4096];

uint splitWork(uint workSize) { return (BS + workSize - 1) / BS; }

[shader("compute")]
[numthreads(BS, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint Cout_idx = gl_WorkGroupID.x;
    const uint tid      = gl_LocalInvocationID.x;
    const uint s0u      = uint(p.s0);
    const uint tmp_len  = BS * s0u + p.K;  // bs*s0 + K is one window past the last touched slot

    for (uint i = 0; i < splitWork(tmp_len); i++) {
        uint idx = i * BS + tid;
        if (idx < tmp_len) { tmp[idx] = D_TYPE(0); }
    }

    const uint L_blocks = splitWork(p.L);
    for (uint L_block_id = 0; L_block_id < L_blocks; L_block_id++) {
        if (L_block_id > 0) {
            GroupMemoryBarrierWithGroupSync();
            // Slide the live window: shift values down by bs*s0, zero the freshly opened tail.
            for (uint i = 0; i < splitWork(tmp_len); i++) {
                uint idx = i * BS + tid;
                if (idx >= BS * s0u && idx < tmp_len) {
                    tmp[idx - BS * s0u] = tmp[idx];
                    tmp[idx] = D_TYPE(0);
                } else if (idx >= p.K && idx < BS * s0u) {
                    tmp[idx] = D_TYPE(0);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        const uint L_idx = L_block_id * BS + tid;
        for (uint K_idx = 0; K_idx < p.K; K_idx++) {
            D_TYPE dp = D_TYPE(0);
            for (uint Cin_idx = 0; Cin_idx < p.Cin; Cin_idx++) {
                A_TYPE elemKrn = data_a[K_idx + Cout_idx * p.nb01 + Cin_idx * p.nb02];
                if (L_idx < p.L) {
                    B_TYPE elemInp = data_b[L_idx + Cin_idx * p.nb11];
                    dp = D_TYPE(mad(FLOAT_TYPE(elemKrn), FLOAT_TYPE(elemInp), FLOAT_TYPE(dp)));
                }
            }
            tmp[tid * s0u + K_idx] += dp;
            GroupMemoryBarrierWithGroupSync();
        }

        const uint KLb_idx = L_block_id * BS * s0u;
        if (L_block_id < L_blocks - 1) {
            for (uint s0_idx = 0; s0_idx < s0u; s0_idx++) {
                uint sh_idx = s0u * tid + s0_idx;
                uint KL_idx = KLb_idx + sh_idx;
                if (KL_idx < p.KL) {
                    data_d[KL_idx + Cout_idx * p.nb1] = tmp[sh_idx];
                }
            }
        }
    }

    for (uint i = 0; i < splitWork(tmp_len); i++) {
        uint idx    = i * BS + tid;
        uint KL_idx = (L_blocks - 1) * BS * s0u + idx;
        if (KL_idx < p.KL) {
            data_d[KL_idx + Cout_idx * p.nb1] = tmp[idx];
        }
    }
}
