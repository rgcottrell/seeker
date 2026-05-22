// Ported from llama.cpp ggml-vulkan rms_norm_partials.comp, MIT license — see LICENSE.llama-cpp
//
// Second pass of split-K rms_norm: sums per-block partials into the channel's running sum,
// then scales each element. Optionally multiplies the scaled output by src1 (do_multiply).

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_SubgroupInvocationID, gl_SubgroupSize, gl_NumWorkGroups

#include "generic_binary_head.slang"
#include "types.slang"

static const uint BLOCK_SIZE = 128;

[SpecializationConstant] const bool do_multiply = false;

[[vk::binding(3, 0)]] StructuredBuffer<float> partial_sums;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID         : SV_GroupID,
          uint3 gl_GlobalInvocationID  : SV_DispatchThreadID) {
    const uint ncols     = p.ne00;
    const uint nrows     = gl_NumWorkGroups.x;
    const uint nchannels = gl_NumWorkGroups.y;

    const uint row     = 0;
    const uint channel = gl_WorkGroupID.y;
    const uint samp    = gl_WorkGroupID.z;
    const uint tid     = gl_GlobalInvocationID.x;

    const uint stride_row     = p.nb01;
    const uint stride_channel = p.nb02;
    const uint stride_sample  = p.nb03;

    const uint a_offset = samp * stride_sample + channel * stride_channel + row * stride_row + get_aoffset();
    const uint b_offset = src1_idx(0, row, channel, samp) + get_boffset();
    const uint d_offset = ((samp * nchannels + channel) * nrows + row) * ncols + get_doffset();

    FLOAT_TYPE sum = FLOAT_TYPE(0);
    const uint num_partials = uint(p.param3);
    for (uint i = gl_SubgroupInvocationID; i < num_partials; i += gl_SubgroupSize) {
        sum += FLOAT_TYPE(partial_sums[i]);
    }
    sum = WaveActiveSum(sum);

    const uint col = tid;
    if (col >= ncols) { return; }

    const FLOAT_TYPE mean  = sum / FLOAT_TYPE(ncols);
    const FLOAT_TYPE scale = rsqrt(mean + FLOAT_TYPE(p.param1));

    if (do_multiply) {
        if (ncols > p.ne10) {
            data_d[d_offset + col] = D_TYPE(scale * FLOAT_TYPE(data_a[a_offset + col]) * FLOAT_TYPE(data_b[b_offset + fastmod(col, p.ne10)]));
        } else {
            data_d[d_offset + col] = D_TYPE(scale * FLOAT_TYPE(data_a[a_offset + col]) * FLOAT_TYPE(data_b[b_offset + col]));
        }
    } else {
        data_d[d_offset + col] = D_TYPE(scale * FLOAT_TYPE(data_a[a_offset + col]));
    }
}
