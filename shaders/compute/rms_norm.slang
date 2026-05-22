// Ported from llama.cpp ggml-vulkan rms_norm.comp, MIT license — see LICENSE.llama-cpp
//
// Phase 2: the RMS_NORM_ROPE_FUSION branch (fused rope path through shared
// memory) is omitted; lands when rope.slang arrives.

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   B_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

import glsl;  // for gl_NumWorkGroups

#include "generic_binary_head.slang"
#include "types.slang"

static const uint BLOCK_SIZE = 512;

[SpecializationConstant] const bool do_multiply = false;

groupshared FLOAT_TYPE sumsh[BLOCK_SIZE];

void rms_norm(uint num_iters,
              uint3 gl_WorkGroupID,
              uint  tid) {
    const uint ncols     = p.ne00;
    const uint nrows     = gl_NumWorkGroups.x;
    const uint nchannels = gl_NumWorkGroups.y;

    const uint row     = gl_WorkGroupID.x;
    const uint channel = gl_WorkGroupID.y;
    const uint samp    = gl_WorkGroupID.z;

    const uint stride_row     = p.nb01;
    const uint stride_channel = p.nb02;
    const uint stride_sample  = p.nb03;

    uint a_offset = samp * stride_sample + channel * stride_channel + row * stride_row + get_aoffset();
    uint b_offset = src1_idx(0, row, channel, samp) + get_boffset();
    uint d_offset = ((samp * nchannels + channel) * nrows + row) * ncols + get_doffset();

    FLOAT_TYPE sum = FLOAT_TYPE(0.0f);

    [unroll] for (uint col = tid, idx = 0; idx < num_iters; col += BLOCK_SIZE, ++idx) {
        FLOAT_TYPE xi = FLOAT_TYPE(0);
        if (col < ncols) {
            xi = FLOAT_TYPE(data_a[a_offset + col]);
        }
        sum += xi * xi;
    }

    sumsh[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (tid < uint(s)) {
            sum += sumsh[tid + s];
            sumsh[tid] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    sum = sumsh[0];

    const FLOAT_TYPE mean  = sum / FLOAT_TYPE(ncols);
    const FLOAT_TYPE scale = rsqrt(mean + FLOAT_TYPE(p.param1));

    if (do_multiply) {
        if (ncols > p.ne10) {
            [unroll] for (uint col = tid, idx = 0; idx < num_iters; col += BLOCK_SIZE, ++idx) {
                if (col >= ncols) { continue; }
                data_d[d_offset + col] = D_TYPE(
                    scale * FLOAT_TYPE(data_a[a_offset + col])
                          * FLOAT_TYPE(data_b[b_offset + fastmod(col, p.ne10)])
                );
            }
        } else {
            [unroll] for (uint col = tid, idx = 0; idx < num_iters; col += BLOCK_SIZE, ++idx) {
                if (col >= ncols) { continue; }
                data_d[d_offset + col] = D_TYPE(
                    scale * FLOAT_TYPE(data_a[a_offset + col])
                          * FLOAT_TYPE(data_b[b_offset + col])
                );
            }
        }
    } else {
        [unroll] for (uint col = tid, idx = 0; idx < num_iters; col += BLOCK_SIZE, ++idx) {
            if (col >= ncols) { continue; }
            data_d[d_offset + col] = D_TYPE(scale * FLOAT_TYPE(data_a[a_offset + col]));
        }
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID    : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    // Instantiate rms_norm at several call-site constants so the inner loops
    // can unroll — the GLSL source does the same.
    const uint tid = gl_LocalInvocationID.x;
    const uint num_blocks = (p.ne00 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if      (num_blocks > 32) { rms_norm(num_blocks, gl_WorkGroupID, tid); }
    else if (num_blocks > 16) { rms_norm(32,         gl_WorkGroupID, tid); }
    else if (num_blocks > 12) { rms_norm(16,         gl_WorkGroupID, tid); }
    else if (num_blocks > 10) { rms_norm(12,         gl_WorkGroupID, tid); }
    else if (num_blocks > 8)  { rms_norm(10,         gl_WorkGroupID, tid); }
    else if (num_blocks > 4)  { rms_norm(8,          gl_WorkGroupID, tid); }
    else if (num_blocks == 4) { rms_norm(4,          gl_WorkGroupID, tid); }
    else if (num_blocks == 3) { rms_norm(3,          gl_WorkGroupID, tid); }
    else if (num_blocks == 2) { rms_norm(2,          gl_WorkGroupID, tid); }
    else if (num_blocks == 1) { rms_norm(1,          gl_WorkGroupID, tid); }
}
