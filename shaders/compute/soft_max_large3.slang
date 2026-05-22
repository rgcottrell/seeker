// Ported from llama.cpp ggml-vulkan soft_max_large3.comp, MIT license — see LICENSE.llama-cpp
//
// Pass 3: reduce per-block (max, sum) into a single row scale and divide data_d in place.

//@variants
//  f32:     A_TYPE=float    B_TYPE=float        D_TYPE=float    FLOAT_TYPE=float
//  f16:     A_TYPE=float    B_TYPE=float16_t    D_TYPE=float    FLOAT_TYPE=float
//@end-variants

import glsl;

#include "soft_max_large_head.slang"

groupshared FLOAT_TYPE sumsh[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID       : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid      = gl_LocalInvocationID.x;
    const uint rowx     = gl_WorkGroupID.y;
    const uint wg_start = gl_WorkGroupID.x * BLOCK_SIZE * num_iters;

    const uint i03 = rowx / (p.ne01 * p.ne02);
    const uint i02 = (rowx - i03 * p.ne01 * p.ne02) / p.ne01;

    if (rowx >= p.nrows_x) { return; }

    FLOAT_TYPE max_val = p.has_sinks == 0 ? FLOAT_TYPE(asfloat(0xFF800000u)) : FLOAT_TYPE(data_c[i02]);
    FLOAT_TYPE sum     = FLOAT_TYPE(0);

    [unroll] for (uint i = 0; i < gl_NumWorkGroups.x; i += BLOCK_SIZE) {
        if (i + tid < gl_NumWorkGroups.x) {
            max_val = max(max_val, FLOAT_TYPE(data_m[rowx * gl_NumWorkGroups.x + i + tid]));
            sum    += FLOAT_TYPE(data_s[rowx * gl_NumWorkGroups.x + i + tid]);
        }
    }

    vals[tid]  = max_val;
    sumsh[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            vals[tid]  = max(max_val, vals[tid + s]);
            sumsh[tid] += sumsh[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    max_val = vals[0];
    sum     = sumsh[0];

    if (p.has_sinks != 0) {
        sum += FLOAT_TYPE(exp(FLOAT_TYPE(data_c[i02]) - max_val));
    }

    const FLOAT_TYPE rcpdivisor = FLOAT_TYPE(1) / sum;

    [unroll] for (uint col0 = wg_start, idx = 0; idx < num_iters; col0 += BLOCK_SIZE, ++idx) {
        const uint col = col0 + tid;
        if (col >= p.KX) { continue; }
        data_d[rowx * p.KX + col] = D_TYPE(FLOAT_TYPE(data_d[rowx * p.KX + col]) * rcpdivisor);
    }
}
