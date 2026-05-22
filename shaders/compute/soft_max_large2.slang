// Ported from llama.cpp ggml-vulkan soft_max_large2.comp, MIT license — see LICENSE.llama-cpp
//
// Pass 2: re-scan with the row max (gathered from data_m partials), emit exp values into
// data_d, and write per-block sums to data_s.

//@variants
//  f32:     A_TYPE=float    B_TYPE=float        D_TYPE=float    FLOAT_TYPE=float
//  f16:     A_TYPE=float    B_TYPE=float16_t    D_TYPE=float    FLOAT_TYPE=float
//@end-variants

import glsl;

#include "soft_max_large_head.slang"

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID       : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid      = gl_LocalInvocationID.x;
    const uint rowx     = gl_WorkGroupID.y;
    const uint wg_start = gl_WorkGroupID.x * BLOCK_SIZE * num_iters;

    const uint i03 = rowx / (p.ne01 * p.ne02);
    const uint i02 = (rowx - i03 * p.ne01 * p.ne02) / p.ne01;
    const uint i01 = rowx % p.ne01;

    uint rowy_start = 0;
    if (p.KY > 0) {
        rowy_start = i01 * p.nb11 + (i02 % p.ne12) * p.nb12 + (i03 % p.ne13) * p.nb13;
    }

    if (rowx >= p.nrows_x) { return; }

    const float slope = get_slope(rowx);

    FLOAT_TYPE max_val = p.has_sinks == 0 ? FLOAT_TYPE(asfloat(0xFF800000u)) : FLOAT_TYPE(data_c[i02]);

    [unroll] for (uint i = 0; i < gl_NumWorkGroups.x; i += BLOCK_SIZE) {
        if (i + tid < gl_NumWorkGroups.x) {
            max_val = max(max_val, FLOAT_TYPE(data_m[rowx * gl_NumWorkGroups.x + i + tid]));
        }
    }

    vals[tid] = max_val;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) { vals[tid] = max(max_val, vals[tid + s]); }
        GroupMemoryBarrierWithGroupSync();
    }

    max_val = vals[0];
    GroupMemoryBarrierWithGroupSync();

    FLOAT_TYPE sum = FLOAT_TYPE(0);

    [unroll] for (uint col0 = wg_start, idx = 0; idx < num_iters; col0 += BLOCK_SIZE, ++idx) {
        const uint col = col0 + tid;
        if (col >= p.KX) { break; }

        const uint i = rowx * p.KX + col;
        const FLOAT_TYPE val = exp(
            FLOAT_TYPE(data_a[i]) * FLOAT_TYPE(p.scale)
            + (p.KY > 0 ? FLOAT_TYPE(slope) * FLOAT_TYPE(data_b[rowy_start + col]) : FLOAT_TYPE(0))
            - max_val
        );
        sum += val;
        data_d[i] = D_TYPE(val);
    }

    vals[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) { vals[tid] += vals[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        data_s[rowx * gl_NumWorkGroups.x + gl_WorkGroupID.x] = float(vals[0]);
    }
}
