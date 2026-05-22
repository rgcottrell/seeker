// Ported from llama.cpp ggml-vulkan soft_max.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       B_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   B_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"

struct SoftMaxParams {
    uint KX;
    uint KY;
    uint ne00;
    uint ne01;
    uint ne02;
    uint ne12;
    uint ne13;
    uint nb11;
    uint nb12;
    uint nb13;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint n_head_log2;
    uint nrows_x;
    uint has_sinks;
};

[[vk::push_constant]] ConstantBuffer<SoftMaxParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE> data_b;
[[vk::binding(2, 0)]] StructuredBuffer<float>  data_c;
[[vk::binding(3, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[SpecializationConstant] const uint BLOCK_SIZE = 32;
static const uint DATA_CACHE_SIZE = 16;

groupshared FLOAT_TYPE vals[BLOCK_SIZE];

void soft_max(uint num_iters, uint3 gl_WorkGroupID, uint tid) {
    const uint rowx = gl_WorkGroupID.z * 262144 + gl_WorkGroupID.y * 512 + gl_WorkGroupID.x;

    const uint i03 = rowx / (p.ne01 * p.ne02);
    const uint i02 = (rowx - i03 * p.ne01 * p.ne02) / p.ne01;
    const uint i01 = rowx % p.ne01;

    uint rowy_start = 0;
    if (p.KY > 0) {
        rowy_start = i01 * p.nb11 + (i02 % p.ne12) * p.nb12 + (i03 % p.ne13) * p.nb13;
    }

    if (rowx >= p.nrows_x) { return; }

    float slope = 1.0f;
    if (p.max_bias > 0.0f) {
        const uint h = (rowx / p.ne01) % p.ne02;
        const float base = h < p.n_head_log2 ? p.m0 : p.m1;
        const uint   exp_ = h < p.n_head_log2 ? h + 1 : 2 * (h - p.n_head_log2) + 1;
        slope = pow(base, float(exp_));
    }

    // Find max. -inf as the seed when no sinks; data_c[i02] otherwise.
    FLOAT_TYPE max_val = p.has_sinks == 0 ? asfloat(0xFF800000) : data_c[i02];

    FLOAT_TYPE data_cache[DATA_CACHE_SIZE];

    [unroll] for (uint col0 = 0, idx = 0; idx < num_iters; col0 += BLOCK_SIZE, ++idx) {
        const uint col = col0 + tid;

        FLOAT_TYPE a = FLOAT_TYPE(0);
        if (col < p.KX) { a = data_a[rowx * p.KX + col]; }

        FLOAT_TYPE b = FLOAT_TYPE(0);
        if (p.KY > 0 && col < p.KX) { b = data_b[rowy_start + col]; }

        FLOAT_TYPE v = a * p.scale + slope * b;

        if (col < p.KX) { max_val = max(max_val, v); }
        if (idx < DATA_CACHE_SIZE) { data_cache[idx] = v; }
    }

    // Reduce max across workgroup.
    vals[tid] = max_val;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) { vals[tid] = max(vals[tid], vals[tid + s]); }
        GroupMemoryBarrierWithGroupSync();
    }
    max_val = vals[0];
    GroupMemoryBarrierWithGroupSync();

    FLOAT_TYPE sum = FLOAT_TYPE(0.0f);

    [unroll] for (uint col0 = 0, idx = 0; idx < num_iters; col0 += BLOCK_SIZE, ++idx) {
        const uint col = col0 + tid;
        if (col >= p.KX) { break; }

        const uint i = rowx * p.KX + col;
        FLOAT_TYPE val;
        if (idx < DATA_CACHE_SIZE) {
            val = exp(data_cache[idx] - max_val);
        } else {
            val = exp(
                FLOAT_TYPE(data_a[i]) * p.scale
                + (p.KY > 0 ? slope * FLOAT_TYPE(data_b[rowy_start + col]) : FLOAT_TYPE(0.0f))
                - max_val
            );
        }
        sum += val;
        if (idx < DATA_CACHE_SIZE) {
            data_cache[idx] = val;
        } else {
            data_d[i] = D_TYPE(val);
        }
    }

    // Reduce sum across workgroup.
    vals[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) { vals[tid] += vals[tid + s]; }
        GroupMemoryBarrierWithGroupSync();
    }
    sum = vals[0];

    if (p.has_sinks != 0) {
        sum += FLOAT_TYPE(exp(FLOAT_TYPE(data_c[i02]) - max_val));
    }

    FLOAT_TYPE rcpdivisor = FLOAT_TYPE(1.0) / sum;

    [unroll] for (uint col0 = 0, idx = 0; idx < num_iters; col0 += BLOCK_SIZE, ++idx) {
        const uint col = col0 + tid;
        if (col >= p.KX) { continue; }

        if (idx < DATA_CACHE_SIZE) {
            data_d[rowx * p.KX + col] = D_TYPE(data_cache[idx] * rcpdivisor);
        } else {
            data_d[rowx * p.KX + col] *= D_TYPE(rcpdivisor);
        }
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID    : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid = gl_LocalInvocationID.x;
    const uint num_blocks = (p.KX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if      (num_blocks > 32) { soft_max(num_blocks, gl_WorkGroupID, tid); }
    else if (num_blocks > 16) { soft_max(32, gl_WorkGroupID, tid); }
    else if (num_blocks > 8)  { soft_max(16, gl_WorkGroupID, tid); }
    else if (num_blocks > 4)  { soft_max(8, gl_WorkGroupID, tid); }
    else if (num_blocks == 4) { soft_max(4, gl_WorkGroupID, tid); }
    else if (num_blocks == 3) { soft_max(3, gl_WorkGroupID, tid); }
    else if (num_blocks == 2) { soft_max(2, gl_WorkGroupID, tid); }
    else if (num_blocks == 1) { soft_max(1, gl_WorkGroupID, tid); }
}
