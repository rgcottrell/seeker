// Ported from llama.cpp ggml-vulkan solve_tri.comp, MIT license — see LICENSE.llama-cpp
//
// Forward substitution for lower-triangular A: solves A · X = B per (batch, column).
// Workgroup loads BATCH_N rows of A and B into shmem, then K threads each solve one
// column of X, accumulating from earlier columns of A held in thread-local X[N].

//@variants
//  f32: A_TYPE=float    B_TYPE=float    D_TYPE=float    FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_WorkGroupSize

#include "types.slang"
#include "generic_binary_head.slang"

[SpecializationConstant] const uint N           = 64;
[SpecializationConstant] const uint K           = 32;
[SpecializationConstant] const uint BATCH_N     = 32;
[SpecializationConstant] const uint BLOCK_SIZE  = 64;

groupshared FLOAT_TYPE shA[BATCH_N * N];
groupshared FLOAT_TYPE shB[BATCH_N * K];

static uint a_base, b_base, x_base;

FLOAT_TYPE get_a(uint r, uint c) { return FLOAT_TYPE(data_a[a_base + r * p.nb01 + c * p.nb00]); }
FLOAT_TYPE get_b(uint r, uint c) { return FLOAT_TYPE(data_b[b_base + r * p.nb11 + c * p.nb10]); }
void store_x(uint r, uint c, FLOAT_TYPE v) { data_d[x_base + r * p.nb21 + c * p.nb20] = D_TYPE(v); }

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint batch = gl_WorkGroupID.z * 262144u + gl_WorkGroupID.y * 512u + gl_WorkGroupID.x;
    const uint tid   = gl_LocalInvocationID.x;

    if (batch >= p.ne02 * p.ne03) { return; }

    const uint i3 = batch / p.ne22;
    const uint i2 = batch % p.ne22;
    a_base = get_aoffset() + i2 * p.nb02 + i3 * p.nb03;
    b_base = get_boffset() + i2 * p.nb12 + i3 * p.nb13;
    x_base = get_doffset() + i2 * p.nb22 + i3 * p.nb23;

    FLOAT_TYPE X[N];

    [unroll] for (uint row_base = 0; row_base < N; row_base += BATCH_N) {
        const uint cur_N = min(BATCH_N, N - row_base);

        [unroll] for (uint i = 0; i < cur_N * N; i += BLOCK_SIZE) {
            uint idx = i + tid;
            if (((cur_N * N) % BLOCK_SIZE == 0) || idx < cur_N * N) {
                shA[idx] = get_a(row_base + idx / N, idx % N);
            }
        }
        [unroll] for (uint i = 0; i < cur_N * K; i += BLOCK_SIZE) {
            uint idx = i + tid;
            if (((cur_N * K) % BLOCK_SIZE == 0) || idx < cur_N * K) {
                shB[idx] = get_b(row_base + idx / K, idx % K);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < K) {
            [unroll] for (uint row_offset = 0; row_offset < cur_N; ++row_offset) {
                const uint r = row_base + row_offset;
                FLOAT_TYPE b = shB[row_offset * K + tid];
                [unroll] for (uint c = 0; c < r; ++c) {
                    b -= shA[row_offset * N + c] * X[c];
                }
                const FLOAT_TYPE x = b / shA[row_offset * N + r];
                X[r] = x;
                store_x(r, tid, x);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
