// Ported from llama.cpp ggml-vulkan mul_mat_vec.comp, MIT license — see LICENSE.llama-cpp
//
// K_PER_ITER auto-selects per DATA_A_X: FP weights run the 2-elements-per-
// thread scalar path, legacy quants run the 8-elements-per-thread vectorized
// path through `dequantize4` + `data_a_packed16` + `data_b_v4`.
//
// Still scoped out: MUL_MAT_ID (MoE), bias/scale fusion, subgroupAdd reduce,
// K-quants, I-quants, BF16.

//@variants
//  f32:    DATA_A_F32=1    B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  f16:    DATA_A_F16=1    B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  q4_0:   DATA_A_Q4_0=1   B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  q4_1:   DATA_A_Q4_1=1   B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  q5_0:   DATA_A_Q5_0=1   B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  q5_1:   DATA_A_Q5_1=1   B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  q8_0:   DATA_A_Q8_0=1   B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  iq4_nl: DATA_A_IQ4_NL=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  iq4_xs: DATA_A_IQ4_XS=1 B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  bf16:   DATA_A_BF16=1   B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//  mxfp4:  DATA_A_MXFP4=1  B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;  // for gl_NumWorkGroups in the row-stride math

#include "types.slang"
#include "mul_mat_vec_head.slang"

// Matches the GLSL source: vectorized 8-elements path for quants, scalar
// 2-elements path for unquantized FP weights.
#if !defined(DATA_A_F32) && !defined(DATA_A_F16) && !defined(DATA_A_BF16)
#define K_PER_ITER 8
#else
#define K_PER_ITER 2
#endif

void iter(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
          uint first_row, uint num_rows, uint tid,
          uint a_offset, uint b_offset, uint y_offset,
          uint i_scaled, bool lastiter) {
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        // GLSL passes `i*K_PER_ITER` as `i_scaled`; col = i_scaled * BLOCK_SIZE + K_PER_ITER * tid
        const uint col  = i_scaled * BLOCK_SIZE + K_PER_ITER * tid;
        const uint iqs  = (col % QUANT_K) / QUANT_R;
        const uint iybs = col - col % QUANT_K;

#if K_PER_ITER == 8
    #if QUANT_R == 2
        const float4 bv02 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + iybs + iqs) / 4]);
        const float4 bv13 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + iybs + iqs + y_offset) / 4]);
        const float4 bv0  = float4(bv02.x, bv13.x, bv02.y, bv13.y);
        const float4 bv1  = float4(bv02.z, bv13.z, bv02.w, bv13.w);
    #else
        const float4 bv0  = float4(data_b_v4[(j * p.batch_stride_b + b_offset + iybs + iqs) / 4]);
        const float4 bv1  = float4(data_b_v4[(j * p.batch_stride_b + b_offset + iybs + iqs) / 4 + 1]);
    #endif
#else
        // If the OOB element falls past ncols on the last iter, fetch only the first.
        const bool OOB = lastiter && (iybs + iqs + y_offset >= p.ncols);

        FLOAT_TYPE b0 = FLOAT_TYPE(data_b[j * p.batch_stride_b + b_offset + iybs + iqs]);
        FLOAT_TYPE b1 = FLOAT_TYPE(0);
        if (!OOB) {
            b1 = FLOAT_TYPE(data_b[j * p.batch_stride_b + b_offset + iybs + iqs + y_offset]);
        }
#endif

        uint ibi = first_row * p.ncols;
        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint ib = (ibi + col) / QUANT_K;
            ibi += p.ncols;

#if K_PER_ITER == 8
            float4 v  = dequantize4(ib, iqs,                 a_offset);
            float4 v2 = dequantize4(ib, iqs + (4 / QUANT_R), a_offset);

            const float2 dm = get_dm(ib, a_offset);
            if (dm.y != 0.0f) {  // quant has a min component
                v  = v  * dm.x + dm.y;
                v2 = v2 * dm.x + dm.y;
            }

            FLOAT_TYPE rowtmp = FLOAT_TYPE(dot(bv0, v));
            rowtmp += FLOAT_TYPE(dot(bv1, v2));
            if (dm.y == 0.0f) {
                rowtmp *= FLOAT_TYPE(dm.x);
            }
            temp[j][n] += rowtmp;
#else
            const float2 v = dequantize(ib, iqs, a_offset);
            temp[j][n] = mad(FLOAT_TYPE(v.x), b0, temp[j][n]);
            if (!OOB) {
                temp[j][n] = mad(FLOAT_TYPE(v.y), b1, temp[j][n]);
            }
#endif
        }
    }
}

void compute_outputs(uint3 wg_id, uint tid, uint first_row, uint num_rows) {
    uint a_offset, b_offset, d_offset;
    get_offsets(wg_id, a_offset, b_offset, d_offset);
    const uint y_offset = QUANT_R == 1 ? 1 : QUANT_K / 2;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint i = 0; i < NUM_ROWS; ++i) {
            temp[j][i] = FLOAT_TYPE(0);
        }
    }

    uint num_iters = p.ncols / (K_PER_ITER * BLOCK_SIZE);
    if (num_iters * K_PER_ITER * BLOCK_SIZE + K_PER_ITER * tid < p.ncols) {
        num_iters++;
    }

    uint unroll_count = 4;
    uint unrolled_iters = num_iters & ~(unroll_count - 1);

    // If ncols is odd, we need lastiter==true on the last iteration so OOB is
    // computed correctly. Skip some unrolling so the trailing tail handles it.
    if ((p.ncols & 1) != 0 && unrolled_iters == num_iters && unrolled_iters > 0) {
        unrolled_iters -= unroll_count;
    }

    uint i = 0;
    while (i < unrolled_iters) {
        [unroll] for (uint k = 0; k < unroll_count; ++k) {
            iter(temp, first_row, num_rows, tid, a_offset, b_offset, y_offset, i * K_PER_ITER, false);
            i++;
        }
    }

    unroll_count = 2;
    unrolled_iters = num_iters & ~(unroll_count - 1);
    if ((p.ncols & 1) != 0 && unrolled_iters == num_iters && unrolled_iters > 0) {
        unrolled_iters -= unroll_count;
    }

    while (i < unrolled_iters) {
        [unroll] for (uint k = 0; k < unroll_count; ++k) {
            iter(temp, first_row, num_rows, tid, a_offset, b_offset, y_offset, i * K_PER_ITER, false);
            i++;
        }
    }
    while (i < num_iters) {
        iter(temp, first_row, num_rows, tid, a_offset, b_offset, y_offset, i * K_PER_ITER, true);
        i++;
    }

    reduce_result(temp, d_offset, first_row, num_rows, tid);
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid = gl_LocalInvocationID.x;
    const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);

#ifdef NEEDS_INIT_IQ_SHMEM
    // I-quant variants need the lookup table copied to groupshared first;
    // every thread must call this before any early return.
    init_iq_shmem(uint3(BLOCK_SIZE, 1, 1));
#endif

    if (first_row + NUM_ROWS <= p.stride_d) {
        compute_outputs(gl_WorkGroupID, tid, first_row, NUM_ROWS);
    } else {
        if (first_row >= p.stride_d) { return; }
        compute_outputs(gl_WorkGroupID, tid, first_row, p.stride_d - first_row);
    }
}
