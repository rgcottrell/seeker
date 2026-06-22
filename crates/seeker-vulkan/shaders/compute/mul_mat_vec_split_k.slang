// Split-K mul_mat_vec for huge-M matmuls (lm_head, n_vocab × hidden).
// Mirrors `mul_mat_vec.slang` but partitions the K-reduction across
// `gl_WorkGroupID.y` workgroups, each writing one partial output per
// row into a `[M, SPLIT_K]` buffer that a follow-up reduce shader
// sums into the final `[M]` logits.
//
// Goal: more workgroup-level parallelism for the lm_head matvec
// (248k × 2048 Q8_0 + F32) on RDNA so the matvec can hit closer to
// the LPDDR5X bandwidth ceiling. The existing single-pass kernel
// already issues ~62k workgroups; split-K multiplies that by SPLIT_K
// at the cost of one reduce dispatch.
//
// Single-batch only — no broadcast/batch support (lm_head is always
// batch_idx=0). Reuses the existing matvec head for bindings so the
// dequant / B-load paths are identical; just the K-range and the
// final write differ.

//@variants
//  q8_0:   DATA_A_Q8_0=1   B_TYPE=float B_TYPEV4=float4 D_TYPE=float FLOAT_TYPE=float
//@end-variants

import glsl;

#include "types.slang"
#include "mul_mat_vec_head.slang"

#define K_PER_ITER 8

[SpecializationConstant] const uint SPLIT_K = 4;

void iter_split(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                uint first_row, uint num_rows, uint tid,
                uint a_offset, uint b_offset, uint y_offset,
                uint i_scaled, bool lastiter) {
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        const uint col  = i_scaled * BLOCK_SIZE + K_PER_ITER * tid;
        const uint iqs  = (col % QUANT_K) / QUANT_R;
        const uint iybs = col - col % QUANT_K;

#if QUANT_R == 2
        const float4 bv02 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + iybs + iqs) / 4    ]);
        const float4 bv13 = float4(data_b_v4[(j * p.batch_stride_b + b_offset + iybs + iqs + y_offset) / 4    ]);
        const float4 bv0  = float4(bv02.x, bv13.x, bv02.y, bv13.y);
        const float4 bv1  = float4(bv02.z, bv13.z, bv02.w, bv13.w);
#else
        const float4 bv0  = float4(data_b_v4[(j * p.batch_stride_b + b_offset + iybs + iqs) / 4]);
        const float4 bv1  = float4(data_b_v4[(j * p.batch_stride_b + b_offset + iybs + iqs) / 4 + 1]);
#endif

        uint ibi = first_row * p.ncols;
        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const uint ib = (ibi + col) / QUANT_K;
            ibi += p.ncols;

            float4 v  = dequantize4(ib, iqs,                 a_offset);
            float4 v2 = dequantize4(ib, iqs + (4 / QUANT_R), a_offset);

            const float2 dm = get_dm(ib, a_offset);
            if (dm.y != 0.0f) {
                v  = v  * dm.x + dm.y;
                v2 = v2 * dm.x + dm.y;
            }

            FLOAT_TYPE rowtmp = FLOAT_TYPE(dot(bv0, v));
            rowtmp += FLOAT_TYPE(dot(bv1, v2));
            if (dm.y == 0.0f) {
                rowtmp *= FLOAT_TYPE(dm.x);
            }
            temp[j][n] += rowtmp;
        }
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid       = gl_LocalInvocationID.x;
    const uint first_row = NUM_ROWS * gl_WorkGroupID.x;
    const uint split_idx = gl_WorkGroupID.y;

    const uint num_rows = (first_row + NUM_ROWS <= p.stride_d)
        ? NUM_ROWS
        : (p.stride_d - first_row);
    if (first_row >= p.stride_d) { return; }

    const uint y_offset = QUANT_R == 1 ? 1 : QUANT_K / 2;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint i = 0; i < NUM_ROWS; ++i) {
            temp[j][i] = FLOAT_TYPE(0);
        }
    }

    // Total iterations across full K.
    const uint num_iters_full = p.ncols / (K_PER_ITER * BLOCK_SIZE);
    // Iteration range owned by this split — uniform across SPLIT_K splits;
    // assumes K is divisible by `SPLIT_K * K_PER_ITER * BLOCK_SIZE` (e.g.
    // K=2048, SPLIT_K=4, K_PER_ITER=8, BLOCK_SIZE=32 → 2 iters/split).
    const uint iters_per_split = num_iters_full / SPLIT_K;
    const uint i_lo = split_idx * iters_per_split;
    const uint i_hi = i_lo + iters_per_split;

    for (uint i = i_lo; i < i_hi; ++i) {
        iter_split(temp, first_row, num_rows, tid,
                   /*a_offset=*/ 0, /*b_offset=*/ 0, y_offset,
                   i * K_PER_ITER, false);
    }

    // Subgroup-reduce the BLOCK_SIZE lanes' partials, then lane 0 writes
    // its (row, split) slot into the [M, SPLIT_K] partials buffer.
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const FLOAT_TYPE sum = WaveActiveSum(temp[j][n]);
            if (tid == 0) {
                // Partials laid out as [M, SPLIT_K] — split-major so a
                // contiguous reduce kernel sweeps one row's K splits with
                // unit stride.
                const uint dst_idx = (first_row + n) * SPLIT_K + split_idx;
                data_d[dst_idx] = D_TYPE(sum);
            }
        }
    }
}
