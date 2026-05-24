// Ported from llama.cpp ggml-vulkan mul_mat_vec_base.glsl + mul_mat_vec_iface.glsl
// (subset), MIT license — see LICENSE.llama-cpp
//
// First-cut matmul-vec foundation:
//   - F32 / F16 weights only (DATA_A_F32, DATA_A_F16)
//   - No MUL_MAT_ID (MoE expert routing) variant
//   - No bias / scale fusion (fusion_flags retained in struct for layout
//     compat, but ignored)
//   - No subgroupAdd path — shared-memory tree-reduce only
//   - No packed16/32/128 buffer aliases
//
// These restrictions will be lifted in follow-up passes as the matching
// types/dequant infrastructure lands.

#ifndef MUL_MAT_VEC_HEAD_SLANG
#define MUL_MAT_VEC_HEAD_SLANG

#include "types.slang"

struct MulMatVecParams {
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;        // reserved; not consumed in this minimal port

    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
};

[[vk::push_constant]] ConstantBuffer<MulMatVecParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
#if defined(A_TYPE_PACKED16)
// Vectorized view of the A buffer. The GLSL source puts this at the same
// `binding = 0` as `data_a` (aliased descriptors); Slang rejects overlapping
// bindings, so we put it at binding 3 and rely on the host to bind the *same*
// VkBuffer to both descriptor slots. Same memory, two views.
[[vk::binding(3, 0)]] StructuredBuffer<A_TYPE_PACKED16> data_a_packed16;
#endif
#if defined(A_TYPE_PACKED32)
// 32-bit packed view of A — used by Q4_K / Q5_K / Q2_K mat-vec kernels. Host
// binds the same VkBuffer to binding 6.
[[vk::binding(6, 0)]] StructuredBuffer<A_TYPE_PACKED32> data_a_packed32;
#endif

[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   data_b;
#if defined(B_TYPEV2)
// vec2 view of `data_b` — used by Q5_K mat-vec. Host binds same VkBuffer to binding 5.
[[vk::binding(5, 0)]] StructuredBuffer<B_TYPEV2> data_b_v2;
#endif
#if defined(B_TYPEV4)
// vec4 view of `data_b`. GLSL aliases at binding = 1; we put it at binding 4.
[[vk::binding(4, 0)]] StructuredBuffer<B_TYPEV4> data_b_v4;
#endif

[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[SpecializationConstant] const uint BLOCK_SIZE = 32;

// NUM_ROWS = how many output rows one workgroup produces. Increasing this
// amortizes the B-vector load across rows (B is fetched once per iter and
// reused across NUM_ROWS A-row dequants), which dominates decode bandwidth.
// llama.cpp uses 4–8 depending on dtype; we start at 2 — register pressure
// goes up linearly and we don't want to overflow Apple's tighter file.
// NUM_COLS is for multi-prompt batching, kept at 1 for now.
static const uint NUM_ROWS = 2;
static const uint NUM_COLS = 1;

// Per-DATA_A_X dequantize / dequantize4 / get_dm overloads live in dequant_funcs.slang.
// Including here after the buffer bindings so the helpers can reference data_a /
// data_a_packed16 at file scope.
#include "dequant_funcs.slang"

void get_offsets(uint3 wg_id, out uint a_offset, out uint b_offset, out uint d_offset) {
    const uint batch_idx = wg_id.y + p.base_work_group_y;

    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;
        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;
        batch_idx_a = i03 * p.ne02 + i02;
    }

    a_offset = batch_idx_a * (p.batch_stride_a / QUANT_K);
    b_offset = batch_idx * p.batch_stride_b;
    d_offset = batch_idx * p.batch_stride_d;
}

// Subgroup-reduce result. We require `computeFullSubgroups` and dispatch
// `numthreads(BLOCK_SIZE = 32, 1, 1)`, which on Apple/Intel/NVIDIA hardware
// (subgroupSize == 32) means one workgroup == one subgroup and a single
// `WaveActiveSum` collapses BLOCK_SIZE partial sums with no shared memory
// and no `GroupMemoryBarrier`. On vendors where subgroupSize differs from
// 32 the pipeline must request matching size via
// VK_EXT_subgroup_size_control's required-size (32) for this to remain
// correct; we already enable the feature in `inference::device`.
void reduce_result(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS],
                   uint d_offset, uint first_row, uint num_rows, uint tid) {
    [unroll] for (uint j = 0; j < NUM_COLS; ++j) {
        [unroll] for (uint n = 0; n < num_rows; ++n) {
            const FLOAT_TYPE sum = WaveActiveSum(temp[j][n]);
            if (tid == 0) {
                data_d[j * p.batch_stride_d + d_offset + first_row + n] = D_TYPE(sum);
            }
        }
    }
}

#endif
