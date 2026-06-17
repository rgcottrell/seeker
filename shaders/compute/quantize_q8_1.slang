// Quantize a contiguous f32 activation `[K, n_cols]` (column-major: column c at
// `c*K`, K a multiple of 32) into `block_q8_1` blocks for the integer-dot (MMQ)
// matvec path. One thread owns one 32-element block.
//
// block_q8_1 = { ds: f16x2, qs[32]: int8 }, ds.x = d (scale), ds.y = d·Σqs
// (the block sum, used by the K-quant min term: Σx = Σ d·q = ds.y). Matches
// ggml's quantize_row_q8_1. The packed32 view (int32 qs[8]) reinterprets the
// same 32 contiguous int8 for `dot4add_i8packed`.

#include "types.slang"

struct Params {
    uint nblocks; // total 32-element blocks = (K/32) * n_cols
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>          data_s; // [K * n_cols]
[[vk::binding(1, 0)]] RWStructuredBuffer<block_q8_1>   data_q; // [nblocks]

static const uint BLOCK_SIZE = 256;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_GlobalInvocationID: SV_DispatchThreadID) {
    const uint b = gl_GlobalInvocationID.x;
    if (b >= p.nblocks) { return; }

    const uint src = b * QUANT_K_Q8_1; // contiguous: block b covers [b*32, b*32+32)
    float amax = 0.0f;
    [unroll] for (uint j = 0; j < QUANT_K_Q8_1; ++j) {
        amax = max(amax, abs(data_s[src + j]));
    }
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    int sum = 0;
    [unroll] for (uint j = 0; j < QUANT_K_Q8_1; ++j) {
        const int q = int(round(data_s[src + j] * id));
        data_q[b].qs[j] = int8_t(q);
        sum += q;
    }
    data_q[b].ds = float16_t2(float16_t(d), float16_t(d * float(sum)));
}
