// Ported from llama.cpp ggml-vulkan mul_mat_split_k_reduce.comp, MIT license — see LICENSE.llama-cpp
//
// Reduces K partial sums from a split-K matmul into the final dst. Vector path is taken
// when ne is a multiple of 4 and the lane's 4-wide stripe fits.

//@variants
//  f32:
//@end-variants

import glsl;

struct SplitKParams {
    uint ne;
    uint k_num;
};

[[vk::push_constant]] ConstantBuffer<SplitKParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>    data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float>  data_d;
[[vk::binding(2, 0)]] StructuredBuffer<float4>   data_a4;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> data_d4;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = gl_GlobalInvocationID.x * 4u;
    if (idx >= p.ne) { return; }

    if (idx + 3 < p.ne && (p.ne % 4) == 0) {
        float4 result = float4(0, 0, 0, 0);
        [unroll] for (uint i = 0; i < p.k_num; i++) {
            result += data_a4[(i * p.ne + idx) / 4u];
        }
        data_d4[idx / 4u] = result;
    } else {
        [unroll] for (uint j = 0; j < 4; ++j) {
            if (idx + j < p.ne) {
                float result = 0.0f;
                [unroll] for (uint i = 0; i < p.k_num; i++) {
                    result += data_a[i * p.ne + idx + j];
                }
                data_d[idx + j] = result;
            }
        }
    }
}
