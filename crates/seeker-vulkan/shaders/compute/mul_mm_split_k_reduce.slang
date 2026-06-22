// Sum the SPLIT_K partial M×N tiles produced by mul_mm_cm's split-K path
// (`[SPLIT_K, M, N]` column-major, partial s at s*M*N) into the final M×N
// output. One thread per output element.

struct Params {
    uint mn;      // M * N
    uint split_k;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>   partials; // [SPLIT_K * M * N]
[[vk::binding(1, 0)]] RWStructuredBuffer<float> data_d;   // [M * N]

[SpecializationConstant] const uint BLOCK_SIZE = 256;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_GlobalInvocationID: SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= p.mn) { return; }
    float acc = 0.0f;
    for (uint s = 0u; s < p.split_k; ++s) {
        acc += partials[s * p.mn + i];
    }
    data_d[i] = acc;
}
