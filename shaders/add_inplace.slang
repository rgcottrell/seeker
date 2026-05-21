// Element-wise `a += b` on fp16 buffers.
// Used as the residual-add step between layers in the decoder.

struct PushConstants {
    uint n_elements;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float16_t> a;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> b;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.n_elements) return;
    a[i] = a[i] + b[i];
}
