// Fused `gated_attn = (rms_norm(gdn_attn) * ssm_norm) * silu(z)` for the
// qwen35moe SSM block. Replaces an `rms_norm` (with `do_multiply`)
// followed by a `swiglu_split(z, attn_normed)` — saves one dispatch
// and one buffer-memory barrier per SSM layer.
//
// Workgroup layout: one workgroup per (head, token) pair, BLOCK_SIZE=128
// threads collectively reducing over `s_v` elements per row. Caller
// dispatches `(num_v, l, 1)` workgroups.

//@variants
//  f32: A_TYPE=float
//@end-variants

[SpecializationConstant] const uint BLOCK_SIZE = 128;

struct SsmNormGateParams {
    uint s_v;         // row width (= 128)
    uint num_v;       // rows-per-token (= 32)
    uint value_dim;   // s_v * num_v (= 4096)
    uint l;           // token count
    float eps;
};

[[vk::push_constant]] ConstantBuffer<SsmNormGateParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   gdn_attn;   // [s_v, num_v, L]
[[vk::binding(1, 0)]] StructuredBuffer<A_TYPE>   ssm_norm;   // [s_v]
[[vk::binding(2, 0)]] StructuredBuffer<A_TYPE>   z_buf;      // [value_dim, L]
[[vk::binding(3, 0)]] RWStructuredBuffer<A_TYPE> dst;        // [value_dim, L]

groupshared float sumsh[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint head  = gl_WorkGroupID.x;        // 0..num_v
    const uint token = gl_WorkGroupID.y;        // 0..L
    const uint tid   = gl_LocalInvocationID.x;  // 0..BLOCK_SIZE

    // Base offsets — gdn_attn is `[s_v, num_v, L]` contiguous on s_v,
    // so row (head, token) starts at `(token * num_v + head) * s_v`.
    // z and dst share `[value_dim, L]` with value_dim = num_v * s_v;
    // same flat layout — head/token map to the same base.
    const uint gdn_base = (token * p.num_v + head) * p.s_v;
    const uint z_base   = gdn_base;
    const uint dst_base = gdn_base;

    // Reduce sum-of-squares over s_v elements.
    float local_sum = 0.0f;
    if (tid < p.s_v) {
        const float x = float(gdn_attn[gdn_base + tid]);
        local_sum = x * x;
    }
    sumsh[tid] = local_sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (tid < uint(s)) {
            sumsh[tid] += sumsh[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const float mean  = sumsh[0] / float(p.s_v);
    const float scale = rsqrt(mean + p.eps);

    if (tid < p.s_v) {
        const float x        = float(gdn_attn[gdn_base + tid]);
        const float w        = float(ssm_norm[tid]);
        const float z        = float(z_buf[z_base + tid]);
        const float silu_z   = z / (1.0f + exp(-z));
        dst[dst_base + tid]  = A_TYPE((x * scale) * w * silu_z);
    }
}
