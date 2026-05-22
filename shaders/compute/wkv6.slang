// Ported from llama.cpp ggml-vulkan wkv6.comp, MIT license — see LICENSE.llama-cpp
//
// RWKV-6 attention forward over T tokens. Per-head state of shape [head_size, head_size]
// is held in a thread-local array `state`, with k/r/tf/td staged through shared memory.

//@variants
//  f32: A_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "types.slang"

static const uint BLOCK_SIZE = 64;

struct Wkv6Params {
    uint B;
    uint T;
    uint C;
    uint H;
};

[[vk::push_constant]] ConstantBuffer<Wkv6Params> pc;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   k;
[[vk::binding(1, 0)]] StructuredBuffer<A_TYPE>   v;
[[vk::binding(2, 0)]] StructuredBuffer<A_TYPE>   r;
[[vk::binding(3, 0)]] StructuredBuffer<A_TYPE>   tf;
[[vk::binding(4, 0)]] StructuredBuffer<A_TYPE>   td;
[[vk::binding(5, 0)]] StructuredBuffer<A_TYPE>   state_in;
[[vk::binding(6, 0)]] RWStructuredBuffer<A_TYPE> dst;

groupshared A_TYPE _k[BLOCK_SIZE], _r[BLOCK_SIZE], _tf[BLOCK_SIZE], _td[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint head_size = BLOCK_SIZE;
    const uint batch_id  = gl_WorkGroupID.x / pc.H;
    const uint head_id   = gl_WorkGroupID.x % pc.H;
    const uint tid       = gl_LocalInvocationID.x;

    const uint state_size   = pc.C * head_size;
    const uint n_seq_tokens = pc.T / pc.B;

    if (batch_id >= pc.B || head_id >= pc.H) { return; }

    A_TYPE state[BLOCK_SIZE];
    [unroll] for (uint i = 0; i < head_size; i++) {
        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size + i * head_size + tid];
    }

    GroupMemoryBarrierWithGroupSync();
    _tf[tid] = tf[head_id * head_size + tid];
    GroupMemoryBarrierWithGroupSync();

    const uint start_t = batch_id       * n_seq_tokens * pc.C + head_id * head_size + tid;
    const uint end_t   = (batch_id + 1) * n_seq_tokens * pc.C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += pc.C) {
        GroupMemoryBarrierWithGroupSync();
        _k[tid]  = k[t];
        _r[tid]  = r[t];
        _td[tid] = td[t];
        GroupMemoryBarrierWithGroupSync();

        const A_TYPE v_val = v[t];
        A_TYPE y = A_TYPE(0);

        [unroll] for (uint j = 0; j < head_size; j += 4) {
            float4 k_vec  = float4(_k[j],  _k[j+1],  _k[j+2],  _k[j+3]);
            float4 r_vec  = float4(_r[j],  _r[j+1],  _r[j+2],  _r[j+3]);
            float4 tf_vec = float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            float4 td_vec = float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            float4 s_vec  = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv   = k_vec * float(v_val);
            float4 temp = tf_vec * kv + s_vec;
            y += A_TYPE(dot(r_vec, temp));

            s_vec = s_vec * td_vec + kv;
            state[j]   = A_TYPE(s_vec.x);
            state[j+1] = A_TYPE(s_vec.y);
            state[j+2] = A_TYPE(s_vec.z);
            state[j+3] = A_TYPE(s_vec.w);
        }

        dst[t] = y;
    }

    [unroll] for (uint i = 0; i < head_size; i++) {
        dst[pc.T * pc.C + batch_id * state_size + head_id * head_size * head_size + i * head_size + tid] = state[i];
    }
}
