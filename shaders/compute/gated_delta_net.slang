// Ported from llama.cpp ggml-vulkan gated_delta_net.comp, MIT license — see LICENSE.llama-cpp
//
// Gated delta-net attention (Qwen3 / Hyena-style). Each workgroup owns one (head, seq)
// pair; LANES_PER_COLUMN threads share a single output column and reduce their per-row
// shards via shared memory. The GLSL source also offers subgroup-add and subgroup-clustered
// fast paths, but Slang lacks a direct equivalent for fixed-cluster reductions, so we ship
// only the shmem fallback — same correctness, leaves perf on the table for later.

//@variants
//  f32_shmem: FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_SubgroupInvocationID

#include "types.slang"

[SpecializationConstant] const uint S_V              = 128;
[SpecializationConstant] const uint KDA              = 0;
[SpecializationConstant] const uint SUBGROUP_SIZE    = 32;
[SpecializationConstant] const uint LANES_PER_COLUMN = 32;

static const uint COLS_PER_WG   = SUBGROUP_SIZE / LANES_PER_COLUMN;
static const uint ROWS_PER_LANE = S_V / LANES_PER_COLUMN;

struct GatedDeltaParams {
    uint H;
    uint n_tokens;
    uint n_seqs;
    uint s_off;
    uint sq1; uint sq2; uint sq3;
    uint sv1; uint sv2; uint sv3;
    uint sb1; uint sb2; uint sb3;
    uint neq1; uint rq3;
    float scale;
    uint K;
};

[[vk::push_constant]] ConstantBuffer<GatedDeltaParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<FLOAT_TYPE>   data_q;
[[vk::binding(1, 0)]] StructuredBuffer<FLOAT_TYPE>   data_k;
[[vk::binding(2, 0)]] StructuredBuffer<FLOAT_TYPE>   data_v;
[[vk::binding(3, 0)]] StructuredBuffer<FLOAT_TYPE>   data_g;
[[vk::binding(4, 0)]] StructuredBuffer<FLOAT_TYPE>   data_beta;
[[vk::binding(5, 0)]] StructuredBuffer<FLOAT_TYPE>   data_state;
[[vk::binding(6, 0)]] RWStructuredBuffer<FLOAT_TYPE> data_dst;

groupshared FLOAT_TYPE temp[SUBGROUP_SIZE];

FLOAT_TYPE reduce_add_shmem(FLOAT_TYPE partial, uint lane) {
    temp[lane] = partial;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = LANES_PER_COLUMN / 2u; s > 0; s >>= 1u) {
        const FLOAT_TYPE other = temp[lane ^ s];
        GroupMemoryBarrierWithGroupSync();
        temp[lane] += other;
        GroupMemoryBarrierWithGroupSync();
    }
    const FLOAT_TYPE result = temp[lane];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID) {
    const uint head_id = gl_WorkGroupID.x;
    const uint seq_id  = gl_WorkGroupID.y;
    const uint lane    = gl_SubgroupInvocationID % LANES_PER_COLUMN;
    const uint col     = gl_WorkGroupID.z * COLS_PER_WG + (gl_SubgroupInvocationID / LANES_PER_COLUMN);

    const uint iq1 = head_id % p.neq1;
    const uint iq3 = seq_id / p.rq3;

    const uint state_size          = S_V * S_V;
    const uint state_in_base       = (seq_id * p.K * p.H + head_id) * state_size;
    const uint state_out_base      = (seq_id * p.H + head_id) * state_size;
    const uint state_size_per_snap = state_size * p.H * p.n_seqs;

    FLOAT_TYPE s_shard[ROWS_PER_LANE];
    [unroll] for (uint r = 0; r < ROWS_PER_LANE; r++) {
        s_shard[r] = data_state[state_in_base + col * S_V + r * LANES_PER_COLUMN + lane];
    }

    const int shift = int(p.n_tokens) - int(p.K);
    uint attn_off   = (seq_id * p.n_tokens * p.H + head_id) * S_V;

    for (uint t = 0; t < p.n_tokens; t++) {
        const uint q_off  = iq3 * p.sq3 + t * p.sq2 + iq1 * p.sq1;
        const uint k_off  = q_off;
        const uint v_off  = seq_id * p.sv3 + t * p.sv2 + head_id * p.sv1;
        const uint gb_off = seq_id * p.sb3 + t * p.sb2 + head_id * p.sb1;
        const FLOAT_TYPE beta_val = data_beta[gb_off];

        FLOAT_TYPE k_reg[ROWS_PER_LANE];
        FLOAT_TYPE q_reg[ROWS_PER_LANE];
        [unroll] for (uint r = 0; r < ROWS_PER_LANE; r++) {
            const uint i = r * LANES_PER_COLUMN + lane;
            k_reg[r] = data_k[k_off + i];
            q_reg[r] = data_q[q_off + i];
        }

        FLOAT_TYPE g_exp[ROWS_PER_LANE];
        if (KDA == 0) {
            const FLOAT_TYPE g_val = exp(data_g[gb_off]);
            [unroll] for (uint r = 0; r < ROWS_PER_LANE; r++) { g_exp[r] = g_val; }
        } else {
            const uint g_base = gb_off * S_V;
            [unroll] for (uint r = 0; r < ROWS_PER_LANE; r++) {
                const uint i = r * LANES_PER_COLUMN + lane;
                g_exp[r] = exp(data_g[g_base + i]);
            }
        }

        const FLOAT_TYPE v_val = data_v[v_off + col];

        FLOAT_TYPE kv_shard = FLOAT_TYPE(0);
        [unroll] for (uint r = 0; r < ROWS_PER_LANE; r++) {
            kv_shard += g_exp[r] * s_shard[r] * k_reg[r];
        }
        const FLOAT_TYPE kv_col = reduce_add_shmem(kv_shard, lane);

        const FLOAT_TYPE delta_col = (v_val - kv_col) * beta_val;

        FLOAT_TYPE attn_partial = FLOAT_TYPE(0);
        [unroll] for (uint r = 0; r < ROWS_PER_LANE; r++) {
            s_shard[r] = g_exp[r] * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const FLOAT_TYPE attn_col = reduce_add_shmem(attn_partial, lane);

        if (lane == 0) {
            data_dst[attn_off + col] = attn_col * p.scale;
        }
        attn_off += S_V * p.H;

        if (p.K > 1u) {
            const int target_slot = int(t) - shift;
            if (target_slot >= 0 && target_slot < int(p.K)) {
                const uint slot_base = p.s_off + uint(target_slot) * state_size_per_snap + state_out_base;
                [unroll] for (uint r = 0; r < ROWS_PER_LANE; r++) {
                    data_dst[slot_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
                }
            }
        }
    }

    if (p.K == 1u) {
        [unroll] for (uint r = 0; r < ROWS_PER_LANE; r++) {
            data_dst[p.s_off + state_out_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
        }
    }
}
