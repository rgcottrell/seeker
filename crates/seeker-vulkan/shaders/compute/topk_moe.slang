// Ported from llama.cpp ggml-vulkan topk_moe.comp, MIT license — see LICENSE.llama-cpp
//
// MoE gating + top-k expert selection. Two layers of routing:
//   1. Apply the gating function (softmax / sigmoid / softmax-after) to the raw logits.
//   2. Iteratively pick the top n_expert_used experts via warp argmax with ties broken by
//      smaller expert index.
// Subgroup intrinsics map: WaveActiveMax/Sum and WaveReadLaneAt for shuffleXor.

//@variants
//  f32: A_TYPE=float       FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_SubgroupID, gl_SubgroupInvocationID

#include "types.slang"

static const uint GATING_FUNC_SOFTMAX        = 0;
static const uint GATING_FUNC_SIGMOID        = 1;
static const uint GATING_FUNC_SOFTMAX_WEIGHT = 2;

struct TopkMoeParams {
    uint  n_rows;
    uint  n_experts_push;
    uint  n_expert_used;
    float clamp_min;
    float clamp_max;
    uint  gating_func;
    uint  has_bias;
    uint  with_norm;
    float output_scale;
    float output_bias;
};

[[vk::push_constant]] ConstantBuffer<TopkMoeParams> p;

[SpecializationConstant] const uint WARP_SIZE          = 32;
[SpecializationConstant] const uint n_experts_spec     = 512;
[SpecializationConstant] const bool nexperts_use_push  = false;

static const uint EXPERTS_PER_THREAD = (n_experts_spec + WARP_SIZE - 1) / WARP_SIZE;

[[vk::binding(0, 0)]] StructuredBuffer<float>   logits;
[[vk::binding(1, 0)]] StructuredBuffer<float>   bias;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> weights;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>  ids;

float shuffle_xor(float v, uint mask) {
    return WaveReadLaneAt(v, WaveGetLaneIndex() ^ mask);
}
uint shuffle_xor_u(uint v, uint mask) {
    return WaveReadLaneAt(v, WaveGetLaneIndex() ^ mask);
}

void softmax_warp_inplace(inout float vals[EXPERTS_PER_THREAD], uint limit, uint lane, bool use_limit) {
    float max_val = -1.0f / 0.0f;
    [unroll] for (uint i = 0; i < EXPERTS_PER_THREAD; i++) {
        const uint idx       = lane + i * WARP_SIZE;
        const bool is_active = !use_limit || (idx < limit);
        if (is_active) { max_val = max(max_val, vals[i]); }
    }
    max_val = WaveActiveMax(max_val);

    float sum = 0.0f;
    [unroll] for (uint i = 0; i < EXPERTS_PER_THREAD; i++) {
        const uint idx       = lane + i * WARP_SIZE;
        const bool is_active = !use_limit || (idx < limit);
        if (is_active) {
            const float val = exp(vals[i] - max_val);
            vals[i] = val;
            sum += val;
        } else {
            vals[i] = 0.0f;
        }
    }
    sum = WaveActiveSum(sum);

    const float inv_sum = 1.0f / sum;
    [unroll] for (uint i = 0; i < EXPERTS_PER_THREAD; i++) {
        const uint idx       = lane + i * WARP_SIZE;
        const bool is_active = !use_limit || (idx < limit);
        if (is_active) { vals[i] *= inv_sum; }
    }
}

[shader("compute")]
[numthreads(32, 4, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID) {
    const uint n_experts = nexperts_use_push ? p.n_experts_push : n_experts_spec;
    const uint row       = gl_WorkGroupID.x * 4u + gl_SubgroupID;
    if (row >= p.n_rows) { return; }

    const uint logits_offset  = n_experts * row;
    const uint bias_offset    = 0;
    const uint weights_offset = p.n_expert_used * row;
    const uint ids_offset     = n_experts * row;
    const uint lane           = gl_SubgroupInvocationID;

    float probs[EXPERTS_PER_THREAD];
    [unroll] for (uint i = 0; i < EXPERTS_PER_THREAD; i++) { probs[i] = -1.0f / 0.0f; }

    [unroll] for (uint i = 0; i < n_experts; i += WARP_SIZE) {
        const uint expert = i + lane;
        probs[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[logits_offset + expert] : -1.0f / 0.0f;
    }

    if (p.gating_func == GATING_FUNC_SOFTMAX) {
        softmax_warp_inplace(probs, n_experts, lane, nexperts_use_push);
    } else if (p.gating_func == GATING_FUNC_SIGMOID) {
        [unroll] for (uint i = 0; i < n_experts; i += WARP_SIZE) {
            const uint expert = i + lane;
            const uint slot = i / WARP_SIZE;
            probs[slot] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? 1.0f / (1.0f + exp(-probs[slot])) : -1.0f / 0.0f;
        }
    }

    float selection_probs[EXPERTS_PER_THREAD];
    if (p.has_bias != 0) {
        [unroll] for (uint i = 0; i < n_experts; i += WARP_SIZE) {
            const uint expert = i + lane;
            const uint slot = i / WARP_SIZE;
            selection_probs[slot] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? probs[slot] + bias[bias_offset + expert] : -1.0f / 0.0f;
        }
    } else {
        [unroll] for (uint i = 0; i < EXPERTS_PER_THREAD; i++) { selection_probs[i] = probs[i]; }
    }

    float wt_sum = 0.0f;
    float output_weights[EXPERTS_PER_THREAD];
    [unroll] for (uint i = 0; i < EXPERTS_PER_THREAD; i++) { output_weights[i] = 0.0f; }

    for (uint k = 0; k < p.n_expert_used; k++) {
        float max_val   = probs[0];
        float max_val_s = selection_probs[0];
        uint  max_expert = lane;

        [unroll] for (uint i = WARP_SIZE; i < n_experts; i += WARP_SIZE) {
            const uint expert = i + lane;
            const uint slot = i / WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_probs[slot] > max_val_s) {
                max_val    = probs[slot];
                max_val_s  = selection_probs[slot];
                max_expert = expert;
            }
        }

        [unroll] for (uint mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = shuffle_xor(max_val,   mask);
            const float val_s  = shuffle_xor(max_val_s, mask);
            const uint  expert = shuffle_xor_u(max_expert, mask);
            if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                max_val    = val;
                max_val_s  = val_s;
                max_expert = expert;
            }
        }

        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            selection_probs[max_expert / WARP_SIZE] = -1.0f / 0.0f;
            ids[ids_offset + k] = max_expert;
            wt_sum += max_val;
        }
    }

    if (p.with_norm != 0) {
        wt_sum = WaveActiveSum(wt_sum);
        wt_sum = clamp(wt_sum, p.clamp_min, p.clamp_max);
        const float inv_sum = 1.0f / wt_sum;
        [unroll] for (uint i = 0; i < EXPERTS_PER_THREAD; ++i) { output_weights[i] *= inv_sum; }
    }

    if (p.gating_func == GATING_FUNC_SOFTMAX_WEIGHT) {
        softmax_warp_inplace(output_weights, p.n_expert_used, lane, true);
    }

    [unroll] for (uint i = 0; i < EXPERTS_PER_THREAD; ++i) {
        const uint idx = i * WARP_SIZE + lane;
        if (idx < p.n_expert_used) {
            weights[weights_offset + idx] = p.output_scale * output_weights[i] + p.output_bias;
        }
    }
}
