// Fused per-(head, token) RMS norm + M-RoPE.
//
// Replaces the two separate dispatches the qwen35moe attention block
// runs back-to-back:
//
//   rms_norm(q_attn_view, attn_q_norm) → q_normed  (intermediate)
//   rope_multi(q_normed, positions)   → q_roped
//
// (and the same chain for K). Eliminates one dispatch + barrier + the
// `q_normed` / `k_normed` scratch buffer per Q/K side per attention
// layer (×10 attention layers).
//
// The kernel runs one workgroup per (head, token, seq), 256 threads
// per workgroup (= head_dim_k). Each thread handles ONE head_dim
// element. Phase 1 reduces the sum-of-squares cooperatively, phase 2
// writes the normalized values to shared memory, phase 3 reads the
// per-thread pair partner from shared memory and writes the rotated
// (or pass-through) value to dst.

//@variants
//  f32:    A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float        USE_DYN_OFFSET=0
//  to_f16: A_TYPE=float       D_TYPE=float16_t   FLOAT_TYPE=float        USE_DYN_OFFSET=1
//@end-variants

#include "types.slang"
#include "rope_params.slang"
#include "reduce_funcs.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 256;

struct rms_norm_rope_params {
    rope_params rope;
    float eps;
};

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>     data_a;
[[vk::binding(1, 0)]] StructuredBuffer<int>        data_pos;
[[vk::binding(2, 0)]] StructuredBuffer<float>      data_ff;     // unused when has_ff=0
[[vk::binding(3, 0)]] RWStructuredBuffer<D_TYPE>   data_d;
[[vk::binding(4, 0)]] StructuredBuffer<A_TYPE>     data_w;      // norm weight, shape [head_dim]
#if USE_DYN_OFFSET
// Per-forward dynamic params — see src/inference/decode_dyn.rs. The
// `to_f16` variant writes directly into the F16 KV cache at a position
// offset that changes every decode token; reading it from a host-written
// buffer (rather than push constants) lets the recorded decode command
// buffer be replayed across tokens.
#include "decode_dyn.slang"
[[vk::binding(5, 0)]] StructuredBuffer<DecodeDyn>  data_dyn;
#endif

[[vk::push_constant]] ConstantBuffer<rms_norm_rope_params> pc;

groupshared float norm_shared[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID,
          uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid = gl_LocalInvocationID.x;
    const uint i1  = gl_WorkGroupID.x;   // head index
    const uint i2  = gl_WorkGroupID.y;   // token index
    const uint i3  = gl_WorkGroupID.z;   // seq index

    const uint a_row = pc.rope.a_offset + i3 * pc.rope.nb03 + i2 * pc.rope.nb02 + i1 * pc.rope.nb01;
#if USE_DYN_OFFSET
    const uint d_offset = data_dyn[0].rope_d_offset;
#else
    const uint d_offset = pc.rope.d_offset;
#endif
    const uint d_row = d_offset + i3 * pc.rope.nb13 + i2 * pc.rope.nb12 + i1 * pc.rope.nb11;

    // Phase 1: load value and contribute partial sum-of-squares.
    float x = 0.0f;
    if (tid < pc.rope.ne00) {
        x = float(data_a[a_row + tid]);
    }

    // Phase 2: workgroup reduction (wave-cooperative).
    const float sum = workgroup_sum(x * x, tid, BLOCK_SIZE);

    // Phase 3: normalize and apply weight.
    const float mean  = sum / float(pc.rope.ne00);
    const float scale = rsqrt(mean + pc.eps);
    float w = 1.0f;
    if (tid < pc.rope.ne00) {
        w = float(data_w[tid]);
    }
    const float x_n = x * scale * w;
    norm_shared[tid] = x_n;
    GroupMemoryBarrierWithGroupSync();

    // Phase 4: rope rotation or pass-through write.
    if (tid >= pc.rope.ne00) { return; }

    if (tid >= pc.rope.n_dims) {
        data_d[d_row + tid] = D_TYPE(x_n);
        return;
    }

    // tid in [0, n_dims). Pair convention: (k, k + n_dims/2) for k in
    // [0, n_dims/2). pair_first is the "k" index; this thread is either
    // the "first" of its pair (writes x0*cos - x1*sin at tid = pair_first)
    // or the "second" (writes x0*sin + x1*cos at tid = pair_first + half).
    const uint half_dims  = pc.rope.n_dims / 2;
    const bool is_first   = (tid < half_dims);
    const uint pair_first = is_first ? tid : tid - half_dims;

    // M-RoPE section dispatch — same logic as rope_multi.slang.
    const int  sect_dims = pc.rope.sections[0] + pc.rope.sections[1]
                         + pc.rope.sections[2] + pc.rope.sections[3];
    const int  sec_w     = pc.rope.sections[1] + pc.rope.sections[0];
    const uint sector    = pair_first % uint(sect_dims);

    float theta_base = 0.0f;
    if (pc.rope.is_imrope != 0) {
        // Qwen-VL imrope sequence; sectors interleave by (sector % 3).
        if (sector % 3 == 1 && int(sector) < 3 * pc.rope.sections[1]) {
            theta_base = float(data_pos[i2 + pc.rope.ne02 * 1]) * pow(pc.rope.theta_scale, float(pair_first));
        } else if (sector % 3 == 2 && int(sector) < 3 * pc.rope.sections[2]) {
            theta_base = float(data_pos[i2 + pc.rope.ne02 * 2]) * pow(pc.rope.theta_scale, float(pair_first));
        } else if (sector % 3 == 0 && int(sector) < 3 * pc.rope.sections[0]) {
            theta_base = float(data_pos[i2]) * pow(pc.rope.theta_scale, float(pair_first));
        } else {
            theta_base = float(data_pos[i2 + pc.rope.ne02 * 3]) * pow(pc.rope.theta_scale, float(pair_first));
        }
    } else {
        if (int(sector) < pc.rope.sections[0]) {
            theta_base = float(data_pos[i2]) * pow(pc.rope.theta_scale, float(pair_first));
        } else if (int(sector) >= pc.rope.sections[0] && int(sector) < sec_w) {
            theta_base = float(data_pos[i2 + pc.rope.ne02 * 1]) * pow(pc.rope.theta_scale, float(pair_first));
        } else if (int(sector) >= sec_w && int(sector) < sec_w + pc.rope.sections[2]) {
            theta_base = float(data_pos[i2 + pc.rope.ne02 * 2]) * pow(pc.rope.theta_scale, float(pair_first));
        } else if (int(sector) >= sec_w + pc.rope.sections[2]) {
            theta_base = float(data_pos[i2 + pc.rope.ne02 * 3]) * pow(pc.rope.theta_scale, float(pair_first));
        }
    }

    const float freq_factor = pc.rope.has_ff != 0 ? data_ff[pair_first] : 1.0f;

    // rope_yarn — ext_factor=0 short path matches qwen_default.
    float mscale = pc.rope.attn_factor;
    float theta  = pc.rope.freq_scale * (theta_base / freq_factor);
    if (pc.rope.ext_factor != 0.0f) {
        const float y = (float(pair_first) - pc.rope.corr_dims[0])
                      / max(0.001f, pc.rope.corr_dims[1] - pc.rope.corr_dims[0]);
        const float ramp_mix = (1.0f - min(1.0f, max(0.0f, y))) * pc.rope.ext_factor;
        const float theta_extrap = theta_base / freq_factor;
        theta  = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / pc.rope.freq_scale);
    }
    if (pc.rope.is_back != 0) { theta = -theta; }
    const float cos_theta = cos(theta) * mscale;
    const float sin_theta = sin(theta) * mscale;

    const float x0 = norm_shared[pair_first];
    const float x1 = norm_shared[pair_first + half_dims];

    if (is_first) {
        data_d[d_row + tid] = D_TYPE(x0 * cos_theta - x1 * sin_theta);
    } else {
        data_d[d_row + tid] = D_TYPE(x0 * sin_theta + x1 * cos_theta);
    }
}
