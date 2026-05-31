// Flash-attention forward pass with optional fused dequant of K/V.
//
// Q is always F32. K and V (homogeneous dtype) can be F32, F16, BF16, or any
// of the cache quants (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL). The mask is
// always F32. Output is F32.
//
// Scope (unchanged from the original scalar reference): MASK_ENABLE-only,
// no MASK_OPT prefilter, no ALiBi slopes, no attention sinks, no split-K,
// no logit softcap, no MMQ integer-dot path. The per-quant access goes
// element-at-a-time inside the inner K-dot-Q / V-accumulator loops; the
// Vulkan compiler hoists the block-scale read out of the per-element loop.
//
// One workgroup per (query-row, head, batch). 32 threads = 1 subgroup
// cooperate across the head_size dim (HSK). The K/V iteration is
// sequential over Bc-wide blocks.

//@variants
//  f32_f32:    Q_TYPE=float    KV_TYPE=float       D_TYPE=float
//  f32_f16:    Q_TYPE=float    KV_TYPE=float16_t   D_TYPE=float
//  f32_bf16:   Q_TYPE=float    KV_TYPE=uint16_t    D_TYPE=float    DATA_KV_BF16=1
//  f32_q4_0:   Q_TYPE=float    KV_TYPE=float       D_TYPE=float    DATA_A_Q4_0=1   DATA_KV_QUANT=1
//  f32_q4_1:   Q_TYPE=float    KV_TYPE=float       D_TYPE=float    DATA_A_Q4_1=1   DATA_KV_QUANT=1
//  f32_q5_0:   Q_TYPE=float    KV_TYPE=float       D_TYPE=float    DATA_A_Q5_0=1   DATA_KV_QUANT=1
//  f32_q5_1:   Q_TYPE=float    KV_TYPE=float       D_TYPE=float    DATA_A_Q5_1=1   DATA_KV_QUANT=1
//  f32_q8_0:   Q_TYPE=float    KV_TYPE=float       D_TYPE=float    DATA_A_Q8_0=1   DATA_KV_QUANT=1
//  f32_iq4_nl: Q_TYPE=float    KV_TYPE=float       D_TYPE=float    DATA_A_IQ4_NL=1 DATA_KV_QUANT=1
//@end-variants

import glsl;  // gl_SubgroupInvocationID, gl_LocalInvocationIndex

#include "types.slang"

[SpecializationConstant] const uint Bc           = 32;
[SpecializationConstant] const uint HSK          = 128;
[SpecializationConstant] const uint HSV          = 128;
[SpecializationConstant] const uint MASK_ENABLE  = 0;
[SpecializationConstant] const uint Clamp        = 0;

static const uint WORKGROUP_SIZE = 32;

struct FaParams {
    uint N;
    uint KV;

    uint ne1;
    uint ne2;
    uint ne3;

    uint neq2;
    uint neq3;
    uint nek2;
    uint nek3;
    uint nev2;
    uint nev3;
    uint nem1;
    uint nem2;
    uint nem3;

    uint nb01;
    uint nb02;
    uint nb03;
    uint nb11;
    uint nb12;
    uint nb13;
    uint nb21;
    uint nb22;
    uint nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    // Length of the always-visible cached prefix (= position_offset). The
    // host-built mask covers only the within-chunk [N × N] causal triangle;
    // KV columns < mask_kv_offset are unconditionally visible. (Repurposed
    // from the unused ALiBi `mask_n_head_log2` slot — keeps push ≤ 128 B.)
    uint  mask_kv_offset;
    float m0;
    float m1;

    uint gqa_ratio;
    uint split_kv;
    uint k_num;
};

[[vk::push_constant]] ConstantBuffer<FaParams> p;

// Q always F32.
[[vk::binding(0, 0)]] StructuredBuffer<Q_TYPE> data_q;

// K and V: type depends on cache dtype. The push-constant strides
// (nb01/nb11/nb21 etc.) stay in ELEMENTS, the same units regardless of
// dtype; for quants the dequant helpers below convert element-indices to
// (block-index, intra-block-index) by dividing by QUANT_K=32.
#if defined(DATA_KV_QUANT)
    // Quant variants: block-struct bindings.
    #if defined(DATA_A_Q4_0)
        [[vk::binding(1, 0)]] StructuredBuffer<block_q4_0> data_k;
        [[vk::binding(2, 0)]] StructuredBuffer<block_q4_0> data_v;
    #elif defined(DATA_A_Q4_1)
        [[vk::binding(1, 0)]] StructuredBuffer<block_q4_1> data_k;
        [[vk::binding(2, 0)]] StructuredBuffer<block_q4_1> data_v;
    #elif defined(DATA_A_Q5_0)
        [[vk::binding(1, 0)]] StructuredBuffer<block_q5_0> data_k;
        [[vk::binding(2, 0)]] StructuredBuffer<block_q5_0> data_v;
    #elif defined(DATA_A_Q5_1)
        [[vk::binding(1, 0)]] StructuredBuffer<block_q5_1> data_k;
        [[vk::binding(2, 0)]] StructuredBuffer<block_q5_1> data_v;
    #elif defined(DATA_A_Q8_0)
        [[vk::binding(1, 0)]] StructuredBuffer<block_q8_0> data_k;
        [[vk::binding(2, 0)]] StructuredBuffer<block_q8_0> data_v;
    #elif defined(DATA_A_IQ4_NL)
        [[vk::binding(1, 0)]] StructuredBuffer<block_iq4_nl> data_k;
        [[vk::binding(2, 0)]] StructuredBuffer<block_iq4_nl> data_v;
    #endif
#elif defined(DATA_KV_BF16)
    // BF16: raw uint16_t storage, expanded to F32 via bit shift on read.
    [[vk::binding(1, 0)]] StructuredBuffer<uint16_t> data_k;
    [[vk::binding(2, 0)]] StructuredBuffer<uint16_t> data_v;
#else
    // F32 or F16: KV_TYPE templated directly.
    [[vk::binding(1, 0)]] StructuredBuffer<KV_TYPE> data_k;
    [[vk::binding(2, 0)]] StructuredBuffer<KV_TYPE> data_v;
#endif

// Mask is always F32 across every variant.
[[vk::binding(3, 0)]] StructuredBuffer<float>      data_m;
// Split-K partials (binding 4): written when p.k_num > 1 — unnormalized O
// plus per-row running (L, M) — for flash_attn_split_k_reduce to merge.
// Declared unconditionally so the descriptor-set layout is identical on
// both the single-pass and split-K paths.
[[vk::binding(4, 0)]] RWStructuredBuffer<float>    data_o_split;
[[vk::binding(5, 0)]] RWStructuredBuffer<D_TYPE>   data_o;

// Per-forward dynamic params (kv_len, k_num, blocks_per_split). These
// change every decode token; host-writes the buffer between submits
// when the recorded cmdbuf is replayed. See src/inference/decode_dyn.rs.
#include "decode_dyn.slang"
[[vk::binding(6, 0)]] StructuredBuffer<DecodeDyn>  data_dyn;

// Shared Q row (already premultiplied by scale).
groupshared float Q_sh[HSK];
// Per-block S values and probabilities.
groupshared float S_sh[Bc];

// Per-element scalar K/V read returning F32. Inlined per dtype.
float read_k_element(uint elem_idx) {
#if defined(DATA_KV_BF16)
    return asfloat(uint(data_k[elem_idx]) << 16);
#elif defined(DATA_A_Q8_0)
    const uint ib  = elem_idx / 32;
    const uint iqs = elem_idx % 32;
    return float(data_k[ib].d) * float(int(data_k[ib].qs[iqs]));
#elif defined(DATA_A_Q4_0)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint nibble;
    if (d < 16) { nibble = uint(data_k[ib].qs[d])      & 0xFu; }
    else        { nibble = uint(data_k[ib].qs[d - 16]) >> 4;   }
    return float(data_k[ib].d) * (float(nibble) - 8.0f);
#elif defined(DATA_A_Q4_1)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint nibble;
    if (d < 16) { nibble = uint(data_k[ib].qs[d])      & 0xFu; }
    else        { nibble = uint(data_k[ib].qs[d - 16]) >> 4;   }
    return float(data_k[ib].d) * float(nibble) + float(data_k[ib].m);
#elif defined(DATA_A_Q5_0)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint low_nibble;
    if (d < 16) { low_nibble = uint(data_k[ib].qs[d])      & 0xFu; }
    else        { low_nibble = uint(data_k[ib].qs[d - 16]) >> 4;   }
    const uint qh = uint(data_k[ib].qh[0]) | (uint(data_k[ib].qh[1]) << 16);
    const uint high_bit = (qh >> d) & 1u;
    return float(data_k[ib].d) * (float(low_nibble | (high_bit << 4)) - 16.0f);
#elif defined(DATA_A_Q5_1)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint low_nibble;
    if (d < 16) { low_nibble = uint(data_k[ib].qs[d])      & 0xFu; }
    else        { low_nibble = uint(data_k[ib].qs[d - 16]) >> 4;   }
    const uint high_bit = (data_k[ib].qh >> d) & 1u;
    return float(data_k[ib].d) * float(low_nibble | (high_bit << 4)) + float(data_k[ib].m);
#elif defined(DATA_A_IQ4_NL)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint nibble;
    if (d < 16) { nibble = uint(data_k[ib].qs[d])      & 0xFu; }
    else        { nibble = uint(data_k[ib].qs[d - 16]) >> 4;   }
    return float(data_k[ib].d) * float(kvalues_iq4nl[nibble]);
#else
    return float(data_k[elem_idx]);
#endif
}

float read_v_element(uint elem_idx) {
#if defined(DATA_KV_BF16)
    return asfloat(uint(data_v[elem_idx]) << 16);
#elif defined(DATA_A_Q8_0)
    const uint ib  = elem_idx / 32;
    const uint iqs = elem_idx % 32;
    return float(data_v[ib].d) * float(int(data_v[ib].qs[iqs]));
#elif defined(DATA_A_Q4_0)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint nibble;
    if (d < 16) { nibble = uint(data_v[ib].qs[d])      & 0xFu; }
    else        { nibble = uint(data_v[ib].qs[d - 16]) >> 4;   }
    return float(data_v[ib].d) * (float(nibble) - 8.0f);
#elif defined(DATA_A_Q4_1)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint nibble;
    if (d < 16) { nibble = uint(data_v[ib].qs[d])      & 0xFu; }
    else        { nibble = uint(data_v[ib].qs[d - 16]) >> 4;   }
    return float(data_v[ib].d) * float(nibble) + float(data_v[ib].m);
#elif defined(DATA_A_Q5_0)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint low_nibble;
    if (d < 16) { low_nibble = uint(data_v[ib].qs[d])      & 0xFu; }
    else        { low_nibble = uint(data_v[ib].qs[d - 16]) >> 4;   }
    const uint qh = uint(data_v[ib].qh[0]) | (uint(data_v[ib].qh[1]) << 16);
    const uint high_bit = (qh >> d) & 1u;
    return float(data_v[ib].d) * (float(low_nibble | (high_bit << 4)) - 16.0f);
#elif defined(DATA_A_Q5_1)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint low_nibble;
    if (d < 16) { low_nibble = uint(data_v[ib].qs[d])      & 0xFu; }
    else        { low_nibble = uint(data_v[ib].qs[d - 16]) >> 4;   }
    const uint high_bit = (data_v[ib].qh >> d) & 1u;
    return float(data_v[ib].d) * float(low_nibble | (high_bit << 4)) + float(data_v[ib].m);
#elif defined(DATA_A_IQ4_NL)
    const uint ib  = elem_idx / 32;
    const uint d   = elem_idx % 32;
    uint nibble;
    if (d < 16) { nibble = uint(data_v[ib].qs[d])      & 0xFu; }
    else        { nibble = uint(data_v[ib].qs[d - 16]) >> 4;   }
    return float(data_v[ib].d) * float(kvalues_iq4nl[nibble]);
#else
    return float(data_v[elem_idx]);
#endif
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
#ifdef NEEDS_INIT_IQ_SHMEM
    // IQ4_NL et al. need their kvalues_iq* lookup tables populated
    // cooperatively before the first read_*_element call.
    init_iq_shmem(uint3(WORKGROUP_SIZE, 1, 1));
#endif

    const uint tid    = gl_LocalInvocationID.x;
    const uint i_row  = gl_WorkGroupID.x;
    const uint iq3    = gl_WorkGroupID.z;
    // Per-forward dynamic params (kv_len, k_num, blocks_per_split)
    // override the corresponding push fields. The push struct still
    // carries the fields for binary compat with the dispatcher but the
    // values are stale at decode-replay time — read from `data_dyn`.
    //
    // `kv_len` is indexed by the batch element `iq3`: batched decode packs
    // one DecodeDyn entry per sequence so each query attends to its own
    // cache length. `k_num` / `blocks_per_split` stay uniform across the
    // batch (sized from the longest sequence; over-long splits run zero
    // iterations and the combine kernel weights them to ~0). With a single
    // sequence (iq3 == 0) this reads entry 0 exactly as before.
    const uint p_KV       = data_dyn[iq3].kv_len;
    const uint p_k_num    = data_dyn[0].k_num;
    const uint p_split_kv = data_dyn[0].blocks_per_split;
    // K/V slab index for this sequence — batched/continuous decode packs one
    // DecodeDyn entry per sequence, each naming the BatchKvCache slab its cache
    // lives in (so a gathered batch can read non-contiguous slabs). The query
    // and output stay packed by `iq3`; only the cache reads are slab-indexed.
    // Single-sequence decode sets slot 0 → identical to the old `iq3 * nb`.
    const uint kv_slot    = data_dyn[iq3].slot;
    // grid.y packs (head, split): head = y / k_num, split = y % k_num.
    // With k_num == 1 this collapses to head = y, split = 0 — identical to
    // the pre-split-K dispatch.
    const uint iq2    = gl_WorkGroupID.y / p_k_num;
    const uint split  = gl_WorkGroupID.y % p_k_num;

    if (i_row >= p.N) { return; }

    const uint rk2 = p.neq2 / p.nek2;
    const uint rk3 = p.neq3 / p.nek3;
    const uint rv2 = p.neq2 / p.nev2;
    const uint rv3 = p.neq3 / p.nev3;
    const uint ik3 = iq3 / rk3;
    const uint ik2 = iq2 / rk2;
    const uint iv3 = iq3 / rv3;
    const uint iv2 = iq2 / rv2;

    const uint q_off = iq3 * p.nb03 + iq2 * p.nb02 + i_row * p.nb01;

    for (uint d = tid; d < HSK; d += WORKGROUP_SIZE) {
        Q_sh[d] = float(data_q[q_off + d]) * p.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint HSV_PER_THREAD = HSV / WORKGROUP_SIZE;
    float Of[HSV_PER_THREAD];
    [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) { Of[d] = 0.0f; }
    float Lf = 0.0f;
    float Mf = -1.0e30f;

    // Mask is the within-chunk [N × N] causal triangle (column stride = p.N),
    // not the full [KV × N]: prefix columns (< mask_kv_offset) are always
    // visible and carry no mask entry. See `write_causal_mask` (host side).
    const uint m_off = (iq3 % p.nem3) * p.nem2 * p.nem1 * p.N
                     + (iq2 % p.nem2) * p.nem1 * p.N
                     + i_row * p.N;

    const uint num_blocks  = (p_KV + Bc - 1) / Bc;
    // This workgroup handles only its split's slice of the KV blocks.
    // k_num == 1 ⇒ split_kv == num_blocks ⇒ [0, num_blocks): full range.
    // A split past the end (block_start >= num_blocks) runs zero iterations
    // and leaves Mf=-1e30, Lf=0 — the combine kernel weights it to ~0.
    const uint block_start = split * p_split_kv;
    const uint block_end   = min(block_start + p_split_kv, num_blocks);
    for (uint j = block_start; j < block_end; ++j) {
        const uint kv_base = j * Bc;

        // 1) S[c] = dot(Q, K[kv_base+c])
        for (uint c0 = 0; c0 < Bc; c0 += WORKGROUP_SIZE) {
            const uint c = c0 + tid;
            float s = 0.0f;
            if (c < Bc && kv_base + c < p_KV) {
                const uint k_off = kv_slot * p.nb13 + ik2 * p.nb12 + (kv_base + c) * p.nb11;
                [unroll] for (uint d = 0; d < HSK; ++d) {
                    s += Q_sh[d] * read_k_element(k_off + d);
                }
                if (MASK_ENABLE != 0 && kv_base + c >= p.mask_kv_offset) {
                    s += data_m[m_off + (kv_base + c - p.mask_kv_offset)];
                }
            } else {
                s = -1.0e30f;
            }
            if (c < Bc) { S_sh[c] = s; }
        }
        GroupMemoryBarrierWithGroupSync();

        // 2) Row max.
        float M_new = Mf;
        [unroll] for (uint c = 0; c < Bc; ++c) {
            M_new = max(M_new, S_sh[c]);
        }

        // 3) Online softmax accumulate.
        const float m_scale = exp(Mf - M_new);
        Lf *= m_scale;
        [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) {
            Of[d] *= m_scale;
        }
        [unroll] for (uint c = 0; c < Bc; ++c) {
            if (kv_base + c >= p_KV) { break; }
            const float p_c = exp(S_sh[c] - M_new);
            Lf += p_c;
            const uint v_off = kv_slot * p.nb23 + iv2 * p.nb22 + (kv_base + c) * p.nb21;
            [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) {
                const uint d_global = d * WORKGROUP_SIZE + tid;
                Of[d] += p_c * read_v_element(v_off + d_global);
            }
        }
        Mf = M_new;
        GroupMemoryBarrierWithGroupSync();
    }

    if (p_k_num <= 1u) {
        // 4) Single pass: normalize + write final, [HSV, n_head, L] layout.
        const float inv_L = Lf == 0.0f ? 0.0f : 1.0f / Lf;
        const uint o_off = iq3 * p.ne2 * p.ne1 * HSV
                         + i_row * p.ne2 * HSV
                         + iq2 * HSV;
        [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) {
            const uint d_global = d * WORKGROUP_SIZE + tid;
            data_o[o_off + d_global] = D_TYPE(Of[d] * inv_L);
        }
    } else {
        // 4') Split-K: write this split's UNNORMALIZED O partial plus its
        // running (L, M) into the layout flash_attn_split_k_reduce decodes:
        //   O:   HSV*ne1*(split + k_num*(iq2 + ne2*iq3)) + HSV*i_row + d
        //   L/M: base + ne1*2*(k_num*(iq2 + ne2*iq3)) + i_row + split*ne1*2
        //        with base = HSV*ne1*ne2*ne3*k_num and M at +ne1 past L.
        // (L and M are identical across the workgroup's 32 threads, so only
        // tid==0 writes them.)
        const uint o_base = HSV * p.ne1 * (split + p_k_num * (iq2 + p.ne2 * iq3))
                          + HSV * i_row;
        [unroll] for (uint d = 0; d < HSV_PER_THREAD; ++d) {
            const uint d_global = d * WORKGROUP_SIZE + tid;
            data_o_split[o_base + d_global] = Of[d];
        }
        if (tid == 0u) {
            const uint lm_base = HSV * p.ne1 * p.ne2 * p.ne3 * p_k_num
                               + p.ne1 * 2u * (p_k_num * (iq2 + p.ne2 * iq3))
                               + i_row + split * (p.ne1 * 2u);
            data_o_split[lm_base]         = Lf;
            data_o_split[lm_base + p.ne1] = Mf;
        }
    }
}
