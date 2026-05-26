// Flash-attention forward pass with cooperative-matrix-accelerated QK^T.
//
// Processes Br=16 query rows per workgroup using a single 16×16 CoopMat
// fragment for matS = matQ @ matK^T. Softmax + PV stay scalar (the latter
// because row-wise rescaling of an O accumulator that lives in CoopMat
// would require Store/Load round-trips through shmem; for the basic
// reference shader that overhead defeats the cm speedup).
//
// Out of scope (same as flash_attn): MMQ, MASK_OPT, ALiBi, sinks,
// split_K, logit_softcap. MASK_ENABLE supported.

//@variants
//  f32_f16: Q_TYPE=float    KV_TYPE=float16_t    D_TYPE=float
//@end-variants

import glsl;

using namespace linalg;

#include "types.slang"

static const uint Br             = 16;   // query rows per workgroup
static const uint Bc             = 16;   // KV cols per outer step
[SpecializationConstant] const uint HSK = 128;
[SpecializationConstant] const uint HSV = 128;
[SpecializationConstant] const uint MASK_ENABLE = 0;

static const uint TK = 16;
static const uint WORKGROUP_SIZE = 32;   // one subgroup

struct FaParams {
    uint N;
    uint KV;
    uint ne1; uint ne2; uint ne3;
    uint neq2; uint neq3;
    uint nek2; uint nek3;
    uint nev2; uint nev3;
    uint nem1; uint nem2; uint nem3;
    uint nb01; uint nb02; uint nb03;
    uint nb11; uint nb12; uint nb13;
    uint nb21; uint nb22; uint nb23;
    float scale;
    float max_bias;
    float logit_softcap;
    uint  mask_n_head_log2;
    float m0;
    float m1;
    uint gqa_ratio;
    uint split_kv;
    uint k_num;
};

[[vk::push_constant]] ConstantBuffer<FaParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<Q_TYPE>   data_q;
[[vk::binding(1, 0)]] StructuredBuffer<KV_TYPE>  data_k;
[[vk::binding(2, 0)]] StructuredBuffer<KV_TYPE>  data_v;
[[vk::binding(3, 0)]] StructuredBuffer<KV_TYPE>  data_m;
[[vk::binding(5, 0)]] RWStructuredBuffer<D_TYPE> data_o;

// Shared Q, K, P, S tiles. Pad inner dim to dodge bank conflicts.
static const uint Q_STRIDE = HSK + 4;
static const uint K_STRIDE = HSK + 4;
static const uint S_STRIDE = Bc + 4;
static const uint P_STRIDE = Bc + 4;

groupshared float16_t Q_sh[Br * Q_STRIDE];
groupshared float16_t K_sh[Bc * K_STRIDE];
groupshared float     S_sh[Br * S_STRIDE];
groupshared float16_t P_sh[Br * P_STRIDE];
groupshared float     O_sh[Br * HSV];
groupshared float     L_sh[Br];
groupshared float     M_sh[Br];
// Per-row rescale factor for this KV block: exp(m_old - m_new). Computed in the
// softmax pass and consumed by the O += P·V step so the running O is correctly
// re-scaled before the new contribution is added.
groupshared float     R_sh[Br];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid    = gl_LocalInvocationID.x;
    const uint i_row0 = gl_WorkGroupID.x * Br;
    const uint iq2    = gl_WorkGroupID.y;
    const uint iq3    = gl_WorkGroupID.z;

    if (i_row0 >= p.N) { return; }

    const uint rk2 = p.neq2 / p.nek2;
    const uint rk3 = p.neq3 / p.nek3;
    const uint rv2 = p.neq2 / p.nev2;
    const uint rv3 = p.neq3 / p.nev3;
    const uint ik3 = iq3 / rk3;
    const uint ik2 = iq2 / rk2;
    const uint iv3 = iq3 / rv3;
    const uint iv2 = iq2 / rv2;

    // Load Q tile (Br × HSK) into shmem, premultiplied by p.scale.
    for (uint i = tid; i < Br * HSK; i += WORKGROUP_SIZE) {
        const uint r = i / HSK;
        const uint d = i % HSK;
        const uint gr = i_row0 + r;
        float16_t v = float16_t(0);
        if (gr < p.N) {
            const uint q_off = iq3 * p.nb03 + iq2 * p.nb02 + gr * p.nb01;
            v = float16_t(float(data_q[q_off + d]) * p.scale);
        }
        Q_sh[r * Q_STRIDE + d] = v;
    }

    // Initialize per-row state.
    for (uint r = tid; r < Br; r += WORKGROUP_SIZE) {
        L_sh[r] = 0.0f;
        M_sh[r] = -1.0e30f;
    }
    // Initialize O = 0 (Br × HSV).
    for (uint i = tid; i < Br * HSV; i += WORKGROUP_SIZE) {
        O_sh[i] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint m_off_base = (iq3 % p.nem3) * p.nem2 * p.nem1 * p.KV
                          + (iq2 % p.nem2) * p.nem1 * p.KV;

    const uint num_blocks = (p.KV + Bc - 1) / Bc;
    for (uint j = 0; j < num_blocks; ++j) {
        const uint kv_base = j * Bc;

        // ── 1) Stage K tile (Bc × HSK) into K_sh.
        for (uint i = tid; i < Bc * HSK; i += WORKGROUP_SIZE) {
            const uint c = i / HSK;
            const uint d = i % HSK;
            float16_t v = float16_t(0);
            if (kv_base + c < p.KV) {
                const uint k_off = iq3 * p.nb13 + ik2 * p.nb12 + (kv_base + c) * p.nb11;
                v = data_k[k_off + d];
            }
            K_sh[c * K_STRIDE + d] = v;
        }
        GroupMemoryBarrierWithGroupSync();

        // ── 2) matS (Br × Bc) = matQ (Br × HSK) @ matK^T (HSK × Bc) via CoopMat.
        var matS = CoopMat<float, MemoryScope.Subgroup, Br, Bc,
                           CoopMatMatrixUse::MatrixAccumulator>(0.0f);
        [unroll] for (uint kk = 0; kk < HSK; kk += TK) {
            // Q is row-major (Br × HSK), K is row-major (Bc × HSK); we want K^T,
            // so loading K with ColumnMajor layout from the (Bc × HSK) shmem with
            // stride HSK+4 effectively yields the (HSK × Bc) K^T view.
            //
            // NOTE: Slang's `CoopMat::Load<>(...)` is a static method that
            // RETURNS the loaded matrix. The instance-method form
            // (`var m = CoopMat(...)(0); m.Load(...)`) emits a Load whose
            // result is silently discarded — verified via SPIR-V disasm
            // when porting mul_mm_cm.slang.
            let matQ = CoopMat<float16_t, MemoryScope.Subgroup, Br, TK,
                               CoopMatMatrixUse::MatrixA>
                .Load<CoopMatMatrixLayout::RowMajor>(Q_sh, kk, Q_STRIDE);
            let matK = CoopMat<float16_t, MemoryScope.Subgroup, TK, Bc,
                               CoopMatMatrixUse::MatrixB>
                .Load<CoopMatMatrixLayout::ColumnMajor>(K_sh, kk, K_STRIDE);
            matS = coopMatMulAdd<float, false>(matQ, matK, matS);
        }
        matS.Store<CoopMatMatrixLayout::RowMajor>(S_sh, 0, S_STRIDE);
        GroupMemoryBarrierWithGroupSync();

        // ── 3) Per-row softmax + rescale of O.
        if (tid < Br) {
            const uint r = tid;
            // Optional mask add.
            if (MASK_ENABLE != 0 && i_row0 + r < p.N) {
                const uint m_off = m_off_base + (i_row0 + r) * p.KV;
                [unroll] for (uint c = 0; c < Bc; ++c) {
                    if (kv_base + c < p.KV) {
                        S_sh[r * S_STRIDE + c] += float(data_m[m_off + kv_base + c]);
                    }
                }
            }
            // Mask out-of-range KV cols with -inf so they don't affect max/sum.
            [unroll] for (uint c = 0; c < Bc; ++c) {
                if (kv_base + c >= p.KV) { S_sh[r * S_STRIDE + c] = -1.0e30f; }
            }

            // m_new = max(m_old, max(S[r, :]))
            float m_new = M_sh[r];
            [unroll] for (uint c = 0; c < Bc; ++c) {
                m_new = max(m_new, S_sh[r * S_STRIDE + c]);
            }
            const float rescale = exp(M_sh[r] - m_new);

            // P[r,c] = exp(S[r,c] - m_new)
            float row_sum = 0.0f;
            [unroll] for (uint c = 0; c < Bc; ++c) {
                const float p_c = exp(S_sh[r * S_STRIDE + c] - m_new);
                P_sh[r * P_STRIDE + c] = float16_t(p_c);
                row_sum += p_c;
            }

            L_sh[r] = rescale * L_sh[r] + row_sum;
            M_sh[r] = m_new;
            R_sh[r] = rescale;
        }
        GroupMemoryBarrierWithGroupSync();

        // ── 4) Rescale running O and accumulate new contribution.
        // For each (r, d) in Br × HSV: O[r,d] = R_sh[r] * O[r,d] + sum_c P[r,c] * V[kv+c, d].
        for (uint i = tid; i < Br * HSV; i += WORKGROUP_SIZE) {
            const uint r = i / HSV;
            const uint d = i % HSV;
            if (i_row0 + r >= p.N) { continue; }

            float acc = 0.0f;
            [unroll] for (uint c = 0; c < Bc; ++c) {
                if (kv_base + c >= p.KV) { break; }
                const uint v_off = iq3 * p.nb23 + iv2 * p.nb22 + (kv_base + c) * p.nb21;
                acc += float(P_sh[r * P_STRIDE + c]) * float(data_v[v_off + d]);
            }
            O_sh[r * HSV + d] = R_sh[r] * O_sh[r * HSV + d] + acc;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // ── 6) Normalize and write back into seeker's `[HSV, n_head, L]`
    // post-permute layout (matches `flash_attn.slang`'s output store,
    // which is `[hidden, L]` contiguous so the next matmul reads it
    // directly without an extra permute). NOT llama.cpp's pre-permute
    // `[HSV, L, n_head]` — the indices of `gr` (= L) and `iq2` (= head)
    // are swapped from the obvious natural order.
    for (uint i = tid; i < Br * HSV; i += WORKGROUP_SIZE) {
        const uint r = i / HSV;
        const uint d = i % HSV;
        const uint gr = i_row0 + r;
        if (gr >= p.N) { continue; }
        const float inv_L = L_sh[r] == 0.0f ? 0.0f : 1.0f / L_sh[r];
        const uint o_off = iq3 * p.ne2 * p.ne1 * HSV
                         + gr * p.ne2 * HSV
                         + iq2 * HSV;
        data_o[o_off + d] = D_TYPE(O_sh[r * HSV + d] * inv_L);
    }
}
