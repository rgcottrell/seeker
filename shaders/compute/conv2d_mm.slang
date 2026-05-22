// Ported from llama.cpp ggml-vulkan conv2d_mm.comp (non-COOPMAT2 path), MIT license — see LICENSE.llama-cpp
//
// Conv2D via a blocked GEMM: kernel is (K × CRS), input is (CRS × NPQ). Each workgroup
// owns one BS_K × BS_NPQ output tile and walks the CRS dimension in BS_CRS-wide steps,
// staging A and B tiles through shmem. The cooperative-matrix fast path lives in the
// upstream COOPMAT2 branch and is deferred until Slang ships cm support.

//@variants
//  f32_unroll:         A_TYPE=float       B_TYPE=float    D_TYPE=float    USE_COLLECTIVES=1    UNROLL_LOOPS=1
//  f32:                A_TYPE=float       B_TYPE=float    D_TYPE=float    USE_COLLECTIVES=1
//  f16_f32_unroll:     A_TYPE=float16_t   B_TYPE=float    D_TYPE=float    USE_COLLECTIVES=1    UNROLL_LOOPS=1
//  f16_f32:            A_TYPE=float16_t   B_TYPE=float    D_TYPE=float    USE_COLLECTIVES=1
//  transpose_f32_unroll:     TRANSPOSE=1  A_TYPE=float       B_TYPE=float    D_TYPE=float    USE_COLLECTIVES=1    UNROLL_LOOPS=1
//  transpose_f32:            TRANSPOSE=1  A_TYPE=float       B_TYPE=float    D_TYPE=float    USE_COLLECTIVES=1
//  transpose_f16_f32_unroll: TRANSPOSE=1  A_TYPE=float16_t   B_TYPE=float    D_TYPE=float    USE_COLLECTIVES=1    UNROLL_LOOPS=1
//  transpose_f16_f32:        TRANSPOSE=1  A_TYPE=float16_t   B_TYPE=float    D_TYPE=float    USE_COLLECTIVES=1
//@end-variants

import glsl;  // gl_SubgroupInvocationID

#include "types.slang"

#ifdef UNROLL_LOOPS
#define UNROLL_ATTR [unroll]
#else
#define UNROLL_ATTR
#endif

struct Conv2dMmParams {
    uint Cout;
    uint Cin;
    uint N;
    uint W;
    uint H;
    uint OW;
    uint OH;
    uint nb01;
    uint nb02;
    uint nb03;
    uint nb11;
    uint nb12;
    uint nb13;
    uint nb1;
    uint nb2;
    uint nb3;
    uint OWmp;   uint OWL;
    uint OWOHmp; uint OWOHL;
};

[[vk::push_constant]] ConstantBuffer<Conv2dMmParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   knl_data;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   src_data;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> dst_data;

[SpecializationConstant] const uint BLOCK_SIZE       = 256;
[SpecializationConstant] const uint BS_K             = 128;
[SpecializationConstant] const uint BS_CRS           = 16;
[SpecializationConstant] const uint BS_NPQ           = 128;
[SpecializationConstant] const uint TS_K             = 8;
[SpecializationConstant] const uint use_collectives  = 1;
[SpecializationConstant] const uint SHMEM_PAD        = 4;
[SpecializationConstant] const uint s0               = 1;
[SpecializationConstant] const uint s1               = 1;
[SpecializationConstant] const uint p0_              = 0;
[SpecializationConstant] const uint p1_              = 0;
[SpecializationConstant] const uint d0               = 1;
[SpecializationConstant] const uint d1               = 1;
[SpecializationConstant] const uint KW               = 1;
[SpecializationConstant] const uint KH               = 1;

static const uint Ash_stride = BS_CRS + SHMEM_PAD;
static const uint Bsh_stride = BS_NPQ + SHMEM_PAD;
static const uint Ash_len    = BS_K * Ash_stride;
static const uint Bsh_len    = BS_CRS * Bsh_stride;

groupshared float Ash[Ash_len];
groupshared float Bsh[Bsh_len];

uint splitWork(uint work_size, uint block_size) {
    return (block_size + work_size - 1) / block_size;
}

uint mulhi_u32(uint a, uint b) { return uint((uint64_t(a) * uint64_t(b)) >> 32); }
uint fastdiv(uint n, uint mp, uint L) { return (mulhi_u32(n, mp) + n) >> L; }

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid     = gl_LocalInvocationID.x;
    const uint WG_SIZE = BLOCK_SIZE;
    const uint TS_NPQ  = BS_K * BS_NPQ / WG_SIZE / TS_K;
    const uint NT_NPQ  = BS_NPQ / TS_NPQ;

    const uint K_total = p.Cout;
    const uint CRS     = p.Cin * KH * KW;
    const uint NPQ     = p.N * p.OH * p.OW;
    const uint NB_CRS  = splitWork(CRS, BS_CRS);

    const uint B_idx_K   = gl_WorkGroupID.x;
    const uint B_idx_NPQ = gl_WorkGroupID.y + gl_WorkGroupID.z * 512u;

    if (B_idx_NPQ * BS_NPQ >= NPQ) { return; }

    const uint T_y = tid / NT_NPQ;
    const uint T_x = tid % NT_NPQ;

    const uint Ar    = tid / BS_CRS;
    const uint Ac    = tid % BS_CRS;
    const uint ArpWg = WG_SIZE / BS_CRS;
    const uint Br    = tid / BS_NPQ;
    const uint Bc    = tid % BS_NPQ;
    const uint BrpWg = WG_SIZE / BS_NPQ;

    float regC[TS_K][BS_K * BS_NPQ / 256 / TS_K];  // TS_K × TS_NPQ
    for (uint T_ly = 0; T_ly < TS_K; T_ly++) {
        for (uint T_lx = 0; T_lx < TS_NPQ; T_lx++) {
            regC[T_ly][T_lx] = 0.0f;
        }
    }

    // Walk the CRS dim block by block.
    for (uint B_idx_CRS = 0; B_idx_CRS < NB_CRS; B_idx_CRS++) {
        uint CRS_idx_a, Cin_idx_a, KH_idx_a, KW_idx_a;
        uint cached_CRS_idx = 0u, cached_Cin_idx = 0u, cached_KH_idx = 0u, cached_KW_idx = 0u;
#ifdef USE_COLLECTIVES
        if (use_collectives == 1u) {
            cached_CRS_idx                = B_idx_CRS * BS_CRS + gl_SubgroupInvocationID;
            cached_Cin_idx                = cached_CRS_idx / (KW * KH);
            const uint cached_CRS_remainder = cached_CRS_idx % (KW * KH);
            cached_KH_idx                 = cached_CRS_remainder / KW;
            cached_KW_idx                 = cached_CRS_remainder % KW;

            CRS_idx_a = WaveReadLaneAt(cached_CRS_idx, Ac);
            Cin_idx_a = WaveReadLaneAt(cached_Cin_idx, Ac);
            KH_idx_a  = WaveReadLaneAt(cached_KH_idx,  Ac);
            KW_idx_a  = WaveReadLaneAt(cached_KW_idx,  Ac);
        } else
#endif
        {
            CRS_idx_a              = B_idx_CRS * BS_CRS + Ac;
            Cin_idx_a              = CRS_idx_a / (KW * KH);
            const uint CRS_rem_a   = CRS_idx_a % (KW * KH);
            KH_idx_a               = CRS_rem_a / KW;
            KW_idx_a               = CRS_rem_a % KW;
        }

        // Stage kernel tile (BS_K × BS_CRS) into Ash.
        UNROLL_ATTR for (uint r_offset = 0; r_offset < BS_K; r_offset += ArpWg) {
            const uint B_ly = r_offset + Ar;
            const uint B_lx = Ac;
            const uint K_idx = B_idx_K * BS_K + B_ly;
#ifdef TRANSPOSE
            const uint knl_idx = min(KW_idx_a + KH_idx_a * p.nb01 + K_idx * p.nb02 + Cin_idx_a * p.nb03, K_total * CRS - 1u);
#else
            const uint knl_idx = min(KW_idx_a + KH_idx_a * p.nb01 + Cin_idx_a * p.nb02 + K_idx * p.nb03, K_total * CRS - 1u);
#endif
            float val = float(knl_data[knl_idx]);
            if (K_idx >= K_total || CRS_idx_a >= CRS) { val = 0.0f; }
            Ash[B_ly * Ash_stride + B_lx] = val;
        }

        // Stage input tile (BS_CRS × BS_NPQ) into Bsh.
        UNROLL_ATTR for (uint r_offset = 0; r_offset < BS_CRS; r_offset += BrpWg) {
            const uint B_ly  = r_offset + Br;
            const uint B_lx  = Bc;
            const uint NPQ_idx = B_idx_NPQ * BS_NPQ + B_lx;
            const uint N_idx   = fastdiv(NPQ_idx, p.OWOHmp, p.OWOHL);
            const uint NPQ_rem = NPQ_idx - N_idx * p.OH * p.OW;
            const uint OH_idx  = fastdiv(NPQ_rem, p.OWmp, p.OWL);
            const uint OW_idx  = NPQ_rem - OH_idx * p.OW;

            uint CRS_idx_b, Cin_idx_b, KH_idx_b, KW_idx_b;
#ifdef USE_COLLECTIVES
            if (use_collectives == 1u) {
                CRS_idx_b = WaveReadLaneAt(cached_CRS_idx, r_offset + Br);
                Cin_idx_b = WaveReadLaneAt(cached_Cin_idx, r_offset + Br);
                KH_idx_b  = WaveReadLaneAt(cached_KH_idx,  r_offset + Br);
                KW_idx_b  = WaveReadLaneAt(cached_KW_idx,  r_offset + Br);
            } else
#endif
            {
                CRS_idx_b                = B_idx_CRS * BS_CRS + B_ly;
                Cin_idx_b                = CRS_idx_b / (KW * KH);
                const uint CRS_rem_b     = CRS_idx_b % (KW * KH);
                KH_idx_b                 = CRS_rem_b / KW;
                KW_idx_b                 = CRS_rem_b % KW;
            }

#ifdef TRANSPOSE
            const uint H_idx_x_s1 = OH_idx - KH_idx_b * d1 + p1_;
            const uint W_idx_x_s0 = OW_idx - KW_idx_b * d0 + p0_;
            const uint H_idx = H_idx_x_s1 / s1;
            const uint W_idx = W_idx_x_s0 / s0;
#else
            const uint H_idx = OH_idx * s1 + KH_idx_b * d1 - p1_;
            const uint W_idx = OW_idx * s0 + KW_idx_b * d0 - p0_;
#endif
            const uint src_idx = min(W_idx + H_idx * p.nb11 + Cin_idx_b * p.nb12 + N_idx * p.nb13,
                                     p.Cin * p.N * p.W * p.H - 1u);
            float val = float(src_data[src_idx]);
            bool oob = (CRS_idx_b >= CRS || NPQ_idx >= NPQ || H_idx >= p.H || W_idx >= p.W);
#ifdef TRANSPOSE
            oob = oob || (H_idx_x_s1 - H_idx * s1 != 0u) || (W_idx_x_s0 - W_idx * s0 != 0u);
#endif
            if (oob) { val = 0.0f; }
            Bsh[B_ly * Bsh_stride + B_lx] = val;
        }
        GroupMemoryBarrierWithGroupSync();

        if (T_y * TS_K < K_total) {
            UNROLL_ATTR for (uint CRS_lidx = 0; CRS_lidx < BS_CRS; CRS_lidx++) {
                float regA[TS_K];
                float regB[BS_K * BS_NPQ / 256 / TS_K];
                for (uint T_ly = 0; T_ly < TS_K; T_ly++) {
                    regA[T_ly] = Ash[(T_y * TS_K + T_ly) * Ash_stride + CRS_lidx];
                }
                for (uint T_lx = 0; T_lx < TS_NPQ; T_lx++) {
                    regB[T_lx] = Bsh[CRS_lidx * Bsh_stride + T_x * TS_NPQ + T_lx];
                }
                for (uint T_ly = 0; T_ly < TS_K; T_ly++) {
                    for (uint T_lx = 0; T_lx < TS_NPQ; T_lx++) {
                        regC[T_ly][T_lx] = mad(regA[T_ly], regB[T_lx], regC[T_ly][T_lx]);
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (T_y * TS_K < K_total) {
        for (uint T_ly = 0; T_ly < TS_K; T_ly++) {
            for (uint T_lx = 0; T_lx < TS_NPQ; T_lx++) {
                const uint K_idx   = B_idx_K * BS_K + T_y * TS_K + T_ly;
                const uint NPQ_idx = B_idx_NPQ * BS_NPQ + T_x * TS_NPQ + T_lx;
                const uint N_idx   = fastdiv(NPQ_idx, p.OWOHmp, p.OWOHL);
                const uint OH_idx  = fastdiv(NPQ_idx - N_idx * p.OH * p.OW, p.OWmp, p.OWL);
                const uint OW_idx  = NPQ_idx - N_idx * p.OH * p.OW - OH_idx * p.OW;
                const uint dst_idx = OW_idx + OH_idx * p.nb1 + K_idx * p.nb2 + N_idx * p.nb3;
                if (K_idx < K_total && NPQ_idx < NPQ) {
                    dst_data[dst_idx] = D_TYPE(regC[T_ly][T_lx]);
                }
            }
        }
    }
}
