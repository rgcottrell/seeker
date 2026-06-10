// Flash-attention forward, cooperative-matrix QK^T, register-held O accumulator.
//
// Faithful re-port of llama.cpp's `flash_attn_cm1.comp` design (the fast coopmat
// FA that encodes the full-res vision tower in ~12 s on Strix Halo, vs seeker's
// old scalar ~22 s). The previous seeker cm1 was a half-port — single subgroup
// (32 threads), O accumulator in f32 groupshared (8 KB), scalar PV — which made
// it occupancy-bound and slower than the scalar path. This version:
//   * 128 threads / 4 subgroups (one WGP-friendly workgroup),
//   * O accumulator in REGISTERS (`Of[rows_per_thread][hsv_per_thread]`),
//   * coopmat 16×16 QK^T (4 subgroups each own MatBc=16 KV cols of Bc=64),
//   * online softmax with per-row subgroup reductions (row-group == subgroup, so
//     `subgroupMax/Add` reduce exactly over one query row's keys),
//   * PV: Phase 1a keeps a SCALAR P·V accumulate into register O (validates the
//     multi-subgroup + register-O + online-softmax structure); Phase 1b swaps it
//     for a coopmat PV staged through `pvsh` (matching llama).
//
// I/O contract matches seeker's existing cm1/scalar path (NOT llama's): one
// workgroup per (Br-row-tile, head, batch); GQA via K/V index broadcast; output
// in seeker's post-permute `[HSV, n_head, L]` layout. F16 K/V only.
//
// Out of scope (vs llama): MASK_OPT, ALiBi, sinks, logit_softcap, SHMEM_STAGING,
// quant dequant, split-K (added in Phase 2). MASK_ENABLE + KV bounds-check kept.

//@variants
//  f32_f16: Q_TYPE=float    KV_TYPE=float16_t    D_TYPE=float
//@end-variants

import glsl;
using namespace linalg;

#include "types.slang"

// ── Fixed tiling. WORKGROUP_SIZE / SUBGROUP_SIZE = ROW_SPLIT = number of
// subgroups, chosen so threads_per_rowgroup == SUBGROUP_SIZE: each query row's
// keys live in exactly one subgroup, so the online-softmax subgroupMax/Add are
// correct without a cross-subgroup reduction.
static const uint WORKGROUP_SIZE = 128;
static const uint SUBGROUP_SIZE  = 32;
static const uint ROW_SPLIT      = WORKGROUP_SIZE / SUBGROUP_SIZE; // 4 subgroups
static const uint Br             = 16;   // query rows per workgroup
static const uint Bc             = 64;   // KV cols per outer step (= ROW_SPLIT*16)
static const uint MatBr          = 16;   // coopmat tile
static const uint MatBc          = 16;

[SpecializationConstant] const uint HSK = 128;
[SpecializationConstant] const uint HSV = 128;
[SpecializationConstant] const uint MASK_ENABLE = 0;
[SpecializationConstant] const uint CLAMP = 0;   // bounds-check KV cols past p.KV

// Head sizes rounded up to a multiple of 16 for the coopmat tiles. Padded
// lanes are zero-filled in Q/K staging so they contribute 0 to QK^T.
static const uint HSK_pad = (HSK + 15) & ~15u;
static const uint HSV_pad = (HSV + 15) & ~15u;

static const uint THREADS_PER_ROWGROUP = WORKGROUP_SIZE / ROW_SPLIT;     // 32
static const uint ROWS_PER_THREAD      = Br / ROW_SPLIT;                 // 4
static const uint COLS_PER_ITER        = WORKGROUP_SIZE / ROW_SPLIT;     // 32
static const uint COLS_PER_THREAD      = Bc / COLS_PER_ITER;             // 2
// HSV columns handled per thread (strided by THREADS_PER_ROWGROUP).
static const uint HSV_PER_THREAD = (HSV + THREADS_PER_ROWGROUP - 1) / THREADS_PER_ROWGROUP;

static const float NEG_BIG = -1.0e30f;

// Online softmax runs in the log2 domain (Q pre-scaled by scale*log2(e),
// mask scaled on add) so every exp() is the cheaper exp2(). Split-K (L, M)
// partials are in the log2 domain too — matches flash_attn.slang and
// flash_attn_split_k_reduce.
static const float LOG2E = 1.44269504088896340736f;

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
    // Length of the always-visible cached prefix (= position_offset). KV columns
    // < this carry no mask entry (visible); columns >= this index the within-
    // chunk [N × N] mask at (col - mask_kv_offset). Matches the scalar path so
    // cm1 serves chunked prefill at any offset, not just the first ubatch.
    uint  mask_kv_offset;
    float m0;          // ALiBi slope (unused)
    uint swa_window;   // sliding-window span (repurposed ALiBi m1 slot); 0 = full causal
    uint gqa_ratio;
    uint split_kv;
    uint k_num;
};

[[vk::push_constant]] ConstantBuffer<FaParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<Q_TYPE>   data_q;
[[vk::binding(1, 0)]] StructuredBuffer<KV_TYPE>  data_k;
[[vk::binding(2, 0)]] StructuredBuffer<KV_TYPE>  data_v;
[[vk::binding(3, 0)]] StructuredBuffer<KV_TYPE>  data_m;
// Split-K partials (unnormalized O + per-row L,M), consumed by
// flash_attn_split_k_reduce. Same layout as the scalar path's binding 4.
[[vk::binding(4, 0)]] RWStructuredBuffer<float>  data_o_split;
[[vk::binding(5, 0)]] RWStructuredBuffer<D_TYPE> data_o;

// Q tile (Br × HSK_pad), premultiplied by scale; K tile (Bc × HSK_pad) per block.
// S^T scores (Bc × Br) from the coopmat QK^T. P (Bc × Br) for the PV step.
static const uint Q_STRIDE = HSK_pad;
static const uint S_STRIDE = Br;
static const uint P_STRIDE = Br;
// pvsh holds one HSV tile of the output: [Br qrows × HSV_PER_TILE hsv cols],
// each subgroup writing its 16-col chunk. Read back into the register O.
static const uint HSV_PER_TILE = ROW_SPLIT * MatBc;             // 64

// K and V are read coopmat-direct from global (no LDS staging) — the key to
// occupancy. Only Q (reused across all KV blocks), the S^T scores, the softmax
// P, and the PV staging tile live in groupshared (~11 KB total at head_dim 128).
groupshared float16_t Q_sh[Br * Q_STRIDE];
groupshared float     S_sh[Bc * S_STRIDE];
groupshared float16_t P_sh[Bc * P_STRIDE];
groupshared float16_t pvsh[Br * HSV_PER_TILE];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    const uint tid    = gl_LocalInvocationID.x;
    const uint sg     = tid / SUBGROUP_SIZE;   // subgroup id 0..3 (== row group)
    const uint lane   = tid % SUBGROUP_SIZE;   // 0..31 (== col_tid)
    const uint row_tid = sg;                   // ROW_SPLIT == num subgroups
    const uint col_tid = lane;

    const uint i_row0 = gl_WorkGroupID.x * Br;
    // Split-K packs (head, split) into grid.y = iq2 * k_num + split. With
    // k_num == 1 (single pass) this is just iq2 = grid.y, split = 0.
    const uint iq2    = gl_WorkGroupID.y / p.k_num;
    const uint split  = gl_WorkGroupID.y % p.k_num;
    const uint iq3    = gl_WorkGroupID.z;

    if (i_row0 >= p.N) { return; }

    // GQA / batch broadcast (seeker convention, same as the scalar path).
    const uint rk2 = p.neq2 / p.nek2;
    const uint rk3 = p.neq3 / p.nek3;
    const uint rv2 = p.neq2 / p.nev2;
    const uint rv3 = p.neq3 / p.nev3;
    const uint ik3 = iq3 / rk3;
    const uint ik2 = iq2 / rk2;
    const uint iv3 = iq3 / rv3;
    const uint iv2 = iq2 / rv2;

    // ── Load Q tile (Br × HSK) into shmem, scaled; zero the [HSK, HSK_pad) pad.
    for (uint i = tid; i < Br * HSK_pad; i += WORKGROUP_SIZE) {
        const uint r = i / HSK_pad;
        const uint d = i % HSK_pad;
        float16_t v = float16_t(0);
        const uint gr = i_row0 + r;
        if (d < HSK && gr < p.N) {
            const uint q_off = iq3 * p.nb03 + iq2 * p.nb02 + gr * p.nb01;
            v = float16_t(float(data_q[q_off + d]) * (p.scale * LOG2E));
        }
        Q_sh[r * Q_STRIDE + d] = v;
    }

    // ── Register state: O accumulator + online-softmax running max/sum, per row.
    float16_t Of[ROWS_PER_THREAD][HSV_PER_THREAD];
    [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
        [unroll] for (uint h = 0; h < HSV_PER_THREAD; ++h) { Of[r][h] = float16_t(0); }
    }
    float Lf[ROWS_PER_THREAD], Mf[ROWS_PER_THREAD];
    [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) { Lf[r] = 0.0f; Mf[r] = NEG_BIG; }

    // Mask is the within-chunk [N × N] causal triangle (column stride = p.N),
    // NOT the full [KV × N]: prefix columns (< mask_kv_offset) are always visible
    // and carry no mask entry. Mirrors the scalar path's `write_causal_mask`.
    const uint m_off_base = (iq3 % p.nem3) * p.nem2 * p.nem1 * p.N
                          + (iq2 % p.nem2) * p.nem1 * p.N;

    // KV block range for this split. `p.split_kv` is the blocks-per-split (host
    // sets k_num=1, split_kv=num_blocks for the single-pass case). Each split
    // walks only [start_block, end_block) so no workgroup walks the whole KV —
    // the per-dispatch watchdog cap that single-pass tripped past ~14k keys.
    uint num_blocks = (p.KV + Bc - 1) / Bc;
    // Causal upper bound: the host mask is the within-chunk causal triangle
    // (see m_off_base comment / write_causal_mask), so the largest row in this
    // tile sees no key past mask_kv_offset + (last row index). Clamp the walk
    // instead of computing fully-masked trailing blocks.
    if (MASK_ENABLE != 0) {
        const uint max_qpos = p.mask_kv_offset + min(i_row0 + Br, p.N) - 1u;
        num_blocks = min(num_blocks, max_qpos / Bc + 1u);
    }
    // Sliding-window lower bound: no row in this tile sees a key below
    // min_qpos - swa_window + 1 (same qpos formula as the per-element SWA
    // mask below). The split ranges are REBASED at this block so splits
    // partition the visible window instead of idling below it; with
    // swa_window == 0 this is 0 and the mapping is identical to plain
    // split-K. Splits past the clamped end run empty (combine-safe).
    uint swa_first_block = 0u;
    if (p.swa_window != 0u) {
        const uint min_qpos = p.mask_kv_offset + i_row0;
        if (min_qpos + 1u > p.swa_window) {
            swa_first_block = (min_qpos + 1u - p.swa_window) / Bc;
        }
    }
    const uint start_block = swa_first_block + split * p.split_kv;
    const uint end_block = min(num_blocks, start_block + p.split_kv);
    for (uint j = start_block; j < end_block; ++j) {
        const uint kv_base = j * Bc;
        GroupMemoryBarrierWithGroupSync();

        // ── QK^T → S^T (Bc × Br), reading K DIRECTLY from global (no LDS
        // staging — that was the occupancy killer). Subgroup `sg` owns KV cols
        // [sg*16, sg*16+16). matK[i][j] = K[t=kv_base+sg*16+i, d+j], RowMajor
        // stride nb11. HSK is 16-aligned (hd_pad = next mult of 32) so there are
        // no partial-d tiles. KV rows past p.KV in the last block read in-bounds
        // slack (the host pads K/V by Bc rows) and are excluded by CLAMP below.
        const uint k_off = iq3 * p.nb13 + ik2 * p.nb12;
        var matS = CoopMat<float, MemoryScope.Subgroup, MatBc, MatBr,
                           CoopMatMatrixUse::MatrixAccumulator>(0.0f);
        [unroll] for (uint d = 0; d < HSK_pad; d += 16) {
            const uint kbase = k_off + (kv_base + sg * MatBc) * p.nb11 + d;
            let matK = CoopMat<float16_t, MemoryScope.Subgroup, MatBc, 16,
                               CoopMatMatrixUse::MatrixA>
                .Load<CoopMatMatrixLayout::RowMajor>(data_k, kbase, p.nb11);
            let matQ = CoopMat<float16_t, MemoryScope.Subgroup, 16, MatBr,
                               CoopMatMatrixUse::MatrixB>
                .Load<CoopMatMatrixLayout::ColumnMajor>(Q_sh, d, Q_STRIDE);
            matS = coopMatMulAdd<float, false>(matK, matQ, matS);
        }
        matS.Store<CoopMatMatrixLayout::RowMajor>(S_sh, sg * MatBc * S_STRIDE, S_STRIDE);
        GroupMemoryBarrierWithGroupSync();

        // ── Optional causal/window mask add (within-chunk [N, N] layout, F16).
        // Columns < mask_kv_offset are the always-visible cached prefix (no mask
        // entry); columns >= it index the within-chunk mask at (col - offset).
        if (MASK_ENABLE != 0) {
            [unroll] for (uint ci = 0; ci < COLS_PER_THREAD; ++ci) {
                const uint c = ci * COLS_PER_ITER + col_tid;
                if (kv_base + c < p.KV && kv_base + c >= p.mask_kv_offset) {
                    [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
                        const uint row = row_tid * ROWS_PER_THREAD + r;
                        if (i_row0 + row < p.N) {
                            const uint m_off = m_off_base + (i_row0 + row) * p.N;
                            // Mask is a natural-log-domain bias (0 / -inf);
                            // scale into the log2 domain of the S scores.
                            S_sh[c * S_STRIDE + row] +=
                                float(data_m[m_off + (kv_base + c - p.mask_kv_offset)]) * LOG2E;
                        }
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // ── Sliding-window mask (analytical, independent of MASK_ENABLE): mask
        // keys outside [qpos-window+1, qpos] for query abs pos qpos.
        if (p.swa_window != 0u) {
            [unroll] for (uint ci = 0; ci < COLS_PER_THREAD; ++ci) {
                const uint c = ci * COLS_PER_ITER + col_tid;
                if (kv_base + c < p.KV) {
                    const uint kpos = kv_base + c;
                    [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
                        const uint row = row_tid * ROWS_PER_THREAD + r;
                        if (i_row0 + row < p.N) {
                            const uint qpos = p.mask_kv_offset + i_row0 + row;
                            if (kpos > qpos || qpos - kpos >= p.swa_window) {
                                S_sh[c * S_STRIDE + row] = NEG_BIG;
                            }
                        }
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // ── Online softmax: per row, max over this block's Bc cols (subgroup
        // reduction), update running M/L, rescale register O, build P.
        float eMf[ROWS_PER_THREAD];
        [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
            const uint row = row_tid * ROWS_PER_THREAD + r;
            float rowmax = NEG_BIG;
            [unroll] for (uint ci = 0; ci < COLS_PER_THREAD; ++ci) {
                const uint c = ci * COLS_PER_ITER + col_tid;
                if (CLAMP != 0 && kv_base + c >= p.KV) { continue; }
                rowmax = max(rowmax, S_sh[c * S_STRIDE + row]);
            }
            rowmax = WaveActiveMax(rowmax);          // reduce over the subgroup (= row)
            const float m_old = Mf[r];
            Mf[r] = max(rowmax, m_old);
            eMf[r] = exp2(m_old - Mf[r]);
            Lf[r] = eMf[r] * Lf[r];
        }

        // Rescale running O by e^(m_old - m_new).
        [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
            [unroll] for (uint h = 0; h < HSV_PER_THREAD; ++h) {
                Of[r][h] = float16_t(eMf[r]) * Of[r][h];
            }
        }

        // P[c, row] = exp(S - M[row]); accumulate per-lane partial L.
        [unroll] for (uint ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const uint c = ci * COLS_PER_ITER + col_tid;
            [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
                const uint row = row_tid * ROWS_PER_THREAD + r;
                float pc = 0.0f;
                if (!(CLAMP != 0 && kv_base + c >= p.KV)) {
                    pc = exp2(S_sh[c * S_STRIDE + row] - Mf[r]);
                    Lf[r] += pc;
                }
                P_sh[c * P_STRIDE + row] = float16_t(pc);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // ── PV (coopmat), reading V DIRECTLY from global (no LDS staging).
        // matPV[qrow][hsv] = Σ_c P[c,qrow]·V[c,hsv], accumulated over Bc/16 KV
        // chunks; each subgroup owns a 16-col HSV chunk of an HSV_PER_TILE(=64)-
        // wide tile. P[c≥KV]=0 (set in the softmax above), so OOB V rows in the
        // last block contribute 0 — bounds-safe given the host's Bc-row K/V pad.
        const uint v_off = iq3 * p.nb23 + iv2 * p.nb22;
        const uint num_hsv_tiles = (HSV + HSV_PER_TILE - 1) / HSV_PER_TILE;
        [unroll] for (uint hsv_tile = 0; hsv_tile < num_hsv_tiles; ++hsv_tile) {
            const uint hsv_offset = (hsv_tile * ROW_SPLIT + sg) * MatBc;
            if (hsv_offset < HSV) {
                var matPV = CoopMat<float16_t, MemoryScope.Subgroup, MatBr, MatBc,
                                    CoopMatMatrixUse::MatrixAccumulator>(float16_t(0));
                [unroll] for (uint bcc = 0; bcc < Bc / MatBc; ++bcc) {
                    // matP[qrow][c_local] = P[kv_chunk c_local, qrow]  (ColumnMajor)
                    let matP = CoopMat<float16_t, MemoryScope.Subgroup, MatBr, 16,
                                       CoopMatMatrixUse::MatrixA>
                        .Load<CoopMatMatrixLayout::ColumnMajor>(P_sh, bcc * MatBc * P_STRIDE, P_STRIDE);
                    // matV[c_local][hsv_local] = V[kv_chunk c_local, hsv_offset+hsv_local]
                    const uint vbase = v_off + (kv_base + bcc * MatBc) * p.nb21 + hsv_offset;
                    let matV = CoopMat<float16_t, MemoryScope.Subgroup, 16, MatBc,
                                       CoopMatMatrixUse::MatrixB>
                        .Load<CoopMatMatrixLayout::RowMajor>(data_v, vbase, p.nb21);
                    matPV = coopMatMulAdd<float16_t, false>(matP, matV, matPV);
                }
                matPV.Store<CoopMatMatrixLayout::RowMajor>(pvsh, sg * MatBc, HSV_PER_TILE);
            }
            GroupMemoryBarrierWithGroupSync();

            // Accumulate this tile's pvsh into the register O (already rescaled).
            const uint tile_base = hsv_tile * HSV_PER_TILE;
            [unroll] for (uint h = 0; h < HSV_PER_THREAD; ++h) {
                const uint d = h * THREADS_PER_ROWGROUP + col_tid;
                if (d < HSV && d >= tile_base && d < tile_base + HSV_PER_TILE) {
                    const uint hsv_local = d - tile_base;
                    [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
                        const uint row = row_tid * ROWS_PER_THREAD + r;
                        Of[r][h] += pvsh[row * HSV_PER_TILE + hsv_local];
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Each lane accumulated only its own columns' P into Lf; the row's total L
    // is the sum over the subgroup's lanes (row-group == subgroup).
    [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
        Lf[r] = WaveActiveSum(Lf[r]);
    }

    if (p.k_num > 1) {
        // ── Split-K: store this split's UNNORMALIZED O + per-row (L, M) to the
        // partials buffer. `flash_attn_split_k_reduce` combines the k_num splits
        // and does the final 1/L. Layout matches the scalar path exactly.
        [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
            const uint row = row_tid * ROWS_PER_THREAD + r;
            const uint gr  = i_row0 + row;
            if (gr >= p.N) { continue; }
            const uint o_base = HSV * p.ne1 * (split + p.k_num * (iq2 + p.ne2 * iq3))
                              + HSV * gr;
            [unroll] for (uint h = 0; h < HSV_PER_THREAD; ++h) {
                const uint d = h * THREADS_PER_ROWGROUP + col_tid;
                if (d >= HSV) { continue; }
                data_o_split[o_base + d] = float(Of[r][h]);
            }
            if (col_tid == 0) {
                const uint lm_base = HSV * p.ne1 * p.ne2 * p.ne3 * p.k_num
                                   + p.ne1 * 2u * (p.k_num * (iq2 + p.ne2 * iq3))
                                   + gr + split * (p.ne1 * 2u);
                data_o_split[lm_base] = Lf[r];
                data_o_split[lm_base + p.ne1] = Mf[r];
            }
        }
        return;
    }

    // ── Single pass: normalize (divide by L) and write back in seeker's
    // [HSV, n_head, L] layout. Each lane holds the rows {row_tid*RPT + r} and
    // the hsv cols {h*THREADS_PER_ROWGROUP + col_tid}.
    [unroll] for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
        const uint row = row_tid * ROWS_PER_THREAD + r;
        const uint gr  = i_row0 + row;
        if (gr >= p.N) { continue; }
        const float inv_L = Lf[r] == 0.0f ? 0.0f : 1.0f / Lf[r];
        const uint o_off = iq3 * p.ne2 * p.ne1 * HSV + gr * p.ne2 * HSV + iq2 * HSV;
        [unroll] for (uint h = 0; h < HSV_PER_THREAD; ++h) {
            const uint d = h * THREADS_PER_ROWGROUP + col_tid;
            if (d >= HSV) { continue; }
            data_o[o_off + d] = D_TYPE(float(Of[r][h]) * inv_L);
        }
    }
}
