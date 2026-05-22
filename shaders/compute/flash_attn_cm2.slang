// NV cooperative_matrix2 flash-attention forward pass.
//
// Mirrors llama.cpp's flash_attn_cm2.comp structure but uses Slang's
// `linalg::CoopMat` cm2 surface end-to-end:
//   - `TensorLayout<2, CoopMatClampMode.Constant>` for hardware-clamped Q/K/V loads
//   - `MapElement` lambdas for scale, mask, exp, output store
//   - `ReduceRow<N>` for row max (proper combiner) and row broadcast (identity combiner)
//   - `convertUse<MatrixA>()` to flip the softmax result from Accumulator to A operand
//
// First-cut scope: F32 Q × F16 K/V → F32 O. MASK_ENABLE supported. Out of scope
// (drop until host bench actually needs them): MASK_OPT, ALiBi, sinks, split_K,
// logit_softcap, quantized K/V decode.

// NOTE: Q is bound as F16 (StructuredBuffer<float16_t>) because Slang's cm2
// TensorLayout Load requires buffer-element type == CoopMat-element type. The
// host pre-converts Q to F16 before binding; scale is applied via MapElement.
//@variants
//  f16: Q_TYPE=float16_t   KV_TYPE=float16_t    D_TYPE=float
//@end-variants

using namespace linalg;

#include "types.slang"

// CoopMat dims must be compile-time constants. BR / BC / HSK / HSV baked in here.
static const uint BR  = 16;
static const uint BC  = 16;
static const uint HSK = 128;
static const uint HSV = 128;
static const uint BK  = 16;

static const uint WORKGROUP_SIZE = 32;

[SpecializationConstant] const uint MASK_ENABLE = 0;

struct FaCm2Params {
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

[[vk::push_constant]] ConstantBuffer<FaCm2Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<Q_TYPE>   data_q;
[[vk::binding(1, 0)]] StructuredBuffer<KV_TYPE>  data_k;
[[vk::binding(2, 0)]] StructuredBuffer<KV_TYPE>  data_v;
[[vk::binding(3, 0)]] StructuredBuffer<KV_TYPE>  data_m;
[[vk::binding(5, 0)]] RWStructuredBuffer<D_TYPE> data_o;

static const float NEG_INF = -3.402823466e+38f;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID) {
    const uint i_row0 = gl_WorkGroupID.x * BR;
    const uint iq2    = gl_WorkGroupID.y;
    const uint iq3    = gl_WorkGroupID.z;

    if (i_row0 >= p.N) { return; }

    // K/V GQA broadcasting.
    const uint rk2 = p.neq2 / p.nek2;
    const uint rk3 = p.neq3 / p.nek3;
    const uint rv2 = p.neq2 / p.nev2;
    const uint rv3 = p.neq3 / p.nev3;
    const uint ik3 = iq3 / rk3;
    const uint ik2 = iq2 / rk2;
    const uint iv3 = iq3 / rv3;
    const uint iv2 = iq2 / rv2;

    // ── Tensor layouts for each tensor's full shape; we'll slice the per-tile
    //    region inside the loop. nb*1 strides are in elements (not bytes).
    TensorLayout<2, CoopMatClampMode.Constant> tlQ;
    TensorLayout<2, CoopMatClampMode.Constant> tlK;
    TensorLayout<2, CoopMatClampMode.Constant> tlV;
    TensorLayout<2, CoopMatClampMode.Constant> tlM;
    let tlQ1 = tlQ.Dimension(p.N,  HSK).Stride(p.nb01, 1);
    let tlK1 = tlK.Dimension(p.KV, HSK).Stride(p.nb11, 1);
    let tlV1 = tlV.Dimension(p.KV, HSV).Stride(p.nb21, 1);
    let tlM1 = tlM.Dimension(p.N,  p.KV).Stride(p.KV,  1);
    TensorView<2, false, 1, 0> tvTr;

    const uint q_base = iq3 * p.nb03 + iq2 * p.nb02;
    const uint k_base = iq3 * p.nb13 + ik2 * p.nb12;
    const uint v_base = iq3 * p.nb23 + iv2 * p.nb22;
    const uint m_base = (iq3 % p.nem3) * p.nem2 * p.nem1 * p.KV
                      + (iq2 % p.nem2) * p.nem1 * p.KV;

    // Running per-row m and L, stored across a Br × BC matrix (broadcast columns).
    var mPrev = CoopMat<float, MemoryScope.Subgroup, BR, BC,
                       CoopMatMatrixUse.MatrixAccumulator>(NEG_INF);
    var lPrev = CoopMat<float, MemoryScope.Subgroup, BR, BC,
                       CoopMatMatrixUse.MatrixAccumulator>(0.0f);
    var oAcc  = CoopMat<float, MemoryScope.Subgroup, BR, HSV,
                       CoopMatMatrixUse.MatrixAccumulator>(0.0f);

    const float scale = p.scale;
    const uint i_row0_local = i_row0;
    const uint num_kv_blocks = (p.KV + BC - 1) / BC;

    for (uint j = 0; j < num_kv_blocks; ++j) {
        const uint kv_base = j * BC;

        // ── Compute matS = matQ @ matK^T.
        var matS = CoopMat<float, MemoryScope.Subgroup, BR, BC,
                           CoopMatMatrixUse.MatrixAccumulator>(0.0f);
        [unroll] for (uint kk = 0; kk < HSK; kk += BK) {
            let qSlice = tlQ1.Slice(i_row0,  BR, kk, BK);
            let kSlice = tlK1.Slice(kv_base, BC, kk, BK);
            let matA = CoopMat<float16_t, MemoryScope.Subgroup, BR, BK,
                               CoopMatMatrixUse.MatrixA>.Load(data_q, q_base, qSlice);
            let matB = CoopMat<float16_t, MemoryScope.Subgroup, BK, BC,
                               CoopMatMatrixUse.MatrixB>.Load(data_k, k_base, kSlice, tvTr);
            matS = coopMatMulAdd<float, false>(matA, matB, matS);
        }

        // ── Scale + mask: per-element. Out-of-range positions become NEG_INF so
        //    they're inert in the upcoming max/sum.
        const uint kv_base_local = kv_base;
        const uint KV_local = p.KV;
        const uint N_local  = p.N;
        if (MASK_ENABLE != 0) {
            // Load mask as F16 (host layout), then widen to F32 via conversion-ctor
            // so it can be added to the F32 matS.
            let matMaskH = CoopMat<float16_t, MemoryScope.Subgroup, BR, BC,
                                   CoopMatMatrixUse.MatrixAccumulator>
                .Load(data_m, m_base, tlM1.Slice(i_row0, BR, kv_base, BC));
            let matMask = CoopMat<float, MemoryScope.Subgroup, BR, BC,
                                  CoopMatMatrixUse.MatrixAccumulator>(matMaskH);
            matS = (matS * scale + matMask).MapElement((uint r, uint c, float elem) => {
                const uint row = i_row0_local + r;
                const uint col = kv_base_local + c;
                return (row < N_local && col < KV_local) ? elem : NEG_INF;
            });
        } else {
            matS = matS.MapElement((uint r, uint c, float elem) => {
                const uint row = i_row0_local + r;
                const uint col = kv_base_local + c;
                return (row < N_local && col < KV_local) ? elem * scale : NEG_INF;
            });
        }

        // ── Row max + online softmax updates.
        let rowMaxBC = matS.ReduceRow<BC>((float a, float b) => max(a, b));
        let mNew     = makeTuple(rowMaxBC, mPrev).MapElement(
                          (uint r, uint c, float a, float b) => max(a, b));
        let scalePrev = makeTuple(mPrev,    mNew).MapElement(
                          (uint r, uint c, float a, float b) => exp(a - b));
        let scaleTile = makeTuple(rowMaxBC, mNew).MapElement(
                          (uint r, uint c, float a, float b) => exp(a - b));
        mPrev = mNew;

        matS = makeTuple(matS, rowMaxBC).MapElement(
            (uint r, uint c, float elem, float rmax) => {
                if (rmax <= NEG_INF * 0.5f) { return 0.0f; }
                return exp(elem - rmax);
            });

        let rowSum = matS.ReduceRow<BC>((float a, float b) => a + b);
        lPrev = lPrev * scalePrev + rowSum * scaleTile;

        // ── Broadcast (Br × Bc) per-row rescales to (Br × HSV) for matO updates.
        let scalePrevD = scalePrev.ReduceRow<HSV>((float a, float b) => a);
        let scaleTileD = scaleTile.ReduceRow<HSV>((float a, float b) => a);

        // ── matO_new = matP @ matV; matO = oAcc * scalePrevD + matO_new * scaleTileD.
        let matP = CoopMat<float16_t, MemoryScope.Subgroup, BR, BC,
                           CoopMatMatrixUse.MatrixAccumulator>(matS)
                       .convertUse<CoopMatMatrixUse.MatrixA>();

        let vSlice = tlV1.Slice(kv_base, BC, 0, HSV);
        let matV   = CoopMat<float16_t, MemoryScope.Subgroup, BC, HSV,
                             CoopMatMatrixUse.MatrixB>.Load(data_v, v_base, vSlice);

        var matO_new = CoopMat<float, MemoryScope.Subgroup, BR, HSV,
                               CoopMatMatrixUse.MatrixAccumulator>(0.0f);
        matO_new = coopMatMulAdd<float, false>(matP, matV, matO_new);
        oAcc = oAcc * scalePrevD + matO_new * scaleTileD;
    }

    // ── Normalize + store. lPrev is (Br × Bc); broadcast its first column to HSV
    //    cols and divide.
    let rowL = lPrev.ReduceRow<HSV>((float a, float b) => a);
    const uint i_row0_final = i_row0;
    const uint iq2_final    = iq2;
    const uint iq3_final    = iq3;
    const uint N_final      = p.N;
    const uint ne1_local    = p.ne1;
    const uint ne2_local    = p.ne2;

    let oOut = makeTuple(oAcc, rowL).MapElement(
        (uint r, uint c, float elem, float l) => {
            const uint row = i_row0_final + r;
            const uint d   = c;
            if (row < N_final && d < HSV) {
                const float inv = (l == 0.0f) ? 0.0f : (1.0f / l);
                const uint o_off = iq3_final * ne2_local * ne1_local * HSV
                                 + iq2_final * ne1_local * HSV
                                 + row * HSV;
                data_o[o_off + d] = D_TYPE(elem * inv);
            }
            return elem;
        });
}
