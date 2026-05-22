// Ported from llama.cpp ggml-vulkan, MIT license — see LICENSE.llama-cpp
//
// GLSL → Slang translation cheat sheet (applied throughout shaders/):
//   #include "foo.glsl"                        → #include "foo.slang"   (Slang supports the C-style preprocessor)
//   #extension GL_… : require                  → omitted; slangc infers
//   layout(local_size_x = N) in;               → [numthreads(N, 1, 1)] on the entry function
//   layout(binding = K) [readonly] buffer …    → [[vk::binding(K, 0)]] StructuredBuffer / RWStructuredBuffer
//   layout(push_constant) uniform …            → [[vk::push_constant]] ConstantBuffer<Params> p;
//   gl_GlobalInvocationID                      → uint3 parameter with SV_DispatchThreadID semantic
//   gl_NumWorkGroups / gl_WorkGroupSize        → `import glsl;` then use the names directly
//   subgroupAdd(x), subgroupMin/Max/Broadcast  → WaveActiveSum, WaveActiveMin/Max, WaveReadLaneAt
//   barrier()                                  → GroupMemoryBarrierWithGroupSync()
//   shared T arr[N];                           → groupshared T arr[N];
//   [[unroll]]                                 → [unroll]
//
// Type system:
//   - For elementwise/activation shaders, pass `-DA_TYPE=float -DD_TYPE=float -DFLOAT_TYPE=float`
//     (variant lines in the .slang source); the #ifndef fallbacks below default to float.
//   - For shaders that work with quantized GGML blocks, pass `-DDATA_A_Q4_K=1` (etc.);
//     the DATA_A_X → A_TYPE selection block below picks the right block struct.

#ifndef GGML_TYPES_SLANG
#define GGML_TYPES_SLANG

// ── Quantized block layouts ──────────────────────────────────────────────
// Field layouts match llama.cpp's types.glsl exactly so SPIR-V output is
// byte-compatible. Phase 4a covers legacy (Q4_0..Q8_1) and K-quants
// (Q2..Q6_K). I-quants and the SIMD-friendly packed16/packed32/packed128
// variants come in a later pass.

#define QUANT_K_Q4_0 32
#define QUANT_R_Q4_0 2
struct block_q4_0          { float16_t d;                  uint8_t  qs[16]; };
struct block_q4_0_packed16 { float16_t d;                  uint16_t qs[8];  };

#define QUANT_K_Q4_1 32
#define QUANT_R_Q4_1 2
struct block_q4_1          { float16_t d; float16_t m;     uint8_t  qs[16]; };
struct block_q4_1_packed16 { float16_t d; float16_t m;     uint16_t qs[8];  };

#define QUANT_K_Q5_0 32
#define QUANT_R_Q5_0 2
struct block_q5_0          { float16_t d; uint16_t qh[2];  uint8_t  qs[16]; };
struct block_q5_0_packed16 { float16_t d; uint16_t qh[2];  uint16_t qs[8];  };

#define QUANT_K_Q5_1 32
#define QUANT_R_Q5_1 2
struct block_q5_1          { float16_t d; float16_t m; uint qh; uint8_t  qs[16]; };
struct block_q5_1_packed16 { float16_t d; float16_t m; uint qh; uint16_t qs[8];  };

#define QUANT_K_Q8_0 32
#define QUANT_R_Q8_0 1
struct block_q8_0          { float16_t d; int8_t  qs[32]; };
struct block_q8_0_packed16 { float16_t d; int16_t qs[16]; };

#define QUANT_K_Q8_1 32
#define QUANT_R_Q8_1 1
struct block_q8_1            { float16_t2 ds;    int8_t  qs[32]; };
struct block_q8_1_packed16   { float16_t2 ds;    int16_t qs[16]; };
struct block_q8_1_packed32   { float16_t2 ds;    int32_t qs[8];  };
// 4 blocks bundled together to allow 128-bit-aligned loads/stores.
struct block_q8_1_x4         { float16_t2 ds[4]; int32_t qs[32]; };
struct block_q8_1_x4_packed128 { float16_t2 ds[4]; int4 qs[8]; };

#define QUANT_K_Q2_K 256
struct block_q2_K {
    uint8_t    scales[16];   // QUANT_K/16
    uint8_t    qs[64];       // QUANT_K/4
    float16_t2 dm;
};
struct block_q2_K_packed16 {
    uint16_t   scales[8];    // QUANT_K/16/2
    uint16_t   qs[32];       // QUANT_K/4/2
    float16_t2 dm;
};
struct block_q2_K_packed32 {
    uint       scales[4];    // QUANT_K/16/4
    uint       qs[16];       // QUANT_K/4/4
    float16_t2 dm;
};

#define QUANT_K_Q3_K 256
struct block_q3_K {
    uint8_t   hmask[32];     // QUANT_K/8
    uint8_t   qs[64];        // QUANT_K/4
    uint8_t   scales[12];
    float16_t d;
};
struct block_q3_K_packed16 {
    uint16_t  hmask[16];     // QUANT_K/8/2
    uint16_t  qs[32];        // QUANT_K/4/2
    uint16_t  scales[6];     // 12/2
    float16_t d;
};

#define QUANT_K_Q4_K 256
struct block_q4_K {
    float16_t2 dm;
    uint8_t    scales[12];   // 3*QUANT_K/64
    uint8_t    qs[128];      // QUANT_K/2
};
struct block_q4_K_packed16 {
    float16_t2 dm;
    uint16_t   scales[6];    // 3*QUANT_K/64/2
    uint16_t   qs[64];       // QUANT_K/2/2
};
struct block_q4_K_packed32 {
    float16_t2 dm;
    uint       scales[3];    // 3*QUANT_K/64/4
    uint       qs[32];       // QUANT_K/2/4
};

#define QUANT_K_Q5_K 256
struct block_q5_K {
    float16_t2 dm;
    uint8_t    scales[12];
    uint8_t    qh[32];       // QUANT_K/8
    uint8_t    qs[128];      // QUANT_K/2
};
struct block_q5_K_packed16 {
    float16_t2 dm;
    uint16_t   scales[6];    // 12/2
    uint16_t   qh[16];       // QUANT_K/8/2
    uint16_t   qs[64];       // QUANT_K/2/2
};
struct block_q5_K_packed32 {
    float16_t2 dm;
    uint       scales[3];    // 12/4
    uint       qh[8];        // QUANT_K/8/4
    uint       qs[32];       // QUANT_K/2/4
};

#define QUANT_K_Q6_K 256
struct block_q6_K {
    uint8_t   ql[128];       // QUANT_K/2
    uint8_t   qh[64];        // QUANT_K/4
    int8_t    scales[16];    // QUANT_K/16
    float16_t d;
};
struct block_q6_K_packed16 {
    uint16_t  ql[64];        // QUANT_K/2/2
    uint16_t  qh[32];        // QUANT_K/4/2
    int16_t   scales[8];     // QUANT_K/16/2
    float16_t d;
};

// ── I-quants (lattice / lookup-driven) ────────────────────────────────────

#define QUANT_K_IQ2_XXS 256
#define QUANT_R_IQ2_XXS 1
struct block_iq2_xxs {
    float16_t d;
    uint8_t   qs[64];        // QUANT_K_IQ2_XXS/4
};
struct block_iq2_xxs_packed16 {
    float16_t d;
    uint16_t  qs[32];        // QUANT_K_IQ2_XXS/8
};

#define QUANT_K_IQ3_XXS 256
#define QUANT_R_IQ3_XXS 1
struct block_iq3_xxs {
    float16_t d;
    uint8_t   qs[96];        // QUANT_K_IQ3_XXS/4 + QUANT_K_IQ3_XXS/8
};
struct block_iq3_xxs_packed16 {
    float16_t d;
    uint16_t  qs[48];        // QUANT_K_IQ3_XXS/8 + QUANT_K_IQ3_XXS/16
};

#define QUANT_K_IQ3_S 256
#define QUANT_R_IQ3_S 1
struct block_iq3_s {
    float16_t d;
    uint8_t   qs[64];        // QUANT_K_IQ3_S/4
    uint8_t   qh[8];         // QUANT_K_IQ3_S/32
    uint8_t   signs[32];     // QUANT_K_IQ3_S/8
    uint8_t   scales[4];     // QUANT_K_IQ3_S/64
};
struct block_iq3_s_packed16 {
    float16_t d;
    uint16_t  qs[32];        // QUANT_K_IQ3_S/4/2
    uint16_t  qh[4];         // QUANT_K_IQ3_S/32/2
    uint16_t  signs[16];     // QUANT_K_IQ3_S/8/2
    uint16_t  scales[2];     // QUANT_K_IQ3_S/64/2
};

#define QUANT_K_Q1_0 128
#define QUANT_R_Q1_0 1
struct block_q1_0 {
    float16_t d;
    uint8_t   qs[16];        // QUANT_K_Q1_0/8
};

#define QUANT_K_MXFP4 32
#define QUANT_R_MXFP4 2
struct block_mxfp4 {
    uint8_t   e;             // E8M0 shared exponent
    uint8_t   qs[16];        // QUANT_K_MXFP4/2
};

#define QUANT_K_IQ4_NL 32
#define QUANT_R_IQ4_NL 2
struct block_iq4_nl {
    float16_t d;
    uint8_t   qs[16];        // QUANT_K_IQ4_NL/2
};
struct block_iq4_nl_packed16 {
    float16_t d;
    uint16_t  qs[8];         // QUANT_K_IQ4_NL/2/2
};

#define QUANT_K_IQ4_XS 256
#define QUANT_R_IQ4_XS 1
struct block_iq4_xs {
    float16_t d;
    uint16_t  scales_h;
    uint8_t   scales_l[4];   // QUANT_K_IQ4_XS/64
    uint8_t   qs[128];       // QUANT_K_IQ4_XS/2
};
struct block_iq4_xs_packed16 {
    float16_t d;
    uint16_t  scales_h;
    uint16_t  scales_l[2];   // QUANT_K_IQ4_XS/128
    uint16_t  qs[64];        // QUANT_K_IQ4_XS/4
};
struct block_iq4_xs_packed32 {
    float16_t d;
    uint16_t  scales_h;
    uint      scales_l;
    uint      qs[32];        // QUANT_K_IQ4_XS/8
};

// ── DATA_A_X → A_TYPE selection ──────────────────────────────────────────
// Consumer shader variants like `q4_k: DATA_A_Q4_K=1` get the right A_TYPE
// and QUANT_K wired up automatically. Mirrors llama.cpp's macro convention.

#if defined(DATA_A_F32)
    #define A_TYPE   float
    #define QUANT_K  1
    #define QUANT_R  1
#elif defined(DATA_A_F16)
    #define A_TYPE   float16_t
    #define QUANT_K  1
    #define QUANT_R  1
#elif defined(DATA_A_BF16)
    // bf16 is stored as a raw uint16 — consumers reinterpret via bf16_to_fp32 helpers.
    #define A_TYPE   uint16_t
    #define QUANT_K  1
    #define QUANT_R  1
#elif defined(DATA_A_Q4_0)
    #define A_TYPE          block_q4_0
    #define A_TYPE_PACKED16 block_q4_0_packed16
    #define QUANT_K         QUANT_K_Q4_0
    #define QUANT_R         QUANT_R_Q4_0
#elif defined(DATA_A_Q4_1)
    #define A_TYPE          block_q4_1
    #define A_TYPE_PACKED16 block_q4_1_packed16
    #define QUANT_K         QUANT_K_Q4_1
    #define QUANT_R         QUANT_R_Q4_1
#elif defined(DATA_A_Q5_0)
    #define A_TYPE          block_q5_0
    #define A_TYPE_PACKED16 block_q5_0_packed16
    #define QUANT_K         QUANT_K_Q5_0
    #define QUANT_R         QUANT_R_Q5_0
#elif defined(DATA_A_Q5_1)
    #define A_TYPE          block_q5_1
    #define A_TYPE_PACKED16 block_q5_1_packed16
    #define QUANT_K         QUANT_K_Q5_1
    #define QUANT_R         QUANT_R_Q5_1
#elif defined(DATA_A_Q8_0)
    #define A_TYPE          block_q8_0
    #define A_TYPE_PACKED16 block_q8_0_packed16
    #define QUANT_K         QUANT_K_Q8_0
    #define QUANT_R         QUANT_R_Q8_0
#elif defined(DATA_A_Q8_1)
    #define A_TYPE          block_q8_1
    #define QUANT_K         QUANT_K_Q8_1
    #define QUANT_R         QUANT_R_Q8_1
#elif defined(DATA_A_Q2_K)
    #define A_TYPE          block_q2_K
    #define A_TYPE_PACKED16 block_q2_K_packed16
    #define A_TYPE_PACKED32 block_q2_K_packed32
    #define QUANT_K         QUANT_K_Q2_K
    #define QUANT_R         1
#elif defined(DATA_A_Q3_K)
    #define A_TYPE          block_q3_K
    #define A_TYPE_PACKED16 block_q3_K_packed16
    #define QUANT_K         QUANT_K_Q3_K
    #define QUANT_R         1
#elif defined(DATA_A_Q4_K)
    #define A_TYPE          block_q4_K
    #define A_TYPE_PACKED16 block_q4_K_packed16
    #define A_TYPE_PACKED32 block_q4_K_packed32
    #define QUANT_K         QUANT_K_Q4_K
    #define QUANT_R         1
#elif defined(DATA_A_Q5_K)
    #define A_TYPE          block_q5_K
    #define A_TYPE_PACKED16 block_q5_K_packed16
    #define A_TYPE_PACKED32 block_q5_K_packed32
    #define QUANT_K         QUANT_K_Q5_K
    #define QUANT_R         1
#elif defined(DATA_A_Q6_K)
    #define A_TYPE          block_q6_K
    #define A_TYPE_PACKED16 block_q6_K_packed16
    #define QUANT_K         QUANT_K_Q6_K
    #define QUANT_R         1
#elif defined(DATA_A_IQ2_XXS)
    #define A_TYPE          block_iq2_xxs
    #define A_TYPE_PACKED16 block_iq2_xxs_packed16
    #define QUANT_K         QUANT_K_IQ2_XXS
    #define QUANT_R         QUANT_R_IQ2_XXS
#elif defined(DATA_A_IQ3_XXS)
    #define A_TYPE          block_iq3_xxs
    #define A_TYPE_PACKED16 block_iq3_xxs_packed16
    #define QUANT_K         QUANT_K_IQ3_XXS
    #define QUANT_R         QUANT_R_IQ3_XXS
#elif defined(DATA_A_IQ3_S)
    #define A_TYPE          block_iq3_s
    #define A_TYPE_PACKED16 block_iq3_s_packed16
    #define QUANT_K         QUANT_K_IQ3_S
    #define QUANT_R         QUANT_R_IQ3_S
#elif defined(DATA_A_Q1_0)
    #define A_TYPE          block_q1_0
    #define QUANT_K         QUANT_K_Q1_0
    #define QUANT_R         QUANT_R_Q1_0
#elif defined(DATA_A_MXFP4)
    #define A_TYPE          block_mxfp4
    #define QUANT_K         QUANT_K_MXFP4
    #define QUANT_R         QUANT_R_MXFP4
#elif defined(DATA_A_IQ4_NL)
    #define A_TYPE          block_iq4_nl
    #define A_TYPE_PACKED16 block_iq4_nl_packed16
    #define QUANT_K         QUANT_K_IQ4_NL
    #define QUANT_R         QUANT_R_IQ4_NL
#elif defined(DATA_A_IQ4_XS)
    #define A_TYPE          block_iq4_xs
    #define A_TYPE_PACKED16 block_iq4_xs_packed16
    #define A_TYPE_PACKED32 block_iq4_xs_packed32
    #define QUANT_K         QUANT_K_IQ4_XS
    #define QUANT_R         QUANT_R_IQ4_XS
#endif

// ── Fallback defaults ────────────────────────────────────────────────────
// Used by shaders that override types directly via -DA_TYPE=… variants
// rather than the DATA_A_X scheme. Only fire when nothing else set them.

#ifndef A_TYPE
    #define A_TYPE float
#endif
#ifndef B_TYPE
    #define B_TYPE float
#endif
#ifndef D_TYPE
    #define D_TYPE float
#endif
#ifndef FLOAT_TYPE
    #define FLOAT_TYPE float
#endif
#ifndef FLOAT_TYPEV2
    #define FLOAT_TYPEV2 float2
#endif

// ── Helpers ──────────────────────────────────────────────────────────────
// GLSL's unpack8(uint32_t) splits a uint into four uint bytes; HLSL/Slang
// don't expose it directly. Manual byte split — the K-quant matmul-vec
// kernels rely on this heavily.
uint4 unpack8_u(uint v) {
    return uint4(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu, (v >> 24) & 0xFFu);
}
// Signed variant — sign-extends each byte via shift-left then arithmetic
// shift-right. Q8_0 dequantize4 uses this through a different path; kept
// here for symmetry / future K-quant scale unpacks.
int4 unpack8_s(int v) {
    return int4((v << 24) >> 24, (v << 16) >> 24, (v << 8) >> 24, v >> 24);
}

// Reinterpret raw bf16 bits (low 16 of a uint) as fp32. Mirrors
// `ggml_compute_bf16_to_fp32`: bf16 is fp32 with the low 16 mantissa bits
// truncated, so left-shifting by 16 reconstitutes the float.
float bf16_to_fp32(uint u) {
    return asfloat(u << 16);
}
uint fp32_to_bf16(float f) {
    uint u = asuint(f);
    u = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
    return u;
}

// E8M0: 8-bit exponent-only float used by MXFP4 block scales. x == 0 → 2^-127
// (smallest subnormal-ish), otherwise the 8-bit value is the biased exponent.
float e8m0_to_fp32(uint x) {
    uint bits = (x == 0u) ? 0x00400000u : (x << 23);
    return asfloat(bits);
}

// ── I-quant lookup tables + shared-memory init ───────────────────────────
// Each I-quant variant ships a small constant lookup table that consumer
// shaders copy into groupshared at workgroup start via init_iq_shmem(...).
// We mirror llama.cpp's pattern: a `const` table next to a `groupshared`
// copy, and a cooperative loop guarded by NEEDS_INIT_IQ_SHMEM.

#if defined(DATA_A_IQ4_NL) || defined(DATA_A_IQ4_XS)
static const int8_t kvalues_iq4nl_const[16] = {
    int8_t(-127), int8_t(-104), int8_t(-83), int8_t(-65), int8_t(-49), int8_t(-35), int8_t(-22), int8_t(-10),
    int8_t(  1),  int8_t(  13), int8_t( 25), int8_t( 38), int8_t( 53), int8_t( 69), int8_t( 89), int8_t(113),
};

groupshared FLOAT_TYPE kvalues_iq4nl[16];

#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uint3 wgsize) {
    // Cooperative copy + sync. Entry shaders must call this before touching
    // kvalues_iq4nl. `gl_LocalInvocationIndex` comes from `import glsl;`.
    for (uint i = gl_LocalInvocationIndex; i < 16u; i += wgsize.x) {
        kvalues_iq4nl[i] = FLOAT_TYPE(kvalues_iq4nl_const[i]);
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif

#if defined(DATA_A_MXFP4)
// 4-bit signed lattice: low nibble selects from positive half, high nibble
// from the symmetric negative half (shifted by 8). Stored as int8 in shmem
// (no fp conversion at init) so the multiply can use the integer fast path.
static const int8_t kvalues_mxfp4_const[16] = {
    int8_t(0), int8_t( 1), int8_t( 2), int8_t( 3), int8_t( 4), int8_t( 6), int8_t( 8), int8_t( 12),
    int8_t(0), int8_t(-1), int8_t(-2), int8_t(-3), int8_t(-4), int8_t(-6), int8_t(-8), int8_t(-12),
};

groupshared int8_t kvalues_mxfp4[16];

#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uint3 wgsize) {
    for (uint i = gl_LocalInvocationIndex; i < 16u; i += wgsize.x) {
        kvalues_mxfp4[i] = kvalues_mxfp4_const[i];
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif

#if defined(DATA_A_IQ2_XXS)
// 256-entry uvec2 lattice grid — the per-block-token quantization codebook.
// Each entry is two uint32 lanes packing 4 nibble-encoded values each (8 weights total).
static const uint2 iq2xxs_grid_const[256] = {
    uint2(0x08080808, 0x08080808), uint2(0x0808082b, 0x08080808), uint2(0x08081919, 0x08080808), uint2(0x08082b08, 0x08080808),
    uint2(0x08082b2b, 0x08080808), uint2(0x08190819, 0x08080808), uint2(0x08191908, 0x08080808), uint2(0x082b0808, 0x08080808),
    uint2(0x082b082b, 0x08080808), uint2(0x082b2b08, 0x08080808), uint2(0x082b2b2b, 0x08080808), uint2(0x19080819, 0x08080808),
    uint2(0x19081908, 0x08080808), uint2(0x19190808, 0x08080808), uint2(0x19192b08, 0x08080808), uint2(0x192b0819, 0x08080808),
    uint2(0x192b1908, 0x08080808), uint2(0x2b080808, 0x08080808), uint2(0x2b08082b, 0x08080808), uint2(0x2b082b2b, 0x08080808),
    uint2(0x2b2b082b, 0x08080808), uint2(0x08080819, 0x08080819), uint2(0x08081908, 0x08080819), uint2(0x08190808, 0x08080819),
    uint2(0x08191919, 0x08080819), uint2(0x19080808, 0x08080819), uint2(0x2b081908, 0x08080819), uint2(0x2b192b08, 0x08080819),
    uint2(0x08080808, 0x0808082b), uint2(0x0808082b, 0x0808082b), uint2(0x082b082b, 0x0808082b), uint2(0x2b08082b, 0x0808082b),
    uint2(0x08080819, 0x08081908), uint2(0x08081908, 0x08081908), uint2(0x08190808, 0x08081908), uint2(0x082b0819, 0x08081908),
    uint2(0x082b1908, 0x08081908), uint2(0x19080808, 0x08081908), uint2(0x1908082b, 0x08081908), uint2(0x19082b08, 0x08081908),
    uint2(0x192b0808, 0x08081908), uint2(0x2b080819, 0x08081908), uint2(0x2b081908, 0x08081908), uint2(0x2b190808, 0x08081908),
    uint2(0x2b2b1908, 0x08081908), uint2(0x08080808, 0x08081919), uint2(0x0808082b, 0x08081919), uint2(0x08082b08, 0x08081919),
    uint2(0x082b0808, 0x08081919), uint2(0x1908192b, 0x08081919), uint2(0x192b2b19, 0x08081919), uint2(0x2b080808, 0x08081919),
    uint2(0x2b190819, 0x08081919), uint2(0x08082b19, 0x0808192b), uint2(0x08190808, 0x0808192b), uint2(0x19080808, 0x0808192b),
    uint2(0x2b081908, 0x0808192b), uint2(0x2b2b1908, 0x0808192b), uint2(0x08080808, 0x08082b08), uint2(0x08081919, 0x08082b08),
    uint2(0x08082b08, 0x08082b08), uint2(0x08191908, 0x08082b08), uint2(0x082b2b08, 0x08082b08), uint2(0x19080819, 0x08082b08),
    uint2(0x19081908, 0x08082b08), uint2(0x19190808, 0x08082b08), uint2(0x1919082b, 0x08082b08), uint2(0x2b082b08, 0x08082b08),
    uint2(0x08081908, 0x08082b19), uint2(0x19080808, 0x08082b19), uint2(0x0808082b, 0x08082b2b), uint2(0x08191908, 0x08082b2b),
    uint2(0x08080819, 0x08190808), uint2(0x08081908, 0x08190808), uint2(0x08190808, 0x08190808), uint2(0x082b0819, 0x08190808),
    uint2(0x19080808, 0x08190808), uint2(0x192b0808, 0x08190808), uint2(0x2b081908, 0x08190808), uint2(0x2b190808, 0x08190808),
    uint2(0x2b191919, 0x08190808), uint2(0x08080808, 0x08190819), uint2(0x08082b08, 0x08190819), uint2(0x082b0808, 0x08190819),
    uint2(0x19190808, 0x08190819), uint2(0x19192b2b, 0x08190819), uint2(0x2b080808, 0x08190819), uint2(0x082b1908, 0x0819082b),
    uint2(0x19081919, 0x0819082b), uint2(0x08080808, 0x08191908), uint2(0x08082b08, 0x08191908), uint2(0x082b0808, 0x08191908),
    uint2(0x082b1919, 0x08191908), uint2(0x19082b19, 0x08191908), uint2(0x2b080808, 0x08191908), uint2(0x08192b08, 0x08191919),
    uint2(0x192b082b, 0x08191919), uint2(0x08080808, 0x0819192b), uint2(0x0819192b, 0x0819192b), uint2(0x08080819, 0x08192b08),
    uint2(0x08081908, 0x08192b08), uint2(0x08190808, 0x08192b08), uint2(0x19080808, 0x08192b08), uint2(0x2b080819, 0x08192b08),
    uint2(0x08080808, 0x08192b19), uint2(0x08081919, 0x08192b19), uint2(0x2b2b0808, 0x08192b19), uint2(0x19190819, 0x08192b2b),
    uint2(0x08080808, 0x082b0808), uint2(0x0808082b, 0x082b0808), uint2(0x08082b2b, 0x082b0808), uint2(0x19081908, 0x082b0808),
    uint2(0x192b0819, 0x082b0808), uint2(0x2b080808, 0x082b0808), uint2(0x2b08082b, 0x082b0808), uint2(0x082b2b19, 0x082b0819),
    uint2(0x19082b08, 0x082b0819), uint2(0x08080808, 0x082b082b), uint2(0x0808082b, 0x082b082b), uint2(0x08080819, 0x082b1908),
    uint2(0x08081908, 0x082b1908), uint2(0x08190808, 0x082b1908), uint2(0x19080808, 0x082b1908), uint2(0x1919192b, 0x082b1908),
    uint2(0x08080808, 0x082b1919), uint2(0x19080819, 0x082b1919), uint2(0x192b1908, 0x082b1919), uint2(0x2b190808, 0x082b192b),
    uint2(0x08082b08, 0x082b2b08), uint2(0x082b0808, 0x082b2b08), uint2(0x2b191908, 0x082b2b08), uint2(0x19081908, 0x082b2b2b),
    uint2(0x08080819, 0x19080808), uint2(0x08081908, 0x19080808), uint2(0x08190808, 0x19080808), uint2(0x08192b08, 0x19080808),
    uint2(0x082b0819, 0x19080808), uint2(0x082b1908, 0x19080808), uint2(0x19080808, 0x19080808), uint2(0x19082b08, 0x19080808),
    uint2(0x1919192b, 0x19080808), uint2(0x192b0808, 0x19080808), uint2(0x2b080819, 0x19080808), uint2(0x2b081908, 0x19080808),
    uint2(0x2b190808, 0x19080808), uint2(0x08080808, 0x19080819), uint2(0x082b0808, 0x19080819), uint2(0x192b0819, 0x19080819),
    uint2(0x2b080808, 0x19080819), uint2(0x2b081919, 0x19080819), uint2(0x08080819, 0x1908082b), uint2(0x08190808, 0x1908082b),
    uint2(0x19082b08, 0x1908082b), uint2(0x1919192b, 0x1908082b), uint2(0x192b2b08, 0x1908082b), uint2(0x08080808, 0x19081908),
    uint2(0x08082b08, 0x19081908), uint2(0x082b0808, 0x19081908), uint2(0x2b080808, 0x19081908), uint2(0x2b192b19, 0x19081908),
    uint2(0x0819082b, 0x19081919), uint2(0x082b1908, 0x19081919), uint2(0x08080808, 0x1908192b), uint2(0x08080819, 0x19082b08),
    uint2(0x08081908, 0x19082b08), uint2(0x08190808, 0x19082b08), uint2(0x19080808, 0x19082b08), uint2(0x19081919, 0x19082b08),
    uint2(0x08080808, 0x19082b19), uint2(0x19192b08, 0x19082b19), uint2(0x192b0819, 0x19082b19), uint2(0x2b08082b, 0x19082b19),
    uint2(0x19081919, 0x19082b2b), uint2(0x2b190808, 0x19082b2b), uint2(0x08080808, 0x19190808), uint2(0x08082b08, 0x19190808),
    uint2(0x08190819, 0x19190808), uint2(0x08192b19, 0x19190808), uint2(0x082b0808, 0x19190808), uint2(0x2b080808, 0x19190808),
    uint2(0x2b082b08, 0x19190808), uint2(0x08081908, 0x19190819), uint2(0x1908082b, 0x19190819), uint2(0x2b2b1908, 0x19190819),
    uint2(0x2b190819, 0x1919082b), uint2(0x2b190808, 0x19191908), uint2(0x2b19082b, 0x19191908), uint2(0x08082b2b, 0x19191919),
    uint2(0x08080819, 0x1919192b), uint2(0x19191908, 0x1919192b), uint2(0x08080808, 0x19192b08), uint2(0x08190819, 0x19192b08),
    uint2(0x08192b19, 0x19192b08), uint2(0x192b1908, 0x19192b08), uint2(0x19080808, 0x19192b19), uint2(0x08082b08, 0x19192b2b),
    uint2(0x08081908, 0x192b0808), uint2(0x08190808, 0x192b0808), uint2(0x19080808, 0x192b0808), uint2(0x192b2b08, 0x192b0808),
    uint2(0x08080808, 0x192b0819), uint2(0x19191919, 0x192b0819), uint2(0x08192b08, 0x192b082b), uint2(0x192b0808, 0x192b082b),
    uint2(0x08080808, 0x192b1908), uint2(0x08081919, 0x192b1908), uint2(0x08190808, 0x192b1919), uint2(0x0819082b, 0x192b1919),
    uint2(0x2b081908, 0x192b1919), uint2(0x1908082b, 0x192b2b08), uint2(0x08080808, 0x2b080808), uint2(0x0808082b, 0x2b080808),
    uint2(0x08082b2b, 0x2b080808), uint2(0x19080819, 0x2b080808), uint2(0x2b08082b, 0x2b080808), uint2(0x08081908, 0x2b080819),
    uint2(0x08192b08, 0x2b080819), uint2(0x19080808, 0x2b080819), uint2(0x08190819, 0x2b08082b), uint2(0x08080819, 0x2b081908),
    uint2(0x08081908, 0x2b081908), uint2(0x08190808, 0x2b081908), uint2(0x08191919, 0x2b081908), uint2(0x19080808, 0x2b081908),
    uint2(0x192b0808, 0x2b081908), uint2(0x08080808, 0x2b081919), uint2(0x1908192b, 0x2b081919), uint2(0x2b191908, 0x2b081919),
    uint2(0x08082b19, 0x2b08192b), uint2(0x19080808, 0x2b08192b), uint2(0x192b0808, 0x2b08192b), uint2(0x0808082b, 0x2b082b08),
    uint2(0x08081908, 0x2b082b19), uint2(0x08190819, 0x2b082b2b), uint2(0x08081908, 0x2b190808), uint2(0x08190808, 0x2b190808),
    uint2(0x082b1908, 0x2b190808), uint2(0x19080808, 0x2b190808), uint2(0x2b2b0819, 0x2b190808), uint2(0x0819192b, 0x2b190819),
    uint2(0x2b080808, 0x2b190819), uint2(0x19081919, 0x2b19082b), uint2(0x08080808, 0x2b191908), uint2(0x082b082b, 0x2b191908),
    uint2(0x19081908, 0x2b191908), uint2(0x19190819, 0x2b191919), uint2(0x2b080819, 0x2b192b08), uint2(0x082b0808, 0x2b192b19),
    uint2(0x0808082b, 0x2b2b0808), uint2(0x19190808, 0x2b2b0808), uint2(0x2b081919, 0x2b2b0808), uint2(0x08082b19, 0x2b2b0819),
    uint2(0x08080808, 0x2b2b082b), uint2(0x08192b08, 0x2b2b1908), uint2(0x19190808, 0x2b2b2b08), uint2(0x08081908, 0x2b2b2b19)
};

groupshared uint2 iq2xxs_grid[256];

#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uint3 wgsize) {
    for (uint i = gl_LocalInvocationIndex; i < 256u; i += wgsize.x) {
        iq2xxs_grid[i] = iq2xxs_grid_const[i];
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif

#if defined(DATA_A_IQ3_XXS)
// 256-entry uint32 grid; each entry packs 4 byte-encoded lattice coords.
static const uint iq3xxs_grid_const[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

groupshared uint iq3xxs_grid[256];

#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uint3 wgsize) {
    for (uint i = gl_LocalInvocationIndex; i < 256u; i += wgsize.x) {
        iq3xxs_grid[i] = iq3xxs_grid_const[i];
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif

#if defined(DATA_A_IQ3_S)
// 512-entry uint32 lattice grid. Each entry packs 4 byte-encoded coords.
static const uint iq3s_grid_const[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

groupshared uint iq3s_grid[512];

#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uint3 wgsize) {
    for (uint i = gl_LocalInvocationIndex; i < 512u; i += wgsize.x) {
        iq3s_grid[i] = iq3s_grid_const[i];
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif

#endif
