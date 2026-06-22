// Shared TurboQuant constants — the Walsh-Hadamard sign tables (used by both
// the standalone rotation `turbo_wht.slang` and the fused quantize-on-write
// `kv_quant_turbo.slang`, which MUST agree byte-for-byte) plus the PolarQuant
// nearest-centroid midpoint thresholds. Transcribed verbatim from the
// llama-cpp-turboquant fork (turbo_wht.comp / copy_to_quant.comp).
//
// The PolarQuant centroid *values* live in types.slang (guarded per dtype, also
// consumed by the flash-attn dequant); these are the quantizer-only thresholds.

#ifndef TURBO_CONSTS_SLANG
#define TURBO_CONSTS_SLANG

static const float TURBO_INV_SQRT_128 = 0.08838834764831845; // 1 / sqrt(128)

// Pre/post scramble sign vectors. Forward WHT: pre=S1, post=S2; inverse swaps.
static const float TURBO_S1[128] = {
    -1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,
     1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1, -1,
    -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,
     1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1, -1,  1,
    -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1,  1,
     1, -1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,  1, -1,
    -1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1,
     1, -1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,  1,  1, -1,  1
};
static const float TURBO_S2[128] = {
     1,  1,  1,  1, -1,  1,  1, -1,  1, -1, -1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1,
     1, -1,  1,  1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1,  1,
     1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1,  1,
    -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1,
     1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1,
    -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1
};

// Midpoint thresholds between adjacent PolarQuant centroids (nearest-centroid
// boundaries). turbo2/turbo3 select via strict `<` nested-ternary; turbo4 via
// `>=` accumulation — matching the fork's exact comparison operators.
static const float TURBO2_THRESH[3]  = { -0.086728, 0.0, 0.086728 };
static const float TURBO3_THRESH[7]  = {
    -0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259
};
static const float TURBO4_THRESH[15] = {
    -0.145561, -0.103361, -0.079142, -0.060009, -0.043430, -0.028293, -0.013964,
     0.0,       0.013964,  0.028293,  0.043430,  0.060009,  0.079142,  0.103361,  0.145561
};

#endif // TURBO_CONSTS_SLANG
