// Ported from llama.cpp ggml-vulkan upscale.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//  f16: A_TYPE=float16_t   D_TYPE=float16_t   FLOAT_TYPE=float
//@end-variants

#include "types.slang"

struct UpscaleParams {
    uint ne; uint a_offset; uint d_offset;
    uint ne00; uint ne01;
    uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13;
    float sf0; float sf1; float sf2; float sf3;
    float pixel_offset;
};

[[vk::push_constant]] ConstantBuffer<UpscaleParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

// from ggml.h: enum ggml_scale_mode, enum ggml_scale_flag
static const uint NEAREST            = 0;
static const uint BILINEAR           = 1;
static const uint BICUBIC            = 2;
static const uint BILINEAR_ANTIALIAS = 513;

[SpecializationConstant] const uint scale_mode = 0;

float fetch_nearest(uint i10, uint i11, uint i12, uint i13) {
    const uint i00 = uint(i10 / p.sf0);
    const uint i01 = uint(i11 / p.sf1);
    const uint i02 = uint(i12 / p.sf2);
    const uint i03 = uint(i13 / p.sf3);
    return float(data_a[p.a_offset + i03 * p.nb03 + i02 * p.nb02 + i01 * p.nb01 + i00 * p.nb00]);
}

float fetch_bilinear(int2 c0, int2 c1, float2 d, uint i12, uint i13) {
    const uint i02 = uint(i12 / p.sf2);
    const uint i03 = uint(i13 / p.sf3);
    const uint base = p.a_offset + i03 * p.nb03 + i02 * p.nb02;

    const float v00 = float(data_a[base + c0.y * p.nb01 + c0.x * p.nb00]);
    const float v01 = float(data_a[base + c0.y * p.nb01 + c1.x * p.nb00]);
    const float v10 = float(data_a[base + c1.y * p.nb01 + c0.x * p.nb00]);
    const float v11 = float(data_a[base + c1.y * p.nb01 + c1.x * p.nb00]);

    return v00 * (1.0f - d.x) * (1.0f - d.y)
         + v01 *         d.x  * (1.0f - d.y)
         + v10 * (1.0f - d.x) *         d.y
         + v11 *         d.x  *         d.y;
}

float interpolate_bilinear(uint i10, uint i11, uint i12, uint i13) {
    const int2 ne0 = int2(p.ne00, p.ne01);
    const float2 c = (float2(i10, i11) + p.pixel_offset) / float2(p.sf0, p.sf1) - p.pixel_offset;
    const float2 c0f = floor(c);
    const float2 d = c - c0f;
    const int2 c0 = max(int2(c0f), int2(0, 0));
    const int2 c1 = min(int2(c0f + 1), ne0 - int2(1, 1));
    return fetch_bilinear(c0, c1, d, i12, i13);
}

float triangle_filter(float x) {
    return max(1.0f - abs(x), 0.0f);
}

float interpolate_bilinear_antialias(uint i10, uint i11, uint i12, uint i13) {
    const float support1  = max(1.0f, 1.0f / p.sf1);
    const float invscale1 = 1.0f / support1;
    const float support0  = max(1.0f, 1.0f / p.sf0);
    const float invscale0 = 1.0f / support0;

    const uint i02 = uint(i12 / p.sf2);
    const uint i03 = uint(i13 / p.sf3);

    const float y = (float(i11) + p.pixel_offset) / p.sf1;
    const float x = (float(i10) + p.pixel_offset) / p.sf0;

    const int x_min = max(int(x - support0 + p.pixel_offset), 0);
    const int x_max = min(int(x + support0 + p.pixel_offset), int(p.ne00));
    const int y_min = max(int(y - support1 + p.pixel_offset), 0);
    const int y_max = min(int(y + support1 + p.pixel_offset), int(p.ne01));

    float val = 0.0f;
    float total_weight = 0.0f;

    for (int sy = y_min; sy < y_max; sy++) {
        const float weight_y = triangle_filter((sy - y + p.pixel_offset) * invscale1);
        for (int sx = x_min; sx < x_max; sx++) {
            const float weight_x = triangle_filter((sx - x + p.pixel_offset) * invscale0);
            const float weight = weight_x * weight_y;
            if (weight <= 0.0f) { continue; }

            const float pixel = float(data_a[p.a_offset + i03 * p.nb03 + i02 * p.nb02 + sy * p.nb01 + sx * p.nb00]);
            val += pixel * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0f) { val /= total_weight; }
    return val;
}

// Bicubic interpolation with alpha = -0.75
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
static const float4 BCOEFFS1 = float4( 1.25f, -2.25f,  0.0f, 1.0f);
static const float4 BCOEFFS2 = float4(-0.75f,  3.75f, -6.0f, 3.0f);

float4 powers(float x) { return float4(x * x * x, x * x, x, 1); }

float bicubic(float p0, float p1, float p2, float p3, float x) {
    return p0 * dot(BCOEFFS2, powers(x + 1))
         + p1 * dot(BCOEFFS1, powers(x    ))
         + p2 * dot(BCOEFFS1, powers(1 - x))
         + p3 * dot(BCOEFFS2, powers(2 - x));
}

float interpolate_bicubic(uint i10, uint i11, uint i12, uint i13) {
    const int2 res = int2(p.ne00 - 1, p.ne01 - 1);
    const float2 coord = (float2(i10, i11) + p.pixel_offset) / float2(p.sf0, p.sf1) - p.pixel_offset;
    const float2 d = frac(coord);
    const int2 i = int2(floor(coord));

    const uint i02 = uint(i12 / p.sf2);
    const uint i03 = uint(i13 / p.sf3);
    const uint base = p.a_offset + i03 * p.nb03 + i02 * p.nb02;

    // FETCH(dx, dy) = data_a[base + clamp(i.x + dx, 0, res.x) * nb00 + clamp(i.y + dy, 0, res.y) * nb01]
    #define FETCH(a, b) float(data_a[base + clamp(i.x + (a), 0, res.x) * p.nb00 + clamp(i.y + (b), 0, res.y) * p.nb01])

    return bicubic(
        bicubic(FETCH(-1, -1), FETCH(0, -1), FETCH(1, -1), FETCH(2, -1), d.x),
        bicubic(FETCH(-1,  0), FETCH(0,  0), FETCH(1,  0), FETCH(2,  0), d.x),
        bicubic(FETCH(-1,  1), FETCH(0,  1), FETCH(1,  1), FETCH(2,  1), d.x),
        bicubic(FETCH(-1,  2), FETCH(0,  2), FETCH(1,  2), FETCH(2,  2), d.x), d.y
    );

    #undef FETCH
}

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
    if (idx >= p.ne) { return; }

    const uint i10 = idx % p.ne10;
    const uint i11 = (idx / p.ne10) % p.ne11;
    const uint i12 = (idx / (p.ne10 * p.ne11)) % p.ne12;
    const uint i13 = (idx / (p.ne10 * p.ne11 * p.ne12)) % p.ne13;

    float result = 0.0f;
    switch (scale_mode) {
        case NEAREST:
            result = fetch_nearest(i10, i11, i12, i13);
            break;
        case BILINEAR:
            result = interpolate_bilinear(i10, i11, i12, i13);
            break;
        case BICUBIC:
            result = interpolate_bicubic(i10, i11, i12, i13);
            break;
        case BILINEAR_ANTIALIAS:
            result = interpolate_bilinear_antialias(i10, i11, i12, i13);
            break;
    }

    data_d[p.d_offset + idx] = D_TYPE(result);
}
