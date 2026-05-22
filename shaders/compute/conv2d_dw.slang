// Ported from llama.cpp ggml-vulkan conv2d_dw.comp, MIT license — see LICENSE.llama-cpp
//
// Depthwise 2D convolution. Two layouts: WHCN (channels-last in N×C×H×W order) and CWHN
// (channels-fastest). Selected by the WHCN macro at variant generation time.

//@variants
//  whcn_f32: A_TYPE=float    B_TYPE=float    D_TYPE=float    FLOAT_TYPE=float   WHCN=1
//  cwhn_f32: A_TYPE=float    B_TYPE=float    D_TYPE=float    FLOAT_TYPE=float
//@end-variants

#include "types.slang"

struct Conv2dDwParams {
    uint ne;
    uint batches;
    uint channels;
    uint dst_w;
    uint dst_h;
    uint src_w;
    uint src_h;
    uint knl_w;
    uint knl_h;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
};

[[vk::push_constant]] ConstantBuffer<Conv2dDwParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   knl_data;
[[vk::binding(1, 0)]] StructuredBuffer<B_TYPE>   src_data;
[[vk::binding(2, 0)]] RWStructuredBuffer<D_TYPE> dst_data;

FLOAT_TYPE conv_2d_dw_whcn(uint idx) {
    uint i0    = idx / p.dst_w;
    uint dst_x = idx - i0 * p.dst_w;
    uint i1    = i0 / p.dst_h;
    uint dst_y = i0 - i1 * p.dst_h;
    uint n     = i1 / p.channels;
    uint c     = i1 - n * p.channels;

    uint src_i = n * p.channels * p.src_h * p.src_w + c * p.src_h * p.src_w;
    uint knl_i = c * p.knl_h * p.knl_w;

    FLOAT_TYPE sum = FLOAT_TYPE(0);
    for (uint knl_y = 0; knl_y < p.knl_h; ++knl_y) {
        uint src_y = uint(int(dst_y) * p.stride_y + int(knl_y) * p.dilation_y - p.pad_y);
        if (src_y >= p.src_h) { continue; }  // negative values wrap to large uint
        for (uint knl_x = 0; knl_x < p.knl_w; ++knl_x) {
            uint src_x = uint(int(dst_x) * p.stride_x + int(knl_x) * p.dilation_x - p.pad_x);
            if (src_x >= p.src_w) { continue; }
            FLOAT_TYPE v = FLOAT_TYPE(src_data[src_i + src_y * p.src_w + src_x]);
            FLOAT_TYPE k = FLOAT_TYPE(knl_data[knl_i + knl_y * p.knl_w + knl_x]);
            sum = mad(v, k, sum);
        }
    }
    return sum;
}

FLOAT_TYPE conv_2d_dw_cwhn(uint idx) {
    uint i0    = idx / p.channels;
    uint c     = idx - i0 * p.channels;
    uint i1    = i0 / p.dst_w;
    uint dst_x = i0 - i1 * p.dst_w;
    uint n     = i1 / p.dst_h;
    uint dst_y = i1 - n * p.dst_h;

    uint src_i   = n * p.channels * p.src_h * p.src_w;
    uint src_row = p.src_w * p.channels;
    uint knl_row = p.knl_w * p.channels;

    FLOAT_TYPE sum = FLOAT_TYPE(0);
    for (uint knl_y = 0; knl_y < p.knl_h; ++knl_y) {
        uint src_y = uint(int(dst_y) * p.stride_y + int(knl_y) * p.dilation_y - p.pad_y);
        if (src_y >= p.src_h) { continue; }
        for (uint knl_x = 0; knl_x < p.knl_w; ++knl_x) {
            uint src_x = uint(int(dst_x) * p.stride_x + int(knl_x) * p.dilation_x - p.pad_x);
            if (src_x >= p.src_w) { continue; }
            FLOAT_TYPE v = FLOAT_TYPE(src_data[src_i + src_y * src_row + src_x * p.channels + c]);
            FLOAT_TYPE k = FLOAT_TYPE(knl_data[        knl_y * knl_row + knl_x * p.channels + c]);
            sum = mad(v, k, sum);
        }
    }
    return sum;
}

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint idx = gl_GlobalInvocationID.z * 262144u + gl_GlobalInvocationID.y * 512u + gl_GlobalInvocationID.x;
    if (idx >= p.ne) { return; }

    FLOAT_TYPE result =
#ifdef WHCN
        conv_2d_dw_whcn(idx);
#else
        conv_2d_dw_cwhn(idx);
#endif
    dst_data[idx] = D_TYPE(result);
}
