// Ported from llama.cpp ggml-vulkan pool2d.comp, MIT license — see LICENSE.llama-cpp

//@variants
//  f32: A_TYPE=float       D_TYPE=float       FLOAT_TYPE=float
//@end-variants

#include "types.slang"

struct Pool2dParams {
    uint IW; uint IH;
    uint OW; uint OH;
    uint OC;
    uint pelements;
    uint op;
    int k0; int k1;
    int s0; int s1;
    int p0; int p1;
};

[[vk::push_constant]] ConstantBuffer<Pool2dParams> p;

static const uint BLOCK_SIZE   = 512;
static const float FLT_MAX     = 3.402823466e+38f;
static const uint OP_POOL_MAX  = 0u;
static const uint OP_POOL_AVG  = 1u;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE>   data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= p.pelements) { return; }

    const uint O_HW   = p.OW * p.OH;
    const uint nc     = idx / O_HW;
    const uint cur_oh = (idx % O_HW) / p.OW;
    const uint cur_ow = (idx % O_HW) % p.OW;

    const int  start_h = int(cur_oh) * p.s0 - p.p0;
    const uint bh      = uint(max(start_h, 0));
    const uint eh      = uint(min(start_h + p.k0, int(p.IH)));

    const int  start_w = int(cur_ow) * p.s1 - p.p1;
    const uint bw      = uint(max(start_w, 0));
    const uint ew      = uint(min(start_w + p.k1, int(p.IW)));

    const float scale = 1.0f / float(p.k0 * p.k1);
    float res;

    if (p.op == OP_POOL_AVG) {
        res = 0.0f;
    } else if (p.op == OP_POOL_MAX) {
        res = -FLT_MAX;
    } else {
        return;
    }

    for (uint i = bh; i < eh; i++) {
        for (uint j = bw; j < ew; j++) {
            const float cur = float(data_a[nc * p.IH * p.IW + i * p.IW + j]);
            if (p.op == OP_POOL_AVG) {
                res += cur * scale;
            } else if (p.op == OP_POOL_MAX) {
                res = max(res, cur);
            }
        }
    }

    data_d[nc * O_HW + cur_oh * p.OW + cur_ow] = D_TYPE(res);
}
