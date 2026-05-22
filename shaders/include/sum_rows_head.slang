// Ported from llama.cpp ggml-vulkan, MIT license — see LICENSE.llama-cpp

#ifndef SUM_ROWS_HEAD_SLANG
#define SUM_ROWS_HEAD_SLANG

struct SumRowsParams {
    uint n_cols;
    uint ne01; uint ne02;
    uint nb01; uint nb02; uint nb03;
    uint nb11; uint nb12; uint nb13;
    float weight;
    uint misalign_offsets;
    uint ne0_12mp; uint ne0_12L;
    uint ne0_1mp;  uint ne0_1L;
};

[[vk::push_constant]] ConstantBuffer<SumRowsParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<A_TYPE> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<D_TYPE> data_d;

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_doffset() { return p.misalign_offsets & 0xFFFF; }

uint mulhi_u32(uint a, uint b) {
    return uint((uint64_t(a) * uint64_t(b)) >> 32);
}

uint fastdiv(uint n, uint mp, uint L) {
    uint msbs = mulhi_u32(n, mp);
    return (msbs + n) >> L;
}

#endif
