// F32 → F16 V-cache write with the destination offset read from the
// `DecodeDyn` slot.
//
// Replaces the `record_cast(v_view, cache_slot)` chain in
// `record_write_fused_nofence` (which baked
// `cache_layer.byte_offset + position * per_token_bytes` into the
// descriptor binding, breaking the persistent-decode-cmdbuf replay
// path). Here the cmdbuf binds the *full* V-cache layer and the
// shader uses `data_dyn[0].v_cache_d_offset` to write at the right
// position each replay.
//
// Element layout: `[head_dim_v, n_head_kv, L]` contiguous F32 on the
// input side; written contiguously starting at offset
// `v_cache_d_offset` (in F16 elements) on the cache side.

//@variants
//  f32:
//@end-variants

#include "decode_dyn.slang"

[SpecializationConstant] const uint BLOCK_SIZE = 256;

struct VCacheWriteParams {
    uint n_elements;  // head_dim_v * n_head_kv * L
};

[[vk::push_constant]] ConstantBuffer<VCacheWriteParams> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>           data_src;
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t>     data_cache;
[[vk::binding(2, 0)]] StructuredBuffer<DecodeDyn>       data_dyn;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= p.n_elements) { return; }
    const uint offset = data_dyn[0].v_cache_d_offset;
    data_cache[offset + i] = float16_t(data_src[i]);
}
