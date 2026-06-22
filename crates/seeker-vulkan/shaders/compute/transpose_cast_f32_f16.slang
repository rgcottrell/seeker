// Transpose + cast one vocab chunk of the dequantized token embedding into the
// self-conditioning weight `sc_embT`. `src` is a `get_rows` slab `[n_embd, CH]`
// (token v's embedding contiguous at `v·n_embd + e`); `dst` is `sc_embT` in
// seeker's matmul-`a` layout `[K=n_vocab, M=n_embd]` (`a[k=v, m=e]` at
// `e·n_vocab + v`), so the write transposes:
//   dst[e·n_vocab + (vocab_offset + v)] = f16(src[v·n_embd + e])
// Run once per chunk at load to build the whole [n_vocab, n_embd] F16 weight.

struct Params {
    uint n_embd;       // embedding width
    uint chunk;        // tokens in this slab (CH)
    uint vocab_offset; // first vocab id of this slab
    uint n_vocab;      // total vocab (sc_embT row stride)
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[[vk::binding(0, 0)]] StructuredBuffer<float>       src; // [n_embd, CH] get_rows slab
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t> dst; // [n_vocab, n_embd] sc_embT

// The host bounds `chunk` so `n_embd·chunk` threads fit one X grid dimension
// (≤ 65535·256), so a 1D dispatch over the global thread id suffices.
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = gid.x;
    const uint total = p.n_embd * p.chunk;
    if (i >= total) { return; }
    const uint e = i % p.n_embd;
    const uint v = i / p.n_embd;
    dst[e * p.n_vocab + (p.vocab_offset + v)] = float16_t(src[v * p.n_embd + e]);
}
