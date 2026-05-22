// Ported from llama.cpp ggml-vulkan glu_main.glsl, MIT license — see LICENSE.llama-cpp
//
// Included AFTER an entry shader has declared `float op(float a, float b)`.
// Textual include semantics let `main()` below resolve `op` to the per-shader
// activation kernel (reglu, swiglu, geglu, …).

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    const uint i = gl_GlobalInvocationID.z * 262144u + gl_GlobalInvocationID.y * 512u + gl_GlobalInvocationID.x;
    if (i >= p.N) { return; }

    const uint row = i / p.ne20;
    const uint col = i - row * p.ne20;

    const uint i3      = row / (p.ne01 * p.ne02);
    const uint i2      = (row % (p.ne01 * p.ne02)) / p.ne01;
    const uint i1      = row % p.ne01;
    const uint src_idx = i3 * p.nb03 + i2 * p.nb02 + i1 * p.nb01 + col;

    const uint dst_i3  = row / (p.ne11 * p.ne12);
    const uint dst_i2  = (row % (p.ne11 * p.ne12)) / p.ne11;
    const uint dst_i1  = row % p.ne11;
    const uint dst_idx = dst_i3 * p.nb13 + dst_i2 * p.nb12 + dst_i1 * p.nb11 + col;

    if (p.mode == 0) {
        const uint offset = p.ne00 / 2;
        data_d[dst_idx] = D_TYPE(op(float(data_a[src_idx]), float(data_a[src_idx + offset])));
    } else if (p.mode == 1) {
        const uint offset = p.ne00 / 2;
        data_d[dst_idx] = D_TYPE(op(float(data_a[src_idx + offset]), float(data_a[src_idx])));
    } else {
        data_d[dst_idx] = D_TYPE(op(float(data_a[src_idx]), float(data_b[src_idx])));
    }
}
