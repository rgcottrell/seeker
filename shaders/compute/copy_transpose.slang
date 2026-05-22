// Ported from llama.cpp ggml-vulkan copy_transpose.comp, MIT license — see LICENSE.llama-cpp
//
// 32×32 tile with 32×8 threads; LSB-swap in shared memory makes both loads and stores
// contiguous. A_TYPE/D_TYPE here are integer (uint16_t/uint) — the cast through uint is
// a raw-bit move, not a value conversion.

//@variants
//  16: A_TYPE=uint16_t  D_TYPE=uint16_t  FLOAT_TYPE=float
//  32: A_TYPE=uint      D_TYPE=uint      FLOAT_TYPE=float
//@end-variants

import glsl;  // gl_NumWorkGroups

#include "types.slang"
#include "generic_unary_head.slang"

static const uint TILE_DIM = 32;

groupshared uint sh[TILE_DIM][TILE_DIM + 1];

void iter(uint3 wg_id, uint3 lid) {
    const uint tile_col = wg_id.x;
    const uint tile_row = wg_id.y;

    const uint tid_col = lid.x;
    const uint tid_row = lid.y;

    const uint i2  = wg_id.z % p.ne12;
    const uint i3  = wg_id.z / p.ne12;
    const uint i02 = i2;
    const uint i03 = i3;

    [unroll] for (uint y = 0; y < 4; ++y) {
        const uint i00 = tile_col * TILE_DIM + tid_row + 8 * y;
        const uint i01 = tile_row * TILE_DIM + tid_col;
        if (i00 < p.ne00 && i01 < p.ne01 && i02 < p.ne02 && i03 < p.ne03) {
            const uint src_idx = i00 * p.nb00 + i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03;
            sh[tid_row + 8 * y][tid_col] = uint(data_a[get_aoffset() + src_idx]);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint y = 0; y < 4; ++y) {
        const uint i0 = tile_col * TILE_DIM + tid_col;
        const uint i1 = tile_row * TILE_DIM + tid_row + 8 * y;
        if (i0 < p.ne10 && i1 < p.ne11 && i2 < p.ne12 && i3 < p.ne13) {
            const uint dst_idx = i0 * p.nb10 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
            data_d[get_doffset() + dst_idx] = D_TYPE(sh[tid_col][tid_row + 8 * y]);
        }
    }
}

uint ceil_div(uint a, uint b) { return (a + b - 1) / b; }

[shader("compute")]
[numthreads(32, 8, 1)]
void main(uint3 gl_WorkGroupID : SV_GroupID, uint3 gl_LocalInvocationID : SV_GroupThreadID) {
    bool need_barrier = false;
    for (uint z = gl_WorkGroupID.z; z < p.ne12 * p.ne13; z += gl_NumWorkGroups.z) {
        for (uint y = gl_WorkGroupID.y; y < ceil_div(p.ne11, TILE_DIM); y += gl_NumWorkGroups.y) {
            for (uint x = gl_WorkGroupID.x; x < ceil_div(p.ne10, TILE_DIM); x += gl_NumWorkGroups.x) {
                if (need_barrier) { GroupMemoryBarrierWithGroupSync(); }
                need_barrier = true;
                iter(uint3(x, y, z), gl_LocalInvocationID);
            }
        }
    }
}
