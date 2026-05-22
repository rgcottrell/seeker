// Ported from llama.cpp ggml-vulkan rope_funcs.glsl, MIT license — see LICENSE.llama-cpp
//
// Phase note: the RMS_NORM_ROPE_FUSION branch is omitted (the fused
// rms_norm→rope path through shared memory). Stand-alone RoPE only.

#ifndef ROPE_FUNCS_SLANG
#define ROPE_FUNCS_SLANG

#include "rope_head.slang"

float rope_yarn_ramp(float low, float high, uint i0) {
    const float y = (float(i0) / 2.0f - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

uint rope_a_coord(uint i0, uint i01, uint i02, uint i03, rope_params p) {
    return p.a_offset + i03 * p.nb03 + i02 * p.nb02 + i01 * p.nb01 + i0;
}

void rope_yarn(float theta_extrap, uint i0,
               out float cos_theta, out float sin_theta,
               rope_params p) {
    float mscale = p.attn_factor;
    float theta_interp = p.freq_scale * theta_extrap;
    float theta = theta_interp;
    if (p.ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(p.corr_dims[0], p.corr_dims[1], i0) * p.ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / p.freq_scale);
    }
    if (p.is_back != 0) { theta = -theta; }
    cos_theta = cos(theta) * mscale;
    sin_theta = sin(theta) * mscale;
}

void rope_norm(uint i0, uint i1, uint i2, uint i3, rope_params p) {
    if (i0 >= p.ne00) { return; }

    uint idst = i0 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
    const uint ix = rope_a_coord(i0, i1, i2, i3, p);

    if (p.set_rows_stride != 0) {
        idst = i1 * p.nb11 + i0;
        idst += rope_data_i[i2].x * p.set_rows_stride;
    }
    idst += p.d_offset;

    if (i0 >= p.n_dims) {
        rope_data_d[idst + 0] = ROPE_D_TYPE(rope_data_a[ix + 0]);
        rope_data_d[idst + 1] = ROPE_D_TYPE(rope_data_a[ix + 1]);
        return;
    }

    const float theta_base  = float(rope_data_pos[i2]) * pow(p.theta_scale, float(i0) / 2.0f);
    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0 / 2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + 1]);

    rope_data_d[idst + 0] = ROPE_D_TYPE(x0 * cos_theta - x1 * sin_theta);
    rope_data_d[idst + 1] = ROPE_D_TYPE(x0 * sin_theta + x1 * cos_theta);
}

void rope_neox(uint i0, uint i1, uint i2, uint i3, rope_params p) {
    if (i0 >= p.ne00) { return; }

    uint idst = i0 / 2 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
    const uint ix = rope_a_coord(i0 / 2, i1, i2, i3, p);

    if (p.set_rows_stride != 0) {
        idst = i1 * p.nb11 + i0 / 2;
        idst += rope_data_i[i2].x * p.set_rows_stride;
    }
    idst += p.d_offset;

    if (i0 >= p.n_dims) {
        rope_data_d[idst + i0 / 2 + 0] = ROPE_D_TYPE(rope_data_a[ix + i0 / 2 + 0]);
        rope_data_d[idst + i0 / 2 + 1] = ROPE_D_TYPE(rope_data_a[ix + i0 / 2 + 1]);
        return;
    }

    const float theta_base  = float(rope_data_pos[i2]) * pow(p.theta_scale, float(i0) / 2.0f);
    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0 / 2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + p.n_dims / 2]);

    rope_data_d[idst + 0]              = ROPE_D_TYPE(x0 * cos_theta - x1 * sin_theta);
    rope_data_d[idst + p.n_dims / 2]   = ROPE_D_TYPE(x0 * sin_theta + x1 * cos_theta);
}

void rope_multi(uint i0, uint i1, uint i2, uint i3, rope_params p) {
    if (i0 >= p.ne00) { return; }

    uint idst = i0 / 2 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
    const uint ix = rope_a_coord(i0 / 2, i1, i2, i3, p);

    if (p.set_rows_stride != 0) {
        idst = i1 * p.nb11 + i0 / 2;
        idst += rope_data_i[i2].x * p.set_rows_stride;
    }
    idst += p.d_offset;

    if (i0 >= p.n_dims) {
        rope_data_d[idst + i0 / 2 + 0] = ROPE_D_TYPE(rope_data_a[ix + i0 / 2 + 0]);
        rope_data_d[idst + i0 / 2 + 1] = ROPE_D_TYPE(rope_data_a[ix + i0 / 2 + 1]);
        return;
    }

    const int  sect_dims = p.sections[0] + p.sections[1] + p.sections[2] + p.sections[3];
    const int  sec_w     = p.sections[1] + p.sections[0];
    const uint sector    = (i0 / 2) % uint(sect_dims);

    float theta_base = 0.0f;
    if (p.is_imrope != 0) {
        if (sector % 3 == 1 && int(sector) < 3 * p.sections[1]) {
            theta_base = float(rope_data_pos[i2 + p.ne02 * 1]) * pow(p.theta_scale, float(i0) / 2.0f);
        } else if (sector % 3 == 2 && int(sector) < 3 * p.sections[2]) {
            theta_base = float(rope_data_pos[i2 + p.ne02 * 2]) * pow(p.theta_scale, float(i0) / 2.0f);
        } else if (sector % 3 == 0 && int(sector) < 3 * p.sections[0]) {
            theta_base = float(rope_data_pos[i2]) * pow(p.theta_scale, float(i0) / 2.0f);
        } else {
            theta_base = float(rope_data_pos[i2 + p.ne02 * 3]) * pow(p.theta_scale, float(i0) / 2.0f);
        }
    } else {
        if (int(sector) < p.sections[0]) {
            theta_base = float(rope_data_pos[i2]) * pow(p.theta_scale, float(i0) / 2.0f);
        } else if (int(sector) >= p.sections[0] && int(sector) < sec_w) {
            theta_base = float(rope_data_pos[i2 + p.ne02 * 1]) * pow(p.theta_scale, float(i0) / 2.0f);
        } else if (int(sector) >= sec_w && int(sector) < sec_w + p.sections[2]) {
            theta_base = float(rope_data_pos[i2 + p.ne02 * 2]) * pow(p.theta_scale, float(i0) / 2.0f);
        } else if (int(sector) >= sec_w + p.sections[2]) {
            theta_base = float(rope_data_pos[i2 + p.ne02 * 3]) * pow(p.theta_scale, float(i0) / 2.0f);
        }
    }

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0 / 2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + p.n_dims / 2]);

    rope_data_d[idst + 0]            = ROPE_D_TYPE(x0 * cos_theta - x1 * sin_theta);
    rope_data_d[idst + p.n_dims / 2] = ROPE_D_TYPE(x0 * sin_theta + x1 * cos_theta);
}

void rope_vision(uint i0, uint i1, uint i2, uint i3, rope_params p) {
    if (i0 >= p.ne00) { return; }

    const uint idst = p.d_offset + i0 / 2 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
    const uint ix   = rope_a_coord(i0 / 2, i1, i2, i3, p);

    const int  sect_dims = p.sections[0] + p.sections[1];
    const int  sec_w     = p.sections[1] + p.sections[0];
    const uint sector    = (i0 / 2) % uint(sect_dims);

    float theta_base = 0.0f;
    if (int(sector) < p.sections[0]) {
        theta_base = float(rope_data_pos[i2]) * pow(p.theta_scale, float(sector));
    } else if (int(sector) >= p.sections[0] && int(sector) < sec_w) {
        const uint p0 = sector - uint(p.sections[0]);
        theta_base = float(rope_data_pos[i2 + p.ne02]) * pow(p.theta_scale, float(p0));
    }

    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0 / 2] : 1.0f;

    float cos_theta, sin_theta;
    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);

    const float x0 = float(rope_data_a[ix + 0]);
    const float x1 = float(rope_data_a[ix + p.n_dims]);

    rope_data_d[idst + 0]         = ROPE_D_TYPE(x0 * cos_theta - x1 * sin_theta);
    rope_data_d[idst + p.n_dims]  = ROPE_D_TYPE(x0 * sin_theta + x1 * cos_theta);
}

#endif
