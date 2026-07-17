//===- eltwise.cc ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
// Element-wise binary AIE microkernels (add, mul) for the NPU embedding forward.
// Hand-written C++ replacing the Python-lambda compute IRON used to synthesise
// for eltwise.py (`lambda a, b: a + b` / `a * b`), which lowered to a *scalar*
// per-element tile loop. These are 16-wide vectorised instead. `add` is the
// transformer residual add; `mul` is the SwiGLU gate*up product (and the RoPE
// rotation). bf16 is the activation working dtype end-to-end; f32 is kept for
// reference/testing. Derived from mlir_aie aie2/add.cc + aie2/mul.cc.
//
// The tile element count arrives as a trailing int32 (`transform_binary_typed`
// appends `tile_size` to every ExternalFunction call), so the kernels are
// size-generic — no hardcoded tile.
//===--------------------------------------------------------------------===//

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T>
void eltwise_add_vec(T *restrict a, T *restrict b, T *restrict c, int32_t n) {
  event0();
  constexpr int vec_factor = 16;
  T *__restrict pa = a;
  T *__restrict pb = b;
  T *__restrict pc = c;
  const int F = n / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<T, vec_factor> va = aie::load_v<vec_factor>(pa);
    pa += vec_factor;
    aie::vector<T, vec_factor> vb = aie::load_v<vec_factor>(pb);
    pb += vec_factor;
    aie::vector<T, vec_factor> vc = aie::add(va, vb);
    aie::store_v(pc, vc);
    pc += vec_factor;
  }
  event1();
}

template <typename T>
void eltwise_mul_vec(T *restrict a, T *restrict b, T *restrict c, int32_t n) {
  event0();
  constexpr int vec_factor = 16;
  T *__restrict pa = a;
  T *__restrict pb = b;
  T *__restrict pc = c;
  const int F = n / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<T, vec_factor> va = aie::load_v<vec_factor>(pa);
    pa += vec_factor;
    aie::vector<T, vec_factor> vb = aie::load_v<vec_factor>(pb);
    pb += vec_factor;
    aie::vector<T, vec_factor> vc = aie::mul(va, vb);
    aie::store_v(pc, vc);
    pc += vec_factor;
  }
  event1();
}

extern "C" {

void eltwise_add_bf16(bfloat16 *restrict a, bfloat16 *restrict b,
                      bfloat16 *restrict c, int32_t n) {
  eltwise_add_vec<bfloat16>(a, b, c, n);
}

void eltwise_mul_bf16(bfloat16 *restrict a, bfloat16 *restrict b,
                      bfloat16 *restrict c, int32_t n) {
  eltwise_mul_vec<bfloat16>(a, b, c, n);
}

void eltwise_add_f32(float *restrict a, float *restrict b, float *restrict c,
                     int32_t n) {
  eltwise_add_vec<float>(a, b, c, n);
}

void eltwise_mul_f32(float *restrict a, float *restrict b, float *restrict c,
                     int32_t n) {
  eltwise_mul_vec<float>(a, b, c, n);
}

} // extern "C"
