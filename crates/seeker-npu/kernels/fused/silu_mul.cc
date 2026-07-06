//===- silu_mul.cc --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
// Fused SwiGLU elementwise epilogue for the FFN: out = SiLU(gate) * up, where
// `gate` and `up` are the two projection outputs (per-core (m, n) tiles). This is
// the piece the shipped swiglu.cc does NOT provide — it fuses x*w1 and silu(x*w2),
// expecting the activation + two weights, whereas here gate and up are already the
// matmul results. Derived from aie2p/silu.cc.
//===--------------------------------------------------------------------===//

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

void silu_mul_tanh_approx_bf16(bfloat16 *restrict gate_vector,
                               bfloat16 *restrict up_vector,
                               bfloat16 *restrict output_vector,
                               const int32_t vector_size) {
  event0();

  int num_elems = vector_size;
  auto it_gate = aie::begin_restrict_vector<16>((bfloat16 *)gate_vector);
  auto it_up = aie::begin_restrict_vector<16>((bfloat16 *)up_vector);
  auto it_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> gate;
  aie::vector<bfloat16, 16> up;
  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 16> register_1 = aie::broadcast<bfloat16, 16>(1.0f);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(64)
  for (int i = 0; i < num_elems; i += 16) {
    gate = *it_gate++;
    up = *it_up++;

    // SiLU(gate) = gate * sigmoid(gate), sigmoid via the tanh approximation
    // sigmoid(x) = 0.5 * (1 + tanh(0.5 x)).
    auto half_x = aie::mul(gate, register_0_5);
    auto tanh_half_x = aie::tanh<bfloat16>(half_x.to_vector<float>());
    auto tanh_half_x_approx = aie::add(tanh_half_x, register_1);
    aie::vector<bfloat16, 16> sigmoid_approx =
        aie::mul(tanh_half_x_approx, register_0_5);
    aie::vector<bfloat16, 16> silu_gate =
        aie::mul(gate, sigmoid_approx).to_vector<bfloat16>();

    // out = SiLU(gate) * up
    auto out = aie::mul(silu_gate, up);
    *it_out++ = out.to_vector<bfloat16>();
  }

  event1();

  return;
}

extern "C" {

void silu_mul_bf16(bfloat16 *restrict gate, bfloat16 *restrict up,
                   bfloat16 *restrict output) {
  int32_t input_size = 1024; // per-core (m, n) tile = 1024 elems
  silu_mul_tanh_approx_bf16(gate, up, output, input_size);
}

} // extern "C"
