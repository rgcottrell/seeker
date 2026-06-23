# Element-wise binary kernels (add, mul) for the NPU embedding forward — IRON
# designs compiled with @iron.jit, modelled on the mlir_aie vector_vector_add
# example. f32 in/out (the activation working dtype between GEMMs). Used for the
# transformer residual adds and the SwiGLU gate*up product.
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions
# (derived from mlir_aie's vector_vector_add example).
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform_binary_typed


@iron.jit
def eltwise_add(
    input0: In,
    input1: In,
    output: Out,
    *,
    num_elements: CompileTime[int],
    dtype: CompileTime[type],
    tile_size: CompileTime[int] = 1024,
):
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    return transform_binary_typed(lambda a, b: a + b, tensor_ty, tile_size=tile_size)


@iron.jit
def eltwise_mul(
    input0: In,
    input1: In,
    output: Out,
    *,
    num_elements: CompileTime[int],
    dtype: CompileTime[type],
    tile_size: CompileTime[int] = 1024,
):
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    return transform_binary_typed(lambda a, b: a * b, tensor_ty, tile_size=tile_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["add", "mul"], required=True)
    p.add_argument("-n", "--num-elements", type=int, default=4096)
    args = p.parse_args()

    in0 = iron.rand((args.num_elements,), dtype=np.float32, device="npu")
    in1 = iron.rand((args.num_elements,), dtype=np.float32, device="npu")
    out = iron.zeros_like(in0)

    design = eltwise_add if args.op == "add" else eltwise_mul
    design(in0, in1, out, num_elements=args.num_elements, dtype=np.float32)

    a = in0.numpy()
    b = in1.numpy()
    expected = (a + b) if args.op == "add" else (a * b)
    actual = out.numpy()
    # f32 tolerance: the AIE vector op rounds in a different order than numpy, so
    # bit-exactness (assert_pass) is too strict — a 1-ULP difference is correct.
    if np.allclose(actual, expected, rtol=1e-3, atol=1e-5):
        print("PASS!")
    else:
        max_err = float(np.abs(actual - expected).max())
        print(f"FAIL! eltwise {args.op} max_abs_err={max_err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
