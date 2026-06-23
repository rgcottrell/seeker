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
from ml_dtypes import bfloat16


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


_DTYPES = {"f32": np.float32, "bf16": bfloat16}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["add", "mul"], required=True)
    p.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    p.add_argument("-n", "--num-elements", type=int, default=4096)
    args = p.parse_args()
    dt = _DTYPES[args.dtype]

    in0 = iron.rand((args.num_elements,), dtype=dt, device="npu")
    in1 = iron.rand((args.num_elements,), dtype=dt, device="npu")
    out = iron.zeros_like(in0)

    design = eltwise_add if args.op == "add" else eltwise_mul
    design(in0, in1, out, num_elements=args.num_elements, dtype=dt)

    a = in0.numpy().astype(np.float32)
    b = in1.numpy().astype(np.float32)
    expected = (a + b) if args.op == "add" else (a * b)
    actual = out.numpy().astype(np.float32)
    # AIE rounds in a different order than numpy; bf16 also loses mantissa bits, so
    # use a dtype-appropriate tolerance (bit-exactness / assert_pass is too strict).
    rtol, atol = (1e-3, 1e-5) if args.dtype == "f32" else (3e-2, 2e-2)
    if np.allclose(actual, expected, rtol=rtol, atol=atol):
        print("PASS!")
    else:
        max_err = float(np.abs(actual - expected).max())
        print(f"FAIL! eltwise {args.op} {args.dtype} max_abs_err={max_err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
