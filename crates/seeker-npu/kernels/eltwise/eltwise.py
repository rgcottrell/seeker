# Element-wise binary kernels (add, mul) for the NPU embedding forward — IRON
# designs compiled with @iron.jit that wire the local eltwise.cc AIE microkernels
# (16-wide vectorised bf16/f32), replacing the earlier Python-lambda compute
# (which IRON lowered to a scalar per-element tile loop). f32 and bf16; used for
# the transformer residual adds and the SwiGLU gate*up product.
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions
# (derived from mlir_aie's vector_vector_add example).
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron.algorithms import transform_binary_typed
from aie.iron.kernels._common import _default_source_path, _include_dirs
from ml_dtypes import bfloat16

_ELTWISE_CC = str(Path(__file__).resolve().parent / "eltwise.cc")
_DTYPE_SUFFIX = {np.float32: "f32", bfloat16: "bf16"}


def _eltwise_kernel(op: str, dtype: type, tile_size: int) -> ExternalFunction:
    """Wire eltwise.cc's `eltwise_<op>_<dtype>` entry point as an ExternalFunction.

    `transform_binary_typed` calls the kernel as `(in0, in1, out, tile_size)` —
    it always appends the tile element count as a trailing int32 — so the arg
    types are three bf16/f32 tiles plus np.int32.
    """
    suffix = _DTYPE_SUFFIX[dtype]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    # eltwise.cc includes "aie_kernel_utils.h" (in the aie_kernels dir) + <aie_api/...>.
    aie_kernels_dir = str(_default_source_path("silu.cc").parent.parent)
    return ExternalFunction(
        f"eltwise_{op}_{suffix}",
        source_file=_ELTWISE_CC,
        arg_types=[tile_ty, tile_ty, tile_ty, np.int32],  # in0, in1, out, tile_size
        include_dirs=_include_dirs() + [aie_kernels_dir],
    )


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
    return transform_binary_typed(
        _eltwise_kernel("add", dtype, tile_size), tensor_ty, tile_size=tile_size
    )


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
    return transform_binary_typed(
        _eltwise_kernel("mul", dtype, tile_size), tensor_ty, tile_size=tile_size
    )


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
