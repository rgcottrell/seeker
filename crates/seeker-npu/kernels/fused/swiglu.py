# Fused SwiGLU elementwise kernel for the FFN: hidden = SiLU(gate) * up, computed in
# one NPU dispatch from the two projection outputs (replacing a host SiLU + host mul).
# Wires the local silu_mul.cc microkernel via transform_parallel_binary_typed (two
# input tiles -> one output). bf16 only; tile 1024 (silu_mul.cc's fixed LUT loop).
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron.algorithms import transform_parallel_binary_typed
from aie.iron.kernels._common import _default_source_path, _include_dirs
from ml_dtypes import bfloat16

_SILU_MUL_CC = str(Path(__file__).resolve().parent / "silu_mul.cc")


def _silu_mul_kernel(tile_size: int) -> ExternalFunction:
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    # silu_mul.cc includes "aie_kernel_utils.h" (in the aie_kernels dir) + <aie_api/...>.
    aie_kernels_dir = str(_default_source_path("silu.cc").parent.parent)
    return ExternalFunction(
        "silu_mul_bf16",
        source_file=_SILU_MUL_CC,
        arg_types=[tile_ty, tile_ty, tile_ty],  # gate, up, out
        include_dirs=_include_dirs() + [aie_kernels_dir],
    )


@iron.jit
def swiglu(
    gate: In,
    up: In,
    output: Out,
    *,
    num_elements: CompileTime[int],
    tile_size: CompileTime[int] = 1024,
):
    tile_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]
    return transform_parallel_binary_typed(
        _silu_mul_kernel(tile_size),
        tile_ty,
        tile_size=tile_size,
        pass_size_to_kernel=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num-elements", type=int, default=1572864)  # 512*3072
    args = p.parse_args()

    gate = iron.rand((args.num_elements,), dtype=bfloat16, device="npu")
    up = iron.rand((args.num_elements,), dtype=bfloat16, device="npu")
    out = iron.zeros_like(gate)
    swiglu(gate, up, out, num_elements=args.num_elements)

    g = gate.numpy().astype(np.float32)
    u = up.numpy().astype(np.float32)
    expected = (g / (1.0 + np.exp(-g))) * u  # SiLU(gate) * up
    actual = out.numpy().astype(np.float32)
    if np.allclose(actual, expected, rtol=3e-2, atol=2e-2):
        print("PASS!")
    else:
        max_err = float(np.abs(actual - expected).max())
        print(f"FAIL! swiglu max_abs_err={max_err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
