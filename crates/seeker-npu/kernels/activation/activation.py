# Activation kernels for the NPU embedding forward — IRON designs that wire the
# shipped mlir_aie AIE2P LUT microkernels (silu.cc etc.) via @iron.jit. bf16 tiles
# (the LUT kernels are bf16, fixed tile_size=1024). `silu` is the SwiGLU gate
# activation (paired with the eltwise `mul` for `silu(gate) * up`).
#
# Derived from mlir_aie's IRON kernel wrappers (Apache-2.0 WITH LLVM-exception).
import argparse

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform_parallel_typed
from ml_dtypes import bfloat16


@iron.jit
def silu(input0: In, output: Out, *, num_elements: CompileTime[int]):
    tile_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]
    # The LUT silu kernel is unary (no trailing size arg), so pass_size_to_kernel
    # must be False. 8 columns x 1024 tile => num_elements must be a multiple of 8192.
    return transform_parallel_typed(
        kernels.silu(tile_size=1024),
        tile_ty,
        tile_size=1024,
        pass_size_to_kernel=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num-elements", type=int, default=8192)
    args = p.parse_args()
    if args.num_elements % 8192 != 0:
        raise SystemExit("num_elements must be a multiple of 8192 (8 cols x 1024 tile)")

    x = iron.rand((args.num_elements,), dtype=bfloat16, device="npu")
    out = iron.zeros_like(x)
    silu(x, out, num_elements=args.num_elements)

    xf = x.numpy().astype(np.float32)
    ref = xf / (1.0 + np.exp(-xf))  # silu(x) = x * sigmoid(x)
    got = out.numpy().astype(np.float32)
    # bf16 LUT activation: a few % relative error is expected and acceptable.
    if np.allclose(got, ref, rtol=3e-2, atol=2e-2):
        print("PASS!")
    else:
        max_rel = float((np.abs(ref - got) / np.maximum(np.abs(ref), 1e-2)).max())
        print(f"FAIL! silu max_rel_err={max_rel}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
