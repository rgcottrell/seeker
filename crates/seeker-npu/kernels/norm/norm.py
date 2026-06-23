# Normalization kernels for the NPU embedding forward — IRON designs over the
# shipped mlir_aie AIE2P microkernels. bf16, fixed tile 1024 (8 cols × 1024 =>
# N multiple of 8192). Qwen3's n_embd == 1024 == the tile, so a per-1024-tile op
# is exactly per-token (per-row).
#
#   rmsnorm:  out = x * invsqrt(mean(x^2) + 1e-5)   (gamma = 1; the learned
#             RMSNorm weight is applied separately via the eltwise `mul` kernel)
#   softmax:  per-1024-tile softmax
#
# Derived from mlir_aie (Apache-2.0 WITH LLVM-exception): rmsnorm wires
# aie2p/rms_norm.cc via ExternalFunction; softmax uses the aie.iron.kernels wrapper.
import argparse

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron.algorithms import transform_parallel_typed
from aie.iron.kernels._common import _default_source_path, _include_dirs
from ml_dtypes import bfloat16

_LUT_TILE = 1024


def _rms_kernel() -> ExternalFunction:
    tile_ty = np.ndarray[(_LUT_TILE,), np.dtype[bfloat16]]
    # aie2p/rms_norm.cc: extern "C" rms_norm(bfloat16* in, bfloat16* out, int32 cols).
    return ExternalFunction(
        "rms_norm",
        source_file=str(_default_source_path("rms_norm.cc")),
        arg_types=[tile_ty, tile_ty, np.int32],
        include_dirs=_include_dirs(),
    )


@iron.jit
def rmsnorm(input0: In, output: Out, *, num_elements: CompileTime[int]):
    tile_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]
    # pass_size_to_kernel=True forwards the tile element count as `cols`.
    return transform_parallel_typed(_rms_kernel(), tile_ty, tile_size=_LUT_TILE)


@iron.jit
def softmax(input0: In, output: Out, *, num_elements: CompileTime[int]):
    tile_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]
    return transform_parallel_typed(kernels.softmax(tile_size=_LUT_TILE), tile_ty, tile_size=_LUT_TILE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["rmsnorm", "softmax"], required=True)
    p.add_argument("-n", "--num-elements", type=int, default=8192)
    args = p.parse_args()
    if args.num_elements % 8192 != 0:
        raise SystemExit("num_elements must be a multiple of 8192 (8 cols x 1024 tile)")

    x = iron.rand((args.num_elements,), dtype=bfloat16, device="npu")
    out = iron.zeros_like(x)
    (rmsnorm if args.op == "rmsnorm" else softmax)(x, out, num_elements=args.num_elements)

    xf = x.numpy().astype(np.float32).reshape(-1, _LUT_TILE)
    got = out.numpy().astype(np.float32).reshape(-1, _LUT_TILE)
    if args.op == "rmsnorm":
        ref = xf * (1.0 / np.sqrt((xf * xf).mean(1, keepdims=True) + 1e-5))
    else:
        e = np.exp(xf - xf.max(1, keepdims=True))
        ref = e / e.sum(1, keepdims=True)

    if np.allclose(got, ref, rtol=5e-2, atol=2e-2):
        print("PASS!")
    else:
        max_err = float(np.abs(ref - got).max())
        print(f"FAIL! {args.op} max_abs_err={max_err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
