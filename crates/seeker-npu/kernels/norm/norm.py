# Normalization kernels for the NPU embedding forward — IRON designs over the
# shipped mlir_aie AIE2P microkernels (bf16).
#
#   rmsnorm:  out = x * invsqrt(mean(x^2) + 1e-5) over each `cols`-wide tile
#             (gamma = 1; the learned RMSNorm weight is applied separately via the
#             eltwise `mul`). cols=1024 == per-token (n_embd); cols=128 == per-head
#             (head_dim) for the Qwen3 q/k-norm. N must be a multiple of cols*8.
#   softmax:  per-1024-tile softmax (LUT kernel, fixed tile 1024; N multiple of 8192).
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

_SOFTMAX_TILE = 1024


def _rms_kernel(cols: int) -> ExternalFunction:
    tile_ty = np.ndarray[(cols,), np.dtype[bfloat16]]
    # aie2p/rms_norm.cc: extern "C" rms_norm(bfloat16* in, bfloat16* out, int32 cols).
    return ExternalFunction(
        "rms_norm",
        source_file=str(_default_source_path("rms_norm.cc")),
        arg_types=[tile_ty, tile_ty, np.int32],
        include_dirs=_include_dirs(),
    )


@iron.jit
def rmsnorm(input0: In, output: Out, *, num_elements: CompileTime[int], cols: CompileTime[int]):
    tile_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]
    # tile_size = cols, so each tile is one normalization group; pass_size_to_kernel
    # (default True) forwards the tile element count as the kernel's `cols` arg.
    return transform_parallel_typed(_rms_kernel(cols), tile_ty, tile_size=cols)


@iron.jit
def softmax(input0: In, output: Out, *, num_elements: CompileTime[int]):
    tile_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]
    return transform_parallel_typed(
        kernels.softmax(tile_size=_SOFTMAX_TILE), tile_ty, tile_size=_SOFTMAX_TILE
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["rmsnorm", "softmax"], required=True)
    p.add_argument("-n", "--num-elements", type=int, default=8192)
    p.add_argument("--cols", type=int, default=1024, help="rmsnorm group width (1024 token / 128 head)")
    args = p.parse_args()
    n = args.num_elements

    x = iron.rand((n,), dtype=bfloat16, device="npu")
    out = iron.zeros_like(x)
    if args.op == "rmsnorm":
        tile = args.cols
        if n % (tile * 8) != 0:
            raise SystemExit(f"num_elements must be a multiple of cols*8 ({tile * 8})")
        rmsnorm(x, out, num_elements=n, cols=tile)
    else:
        tile = _SOFTMAX_TILE
        if n % (tile * 8) != 0:
            raise SystemExit("num_elements must be a multiple of 8192 (8 cols x 1024 tile)")
        softmax(x, out, num_elements=n)

    xf = x.numpy().astype(np.float32).reshape(-1, tile)
    got = out.numpy().astype(np.float32).reshape(-1, tile)
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
