#!/usr/bin/env bash
#
# Build a fixed-shape bf16->f32 GEMM xclbin (+ instruction blob) for the Strix
# Halo NPU from the vendored IRON whole_array design, and copy the artifacts into
# build/gemm_<M>x<K>x<N>.{xclbin,insts.bin}.
#
# Usage:  build.sh <M> <K> <N>        # e.g. build.sh 2048 1024 256  (Qwen3 wq, L=256)
#
# Requires the AIE toolchain (Python 3.12 venv with mlir_aie + llvm-aie/Peano) and
# XRT. The NPU also needs RLIMIT_MEMLOCK raised (XRT locks tens of MB). The IRON
# JIT writes final.xclbin + insts.bin into ~/.npu/cache/<hash>/; we copy the newest.
set -euo pipefail

M=${1:?usage: build.sh M K N}
K=${2:?usage: build.sh M K N}
N=${3:?usage: build.sh M K N}

# Env overrides (defaults match the gpu-npu-demo bring-up on this box).
XRT_SETUP=${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}
VENV=${SEEKER_NPU_VENV:-$HOME/workspace/gpu-npu-demo/.venv}

here=$(cd "$(dirname "$0")" && pwd)
outdir="$here/build"
mkdir -p "$outdir"

# shellcheck disable=SC1090
source "$XRT_SETUP" >/dev/null 2>&1
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "### building GEMM ${M}x${K}x${N} (bf16->f32, 8 cols) ..."
# Run from a scratch dir so the design's relative imports/cache behave like the
# reference sweep; the design self-validates on the NPU and prints PASS/FAIL.
workdir=$(mktemp -d)
cp "$here/whole_array.py" "$workdir/wa.py"
(
  cd "$workdir"
  python wa.py -M "$M" -K "$K" -N "$N" \
    --dtype_in bf16 --dtype_out f32 --n-aie-cols 8 --dev npu2 -w 1 -i 1
)

cache=$(ls -td "$HOME"/.npu/cache/*/ | head -1)
cp "$cache/final.xclbin" "$outdir/gemm_${M}x${K}x${N}.xclbin"
cp "$cache/insts.bin" "$outdir/gemm_${M}x${K}x${N}.insts.bin"
rm -rf "$workdir"
echo "wrote $outdir/gemm_${M}x${K}x${N}.{xclbin,insts.bin}"
