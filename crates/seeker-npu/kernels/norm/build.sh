#!/usr/bin/env bash
#
# Build a fixed-size bf16 normalization xclbin (+ instruction blob) for the NPU.
#   rmsnorm -> build/rmsnorm_<cols>_<n>.{xclbin,insts.bin}   (cols = group width)
#   softmax -> build/softmax_<n>.{xclbin,insts.bin}
#
# Usage:  build.sh rmsnorm <n> [cols]    # cols = 1024 (per-token, default) | 128 (per-head)
#         build.sh softmax <n>           # n a multiple of 8192
#
# Same prerequisites as kernels/gemm/build.sh (XRT, the AIE venv, raised memlock).
set -euo pipefail

OP=${1:?usage: build.sh <rmsnorm|softmax> <n> [cols]}
N=${2:?usage: build.sh <rmsnorm|softmax> <n> [cols]}
COLS=${3:-1024}

XRT_SETUP=${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}
VENV=${SEEKER_NPU_VENV:-$HOME/workspace/gpu-npu-demo/.venv}

here=$(cd "$(dirname "$0")" && pwd)
outdir="$here/build"
mkdir -p "$outdir"

# shellcheck disable=SC1090
source "$XRT_SETUP" >/dev/null 2>&1
# shellcheck disable=SC1091
source "$VENV/bin/activate"

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
cp "$here/norm.py" "$workdir/norm.py"

# rmsnorm artifacts carry the group width; softmax is always per-1024-tile.
if [ "$OP" = "rmsnorm" ]; then
  stem="rmsnorm_${COLS}_${N}"
else
  stem="softmax_${N}"
fi

echo "### building ${stem} (bf16) ..."
# Isolated IRON cache (HOME -> workdir) so we copy exactly this build's artifact.
(
  cd "$workdir"
  HOME="$workdir" python norm.py --op "$OP" -n "$N" --cols "$COLS"
)

cache=$(ls -td "$workdir"/.npu/cache/*/ 2>/dev/null | head -1)
if [ -z "$cache" ] || [ ! -f "$cache/final.xclbin" ]; then
  echo "error: build produced no final.xclbin under the isolated cache" >&2
  exit 1
fi
cp "$cache/final.xclbin" "$outdir/${stem}.xclbin"
cp "$cache/insts.bin" "$outdir/${stem}.insts.bin"
echo "wrote $outdir/${stem}.{xclbin,insts.bin}"
