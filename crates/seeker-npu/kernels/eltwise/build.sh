#!/usr/bin/env bash
#
# Build a fixed-size f32 element-wise kernel xclbin (+ instruction blob) for the
# NPU and copy the artifacts to build/eltwise_<op>_<n>.{xclbin,insts.bin}.
#
# Usage:  build.sh <op> <n>          # op = add|mul,  n = element count (multiple of 1024)
#
# Same prerequisites as kernels/gemm/build.sh (XRT, the AIE venv, raised memlock).
set -euo pipefail

OP=${1:?usage: build.sh <add|mul> <n>}
N=${2:?usage: build.sh <add|mul> <n>}

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
cp "$here/eltwise.py" "$workdir/eltwise.py"

echo "### building eltwise ${OP} n=${N} (f32) ..."
# Isolated IRON cache (HOME -> workdir) so we copy exactly this build's artifact.
(
  cd "$workdir"
  HOME="$workdir" python eltwise.py --op "$OP" -n "$N"
)

cache=$(ls -td "$workdir"/.npu/cache/*/ 2>/dev/null | head -1)
if [ -z "$cache" ] || [ ! -f "$cache/final.xclbin" ]; then
  echo "error: build produced no final.xclbin under the isolated cache" >&2
  exit 1
fi
cp "$cache/final.xclbin" "$outdir/eltwise_${OP}_${N}.xclbin"
cp "$cache/insts.bin" "$outdir/eltwise_${OP}_${N}.insts.bin"
echo "wrote $outdir/eltwise_${OP}_${N}.{xclbin,insts.bin}"
