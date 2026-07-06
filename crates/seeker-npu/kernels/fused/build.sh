#!/usr/bin/env bash
#
# Build a fixed-shape FUSED FFN AIE kernel (xclbin + insts) from fused/ffn.py — the
# GEMM design with an on-chip activation epilogue. Artifacts land in
# fused/build/<stem>.{xclbin,insts.bin} (gitignored). Mirrors gemm/build.sh's
# isolated-HOME cache pattern.
#
# Usage:  build.sh <M> <K> <N> [silu=0|1] [b_col_maj=0|1]
#   e.g.  build.sh 512 1024 3072 1 1   (FFN gate: A=x2[512,1024] · Wgateᵀ -> SiLU)
#         build.sh swiglu <n>          (fused SwiGLU elementwise: silu(gate)*up, swiglu.py)
#
# Inputs bf16, output bf16. The fused SiLU epilogue requires the per-core (m, n)
# output tile to be 1024 elems (silu.cc is a fixed-1024 bf16 LUT), so this script
# pins -m 32 -n 32. Output stem: ffn_silu_<M>x<K>x<N>[_bcm] / ffn_<M>x<K>x<N>[_bcm].
set -euo pipefail

# ── fused SwiGLU elementwise kernel (swiglu.py + silu_mul.cc) ──
if [ "${1:-}" = "swiglu" ]; then
  n=${2:?usage: build.sh swiglu <n>}
  here=$(cd "$(dirname "$0")" && pwd)
  mkdir -p "$here/build"
  source "${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}" >/dev/null 2>&1
  source "${SEEKER_NPU_VENV:-$HOME/workspace/gpu-npu-demo/.venv}/bin/activate"
  echo "### building fused SwiGLU n=${n} (silu(gate)*up, bf16) ..."
  wd=$(mktemp -d)
  trap 'rm -rf "$wd"' EXIT
  ( cd "$wd"; HOME="$wd" python "$here/swiglu.py" -n "$n" )
  cache=$(ls -td "$wd"/.npu/cache/*/ 2>/dev/null | head -1)
  if [ -z "$cache" ] || [ ! -f "$cache/final.xclbin" ]; then
    echo "error: no final.xclbin under the isolated cache" >&2
    exit 1
  fi
  cp "$cache/final.xclbin" "$here/build/swiglu_${n}.xclbin"
  cp "$cache/insts.bin" "$here/build/swiglu_${n}.insts.bin"
  echo "wrote $here/build/swiglu_${n}.{xclbin,insts.bin}"
  exit 0
fi

M=${1:?usage: build.sh M K N [silu=0|1] [b_col_maj=0|1]}
K=${2:?usage: build.sh M K N [silu=0|1] [b_col_maj=0|1]}
N=${3:?usage: build.sh M K N [silu=0|1] [b_col_maj=0|1]}
SILU=${4:-1}
BCM=${5:-1}

stem="ffn"
[ "$SILU" = "1" ] && stem="${stem}_silu"
stem="${stem}_${M}x${K}x${N}"
[ "$BCM" = "1" ] && stem="${stem}_bcm"

XRT_SETUP=${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}
VENV=${SEEKER_NPU_VENV:-$HOME/workspace/gpu-npu-demo/.venv}

here=$(cd "$(dirname "$0")" && pwd)
outdir="$here/build"
mkdir -p "$outdir"

# shellcheck disable=SC1090
source "$XRT_SETUP" >/dev/null 2>&1
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "### building fused FFN ${M}x${K}x${N} (silu=${SILU}, bf16, b_col_maj=${BCM}, m=n=32, 8 cols) ..."
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
cp "$here/ffn.py" "$workdir/ffn.py"

(
  cd "$workdir"
  HOME="$workdir" python ffn.py -M "$M" -K "$K" -N "$N" -m 32 -k 64 -n 32 \
    --dtype_in bf16 --dtype_out bf16 --b-col-maj "$BCM" --silu "$SILU" \
    --n-aie-cols 8 --dev npu2 -w 1 -i 1
)

cache=$(ls -td "$workdir"/.npu/cache/*/ 2>/dev/null | head -1)
if [ -z "$cache" ] || [ ! -f "$cache/final.xclbin" ]; then
  echo "error: build produced no final.xclbin under the isolated cache" >&2
  exit 1
fi
cp "$cache/final.xclbin" "$outdir/${stem}.xclbin"
cp "$cache/insts.bin" "$outdir/${stem}.insts.bin"
echo "wrote $outdir/${stem}.{xclbin,insts.bin}"
