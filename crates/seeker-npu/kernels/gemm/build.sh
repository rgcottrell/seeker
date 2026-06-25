#!/usr/bin/env bash
#
# Build a fixed-shape bf16 GEMM xclbin (+ instruction blob) for the Strix Halo NPU
# from the vendored IRON whole_array design, and copy the artifacts into
# build/gemm_<M>x<K>x<N>[_bcm][_<dtype_out>].{xclbin,insts.bin}.
#
# Usage:  build.sh <M> <K> <N> [dtype_out] [b_col_maj]
#           dtype_out = f32 (default) | bf16      b_col_maj = 0 (default) | 1
#   e.g.  build.sh 2048 1024 256          (Qwen3 wq, bf16->f32; gemm_2048x1024x256)
#         build.sh 2048 1024 256 bf16     (resident-activation forward; gemm_..._bf16)
#         build.sh 512 1024 2048 bf16 1   (layer projection, b_col_maj; gemm_..._bcm_bf16)
#
# Inputs are always bf16 (f32 accumulation internally). dtype_out=bf16 keeps the
# whole forward in bf16 so no f32<->bf16 cast is needed between ops. NOTE: M must be
# a multiple of 512 (transfer-block constraint, unconditional for the default c_col_maj).
#
# Requires the AIE toolchain (Python 3.12 venv with mlir_aie + llvm-aie/Peano) and
# XRT. The NPU also needs RLIMIT_MEMLOCK raised (XRT locks tens of MB).
set -euo pipefail

M=${1:?usage: build.sh M K N [dtype_out=f32|bf16] [b_col_maj=0|1]}
K=${2:?usage: build.sh M K N [dtype_out=f32|bf16] [b_col_maj=0|1]}
N=${3:?usage: build.sh M K N [dtype_out=f32|bf16] [b_col_maj=0|1]}
DTYPE_OUT=${4:-f32}
# B column-major: B stored [N,K] (logical Bᵀ). The Qwen3 layer uses this so the
# weight operand is fed exactly as GGUF stores it ([out][in]) while activations
# stay token-major (A = x[L,in]), giving a transpose-free op chain.
BCM=${5:-0}
# f32 + row-major keep the bare name (back-compat); col-major / other dtypes suffix.
suffix=""
[ "$BCM" = "1" ] && suffix="${suffix}_bcm"
[ "$DTYPE_OUT" != "f32" ] && suffix="${suffix}_${DTYPE_OUT}"

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

echo "### building GEMM ${M}x${K}x${N} (bf16->${DTYPE_OUT}, b_col_maj=${BCM}, 8 cols) ..."
# Run from a scratch dir so the design's relative imports/cache behave like the
# reference sweep; the design self-validates on the NPU and prints PASS/FAIL.
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
cp "$here/whole_array.py" "$workdir/wa.py"

# Build into an ISOLATED IRON cache by pointing HOME at the scratch dir, so the
# JIT's `~/.npu/cache` contains exactly THIS build's artifact. (Scraping the
# shared ~/.npu/cache with `ls -td` could pick up an unrelated or concurrent
# compile.) The trade-off is no warm-cache reuse across invocations — fine for a
# rarely-run, correctness-first build script.
(
  cd "$workdir"
  HOME="$workdir" python wa.py -M "$M" -K "$K" -N "$N" \
    --dtype_in bf16 --dtype_out "$DTYPE_OUT" --b-col-maj "$BCM" \
    --n-aie-cols 8 --dev npu2 -w 1 -i 1
)

cache=$(ls -td "$workdir"/.npu/cache/*/ 2>/dev/null | head -1)
if [ -z "$cache" ] || [ ! -f "$cache/final.xclbin" ]; then
  echo "error: build produced no final.xclbin under the isolated cache" >&2
  exit 1
fi
cp "$cache/final.xclbin" "$outdir/gemm_${M}x${K}x${N}${suffix}.xclbin"
cp "$cache/insts.bin" "$outdir/gemm_${M}x${K}x${N}${suffix}.insts.bin"
echo "wrote $outdir/gemm_${M}x${K}x${N}${suffix}.{xclbin,insts.bin}"
