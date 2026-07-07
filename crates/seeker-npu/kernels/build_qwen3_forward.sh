#!/usr/bin/env bash
#
# Build every AIE xclbin the Qwen3-Embedding NPU forward (src/qwen3.rs) needs.
# Artifacts land in <kind>/build/ (gitignored) and are resolved at runtime from the
# crate's kernels dir (or $SEEKER_NPU_KERNEL_DIR). Requires the AIE toolchain + XRT;
# see the per-kind build.sh headers. Run from anywhere:
#
#   crates/seeker-npu/kernels/build_qwen3_forward.sh
#
# The DEFAULT (hybrid) forward runs only the GEMMs on the NPU — matmul with f32
# accumulation (f32-output) is what holds accuracy; RMSNorm/RoPE/softmax/SiLU run in
# f32 on the host. Pass `--onchip` to also build the bf16 norm/softmax/silu kernels for
# the SEEKER_NPU_ONCHIP_OPS path (lower accuracy; future f32-NPU-kernel work).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)

echo "### f32-output GEMMs (token block M=512; the hybrid forward's NPU work) ###"
# b_col_maj weight GEMMs: wq, wk/wv, wo, ffn gate/up, ffn down, and QKᵀ.
"$here/gemm/build.sh" 512 1024 2048 f32 1   # wq      (K=n_embd,  N=q_dim)
"$here/gemm/build.sh" 512 1024 1024 f32 1   # wk, wv  (K=n_embd,  N=kv_dim)
"$here/gemm/build.sh" 512 2048 1024 f32 1   # wo      (K=q_dim,   N=n_embd)
"$here/gemm/build.sh" 512 1024 3072 f32 1   # gate/up (K=n_embd,  N=n_ff)
"$here/gemm/build.sh" 512 3072 1024 f32 1   # down    (K=n_ff,    N=n_embd)
# Attention (GQA-batched: M=1024 = 2 Q-heads stacked, sharing one KV head's K/V).
"$here/gemm/build.sh" 1024 128 1024 f32 1   # QKᵀ     (K=head_dim, N=keys=1024)
# Row-major-B ·V GEMM (V feature dim padded 128->256 for the N%256 rule).
"$here/gemm/build.sh" 1024 1024 256 f32 0   # ·V      (K=keys, N=256)

echo "### 128-token bucket (short-prompt kernels; m=16 weights, m=32/n=16 attention) ###"
# Same shapes at M=128 so prompts of <=128 tokens run far fewer GEMM FLOPs (the
# forward picks the smallest bucket >= l). The smaller M needs -m 16 (M % 2·m·4 == 0);
# attention (N=128, no key/value padding at this size) also needs -n 16. The weight
# GEMMs also take -k 256 (8th arg): the deeper K-tile is ~20–30% faster at m=16 and
# still fits L1 (bit-identical result). The K=128 attention GEMMs keep the default k.
"$here/gemm/build.sh" 128 1024 2048 f32 1 16 "" 256   # wq
"$here/gemm/build.sh" 128 1024 1024 f32 1 16 "" 256   # wk, wv
"$here/gemm/build.sh" 128 2048 1024 f32 1 16 "" 256   # wo
"$here/gemm/build.sh" 128 1024 3072 f32 1 16 "" 256   # gate/up
"$here/gemm/build.sh" 128 3072 1024 f32 1 16 "" 256   # down
"$here/gemm/build.sh" 256 128 128 f32 1 32 16         # QKᵀ (M=gqa·128, keys=128)
"$here/gemm/build.sh" 256 128 128 f32 0 32 16         # ·V  (keys=128, vpad=128)

echo "### 256-token bucket (128 < l <= 256; m=32 weights) ###"
# M=256 needs -m 32. Attention is M=gqa·256=512 (default m=64); keys=256 and vpad=128
# are tile-aligned, so QKᵀ keeps the default -n and ·V needs -n 16 for N=128.
"$here/gemm/build.sh" 256 1024 2048 f32 1 32     # wq
"$here/gemm/build.sh" 256 1024 1024 f32 1 32     # wk, wv
"$here/gemm/build.sh" 256 2048 1024 f32 1 32     # wo
"$here/gemm/build.sh" 256 1024 3072 f32 1 32     # gate/up
"$here/gemm/build.sh" 256 3072 1024 f32 1 32     # down
"$here/gemm/build.sh" 512 128 256 f32 1          # QKᵀ (M=gqa·256, keys=256)
"$here/gemm/build.sh" 512 256 128 f32 0 64 16    # ·V  (keys=256, vpad=128)

if [ "${1:-}" = "--onchip" ]; then
  echo "### bf16 on-chip norm/softmax/silu (SEEKER_NPU_ONCHIP_OPS path) ###"
  "$here/norm/build.sh" rmsnorm 524288 1024     # input / ffn RMSNorm
  "$here/norm/build.sh" rmsnorm 1048576 128     # per-head q-norm
  "$here/norm/build.sh" rmsnorm 524288 128      # per-head k-norm
  "$here/norm/build.sh" softmax 1048576         # per-KV-pair attention softmax (mb·KEYS)
  "$here/eltwise/build.sh" mul bf16 524288      # ·norm-weight (n_embd width) + k rope muls
  "$here/eltwise/build.sh" mul bf16 1048576     # q norm-weight + q rope muls
  "$here/eltwise/build.sh" mul bf16 1572864     # SwiGLU silu(gate) * up
  "$here/eltwise/build.sh" add bf16 524288      # k rope add
  "$here/eltwise/build.sh" add bf16 1048576     # q rope add + per-KV-pair attention mask add
  "$here/activation/build.sh" 1572864           # SiLU
fi

echo "wrote $(ls "$here"/gemm/build/gemm_512x*_bcm.xclbin "$here"/gemm/build/gemm_512x1024x256.xclbin 2>/dev/null | wc -l) GEMM xclbins"
