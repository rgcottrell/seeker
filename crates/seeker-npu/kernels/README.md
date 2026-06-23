# seeker-npu AIE kernels

AIE (XDNA2) compute kernels for the NPU embedding backend, authored as IRON
designs and compiled offline to `xclbin` + instruction blobs with the AMD AIE
toolchain (Python 3.12 venv with `mlir_aie` + `llvm-aie`/Peano). The compiled
artifacts are device- and shape-specific and **not** checked in (see
`../.gitignore`); regenerate them with the per-kernel build scripts.

## Prerequisites

- A Strix Halo NPU + XRT (`/opt/xilinx/xrt`).
- The AIE Python toolchain. The bring-up venv at `~/workspace/gpu-npu-demo/.venv`
  works; override with `SEEKER_NPU_VENV`.
- `RLIMIT_MEMLOCK` raised (XRT locks tens of MB; the default 8 MB cap fails with
  an `mmap … MAP_LOCKED … EAGAIN`). e.g. `memlock unlimited` in limits.d.

## gemm/ — bf16→f32 matmul (`C = A @ B`)

The whole_array IRON design (`whole_array.py`, vendored from the gpu-npu-demo
bring-up, Apache-2.0). Fixed-shape per (M, K, N); A is `[M,K]` row-major bf16, B
is `[K,N]` row-major bf16, C is `[M,N]` row-major f32. Tiling requires
`M % 256 == 0`, `K % 64 == 0`, `N % 256 == 0` (8 columns, default 64×64×32 tile);
pad the token dim N up to a 256 bucket.

```sh
# Build one shape (e.g. Qwen3 wq: q_dim x n_embd x L_bucket):
crates/seeker-npu/kernels/gemm/build.sh 2048 1024 256
# Validate it on the NPU against a host f32 reference:
cargo run -p seeker-npu --example gemm        # uses 2048x1024x256 by default
```

## eltwise/ — f32 element-wise add / mul

IRON `transform_binary` designs (`eltwise.py`), one xclbin per (op, N). `add` is
the transformer residual add; `mul` is the SwiGLU `gate * up` product. Both are
f32 in/out (the activation working dtype between GEMMs). N is the element count
(a multiple of the 1024 tile).

```sh
crates/seeker-npu/kernels/eltwise/build.sh add 4096
crates/seeker-npu/kernels/eltwise/build.sh mul 4096
cargo run -p seeker-npu --example eltwise      # validates add + mul vs host f32
```

## activation/ — bf16 SiLU

`activation.py` wires the shipped mlir_aie AIE2P LUT microkernel (`silu.cc`) into
an IRON design via `transform_parallel_typed` + `aie.iron.kernels.silu`. bf16
in/out, fixed LUT tile of 1024 → N must be a multiple of 8192 (8 cols × 1024).
SiLU is the SwiGLU gate activation; pair it with the eltwise `mul` for
`silu(gate) * up`. As a LUT bf16 kernel it matches the host reference within a few
percent (numpy `allclose` atol 2e-2 / rtol 3e-2).

```sh
crates/seeker-npu/kernels/activation/build.sh 8192
cargo run -p seeker-npu --example silu         # validates vs host x*sigmoid(x)
```

## norm/ — bf16 RMSNorm + softmax

`norm.py` builds two per-tile bf16 kernels (fixed tile 1024, N a multiple of 8192).
Qwen3's `n_embd == 1024 == tile`, so a per-1024-tile op is exactly per-token.

- **rmsnorm** wires `aie2p/rms_norm.cc` via `ExternalFunction`: `out = x *
  invsqrt(mean(x²) + 1e-5)` with **gamma = 1** — the learned RMSNorm weight is
  applied separately with the eltwise `mul`.
- **softmax** uses the `aie.iron.kernels.softmax` wrapper (per-1024-tile softmax;
  attention will lay scores out so each query's padded row fills a tile).

```sh
crates/seeker-npu/kernels/norm/build.sh rmsnorm 8192
crates/seeker-npu/kernels/norm/build.sh softmax 8192
cargo run -p seeker-npu --example norm         # validates both vs host f32
```

