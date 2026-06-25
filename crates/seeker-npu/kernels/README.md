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

## gemm/ — bf16 matmul (`C = A @ B`)

The whole_array IRON design (`whole_array.py`, vendored from the gpu-npu-demo
bring-up, Apache-2.0). Fixed-shape per (M, K, N); A is `[M,K]` row-major bf16, B
is `[K,N]` row-major bf16. Inputs are always bf16 (f32 accumulation internally);
the **output dtype is selectable** — `f32` (default) or `bf16`. The forward uses
`bf16` output so activations stay bf16 end-to-end (no f32↔bf16 cast). Tiling
requires `M % 256 == 0`, `K % 64 == 0`, `N % 256 == 0` (8 columns, 64×64×32 tile).

A 5th arg builds **`b_col_maj`** — B stored `[N,K]` (logical Bᵀ). The Qwen3 layer
uses this so the weight is fed exactly as GGUF stores it (`[out][in]`) while
activations stay token-major (`A = x[L,in]`), giving a transpose-free op chain
where the output `q[L,out]` keeps per-head 128-chunks contiguous. (b_col_maj adds
a transfer-block constraint: the token dim M must be a multiple of 512.)

```sh
# Generic (Qwen3 wq: q_dim x n_embd x L_bucket). 4th arg = output dtype.
crates/seeker-npu/kernels/gemm/build.sh 2048 1024 256        # bf16->f32 (gemm_2048x1024x256)
crates/seeker-npu/kernels/gemm/build.sh 2048 1024 256 bf16   # bf16->bf16 (gemm_..._bf16)
# Layer-orientation wq: M=L(512) K=n_embd N=q_dim, bf16, b_col_maj (5th arg = 1):
crates/seeker-npu/kernels/gemm/build.sh 512 1024 2048 bf16 1 # gemm_512x1024x2048_bcm_bf16
cargo run -p seeker-npu --example gemm                       # generic GEMM vs host f32
cargo run -p seeker-npu --example layer_proj                 # real-weight wq projection, cosine vs host
```

## eltwise/ — element-wise add / mul (f32 or bf16)

IRON `transform_binary` designs (`eltwise.py`), one xclbin per (op, dtype, N).
`add` is the transformer residual add; `mul` is the SwiGLU `gate * up` product
(and the RoPE rotation, paired with host sin/cos tables). N is the element count
(a multiple of the 1024 tile). The forward runs **bf16 activations end-to-end**
(GEMM is built bf16→bf16, with f32 accumulation internally), so no f32↔bf16 cast
kernel is needed; the `f32` variants are kept for reference/testing.

```sh
crates/seeker-npu/kernels/eltwise/build.sh add bf16 4096
crates/seeker-npu/kernels/eltwise/build.sh mul bf16 4096
cargo run -p seeker-npu --example eltwise      # validates add+mul × f32+bf16 vs host
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

`norm.py` builds per-tile bf16 kernels (N a multiple of `tile·8`).

- **rmsnorm** wires `aie2p/rms_norm.cc` via `ExternalFunction`: `out = x *
  invsqrt(mean(x²) + 1e-5)` over each `cols`-wide tile, with **gamma = 1** — the
  learned weight is applied separately with the eltwise `mul`. `cols=1024` is
  per-token (Qwen3 `n_embd`); `cols=128` is per-head (`head_dim`) for the q/k-norm.
- **softmax** uses the `aie.iron.kernels.softmax` wrapper (per-1024-tile LUT softmax;
  attention lays each query's padded score row into a tile).

```sh
crates/seeker-npu/kernels/norm/build.sh rmsnorm 8192 1024   # per-token  -> rmsnorm_1024_8192
crates/seeker-npu/kernels/norm/build.sh rmsnorm 8192 128    # per-head   -> rmsnorm_128_8192
crates/seeker-npu/kernels/norm/build.sh softmax 8192        # -> softmax_8192
cargo run -p seeker-npu --example norm                      # validates all vs host f32
```

