// C ABI over XRT's C++ API for driving an AIE (XDNA2 NPU) kernel.
// All fallible calls return an int status (0 == NPU_OK); on error, npu_last_error()
// returns a thread-local message. Opaque handles hide all xrt::* C++ types from Rust.
//
// Vendored from ~/workspace/gpu-npu-demo (npu-sys/shim) — the Strix Halo bring-up.
#ifndef NPU_SHIM_H
#define NPU_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_OK 0
#define NPU_ERR_INVALID_ARG 1
#define NPU_ERR_RUNTIME 2 /* xrt::error / std::exception — see npu_last_error() */

typedef struct npu_context *npu_context_t;
typedef struct npu_buffer *npu_buffer_t;

/* Buffer kinds map to the XRT allocation proven for the IRON "MLIR_AIE" kernel:
 *   NPU_BUF_DATA  -> host_only, group_id 0   (kernel data args)
 *   NPU_BUF_INSTR -> cacheable, group_id 1   (instruction sequence) */
typedef enum { NPU_BUF_DATA = 0, NPU_BUF_INSTR = 1 } npu_buf_kind_t;

/* Open device 0, load the xclbin, create a hw_context, and get the kernel by name. */
int npu_context_create(const char *xclbin_path, const char *kernel_name, npu_context_t *out);
void npu_context_destroy(npu_context_t ctx);

/* Allocate `size` bytes. The returned buffer is host-mapped; do I/O via npu_buffer_map(). */
int npu_buffer_alloc(npu_context_t ctx, size_t size, npu_buf_kind_t kind, npu_buffer_t *out);
/* Wrap an EXISTING page-aligned, host-pinned buffer as a userptr BO (uses the
 * hw_context + data memory group), so the NPU shares it zero-copy with the host/GPU.
 * The BO aliases `host_ptr`; freeing the BO does NOT free `host_ptr` (caller owns it). */
int npu_buffer_import_userptr(npu_context_t ctx, void *host_ptr, size_t size, npu_buffer_t *out);
void npu_buffer_free(npu_buffer_t buf);
void *npu_buffer_map(npu_buffer_t buf);
size_t npu_buffer_size(npu_buffer_t buf);
/* Every buffer (including outputs) must be synced TO_DEVICE before a run so it is
 * resident for the NPU; sync outputs FROM_DEVICE after the run before reading. */
int npu_buffer_sync_to_device(npu_buffer_t buf);
int npu_buffer_sync_from_device(npu_buffer_t buf);
/* Stretch (m4): export as a dma-buf fd for GPU import. Returns fd >= 0, or -1 on error. */
int npu_buffer_export_dmabuf(npu_buffer_t buf);

/* Run the kernel with opcode=3: instruction buffer + its byte count, then n data
 * buffers bound to kernel args 3..3+n. Blocks until completion. */
int npu_run(npu_context_t ctx, npu_buffer_t instr, uint32_t ninstr_bytes,
            npu_buffer_t *buffers, size_t n_buffers);

const char *npu_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* NPU_SHIM_H */
