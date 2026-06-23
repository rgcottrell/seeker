// XRT-free stub implementation of npu_shim.h, compiled by build.rs when XILINX_XRT
// is absent (CI, non-Strix hosts). Every call fails cleanly so seeker-npu still
// compiles + links everywhere; real NPU execution requires building with XRT.
#include "npu_shim.h"

extern "C" {

int npu_context_create(const char *, const char *, npu_context_t *) { return NPU_ERR_RUNTIME; }
void npu_context_destroy(npu_context_t) {}
int npu_buffer_alloc(npu_context_t, size_t, npu_buf_kind_t, npu_buffer_t *) { return NPU_ERR_RUNTIME; }
int npu_buffer_import_userptr(npu_context_t, void *, size_t, npu_buffer_t *) { return NPU_ERR_RUNTIME; }
void npu_buffer_free(npu_buffer_t) {}
void *npu_buffer_map(npu_buffer_t) { return nullptr; }
size_t npu_buffer_size(npu_buffer_t) { return 0; }
int npu_buffer_sync_to_device(npu_buffer_t) { return NPU_ERR_RUNTIME; }
int npu_buffer_sync_from_device(npu_buffer_t) { return NPU_ERR_RUNTIME; }
int npu_buffer_export_dmabuf(npu_buffer_t) { return -1; }
int npu_run(npu_context_t, npu_buffer_t, uint32_t, npu_buffer_t *, size_t) { return NPU_ERR_RUNTIME; }
const char *npu_last_error(void) {
  return "seeker-npu was built without XRT (XILINX_XRT not found at build time)";
}

}  // extern "C"
