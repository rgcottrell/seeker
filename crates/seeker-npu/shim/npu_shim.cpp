// Implementation of the C ABI in npu_shim.h over XRT's C++ API.
// Encodes the run convention validated against the IRON-generated vector_vector_add
// kernel: kernel "MLIR_AIE", opcode=3, instr BO (cacheable, group_id 1) + byte count,
// then host_only data BOs. Every body is wrapped in try/catch so no C++ exception
// crosses the C boundary.
//
// Vendored from ~/workspace/gpu-npu-demo (npu-sys/shim) — the Strix Halo bring-up.
#include "npu_shim.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <exception>
#include <string>

namespace {
thread_local std::string g_last_error;
void set_error(const std::string &m) { g_last_error = m; }
}  // namespace

struct npu_context {
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
};

struct npu_buffer {
  xrt::bo bo;
  size_t size = 0;
  void *mapped = nullptr;
};

extern "C" {

int npu_context_create(const char *xclbin_path, const char *kernel_name, npu_context_t *out) {
  if (!xclbin_path || !kernel_name || !out) return NPU_ERR_INVALID_ARG;
  try {
    auto *c = new npu_context();
    c->device = xrt::device(0);
    c->xclbin = xrt::xclbin(std::string(xclbin_path));
    c->device.register_xclbin(c->xclbin);
    c->context = xrt::hw_context(c->device, c->xclbin.get_uuid());
    c->kernel = xrt::kernel(c->context, kernel_name);
    *out = c;
    return NPU_OK;
  } catch (const std::exception &e) {
    set_error(e.what());
    return NPU_ERR_RUNTIME;
  }
}

void npu_context_destroy(npu_context_t ctx) { delete ctx; }

int npu_buffer_alloc(npu_context_t ctx, size_t size, npu_buf_kind_t kind, npu_buffer_t *out) {
  if (!ctx || !out || size == 0) return NPU_ERR_INVALID_ARG;
  try {
    auto *b = new npu_buffer();
    if (kind == NPU_BUF_INSTR) {
      b->bo = xrt::bo(ctx->device, size, xrt::bo::flags::cacheable, ctx->kernel.group_id(1));
    } else {
      b->bo = xrt::bo(ctx->device, size, xrt::bo::flags::host_only, 0);
    }
    b->size = size;
    b->mapped = b->bo.map<void *>();
    *out = b;
    return NPU_OK;
  } catch (const std::exception &e) {
    set_error(e.what());
    return NPU_ERR_RUNTIME;
  }
}

int npu_buffer_import_userptr(npu_context_t ctx, void *host_ptr, size_t size, npu_buffer_t *out) {
  if (!ctx || !host_ptr || !out || size == 0) return NPU_ERR_INVALID_ARG;
  try {
    auto *b = new npu_buffer();
    // userptr ctor takes the hw_context + memory group 0 (the host_only data bank),
    // which is what makes the imported pages reachable by the AIE compute DMA.
    b->bo = xrt::bo(ctx->context, host_ptr, size, xrt::bo::flags::host_only, 0);
    b->size = size;
    b->mapped = b->bo.map<void *>();
    *out = b;
    return NPU_OK;
  } catch (const std::exception &e) {
    set_error(e.what());
    return NPU_ERR_RUNTIME;
  }
}

void npu_buffer_free(npu_buffer_t buf) { delete buf; }
void *npu_buffer_map(npu_buffer_t buf) { return buf ? buf->mapped : nullptr; }
size_t npu_buffer_size(npu_buffer_t buf) { return buf ? buf->size : 0; }

int npu_buffer_sync_to_device(npu_buffer_t buf) {
  if (!buf) return NPU_ERR_INVALID_ARG;
  try {
    buf->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return NPU_OK;
  } catch (const std::exception &e) {
    set_error(e.what());
    return NPU_ERR_RUNTIME;
  }
}

int npu_buffer_sync_from_device(npu_buffer_t buf) {
  if (!buf) return NPU_ERR_INVALID_ARG;
  try {
    buf->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    return NPU_OK;
  } catch (const std::exception &e) {
    set_error(e.what());
    return NPU_ERR_RUNTIME;
  }
}

int npu_buffer_export_dmabuf(npu_buffer_t buf) {
  if (!buf) return -1;
  try {
    // On Linux xrt::bo::export_handle is an int32_t dma-buf fd. Valid while `buf`
    // is alive; ownership of the returned fd passes to the caller (e.g. Vulkan).
    return static_cast<int>(buf->bo.export_buffer());
  } catch (const std::exception &e) {
    set_error(e.what());
    return -1;
  }
}

int npu_run(npu_context_t ctx, npu_buffer_t instr, uint32_t ninstr_bytes, npu_buffer_t *buffers,
            size_t n_buffers) {
  if (!ctx || !instr || (!buffers && n_buffers)) return NPU_ERR_INVALID_ARG;
  try {
    xrt::run run(ctx->kernel);
    run.set_arg(0, static_cast<uint64_t>(3));  // opcode = 3 (execute instruction buffer)
    run.set_arg(1, instr->bo);                 // instruction sequence BO
    run.set_arg(2, ninstr_bytes);              // instruction byte count
    for (size_t i = 0; i < n_buffers; ++i)
      run.set_arg(static_cast<int>(3 + i), buffers[i]->bo);
    run.start();
    ert_cmd_state state = run.wait();
    if (state != ERT_CMD_STATE_COMPLETED) {
      set_error("kernel run did not complete: ert_cmd_state " + std::to_string(static_cast<int>(state)));
      return NPU_ERR_RUNTIME;
    }
    return NPU_OK;
  } catch (const std::exception &e) {
    set_error(e.what());
    return NPU_ERR_RUNTIME;
  }
}

const char *npu_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"
