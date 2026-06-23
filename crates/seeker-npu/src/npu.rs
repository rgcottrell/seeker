//! Safe wrapper over the [`crate::sys`] FFI for running AIE kernels on the Strix
//! Halo XDNA2 NPU.
//!
//! Usage mirrors the proven XRT convention: allocate an instruction buffer and
//! data buffers, write inputs through the mapped slices, sync every buffer to the
//! device, run, then sync the output back. Vendored from `~/workspace/gpu-npu-demo`
//! (the `npu` crate).
use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::path::Path;
use std::ptr;

use crate::sys;

/// An error returned by an NPU operation.
#[derive(Debug)]
pub struct Error {
    pub code: i32,
    pub msg: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "npu error (code {}): {}", self.code, self.msg)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn last_error() -> String {
    unsafe {
        let p = sys::npu_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn check(code: c_int) -> Result<()> {
    if code == sys::NPU_OK as c_int {
        Ok(())
    } else {
        Err(Error {
            code,
            msg: last_error(),
        })
    }
}

fn invalid<E: std::fmt::Display>(e: E) -> Error {
    Error {
        code: sys::NPU_ERR_INVALID_ARG as i32,
        msg: e.to_string(),
    }
}

/// An open NPU device + loaded xclbin + kernel.
pub struct Context {
    ptr: sys::npu_context_t,
}

// The underlying XRT handles are owned exclusively by this object.
unsafe impl Send for Context {}

impl Context {
    /// Open device 0, load `xclbin`, and bind the kernel named `kernel_name`.
    pub fn new(xclbin: &Path, kernel_name: &str) -> Result<Self> {
        let xclbin_c = CString::new(xclbin.to_string_lossy().as_bytes()).map_err(invalid)?;
        let kname_c = CString::new(kernel_name).map_err(invalid)?;
        let mut p: sys::npu_context_t = ptr::null_mut();
        check(unsafe { sys::npu_context_create(xclbin_c.as_ptr(), kname_c.as_ptr(), &mut p) })?;
        Ok(Context { ptr: p })
    }

    /// Allocate a host-mapped data buffer (kernel input/output).
    pub fn alloc_data(&self, size: usize) -> Result<Buffer> {
        self.alloc(size, sys::npu_buf_kind_t::NPU_BUF_DATA)
    }

    /// Allocate a cacheable instruction-sequence buffer.
    pub fn alloc_instr(&self, size: usize) -> Result<Buffer> {
        self.alloc(size, sys::npu_buf_kind_t::NPU_BUF_INSTR)
    }

    /// Wrap an existing page-aligned, host-pinned buffer as a userptr BO that the
    /// NPU shares zero-copy. The BO aliases `ptr` (so [`Buffer::map`]/slices see
    /// the same pages the host/GPU see); the caller owns `ptr` and must keep it
    /// alive for the returned buffer's lifetime.
    ///
    /// # Safety
    /// `ptr` must be valid, page-aligned, and at least `len` bytes for the buffer's life.
    pub unsafe fn import_host_ptr(&self, ptr: *mut u8, len: usize) -> Result<Buffer> {
        let mut p: sys::npu_buffer_t = ptr::null_mut();
        check(unsafe {
            sys::npu_buffer_import_userptr(self.ptr, ptr as *mut std::ffi::c_void, len, &mut p)
        })?;
        Ok(Buffer { ptr: p, size: len })
    }

    fn alloc(&self, size: usize, kind: sys::npu_buf_kind_t) -> Result<Buffer> {
        let mut p: sys::npu_buffer_t = ptr::null_mut();
        check(unsafe { sys::npu_buffer_alloc(self.ptr, size, kind, &mut p) })?;
        Ok(Buffer { ptr: p, size })
    }

    /// Run the kernel (opcode 3): instruction buffer + its byte count, then the
    /// data buffers bound to kernel args 3.. . Blocks until the run completes.
    pub fn run(&self, instr: &Buffer, ninstr_bytes: u32, buffers: &[&Buffer]) -> Result<()> {
        let mut raw: Vec<sys::npu_buffer_t> = buffers.iter().map(|b| b.ptr).collect();
        check(unsafe {
            sys::npu_run(
                self.ptr,
                instr.ptr,
                ninstr_bytes,
                raw.as_mut_ptr(),
                raw.len(),
            )
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { sys::npu_context_destroy(self.ptr) }
    }
}

/// A host-mapped XRT buffer object.
pub struct Buffer {
    ptr: sys::npu_buffer_t,
    size: usize,
}

unsafe impl Send for Buffer {}

impl Buffer {
    pub fn len_bytes(&self) -> usize {
        self.size
    }

    fn map_ptr(&self) -> *mut u8 {
        unsafe { sys::npu_buffer_map(self.ptr) as *mut u8 }
    }

    /// Raw host pointer to the buffer's mapped memory (page-aligned). Used for
    /// zero-copy import into the GPU via VK_EXT_external_memory_host.
    pub fn host_ptr(&self) -> *mut u8 {
        self.map_ptr()
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.map_ptr(), self.size) }
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.map_ptr(), self.size) }
    }

    pub fn as_slice<T: Copy>(&self) -> &[T] {
        let n = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(self.map_ptr() as *const T, n) }
    }

    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        let n = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(self.map_ptr() as *mut T, n) }
    }

    /// Make host writes visible to the NPU (required before a run, even for outputs).
    pub fn sync_to_device(&self) -> Result<()> {
        check(unsafe { sys::npu_buffer_sync_to_device(self.ptr) })
    }

    /// Make NPU writes visible to the host (required after a run, before reading).
    pub fn sync_from_device(&self) -> Result<()> {
        check(unsafe { sys::npu_buffer_sync_from_device(self.ptr) })
    }

    /// Export this buffer as a dma-buf fd for zero-copy import by the GPU. The fd
    /// is valid while this `Buffer` is alive; ownership of the fd passes to the caller.
    pub fn export_dmabuf(&self) -> Result<i32> {
        let fd = unsafe { sys::npu_buffer_export_dmabuf(self.ptr) };
        if fd < 0 {
            Err(Error {
                code: -1,
                msg: last_error(),
            })
        } else {
            Ok(fd)
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { sys::npu_buffer_free(self.ptr) }
    }
}
