//! Architecture-agnostic weight upload: walks every tensor in the GGUF and
//! memcpys it into the weights region. Models look up tensors by name
//! through the returned [`WeightsHandle`].

use std::collections::HashMap;
use std::error::Error;

use ash::vk;

use crate::gguf::{GgmlType, GgufFile};

use std::sync::Arc;

use super::buffer::BufferRange;
use super::device::{Device, DeviceShared};
use super::memory::DeviceBuffer;

/// Logical view of a tensor in GPU memory. Strides follow ggml convention:
/// `stride[0] = element_size`, `stride[i] = dims[i-1] * stride[i-1]`.
/// `byte_offset` is from the start of `buffer`.
#[derive(Debug, Clone, Copy)]
pub struct TensorView {
    pub buffer: vk::Buffer,
    pub byte_offset: u64,
    pub byte_size: u64,
    pub dims: [u64; 4],
    pub byte_stride: [u64; 4],
    pub element_stride: [u64; 4],
    pub dtype: GgmlType,
}

impl TensorView {
    pub fn range(&self) -> BufferRange {
        BufferRange {
            buffer: self.buffer,
            offset: self.byte_offset,
            size: self.byte_size,
        }
    }

    /// Buffer range shifted by `offset_bytes` from this view's start.
    /// Used to address per-token slices inside a multi-token tensor
    /// without rebuilding the whole TensorView.
    pub fn range_with_offset(&self, offset_bytes: u64) -> BufferRange {
        BufferRange {
            buffer: self.buffer,
            offset: self.byte_offset + offset_bytes,
            size: self.byte_size.saturating_sub(offset_bytes),
        }
    }
}

pub struct WeightsHandle {
    /// One buffer per tensor (no single buffer exceeds `maxBufferSize`).
    /// Owned here and freed in `Drop`.
    pub buffers: Vec<DeviceBuffer>,
    pub views: HashMap<String, TensorView>,
    /// Sum of tensor byte sizes uploaded (for logging).
    pub total_bytes: u64,
    /// Refcounted device owner — keeps the logical device alive until
    /// `Drop` has freed every per-tensor buffer, independent of whether the
    /// owning model/engine drops first.
    device: Arc<DeviceShared>,
}

impl WeightsHandle {
    pub fn view(&self, name: &str) -> Result<TensorView, Box<dyn Error>> {
        self.views
            .get(name)
            .copied()
            .ok_or_else(|| format!("weight tensor not found: {name}").into())
    }

    pub fn range(&self, name: &str) -> Result<BufferRange, Box<dyn Error>> {
        Ok(self.view(name)?.range())
    }

    /// Host base pointer of the buffer backing `view`. Weights are
    /// device-local, so this is always `None` now — the `gpu_debug`
    /// reference dumps that call it therefore fail fast (they predate the
    /// device-local move and would need a staging readback to work again).
    /// Kept as the natural hook if that's ever added.
    #[cfg(feature = "gpu_debug")]
    pub fn debug_host_base(&self, view: &TensorView) -> Option<*const u8> {
        self.buffers
            .iter()
            .find(|b| b.buffer == view.buffer)
            .and_then(|b| b.host_ptr)
            .map(|p| p as *const u8)
    }
}

impl Drop for WeightsHandle {
    fn drop(&mut self) {
        let dev = self.device.raw();
        for b in &self.buffers {
            b.destroy(dev);
        }
    }
}

/// Build the logical [`TensorView`] for a tensor living at offset 0 in its
/// own `buffer`.
fn build_view(t: &crate::gguf::TensorInfo, buffer: vk::Buffer) -> TensorView {
    let mut dims = [1u64; 4];
    for (i, d) in t.dims.iter().enumerate().take(4) {
        dims[i] = *d;
    }
    let element_size = element_size_bytes(t.ggml_type);
    let byte_stride = ggml_byte_strides(&dims, t.ggml_type);
    let mut element_stride = [0u64; 4];
    for i in 0..4 {
        element_stride[i] = byte_stride[i] / element_size.max(1);
    }
    TensorView {
        buffer,
        byte_offset: 0,
        byte_size: t.byte_size as u64,
        dims,
        byte_stride,
        element_stride,
        dtype: t.ggml_type,
    }
}

/// Borrow a tensor's bytes from the GGUF, erroring if the slice length
/// disagrees with the header.
fn tensor_bytes<'a>(
    gguf: &'a GgufFile,
    t: &crate::gguf::TensorInfo,
) -> Result<&'a [u8], Box<dyn Error>> {
    let data = gguf
        .tensor_data(&t.name)
        .ok_or_else(|| format!("tensor {} has no data slice", t.name))?;
    if data.len() != t.byte_size {
        return Err(format!(
            "tensor {}: data slice {} bytes != header byte_size {}",
            t.name,
            data.len(),
            t.byte_size
        )
        .into());
    }
    Ok(data)
}

/// Upload every tensor into **its own device-local buffer** — so no single
/// VkBuffer or allocation exceeds the device's `maxBufferSize` /
/// `maxMemoryAllocationSize` (the old single 27 GB region tripped both on
/// RADV). Each buffer is filled through one reused host-visible staging
/// buffer + `cmdCopyBuffer`, so weights land on the GPU's native heap and
/// are never host-mapped.
///
/// `cmd_pool` / `queue` / `fence` come from the engine (no forward pass is
/// in flight yet) and drive the staging copies.
pub fn upload(
    device: &Device,
    gguf: &GgufFile,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    fence: vk::Fence,
) -> Result<WeightsHandle, Box<dyn Error>> {
    let tensors = gguf.tensors();
    let mut views: HashMap<String, TensorView> = HashMap::with_capacity(tensors.len());
    let mut buffers: Vec<DeviceBuffer> = Vec::with_capacity(tensors.len());
    let mut total_bytes: u64 = 0;

    // One staging buffer sized to the largest tensor, reused for every copy;
    // one transient command buffer, reset + resubmitted per tensor (serial,
    // fence-synchronized).
    let max_size = tensors
        .iter()
        .map(|t| t.byte_size as u64)
        .max()
        .unwrap_or(1)
        .max(1);
    let staging = DeviceBuffer::new(
        device,
        max_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let staging_ptr = staging.host_ptr.ok_or("staging buffer was not mapped")?;

    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.device.allocate_command_buffers(&alloc) }?[0];

    let upload_result = (|| -> Result<(), Box<dyn Error>> {
        for t in tensors {
            let data = tensor_bytes(gguf, t)?;
            let dst = DeviceBuffer::new(
                device,
                t.byte_size as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            // SAFETY: staging.size == max tensor size >= data.len().
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), staging_ptr, data.len()) };
            unsafe {
                device
                    .device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device.device.begin_command_buffer(cmd, &begin)?;
                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: t.byte_size as u64,
                };
                device.device.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    dst.buffer,
                    std::slice::from_ref(&region),
                );
                device.device.end_command_buffer(cmd)?;
                device.device.reset_fences(std::slice::from_ref(&fence))?;
                let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
                device
                    .device
                    .queue_submit(queue, std::slice::from_ref(&submit), fence)?;
                device
                    .device
                    .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
            }
            views.insert(t.name.clone(), build_view(t, dst.buffer));
            total_bytes += t.byte_size as u64;
            buffers.push(dst);
        }
        Ok(())
    })();

    // Tear down upload-only resources regardless of success.
    staging.destroy(&device.device);
    unsafe {
        device
            .device
            .free_command_buffers(cmd_pool, std::slice::from_ref(&cmd))
    };
    upload_result?;

    Ok(WeightsHandle {
        buffers,
        views,
        total_bytes,
        device: device.shared(),
    })
}

/// Element size in bytes, used for computing strides. For quantized types
/// (blocks of K elements packed into M bytes), this is M/K rounded — used
/// here only for stride bookkeeping. Quantized matmul shaders read blocks
/// directly via their own offset math.
fn element_size_bytes(ty: GgmlType) -> u64 {
    let (block_size, type_size) = ty.block_layout();
    // Average bytes per element (rounded for quant). For unquantized types
    // block_size = 1.
    (type_size as u64 + (block_size as u64).saturating_sub(1)) / (block_size as u64).max(1)
}

/// ggml strides: `nb[0] = type_size / block_size` (1 for unquantized),
/// `nb[1] = nb[0] * ne[0]`, etc. For quantized, `nb[0] = type_size` and the
/// element_count along dim 0 is ne[0] / block_size — but we keep `dims`
/// representing logical element counts and shaders compute block offsets
/// internally.
fn ggml_byte_strides(dims: &[u64; 4], ty: GgmlType) -> [u64; 4] {
    let (block_size, type_size) = ty.block_layout();
    let mut nb = [0u64; 4];
    nb[0] = type_size as u64 / (block_size as u64).max(1);
    if block_size > 1 {
        nb[0] = type_size as u64;
    }
    // nb[1] traverses one row of length ne[0] elements. For unquantized,
    // that's nb[0] * ne[0]. For quantized, ne[0] / block_size blocks * type_size.
    if block_size > 1 {
        nb[1] = (dims[0] / block_size as u64).max(1) * type_size as u64;
    } else {
        nb[1] = nb[0] * dims[0];
    }
    nb[2] = nb[1] * dims[1];
    nb[3] = nb[2] * dims[2];
    nb
}
