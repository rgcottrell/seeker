//! Architecture-agnostic weight upload: walks every tensor in the GGUF and
//! memcpys it into the weights region. Models look up tensors by name
//! through the returned [`WeightsHandle`].

use std::collections::HashMap;
use std::error::Error;

use ash::vk;

use crate::gguf::{GgmlType, GgufFile};

use super::buffer::BufferRange;
use super::device::Device;
use super::memory::Region;

/// Logical view of a tensor uploaded to the GPU. Strides follow ggml
/// convention: `stride[0] = element_size`, `stride[i] = dims[i-1] * stride[i-1]`.
/// `byte_offset` is from the start of the weights buffer.
#[derive(Debug, Clone, Copy)]
pub struct TensorView {
    pub byte_offset: u64,
    pub byte_size: u64,
    pub dims: [u64; 4],
    pub byte_stride: [u64; 4],
    pub element_stride: [u64; 4],
    pub dtype: GgmlType,
}

impl TensorView {
    pub fn range(&self, buffer: vk::Buffer) -> BufferRange {
        BufferRange {
            buffer,
            offset: self.byte_offset,
            size: self.byte_size,
        }
    }
}

pub struct WeightsHandle {
    pub region: Region,
    pub views: HashMap<String, TensorView>,
}

impl WeightsHandle {
    pub fn view(&self, name: &str) -> Result<TensorView, Box<dyn Error>> {
        self.views
            .get(name)
            .copied()
            .ok_or_else(|| format!("weight tensor not found: {name}").into())
    }

    pub fn range(&self, name: &str) -> Result<BufferRange, Box<dyn Error>> {
        Ok(self.view(name)?.range(self.region.buffer))
    }
}

/// Compute the total bytes needed for all tensors, padded per-tensor to the
/// device's storage-buffer offset alignment.
pub fn required_bytes(device: &Device, gguf: &GgufFile) -> u64 {
    let align = device.limits.min_storage_buffer_offset_alignment.max(1);
    let mut total: u64 = 0;
    for t in gguf.tensors() {
        total = align_up(total, align);
        total += t.byte_size as u64;
    }
    total = align_up(total, align);
    total.max(1)
}

/// Allocate a weights region sized for this GGUF, copy each tensor's bytes
/// in via memcpy (host-visible path) or staging buffer (device-local-only).
/// MVP: require host-visible memory (true on Apple Silicon's unified memory
/// and on most discrete GPUs via BAR / ReBAR). Errors loudly otherwise.
pub fn upload(device: &Device, gguf: &GgufFile) -> Result<WeightsHandle, Box<dyn Error>> {
    let bytes = required_bytes(device, gguf);
    let region = Region::new(
        device,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let host_ptr = region
        .host_ptr
        .ok_or("weights region is not host-visible — staging-buffer upload not implemented yet")?;

    let mut views: HashMap<String, TensorView> = HashMap::with_capacity(gguf.tensors().len());
    let mut cursor: u64 = 0;
    let align = device.limits.min_storage_buffer_offset_alignment.max(1);

    for t in gguf.tensors() {
        cursor = align_up(cursor, align);
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
        // SAFETY: cursor + len <= region.size by construction (required_bytes
        // already accounts for the per-tensor padding); host_ptr is valid for
        // the entire region while the Region is alive.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                host_ptr.add(cursor as usize),
                data.len(),
            );
        }

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

        views.insert(
            t.name.clone(),
            TensorView {
                byte_offset: cursor,
                byte_size: t.byte_size as u64,
                dims,
                byte_stride,
                element_stride,
                dtype: t.ggml_type,
            },
        );
        cursor += t.byte_size as u64;
    }

    let mut region = region;
    region.cursor = cursor;
    Ok(WeightsHandle { region, views })
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

fn align_up(v: u64, alignment: u64) -> u64 {
    (v + alignment - 1) & !(alignment - 1)
}
