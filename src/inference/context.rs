//! Dispatch context — the handle a [`crate::models::Model`]'s
//! `record_forward` writes into. Owns the active command buffer, the scratch
//! region, the descriptor pool, and a borrow of the pipeline cache + device.

use std::error::Error;

use ash::vk;

use crate::gguf::GgmlType;

use super::buffer::BufferRange;
use super::descriptor::DescriptorAllocator;
use super::device::Device;
use super::memory::Region;
use super::pipeline::PipelineCache;
use super::weights::{TensorView, WeightsHandle};

pub struct DispatchContext<'a> {
    pub device: &'a Device,
    pub weights: &'a WeightsHandle,
    pub scratch: &'a mut Region,
    pub pipelines: &'a mut PipelineCache,
    pub descriptors: &'a DescriptorAllocator,
    pub cmd: vk::CommandBuffer,
}

impl<'a> DispatchContext<'a> {
    /// Reserve a `bytes`-byte slot in scratch and return its `BufferRange`.
    /// Cursor advances; the slot is valid only until the next forward pass.
    pub fn alloc_scratch(&mut self, bytes: u64) -> Result<BufferRange, Box<dyn Error>> {
        let off = self.scratch.alloc(bytes)?;
        Ok(BufferRange {
            buffer: self.scratch.buffer,
            offset: off,
            size: bytes,
        })
    }

    /// Reserve scratch space for a tensor with logical shape `dims` and the
    /// given `dtype`, contiguous layout (ggml convention).
    pub fn alloc_tensor(
        &mut self,
        dims: [u64; 4],
        dtype: GgmlType,
    ) -> Result<TensorView, Box<dyn Error>> {
        let (block_size, type_size) = dtype.block_layout();
        let mut byte_stride = [0u64; 4];
        byte_stride[0] = type_size as u64 / (block_size as u64).max(1);
        if block_size > 1 {
            byte_stride[0] = type_size as u64;
            byte_stride[1] = (dims[0] / block_size as u64).max(1) * type_size as u64;
        } else {
            byte_stride[1] = byte_stride[0] * dims[0];
        }
        byte_stride[2] = byte_stride[1] * dims[1];
        byte_stride[3] = byte_stride[2] * dims[2];
        let byte_size = byte_stride[3] * dims[3].max(1);

        let mut element_stride = [0u64; 4];
        let element_size = byte_stride[0].max(1);
        for i in 0..4 {
            element_stride[i] = byte_stride[i] / element_size;
        }

        let offset = self.scratch.alloc(byte_size)?;
        Ok(TensorView {
            buffer: self.scratch.buffer,
            byte_offset: offset,
            byte_size,
            dims,
            byte_stride,
            element_stride,
            dtype,
        })
    }
}
