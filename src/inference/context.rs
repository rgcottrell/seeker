//! Dispatch context — the handle a [`crate::models::Model`]'s
//! `record_forward` writes into. Owns the active command buffer, the scratch
//! region, the descriptor pool, and a borrow of the pipeline cache + device.

use std::error::Error;

use ash::vk;

use super::descriptor::DescriptorAllocator;
use super::device::Device;
use super::memory::Region;
use super::pipeline::PipelineCache;
use super::weights::WeightsHandle;

pub struct DispatchContext<'a> {
    pub device: &'a Device,
    pub weights: &'a WeightsHandle,
    pub scratch: &'a mut Region,
    pub pipelines: &'a mut PipelineCache,
    pub descriptors: &'a DescriptorAllocator,
    pub cmd: vk::CommandBuffer,
}

impl<'a> DispatchContext<'a> {
    /// Reserve a `bytes`-byte slot in scratch and return its
    /// `BufferRange`. Cursor advances; the slot is valid only until the
    /// next scratch reset (which happens once per forward pass).
    pub fn alloc_scratch(&mut self, bytes: u64) -> Result<super::buffer::BufferRange, Box<dyn Error>> {
        let off = self.scratch.alloc(bytes)?;
        Ok(super::buffer::BufferRange {
            buffer: self.scratch.buffer,
            offset: off,
            size: bytes,
        })
    }
}
