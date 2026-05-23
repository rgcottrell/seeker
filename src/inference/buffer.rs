//! Lightweight typed view over a region of a Vulkan buffer. Doesn't own
//! the underlying memory — that's [`super::memory::Region`].

use ash::vk;

#[derive(Debug, Clone, Copy)]
pub struct BufferRange {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
}

impl BufferRange {
    pub fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: self.offset,
            range: self.size,
        }
    }
}
