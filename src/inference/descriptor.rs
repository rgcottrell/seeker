//! Single-pool descriptor allocator. The pool is sized generously for one
//! forward pass and reset between passes — no per-set deallocation.

use std::error::Error;

use ash::vk;

use super::buffer::BufferRange;
use super::device::Device;

pub const POOL_SET_CAPACITY: u32 = 1024;
pub const POOL_BUFFER_CAPACITY: u32 = 8192;

pub struct DescriptorAllocator {
    pub pool: vk::DescriptorPool,
}

impl DescriptorAllocator {
    pub fn new(device: &Device) -> Result<Self, Box<dyn Error>> {
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: POOL_BUFFER_CAPACITY,
        }];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(POOL_SET_CAPACITY)
            .pool_sizes(&sizes);
        let pool = unsafe { device.device.create_descriptor_pool(&info, None) }?;
        Ok(Self { pool })
    }

    /// Allocate one descriptor set with the given layout and bind the
    /// `bindings` slice into slots 0..N.
    pub fn allocate_and_write(
        &self,
        device: &Device,
        set_layout: vk::DescriptorSetLayout,
        bindings: &[BufferRange],
    ) -> Result<vk::DescriptorSet, Box<dyn Error>> {
        let layouts = [set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        let sets = unsafe { device.device.allocate_descriptor_sets(&alloc_info) }?;
        let set = sets[0];

        let infos: Vec<vk::DescriptorBufferInfo> =
            bindings.iter().map(|b| b.descriptor_info()).collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { device.device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }

    pub fn reset(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe {
            device
                .device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
        }?;
        Ok(())
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.device.destroy_descriptor_pool(self.pool, None) };
    }
}
