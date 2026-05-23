//! Bump-allocated VkDeviceMemory regions. Three regions total (weights,
//! scratch, readback) — see [`crate::inference`] module docs.

use std::error::Error;

use ash::vk;

use super::device::Device;

/// One contiguous VkDeviceMemory region with a single bound VkBuffer over it.
/// Suballocation is bump-style: handed out in offset order, never freed
/// individually. The whole region is freed on Drop.
pub struct Region {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub cursor: u64,
    pub alignment: u64,
    pub host_ptr: Option<*mut u8>,
    pub host_visible: bool,
    pub device_ptr_dropper: Option<()>, // placeholder for future explicit cleanup
}

unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// Pick a memory type satisfying `usage` + `required_flags`, allocate
    /// `size` bytes, create a single VkBuffer over the entire region. If
    /// the chosen memory type is host-visible, map it persistently.
    pub fn new(
        device: &Device,
        size: u64,
        usage: vk::BufferUsageFlags,
        required_flags: vk::MemoryPropertyFlags,
    ) -> Result<Self, Box<dyn Error>> {
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.device.create_buffer(&buf_info, None) }?;
        let reqs = unsafe { device.device.get_buffer_memory_requirements(buffer) };

        let mem_type = pick_memory_type(&device.mem_props, reqs.memory_type_bits, required_flags)
            .ok_or_else(|| {
                format!(
                    "no memory type satisfies bits={:#x} flags={:?}",
                    reqs.memory_type_bits, required_flags
                )
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type);
        let memory = match unsafe { device.device.allocate_memory(&alloc_info, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.device.destroy_buffer(buffer, None) };
                return Err(format!("vkAllocateMemory failed: {e:?} (size={})", reqs.size).into());
            }
        };
        unsafe { device.device.bind_buffer_memory(buffer, memory, 0) }?;

        let mt = device.mem_props.memory_types[mem_type as usize];
        let host_visible = mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        let host_ptr = if host_visible {
            let ptr = unsafe {
                device
                    .device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }?;
            Some(ptr as *mut u8)
        } else {
            None
        };

        Ok(Self {
            buffer,
            memory,
            size: reqs.size,
            cursor: 0,
            alignment: reqs.alignment.max(device.limits.min_storage_buffer_offset_alignment),
            host_ptr,
            host_visible,
            device_ptr_dropper: None,
        })
    }

    /// Reserve a `size`-byte slot aligned up to `self.alignment`. Returns the
    /// offset into the region's buffer. Errors when out of space.
    pub fn alloc(&mut self, size: u64) -> Result<u64, Box<dyn Error>> {
        let off = align_up(self.cursor, self.alignment);
        let end = off.checked_add(size).ok_or("alloc overflow")?;
        if end > self.size {
            return Err(format!(
                "region OOM: needed {size} at offset {off}, region size {}",
                self.size
            )
            .into());
        }
        self.cursor = end;
        Ok(off)
    }

    /// Reset the bump cursor to the start of the region. Use between forward
    /// passes for the scratch region.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            if self.host_ptr.is_some() {
                device.device.unmap_memory(self.memory);
                self.host_ptr = None;
            }
            device.device.destroy_buffer(self.buffer, None);
            device.device.free_memory(self.memory, None);
        }
    }
}

fn pick_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if type_bits & (1 << i) == 0 {
            continue;
        }
        let t = mem_props.memory_types[i as usize];
        if t.property_flags.contains(required) {
            return Some(i);
        }
    }
    None
}

fn align_up(v: u64, alignment: u64) -> u64 {
    (v + alignment - 1) & !(alignment - 1)
}
