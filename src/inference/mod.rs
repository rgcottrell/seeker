//! Vulkan inference runtime — architecture-agnostic.
//!
//! Owns the Vulkan device, GPU memory regions, pipeline cache, descriptor
//! pool, and the per-op dispatch recorders in [`ops`]. Knows nothing about
//! LLaMA, Qwen, or any specific transformer arch — those live in
//! `crate::models::*` and use [`context::DispatchContext`] + the [`ops`]
//! helpers to record their forward pass.

pub mod buffer;
pub mod command;
pub mod context;
pub mod descriptor;
pub mod device;
pub mod kv_cache;
pub mod memory;
pub mod ops;
pub mod pipeline;
pub mod sample;
pub mod weights;

use std::error::Error;

use ash::vk;

use crate::gguf::GgufFile;

use buffer::BufferRange;
use context::DispatchContext;
use descriptor::DescriptorAllocator;
use device::Device;
use memory::Region;
use pipeline::PipelineCache;
use weights::WeightsHandle;

/// Top-level runtime. Built once, then used to upload weights and run
/// multiple forward passes against any [`crate::models::Model`].
pub struct Engine {
    pub device: Device,
    pub pipelines: PipelineCache,
    pub descriptors: DescriptorAllocator,
    pub scratch: Region,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl Engine {
    pub fn new(scratch_bytes: u64) -> Result<Self, Box<dyn Error>> {
        let device = Device::new()?;
        let pipelines = PipelineCache::new();
        let descriptors = DescriptorAllocator::new(&device)?;

        // Scratch is host-visible + device-local. On Apple Silicon (unified
        // memory) this maps trivially; on discrete GPUs with BAR/ReBAR this
        // also works. If neither is available we'd need a staging path.
        let scratch = Region::new(
            &device,
            scratch_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.device.create_command_pool(&pool_info, None) }?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.device.allocate_command_buffers(&alloc_info) }?[0];

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.device.create_fence(&fence_info, None) }?;

        Ok(Self {
            device,
            pipelines,
            descriptors,
            scratch,
            command_pool,
            command_buffer,
            fence,
        })
    }

    /// Upload every tensor in `gguf` into a new dedicated weights region.
    pub fn upload_weights(&self, gguf: &GgufFile) -> Result<WeightsHandle, Box<dyn Error>> {
        weights::upload(&self.device, gguf)
    }

    /// Allocate a KV cache sized for the given architecture. Caller picks
    /// dtypes (independently for K and V) and `max_seq_len`.
    pub fn allocate_kv_cache(
        &self,
        n_layer: u32,
        head_dim: u32,
        n_head_kv: u32,
        config: kv_cache::KvCacheConfig,
    ) -> Result<kv_cache::KvCache, Box<dyn Error>> {
        kv_cache::KvCache::new(&self.device, n_layer, head_dim, n_head_kv, config)
    }

    /// Run a forward pass: the closure records dispatches into the
    /// `DispatchContext` and returns the `BufferRange` containing the final
    /// logits (vocab_size F32s). The engine handles begin/end/submit/wait
    /// and reads the logits back as a `Vec<f32>`.
    pub fn forward<F>(
        &mut self,
        weights: &WeightsHandle,
        record: F,
    ) -> Result<Vec<f32>, Box<dyn Error>>
    where
        F: FnOnce(&mut DispatchContext) -> Result<BufferRange, Box<dyn Error>>,
    {
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;
        crate::inference::context::refresh_diff_dump_flag();

        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }

        let (logits_range, taps) = {
            let mut ctx = DispatchContext {
                device: &self.device,
                weights,
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd: self.command_buffer,
                taps: Vec::new(),
            };
            let r = record(&mut ctx)?;
            (r, ctx.taps)
        };

        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        // Read logits back from scratch's host pointer.
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region is not host-visible — readback path requires a staging buffer")?;
        if logits_range.size % 4 != 0 {
            return Err(format!("logits size {} not 4-byte aligned", logits_range.size).into());
        }
        let count = (logits_range.size / 4) as usize;
        let mut out = vec![0f32; count];
        // SAFETY: logits_range refers to a region inside self.scratch (its
        // buffer is self.scratch.buffer) and is fully within its bounds.
        unsafe {
            let src = host_ptr.add(logits_range.offset as usize) as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), count);
        }

        // Print sums for any taps the model recorded. Used for layer-by-layer
        // diff dumps vs llama.cpp's `cb()` callback. Output is one line per
        // tap: `TAP <name> n=<count> sum=<value> max_abs=<value>`.
        for (name, range) in &taps {
            if range.size % 4 != 0 {
                eprintln!("TAP {name}: size {} not 4-byte aligned, skipping", range.size);
                continue;
            }
            let n = (range.size / 4) as usize;
            let mut buf = vec![0f32; n];
            unsafe {
                let src = host_ptr.add(range.offset as usize) as *const f32;
                std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), n);
            }
            let sum: f32 = buf.iter().sum();
            let max_abs: f32 = buf.iter().map(|x| x.abs()).fold(0.0, f32::max);
            let head: Vec<String> = buf.iter().take(5).map(|v| format!("{v:.4}")).collect();
            println!("TAP {name} n={n} off={} sum={sum:.6} max_abs={max_abs:.6} head=[{}]", range.offset, head.join(", "));
        }
        Ok(out)
    }

    /// Run a forward pass and sample a token, all on the GPU. The closure
    /// records the model forward and returns the logits `TensorView`; the
    /// `sampler` then appends its chain into the same command buffer. After
    /// submit/wait the engine reads back exactly 4 bytes — the sampled token
    /// id — instead of pulling the full logits buffer to host. The sampler's
    /// recent-token window is updated automatically via `accept`.
    pub fn forward_sampled<F>(
        &mut self,
        weights: &WeightsHandle,
        sampler: &mut sample::Sampler,
        record_logits: F,
    ) -> Result<u32, Box<dyn Error>>
    where
        F: FnOnce(&mut DispatchContext) -> Result<weights::TensorView, Box<dyn Error>>,
    {
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;
        crate::inference::context::refresh_diff_dump_flag();

        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }

        let (token_range, taps) = {
            let mut ctx = DispatchContext {
                device: &self.device,
                weights,
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd: self.command_buffer,
                taps: Vec::new(),
            };
            let logits = record_logits(&mut ctx)?;
            let r = sampler.record_chain(&mut ctx, logits)?;
            (r, ctx.taps)
        };

        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        if token_range.size < 4 {
            return Err(format!("sampler output too small: {} bytes", token_range.size).into());
        }
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region is not host-visible — readback requires host-visible scratch")?;
        let token = unsafe {
            let src = host_ptr.add(token_range.offset as usize) as *const u32;
            std::ptr::read(src)
        };
        // Print tap summaries (same logic as in `forward`). Used for diff
        // dumps vs llama.cpp's cb() callback.
        for (name, range) in &taps {
            if range.size % 4 != 0 {
                eprintln!("TAP {name}: size {} not 4-byte aligned, skipping", range.size);
                continue;
            }
            let n = (range.size / 4) as usize;
            let mut buf = vec![0f32; n];
            unsafe {
                let src = host_ptr.add(range.offset as usize) as *const f32;
                std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), n);
            }
            let sum: f32 = buf.iter().sum();
            let max_abs: f32 = buf.iter().map(|x| x.abs()).fold(0.0, f32::max);
            let head: Vec<String> = buf.iter().take(5).map(|v| format!("{v:.4}")).collect();
            println!("TAP {name} n={n} off={} sum={sum:.6} max_abs={max_abs:.6} head=[{}]", range.offset, head.join(", "));
        }
        sampler.accept(token);
        Ok(token)
    }

    /// Write F32 data into a scratch slot via the mapped host pointer. Used
    /// for inputs that originate on the CPU side (token id positions, etc.).
    pub fn write_scratch_f32(&self, range: BufferRange, data: &[f32]) -> Result<(), Box<dyn Error>> {
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible")?;
        let bytes = std::mem::size_of_val(data);
        if bytes as u64 > range.size {
            return Err(format!("write_scratch_f32: {bytes} > range.size {}", range.size).into());
        }
        unsafe {
            let dst = host_ptr.add(range.offset as usize) as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }

    /// Write u32 data into a scratch slot (e.g. token ids for get_rows).
    pub fn write_scratch_u32(&self, range: BufferRange, data: &[u32]) -> Result<(), Box<dyn Error>> {
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible")?;
        let bytes = std::mem::size_of_val(data);
        if bytes as u64 > range.size {
            return Err(format!("write_scratch_u32: {bytes} > range.size {}", range.size).into());
        }
        unsafe {
            let dst = host_ptr.add(range.offset as usize) as *mut u32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device.device_wait_idle();
            self.device.device.destroy_fence(self.fence, None);
            self.device.device.destroy_command_pool(self.command_pool, None);
        }
        self.scratch.destroy(&self.device);
        self.descriptors.destroy(&self.device);
        self.pipelines.destroy(&self.device);
    }
}
