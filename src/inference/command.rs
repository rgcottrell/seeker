//! Thin helpers around `vkCmd*` for the recording phase. The dispatch
//! context owns the active command buffer; these helpers just take it as
//! an argument.

use ash::vk;
use ash::vk::Handle as _;

use super::buffer::BufferRange;
use super::device::Device;
use super::pipeline::CachedPipeline;

/// Bind pipeline + descriptor set + push constants, then dispatch.
///
/// Used by the descriptor-pool dispatch path (`device.push_descriptor`
/// is false). `push` is the raw push-constant bytes — caller matches
/// the shader struct layout.
pub fn record_dispatch(
    device: &Device,
    cmd: vk::CommandBuffer,
    pipeline: &CachedPipeline,
    set: vk::DescriptorSet,
    push: &[u8],
    workgroups: [u32; 3],
) {
    unsafe {
        device.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        device.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[set],
            &[],
        );
        if !push.is_empty() {
            device.device.cmd_push_constants(
                cmd,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            );
        }
        device
            .device
            .cmd_dispatch(cmd, workgroups[0], workgroups[1], workgroups[2]);
    }
}

/// Push-descriptor dispatch path (Vulkan 1.4 core). Builds the
/// descriptor writes inline and emits `vkCmdPushDescriptorSet` instead
/// of allocating a set from the pool — saves one
/// alloc/update/bind round-trip per dispatch. The pipeline layout's
/// descriptor-set layout must have been created with
/// `PUSH_DESCRIPTOR_BIT_KHR` (see `pipeline::compile_compute_pipeline`).
pub fn record_dispatch_push(
    device: &Device,
    cmd: vk::CommandBuffer,
    pipeline: &CachedPipeline,
    binding_indices: &[u32],
    bindings: &[crate::inference::buffer::BufferRange],
    push: &[u8],
    workgroups: [u32; 3],
) {
    debug_assert_eq!(binding_indices.len(), bindings.len());
    // Stack-allocated scratch arrays — every dispatch was Vec-allocating
    // two short collections (DescriptorBufferInfo + WriteDescriptorSet)
    // and freeing them right after. 1500 dispatches/forward × 2 small
    // heap allocs added up. Cap at 8 bindings (max wired in any shader,
    // e.g. mul_mat_vec_q4_k_id at slot 7); panics on overflow.
    const MAX: usize = 8;
    debug_assert!(bindings.len() <= MAX, "raise MAX in record_dispatch_push");
    let mut infos: [vk::DescriptorBufferInfo; MAX] = [vk::DescriptorBufferInfo::default(); MAX];
    let mut writes: [vk::WriteDescriptorSet<'_>; MAX] = [vk::WriteDescriptorSet::default(); MAX];
    let n = bindings.len();
    for i in 0..n {
        infos[i] = bindings[i].descriptor_info();
    }
    for i in 0..n {
        writes[i] = vk::WriteDescriptorSet::default()
            .dst_binding(binding_indices[i])
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[i]));
    }
    let writes = &writes[..n];
    unsafe {
        device.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        device.device.cmd_push_descriptor_set(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            writes,
        );
        if !push.is_empty() {
            device.device.cmd_push_constants(
                cmd,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            );
        }
        device
            .device
            .cmd_dispatch(cmd, workgroups[0], workgroups[1], workgroups[2]);
    }
}

/// Compute→Compute barrier on a single buffer range. Replaces the prior
/// whole-buffer barrier — finer-grained scope lets RADV avoid flushing L2
/// for unrelated parts of the scratch region, which adds up across the
/// ~250 barriers a Llama-1B forward pass emits per token.
///
/// `SEEKER_BARRIER_PARANOID=1` falls back to a whole-buffer barrier — use
/// it to A/B-test correctness if a subtle race shows up downstream.
pub fn record_compute_barrier(device: &Device, cmd: vk::CommandBuffer, range: BufferRange) {
    record_compute_barriers(device, cmd, std::slice::from_ref(&range))
}

/// Compute→Compute barrier across several disjoint ranges in one
/// vkCmdPipelineBarrier call. Use after a batch of nofence dispatches
/// that wrote disjoint regions (Q/K/V matmuls, FFN gate/up, etc.) to
/// fence them all at once before downstream reads.
pub fn record_compute_barriers(device: &Device, cmd: vk::CommandBuffer, ranges: &[BufferRange]) {
    // Cached at first access; LazyLock keeps subsequent reads to a
    // single atomic-pointer load. This used to be `std::env::var(…)`,
    // which is a getenv + string-alloc per call — and this function
    // runs hundreds to ~1500 times per decode forward.
    let paranoid = *crate::runtime_flags::BARRIER_PARANOID;
    let bars: Vec<vk::BufferMemoryBarrier> = if paranoid {
        // Fall back to a whole-buffer barrier per unique buffer in the set.
        let mut buffers: Vec<vk::Buffer> = ranges.iter().map(|r| r.buffer).collect();
        buffers.sort_by_key(|b| b.as_raw());
        buffers.dedup();
        buffers
            .into_iter()
            .map(|buffer| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect()
    } else {
        ranges
            .iter()
            .map(|r| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(r.buffer)
                    .offset(r.offset)
                    .size(r.size)
            })
            .collect()
    };
    if bars.is_empty() {
        return;
    }
    unsafe {
        device.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &bars,
            &[],
        );
    }
}

/// Heavyweight memory barrier that synchronizes COMPUTE and TRANSFER access
/// to every buffer. Used between dispatches that share data through
/// `vkCmdCopyBuffer` (e.g. KV cache write / read).
pub fn record_global_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let bar = vk::MemoryBarrier::default()
        .src_access_mask(
            vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::TRANSFER_WRITE
                | vk::AccessFlags::TRANSFER_READ,
        )
        .dst_access_mask(
            vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::TRANSFER_WRITE
                | vk::AccessFlags::TRANSFER_READ,
        );
    unsafe {
        device.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&bar),
            &[],
            &[],
        );
    }
}

/// Issue a buffer-to-buffer copy. Used to pull logits from scratch into the
/// host-visible readback region at end-of-pass.
pub fn record_copy(
    device: &Device,
    cmd: vk::CommandBuffer,
    src: BufferRange,
    dst: BufferRange,
    size: u64,
) {
    let region = vk::BufferCopy {
        src_offset: src.offset,
        dst_offset: dst.offset,
        size,
    };
    unsafe {
        device
            .device
            .cmd_copy_buffer(cmd, src.buffer, dst.buffer, std::slice::from_ref(&region))
    };
}
