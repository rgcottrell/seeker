//! Thin helpers around `vkCmd*` for the recording phase. The dispatch
//! context owns the active command buffer; these helpers just take it as
//! an argument.

use ash::vk;

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
    let infos: Vec<vk::DescriptorBufferInfo> =
        bindings.iter().map(|b| b.descriptor_info()).collect();
    let writes: Vec<vk::WriteDescriptorSet<'_>> = binding_indices
        .iter()
        .zip(infos.iter())
        .map(|(&binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
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
            &writes,
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

/// Compute→Compute barrier on the whole scratch buffer.
pub fn record_compute_barrier(device: &Device, cmd: vk::CommandBuffer, buffer: vk::Buffer) {
    let bar = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    unsafe {
        device.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&bar),
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
