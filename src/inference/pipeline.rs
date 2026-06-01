//! VkPipeline cache. Builds compute pipelines from one of the
//! `crate::shaders::<NAME>_SPV` constants on first request and caches them.
//!
//! Cache key includes spec-constant values so the same shader compiled with
//! different specializations (e.g. `do_multiply=false` vs `true` for
//! rms_norm) gets separate pipelines.

use std::collections::HashMap;
use std::error::Error;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::device::Device;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PipelineKey {
    pub name: String,
    /// Explicit binding indices used by the shader (set 0). Most shaders use
    /// the contiguous range `0..N`; flash_attn skips binding 4.
    pub binding_indices: Vec<u32>,
    pub push_size: u32,
    pub spec_constants: Vec<u32>,
    /// When set, the pipeline is built with
    /// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` so the driver
    /// can't pick a different wave width. Required by any shader that uses
    /// `WaveActiveSum` / cooperative matrix / hardcoded `SUBGROUP_SIZE`.
    /// Skipped (`None`) when the shader is wave-width agnostic.
    pub required_subgroup_size: Option<u32>,
}

impl PipelineKey {
    /// Helper for the common case of contiguous `0..n` bindings.
    pub fn dense(name: &str, n: u32, push_size: u32, spec_constants: Vec<u32>) -> Self {
        Self {
            name: name.to_string(),
            binding_indices: (0..n).collect(),
            push_size,
            spec_constants,
            required_subgroup_size: None,
        }
    }

    /// Pin the pipeline to a specific subgroup size. Use when the shader
    /// hardcodes a wave width (e.g. `WaveActiveSum` over 32 lanes).
    pub fn with_subgroup_size(mut self, size: u32) -> Self {
        self.required_subgroup_size = Some(size);
        self
    }
}

#[derive(Clone, Copy)]
pub struct CachedPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
}

pub struct PipelineCache {
    pipelines: HashMap<PipelineKey, CachedPipeline>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    /// Get or build a compute pipeline.
    ///
    /// - `key.name` identifies the shader (e.g. "rms_norm_f32").
    /// - `spirv` is the bytecode (4-byte aligned).
    /// - `key.bindings` is the descriptor set's binding count (all storage buffers).
    /// - `key.push_size` is the push-constant block byte size.
    /// - `key.spec_constants` is a list of u32 spec-constant values applied to
    ///   constant_id 0, 1, 2, … in order.
    pub fn get(
        &mut self,
        device: &Device,
        key: PipelineKey,
        spirv: &[u8],
    ) -> Result<&CachedPipeline, Box<dyn Error>> {
        if !self.pipelines.contains_key(&key) {
            let built = build_pipeline(device, spirv, &key)?;
            self.pipelines.insert(key.clone(), built);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            for (_, p) in self.pipelines.drain() {
                device.device.destroy_pipeline(p.pipeline, None);
                device.device.destroy_pipeline_layout(p.layout, None);
                device
                    .device
                    .destroy_descriptor_set_layout(p.set_layout, None);
            }
        }
    }
}

fn build_pipeline(
    device: &Device,
    spirv: &[u8],
    key: &PipelineKey,
) -> Result<CachedPipeline, Box<dyn Error>> {
    if !spirv.len().is_multiple_of(4) {
        return Err(format!("SPIR-V size {} not 4-byte aligned", spirv.len()).into());
    }
    // SAFETY: SPIR-V binaries are emitted by build.rs into Shader blocks
    // with 4-byte alignment, so a u32-aligned reinterpret is sound.
    let words =
        unsafe { std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4) };
    let module_info = vk::ShaderModuleCreateInfo::default().code(words);
    let module = unsafe { device.device.create_shader_module(&module_info, None) }?;

    let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = key
        .binding_indices
        .iter()
        .map(|&i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    // When `push_descriptor` is available (core in Vulkan 1.4), flag the
    // set layout `PUSH_DESCRIPTOR_BIT_KHR` so the dispatch path can use
    // `vkCmdPushDescriptorSet` and skip per-dispatch
    // alloc/update/bind on the descriptor pool. A layout created with
    // this flag is push-only — sets cannot be allocated from it — so the
    // dispatch helper must branch on `device.push_descriptor` to match.
    let set_layout_flags = if device.push_descriptor {
        vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
    } else {
        vk::DescriptorSetLayoutCreateFlags::empty()
    };
    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(set_layout_flags)
        .bindings(&layout_bindings);
    let set_layout = unsafe {
        device
            .device
            .create_descriptor_set_layout(&set_layout_info, None)
    }?;

    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(key.push_size);
    let push_ranges = if key.push_size > 0 {
        vec![push_range]
    } else {
        Vec::new()
    };
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&set_layout))
        .push_constant_ranges(&push_ranges);
    let layout = unsafe { device.device.create_pipeline_layout(&layout_info, None) }?;

    // Spec constants: pack values back-to-back, one entry per u32 with
    // constant_id 0, 1, 2, …
    let spec_data: Vec<u8> = key
        .spec_constants
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let spec_entries: Vec<vk::SpecializationMapEntry> = (0..key.spec_constants.len() as u32)
        .map(|i| vk::SpecializationMapEntry {
            constant_id: i,
            offset: i * 4,
            size: 4,
        })
        .collect();
    let spec_info = vk::SpecializationInfo::default()
        .map_entries(&spec_entries)
        .data(&spec_data);

    let entry = c"main";
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(entry);
    if !key.spec_constants.is_empty() {
        stage = stage.specialization_info(&spec_info);
    }
    // `req_sgs_info` must outlive the pipeline-create call because it's
    // chained onto `stage.pNext`; declare it at this scope so the lifetime
    // covers the call below.
    let mut req_sgs_info: vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'_>;
    if let Some(size) = key.required_subgroup_size {
        if !device
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
        {
            return Err(format!(
                "pipeline {:?} requested requiredSubgroupSize but device does not advertise \
                 subgroup_size_control on COMPUTE",
                key.name,
            )
            .into());
        }
        if size < device.min_subgroup_size || size > device.max_subgroup_size {
            return Err(format!(
                "pipeline {:?} requested requiredSubgroupSize={} outside device range [{}, {}]",
                key.name, size, device.min_subgroup_size, device.max_subgroup_size,
            )
            .into());
        }
        req_sgs_info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(size);
        stage = stage.push(&mut req_sgs_info);
    }
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipelines = unsafe {
        device.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create_compute_pipelines failed: {e:?}"))?;
    let pipeline = pipelines[0];

    unsafe { device.device.destroy_shader_module(module, None) };

    Ok(CachedPipeline {
        pipeline,
        layout,
        set_layout,
    })
}
