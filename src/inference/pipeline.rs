//! VkPipeline cache. Builds compute pipelines from one of the
//! `crate::shaders::<NAME>_SPV` constants on first request and caches them.

use std::collections::HashMap;
use std::error::Error;

use ash::vk;

use super::device::Device;

pub struct PipelineCache {
    pipelines: HashMap<(String, u32, u32), CachedPipeline>,
}

pub struct CachedPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    /// Get or build a compute pipeline. `key` identifies the shader (e.g.
    /// "rms_norm_f32"), `spirv` is the bytecode, `bindings` is the number of
    /// storage-buffer bindings (set=0, binding=0..n-1), `push_size` is the
    /// push-constant block byte size.
    pub fn get(
        &mut self,
        device: &Device,
        key: &str,
        spirv: &[u8],
        bindings: u32,
        push_size: u32,
    ) -> Result<&CachedPipeline, Box<dyn Error>> {
        let cache_key = (key.to_string(), bindings, push_size);
        if !self.pipelines.contains_key(&cache_key) {
            let built = build_pipeline(device, spirv, bindings, push_size)?;
            self.pipelines.insert(cache_key.clone(), built);
        }
        Ok(self.pipelines.get(&cache_key).unwrap())
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            for (_, p) in self.pipelines.drain() {
                device.device.destroy_pipeline(p.pipeline, None);
                device.device.destroy_pipeline_layout(p.layout, None);
                device.device.destroy_descriptor_set_layout(p.set_layout, None);
            }
        }
    }
}

fn build_pipeline(
    device: &Device,
    spirv: &[u8],
    bindings: u32,
    push_size: u32,
) -> Result<CachedPipeline, Box<dyn Error>> {
    if spirv.len() % 4 != 0 {
        return Err(format!("SPIR-V size {} not 4-byte aligned", spirv.len()).into());
    }
    // SAFETY: SPIR-V binaries are emitted by build.rs into Aligned4 blocks,
    // so a u32-aligned reinterpret is sound. We require alignment above for
    // any future callers that bypass the static path.
    let words = unsafe {
        std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4)
    };
    let module_info = vk::ShaderModuleCreateInfo::default().code(words);
    let module = unsafe { device.device.create_shader_module(&module_info, None) }?;

    let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
    let set_layout =
        unsafe { device.device.create_descriptor_set_layout(&set_layout_info, None) }?;

    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_size);
    let push_ranges = if push_size > 0 { vec![push_range] } else { Vec::new() };
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&set_layout))
        .push_constant_ranges(&push_ranges);
    let layout = unsafe { device.device.create_pipeline_layout(&layout_info, None) }?;

    let entry = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(entry);
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
