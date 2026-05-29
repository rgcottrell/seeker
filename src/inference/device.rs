//! Vulkan 1.4 instance / physical-device pick / logical device / queue.
//!
//! Hard-fails when required features are absent — no silent fallback. The
//! MVP's contract is "Vulkan 1.4 with these features or bust". All four
//! extensions we used to request explicitly (16-bit storage,
//! shader_float16_int8, maintenance4, subgroup_size_control) are core in
//! 1.4 and are now driven through the `PhysicalDeviceVulkanXFeatures`
//! rollup structs instead.
//!
//! Optional 1.4-era usability features (`maintenance5`, `maintenance6`,
//! `push_descriptor`, `shader_float_controls2`, `shader_expect_assume`,
//! `shader_subgroup_rotate`/`_clustered`) are probed and enabled when the
//! driver reports support; availability is exposed on `Device` so call
//! sites can branch. Cooperative-matrix support (`VK_KHR_cooperative_matrix`
//! and `VK_NV_cooperative_matrix2`) is treated the same way.
//!
//! Debug builds (`cfg(debug_assertions)`) and any build with the
//! `gpu_debug` feature additionally enable the `VK_LAYER_KHRONOS_validation`
//! layer and a `VK_EXT_debug_utils` messenger that funnels
//! driver/validation diagnostics into `tracing` under the `vulkan` target.
//! Release builds without `gpu_debug` carry none of this. The layer +
//! extension are best-effort: if either is missing (e.g. SDK not
//! installed) we log and continue.

use std::error::Error;
use std::ffi::{c_char, CStr};

use ash::{vk, Entry, Instance};
use vk::TaggedStructure as _;

const REQUIRED_API_VERSION: u32 = vk::make_api_version(0, 1, 4, 0);

/// Vulkan handles needed throughout the inference module. Owns the
/// `ash::Entry` so the loader stays alive for the device's lifetime.
pub struct Device {
    pub entry: Entry,
    pub instance: Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub limits: vk::PhysicalDeviceLimits,
    pub api_version: u32,
    pub portability: bool,
    pub maintenance5: bool,
    pub maintenance6: bool,
    pub push_descriptor: bool,
    pub shader_float_controls2: bool,
    pub shader_expect_assume: bool,
    pub shader_subgroup_rotate: bool,
    pub shader_subgroup_rotate_clustered: bool,
    pub coop_matrix: bool,
    pub coop_matrix2: bool,
    /// Compute-unit count (AMD `activeComputeUnitCount`), used to size
    /// flash-attention split-K the way llama.cpp does. 0 when the device
    /// doesn't advertise `VK_AMD_shader_core_properties2`.
    pub shader_core_count: u32,
    /// Smallest subgroup size the driver will assign for compute (also the
    /// smallest value that can be passed to
    /// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`).
    pub min_subgroup_size: u32,
    /// Largest subgroup size the driver will assign for compute.
    pub max_subgroup_size: u32,
    /// Shader stages on which `requiredSubgroupSize` may be requested. We
    /// only care about COMPUTE; this is exposed so callers can sanity-check.
    pub required_subgroup_size_stages: vk::ShaderStageFlags,
    #[cfg(any(debug_assertions, feature = "gpu_debug"))]
    debug: Option<validation::Messenger>,
}

impl Device {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::load() }?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"seeker")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"seeker")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 4, 0));

        // Inspect instance-level extensions so we can request portability_enumeration
        // on platforms where it's reported (MoltenVK).
        let avail_inst_exts = unsafe { entry.enumerate_instance_extension_properties(None) }?;
        let mut inst_ext_names: Vec<*const c_char> = Vec::new();
        let inst_layer_names: Vec<*const c_char>;
        let mut inst_flags = vk::InstanceCreateFlags::empty();
        let portability_inst = vk::KHR_PORTABILITY_ENUMERATION_NAME;
        if avail_inst_exts.iter().any(|e| ext_name(&e.extension_name) == portability_inst) {
            inst_ext_names.push(portability_inst.as_ptr());
            inst_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        #[cfg(any(debug_assertions, feature = "gpu_debug"))]
        let validation_enabled = {
            let layer_ok = validation::layer_available(&entry);
            let ext_ok = avail_inst_exts
                .iter()
                .any(|e| ext_name(&e.extension_name) == vk::EXT_DEBUG_UTILS_NAME);
            let enabled = layer_ok && ext_ok;
            if enabled {
                inst_ext_names.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
                inst_layer_names = vec![validation::LAYER.as_ptr()];
                tracing::info!("Vulkan validation layer + debug_utils enabled (debug build / gpu_debug)");
            } else {
                inst_layer_names = Vec::new();
                if !layer_ok {
                    tracing::warn!(
                        "VK_LAYER_KHRONOS_validation not available; install the Vulkan SDK \
                         to get debug-build diagnostics",
                    );
                } else {
                    tracing::warn!(
                        "VK_EXT_debug_utils not available; skipping debug-build diagnostics",
                    );
                }
            }
            enabled
        };
        #[cfg(not(any(debug_assertions, feature = "gpu_debug")))]
        {
            inst_layer_names = Vec::new();
        }

        #[allow(unused_mut)] // reassigned via push_next when validation is enabled
        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&inst_ext_names)
            .enabled_layer_names(&inst_layer_names)
            .flags(inst_flags);

        // `debug_info` must outlive `create_instance` because it's chained
        // via `push_next` so the validation layer can report errors that
        // occur during instance creation/destruction itself.
        #[cfg(any(debug_assertions, feature = "gpu_debug"))]
        let mut debug_info = validation::build_create_info();
        #[cfg(any(debug_assertions, feature = "gpu_debug"))]
        if validation_enabled {
            instance_info = instance_info.push(&mut debug_info);
        }

        let instance = unsafe { entry.create_instance(&instance_info, None) }?;

        #[cfg(any(debug_assertions, feature = "gpu_debug"))]
        let debug = if validation_enabled {
            match validation::Messenger::create(&entry, &instance) {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!(error = %e, "failed to create debug_utils messenger");
                    None
                }
            }
        } else {
            None
        };

        let DevicePick {
            physical,
            queue_family,
            api_version,
            portability,
            coop_matrix_ext,
            coop_matrix2_ext,
            shader_core_props2_ext,
        } = pick_physical_device(&instance)?;

        // Pull the legacy 1.0 properties (for `limits`) and the
        // `subgroup_size_control` properties via the 1.1+ `GetProperties2`
        // pNext chain so we know what `requiredSubgroupSize` values the
        // driver will accept on COMPUTE pipelines.
        let mut props_sgs = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        // `activeComputeUnitCount` (AMD) — only chained when the extension is
        // advertised, since the driver fills it only if it recognizes the
        // struct. Queried as a physical-device property (no enable needed).
        let mut props_score = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut props_sgs);
        if shader_core_props2_ext {
            props2 = props2.push(&mut props_score);
        }
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let props = props2.properties;
        let shader_core_count = if shader_core_props2_ext {
            props_score.active_compute_unit_count
        } else {
            0
        };
        let device_name = props
            .device_name_as_c_str()
            .ok()
            .and_then(|s| s.to_str().ok())
            .unwrap_or("<?>")
            .to_string();
        let min_subgroup_size = props_sgs.min_subgroup_size;
        let max_subgroup_size = props_sgs.max_subgroup_size;
        let required_subgroup_size_stages = props_sgs.required_subgroup_size_stages;

        // Probe feature support via the Vulkan 1.4 rollup structs (plus
        // coop-matrix structs only when the corresponding extension is
        // advertised — otherwise the driver may ignore them).
        // `PhysicalDeviceFeatures2.features` itself carries the base 1.0
        // boolean features (`shaderInt16`/`shaderInt64`/`shaderFloat64`)
        // that the K-quant SPIR-V uses (block_q6_K has `int16_t scales`,
        // some i-quants emit `int64_t`).
        let mut q11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut q12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut q13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut q14 = vk::PhysicalDeviceVulkan14Features::default();
        let mut q_cm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut q_cm2 = vk::PhysicalDeviceCooperativeMatrix2FeaturesNV::default();
        // Run the query inside a scope so the &mut borrows of q11/q12/...
        // are released before we read their fields below. Also snapshot
        // the base-1.0 booleans we care about into locals here.
        let (shader_int16, shader_int64) = {
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push(&mut q11)
                .push(&mut q12)
                .push(&mut q13)
                .push(&mut q14);
            if coop_matrix_ext {
                features2 = features2.push(&mut q_cm);
            }
            if coop_matrix2_ext {
                features2 = features2.push(&mut q_cm2);
            }
            unsafe { instance.get_physical_device_features2(physical, &mut features2) };
            (
                features2.features.shader_int16,
                features2.features.shader_int64,
            )
        };

        let mut missing: Vec<&'static str> = Vec::new();
        let mut require = |name: &'static str, supported: vk::Bool32| {
            if supported != vk::TRUE {
                missing.push(name);
            }
        };
        require("storage_buffer16_bit_access", q11.storage_buffer16_bit_access);
        require(
            "uniform_and_storage_buffer16_bit_access",
            q11.uniform_and_storage_buffer16_bit_access,
        );
        require("shader_float16", q12.shader_float16);
        require("shader_int8", q12.shader_int8);
        require("storage_buffer8_bit_access", q12.storage_buffer8_bit_access);
        require(
            "uniform_and_storage_buffer8_bit_access",
            q12.uniform_and_storage_buffer8_bit_access,
        );
        require("scalar_block_layout", q12.scalar_block_layout);
        require("vulkan_memory_model", q12.vulkan_memory_model);
        require(
            "vulkan_memory_model_device_scope",
            q12.vulkan_memory_model_device_scope,
        );
        require("timeline_semaphore", q12.timeline_semaphore);
        require("maintenance4", q13.maintenance4);
        require("subgroup_size_control", q13.subgroup_size_control);
        require("compute_full_subgroups", q13.compute_full_subgroups);
        require("shader_integer_dot_product", q13.shader_integer_dot_product);
        require("synchronization2", q13.synchronization2);
        // Base 1.0 features required by K/I-quant shaders.
        require("shader_int16", shader_int16);
        require("shader_int64", shader_int64);
        if !missing.is_empty() {
            return Err(format!(
                "physical device {} missing required Vulkan 1.4 features: {}",
                device_name,
                missing.join(", "),
            )
            .into());
        }

        // Optional features the device may report. For each we record the
        // bit on `Device` and propagate it into the enable-side struct
        // below.
        let maintenance5 = q14.maintenance5 == vk::TRUE;
        let maintenance6 = q14.maintenance6 == vk::TRUE;
        let push_descriptor = q14.push_descriptor == vk::TRUE;
        let shader_float_controls2 = q14.shader_float_controls2 == vk::TRUE;
        let shader_expect_assume = q14.shader_expect_assume == vk::TRUE;
        let shader_subgroup_rotate = q14.shader_subgroup_rotate == vk::TRUE;
        let shader_subgroup_rotate_clustered = q14.shader_subgroup_rotate_clustered == vk::TRUE;
        let coop_matrix = coop_matrix_ext && q_cm.cooperative_matrix == vk::TRUE;
        let coop_matrix2 = coop_matrix2_ext && q_cm2.cooperative_matrix_workgroup_scope == vk::TRUE;

        tracing::info!(
            device = %device_name,
            queue_family,
            api_version = format_args!(
                "{}.{}.{}",
                vk::api_version_major(api_version),
                vk::api_version_minor(api_version),
                vk::api_version_patch(api_version),
            ),
            portability,
            maintenance5,
            maintenance6,
            push_descriptor,
            shader_float_controls2,
            shader_expect_assume,
            shader_subgroup_rotate,
            shader_subgroup_rotate_clustered,
            coop_matrix,
            coop_matrix2,
            shader_core_count,
            min_subgroup_size,
            max_subgroup_size,
            required_subgroup_size_compute =
                required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE),
            "picked physical device",
        );

        // Enable-side feature structs: only the bits we actually want.
        // PhysicalDeviceFeatures2 carries the base 1.0 booleans
        // (`shaderInt16`/`shaderInt64`) the K-quant SPIR-V references.
        // Mixing it with the Vulkan1{1,2,3,4}Features rollups in the
        // pNext chain is explicitly allowed (the only restriction is that
        // `pEnabledFeatures` must be NULL, which ash leaves unset here).
        let mut feat2 = vk::PhysicalDeviceFeatures2::default().features(
            vk::PhysicalDeviceFeatures::default()
                .shader_int16(true)
                .shader_int64(true),
        );
        let mut feat11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true);
        let mut feat12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(true)
            .shader_int8(true)
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .scalar_block_layout(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .timeline_semaphore(true);
        let mut feat13 = vk::PhysicalDeviceVulkan13Features::default()
            .maintenance4(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true)
            .shader_integer_dot_product(true)
            .synchronization2(true);
        let mut feat14 = vk::PhysicalDeviceVulkan14Features::default()
            .maintenance5(maintenance5)
            .maintenance6(maintenance6)
            .push_descriptor(push_descriptor)
            .shader_float_controls2(shader_float_controls2)
            .shader_expect_assume(shader_expect_assume)
            .shader_subgroup_rotate(shader_subgroup_rotate)
            .shader_subgroup_rotate_clustered(shader_subgroup_rotate_clustered);
        // For cooperative matrix, enable every sub-bit the device reported.
        // Either struct is only chained in below if its extension is in
        // the enabled list.
        let mut feat_cm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(q_cm.cooperative_matrix == vk::TRUE)
            .cooperative_matrix_robust_buffer_access(
                q_cm.cooperative_matrix_robust_buffer_access == vk::TRUE,
            );
        let mut feat_cm2 = vk::PhysicalDeviceCooperativeMatrix2FeaturesNV::default()
            .cooperative_matrix_workgroup_scope(
                q_cm2.cooperative_matrix_workgroup_scope == vk::TRUE,
            )
            .cooperative_matrix_flexible_dimensions(
                q_cm2.cooperative_matrix_flexible_dimensions == vk::TRUE,
            )
            .cooperative_matrix_reductions(q_cm2.cooperative_matrix_reductions == vk::TRUE)
            .cooperative_matrix_conversions(q_cm2.cooperative_matrix_conversions == vk::TRUE)
            .cooperative_matrix_per_element_operations(
                q_cm2.cooperative_matrix_per_element_operations == vk::TRUE,
            )
            .cooperative_matrix_tensor_addressing(
                q_cm2.cooperative_matrix_tensor_addressing == vk::TRUE,
            )
            .cooperative_matrix_block_loads(q_cm2.cooperative_matrix_block_loads == vk::TRUE);

        let mut device_exts: Vec<*const c_char> = Vec::new();
        if portability {
            device_exts.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }
        if coop_matrix {
            device_exts.push(vk::KHR_COOPERATIVE_MATRIX_NAME.as_ptr());
        }
        if coop_matrix2 {
            device_exts.push(vk::NV_COOPERATIVE_MATRIX2_NAME.as_ptr());
        }

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0]);

        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_exts)
            .push(&mut feat2)
            .push(&mut feat11)
            .push(&mut feat12)
            .push(&mut feat13)
            .push(&mut feat14);
        if coop_matrix {
            device_info = device_info.push(&mut feat_cm);
        }
        if coop_matrix2 {
            device_info = device_info.push(&mut feat_cm2);
        }
        let device = unsafe { instance.create_device(physical, &device_info, None) }?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let limits = props.limits;

        Ok(Self {
            entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            mem_props,
            limits,
            api_version,
            portability,
            maintenance5,
            maintenance6,
            push_descriptor,
            shader_float_controls2,
            shader_expect_assume,
            shader_subgroup_rotate,
            shader_subgroup_rotate_clustered,
            coop_matrix,
            coop_matrix2,
            shader_core_count,
            min_subgroup_size,
            max_subgroup_size,
            required_subgroup_size_stages,
            #[cfg(any(debug_assertions, feature = "gpu_debug"))]
            debug,
        })
    }

    pub fn name(&self) -> String {
        let props = unsafe { self.instance.get_physical_device_properties(self.physical) };
        props
            .device_name_as_c_str()
            .ok()
            .and_then(|s| s.to_str().ok())
            .unwrap_or("<unknown>")
            .to_string()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            #[cfg(any(debug_assertions, feature = "gpu_debug"))]
            if let Some(d) = self.debug.take() {
                d.destroy();
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn ext_name(name: &[c_char; 256]) -> &CStr {
    unsafe { CStr::from_ptr(name.as_ptr()) }
}

struct DevicePick {
    physical: vk::PhysicalDevice,
    queue_family: u32,
    api_version: u32,
    portability: bool,
    coop_matrix_ext: bool,
    coop_matrix2_ext: bool,
    shader_core_props2_ext: bool,
}

fn pick_physical_device(instance: &Instance) -> Result<DevicePick, Box<dyn Error>> {
    let phys = unsafe { instance.enumerate_physical_devices() }?;
    if phys.is_empty() {
        return Err("no Vulkan physical devices found".into());
    }

    for p in phys {
        let props = unsafe { instance.get_physical_device_properties(p) };
        if props.api_version < REQUIRED_API_VERSION {
            tracing::debug!(
                api_version = format_args!(
                    "{}.{}.{}",
                    vk::api_version_major(props.api_version),
                    vk::api_version_minor(props.api_version),
                    vk::api_version_patch(props.api_version),
                ),
                "skipping physical device — Vulkan 1.4 required",
            );
            continue;
        }

        let avail = unsafe { instance.enumerate_device_extension_properties(p) }?;
        let has_ext = |needle: &CStr| {
            avail.iter().any(|e| ext_name(&e.extension_name) == needle)
        };
        let portability = has_ext(vk::KHR_PORTABILITY_SUBSET_NAME);
        let coop_matrix_ext = has_ext(vk::KHR_COOPERATIVE_MATRIX_NAME);
        let coop_matrix2_ext = has_ext(vk::NV_COOPERATIVE_MATRIX2_NAME);
        let shader_core_props2_ext = has_ext(vk::AMD_SHADER_CORE_PROPERTIES2_NAME);

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(p) };
        let qf = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
        let Some(qf) = qf else { continue };

        return Ok(DevicePick {
            physical: p,
            queue_family: qf as u32,
            api_version: props.api_version,
            portability,
            coop_matrix_ext,
            coop_matrix2_ext,
            shader_core_props2_ext,
        });
    }
    Err("no Vulkan 1.4 physical device with a compute queue found".into())
}

#[cfg(any(debug_assertions, feature = "gpu_debug"))]
mod validation {
    use std::ffi::{c_void, CStr};

    use ash::{ext, vk, Entry, Instance, VkResult};

    pub const LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

    pub struct Messenger {
        loader: ext::debug_utils::Instance,
        handle: vk::DebugUtilsMessengerEXT,
    }

    impl Messenger {
        pub fn create(entry: &Entry, instance: &Instance) -> VkResult<Self> {
            let loader = ext::debug_utils::Instance::load(entry, instance);
            let info = build_create_info();
            let handle = unsafe { loader.create_debug_utils_messenger(&info, None) }?;
            Ok(Self { loader, handle })
        }

        pub fn destroy(self) {
            unsafe {
                self.loader.destroy_debug_utils_messenger(self.handle, None);
            }
        }
    }

    pub fn layer_available(entry: &Entry) -> bool {
        match unsafe { entry.enumerate_instance_layer_properties() } {
            Ok(layers) => layers.iter().any(|l| {
                let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
                name == LAYER
            }),
            Err(_) => false,
        }
    }

    pub fn build_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(callback))
    }

    unsafe extern "system" fn callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user: *mut c_void,
    ) -> vk::Bool32 {
        let data = unsafe { &*data };
        let message = if data.p_message.is_null() {
            std::borrow::Cow::Borrowed("<no message>")
        } else {
            unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy()
        };
        let kind = if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
            "validation"
        } else if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
            "performance"
        } else {
            "general"
        };
        if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            tracing::error!(target: "vulkan", kind, "{}", message);
        } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
            tracing::warn!(target: "vulkan", kind, "{}", message);
        } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
            tracing::info!(target: "vulkan", kind, "{}", message);
        } else {
            tracing::debug!(target: "vulkan", kind, "{}", message);
        }
        vk::FALSE
    }
}
