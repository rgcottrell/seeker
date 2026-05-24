//! Vulkan 1.4 instance / physical-device pick / logical device / queue.
//!
//! Hard-fails when required extensions or features are absent — no silent
//! fallback. The MVP's contract is "Vulkan 1.4 with these caps or bust".
//!
//! Debug builds (`cfg(debug_assertions)`) additionally enable the
//! `VK_LAYER_KHRONOS_validation` layer and a `VK_EXT_debug_utils` messenger
//! that funnels driver/validation diagnostics into `tracing` under the
//! `vulkan` target. The layer + extension are best-effort: if either is
//! missing (e.g. SDK not installed) we log and continue.

use std::error::Error;
use std::ffi::{c_char, CStr, CString};

use ash::{vk, Entry, Instance};
use vk::TaggedStructure as _;

const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    vk::KHR_16BIT_STORAGE_NAME,
    vk::KHR_SHADER_FLOAT16_INT8_NAME,
    vk::KHR_MAINTENANCE4_NAME,
    vk::EXT_SUBGROUP_SIZE_CONTROL_NAME,
];

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
    pub portability: bool,
    #[cfg(debug_assertions)]
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

        #[cfg(debug_assertions)]
        let validation_enabled = {
            let layer_ok = validation::layer_available(&entry);
            let ext_ok = avail_inst_exts
                .iter()
                .any(|e| ext_name(&e.extension_name) == vk::EXT_DEBUG_UTILS_NAME);
            let enabled = layer_ok && ext_ok;
            if enabled {
                inst_ext_names.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
                inst_layer_names = vec![validation::LAYER.as_ptr()];
                tracing::info!("Vulkan validation layer + debug_utils enabled (debug build)");
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
        #[cfg(not(debug_assertions))]
        {
            inst_layer_names = Vec::new();
        }

        #[allow(unused_mut)] // reassigned via push_next under cfg(debug_assertions)
        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&inst_ext_names)
            .enabled_layer_names(&inst_layer_names)
            .flags(inst_flags);

        // `debug_info` must outlive `create_instance` because it's chained
        // via `push_next` so the validation layer can report errors that
        // occur during instance creation/destruction itself.
        #[cfg(debug_assertions)]
        let mut debug_info = validation::build_create_info();
        #[cfg(debug_assertions)]
        if validation_enabled {
            instance_info = instance_info.push(&mut debug_info);
        }

        let instance = unsafe { entry.create_instance(&instance_info, None) }?;

        #[cfg(debug_assertions)]
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

        let (physical, queue_family, portability) = pick_physical_device(&instance)?;
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let device_name = props
            .device_name_as_c_str()
            .ok()
            .and_then(|s| s.to_str().ok())
            .unwrap_or("<?>")
            .to_string();
        tracing::info!(
            device = %device_name,
            queue_family,
            portability,
            "picked physical device",
        );

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0]);

        let mut device_exts: Vec<*const c_char> =
            REQUIRED_DEVICE_EXTENSIONS.iter().map(|c| c.as_ptr()).collect();
        if portability {
            device_exts.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }

        let mut feat_16bit = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true);
        let mut feat_f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(true);
        let mut feat_subgroup = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);
        let mut feat_maint4 = vk::PhysicalDeviceMaintenance4Features::default()
            .maintenance4(true);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_exts)
            .push(&mut feat_16bit)
            .push(&mut feat_f16)
            .push(&mut feat_subgroup)
            .push(&mut feat_maint4);
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
            portability,
            #[cfg(debug_assertions)]
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
            #[cfg(debug_assertions)]
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

fn pick_physical_device(
    instance: &Instance,
) -> Result<(vk::PhysicalDevice, u32, bool), Box<dyn Error>> {
    let phys = unsafe { instance.enumerate_physical_devices() }?;
    if phys.is_empty() {
        return Err("no Vulkan physical devices found".into());
    }

    for p in phys {
        let avail = unsafe { instance.enumerate_device_extension_properties(p) }?;
        let avail_names: Vec<&CStr> = avail.iter().map(|e| ext_name(&e.extension_name)).collect();
        let missing: Vec<&CStr> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .copied()
            .filter(|req| !avail_names.iter().any(|n| n == req))
            .collect();
        if !missing.is_empty() {
            tracing::debug!(
                missing = ?missing.iter().map(|s| s.to_string_lossy().into_owned()).collect::<Vec<_>>(),
                "skipping physical device — missing extensions",
            );
            continue;
        }
        let portability = avail_names.iter().any(|n| *n == vk::KHR_PORTABILITY_SUBSET_NAME);

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(p) };
        let qf = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
        let Some(qf) = qf else { continue };

        return Ok((p, qf as u32, portability));
    }
    Err(format!(
        "no physical device satisfies required extensions: {}",
        REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|e| e.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .into())
}

/// Pre-built CString cache for required extension diagnostics. Currently
/// unused but kept private so future error paths can reference it.
#[allow(dead_code)]
fn missing_extension_message(missing: &[&CStr]) -> CString {
    let s = missing
        .iter()
        .map(|e| e.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(", ");
    CString::new(s).unwrap_or_default()
}

#[cfg(debug_assertions)]
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
