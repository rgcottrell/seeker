//! Vulkan 1.4 instance / physical-device pick / logical device / queue.
//!
//! Hard-fails when required extensions or features are absent — no silent
//! fallback. The MVP's contract is "Vulkan 1.4 with these caps or bust".

use std::error::Error;
use std::ffi::{c_char, CStr, CString};

use ash::{vk, Entry, Instance};

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
        let mut inst_flags = vk::InstanceCreateFlags::empty();
        let portability_inst = vk::KHR_PORTABILITY_ENUMERATION_NAME;
        if avail_inst_exts.iter().any(|e| ext_name(&e.extension_name) == portability_inst) {
            inst_ext_names.push(portability_inst.as_ptr());
            inst_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&inst_ext_names)
            .flags(inst_flags);
        let instance = unsafe { entry.create_instance(&instance_info, None) }?;

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
            .push_next(&mut feat_16bit)
            .push_next(&mut feat_f16)
            .push_next(&mut feat_subgroup)
            .push_next(&mut feat_maint4);
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
