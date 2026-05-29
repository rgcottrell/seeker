//! KV cache. Per-layer K and V buffers persisting across `Engine::forward`
//! calls so prompt prefill happens once and subsequent decode steps run in
//! `O(1)` per token. K and V dtypes are independently configurable from the
//! 9-entry list `{F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1}`.

use std::error::Error;

use ash::vk;

use crate::gguf::GgmlType;

use super::device::Device;
use super::memory::Region;
use super::weights::TensorView;

#[derive(Debug, Clone, Copy)]
pub struct KvCacheConfig {
    pub k_dtype: GgmlType,
    pub v_dtype: GgmlType,
    pub max_seq_len: u32,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            k_dtype: GgmlType::F16,
            v_dtype: GgmlType::F16,
            max_seq_len: 2048,
        }
    }
}

/// Dtypes the cache is willing to store K or V in.
pub const SUPPORTED_DTYPES: &[(GgmlType, &str)] = &[
    (GgmlType::F32, "f32"),
    (GgmlType::F16, "f16"),
    (GgmlType::BF16, "bf16"),
    (GgmlType::Q8_0, "q8_0"),
    (GgmlType::Q4_0, "q4_0"),
    (GgmlType::Q4_1, "q4_1"),
    (GgmlType::IQ4_NL, "iq4_nl"),
    (GgmlType::Q5_0, "q5_0"),
    (GgmlType::Q5_1, "q5_1"),
];

pub fn parse_dtype(s: &str) -> Result<GgmlType, String> {
    SUPPORTED_DTYPES
        .iter()
        .find_map(|(ty, name)| if *name == s { Some(*ty) } else { None })
        .ok_or_else(|| {
            let valid = SUPPORTED_DTYPES
                .iter()
                .map(|(_, n)| *n)
                .collect::<Vec<_>>()
                .join(", ");
            format!("unknown KV cache dtype {s:?}; expected one of: {valid}")
        })
}

pub struct KvCache {
    pub config: KvCacheConfig,
    pub region: Region,
    pub k_layers: Vec<TensorView>,
    pub v_layers: Vec<TensorView>,
    /// Number of token positions already written into the cache.
    pub position: u32,
    /// Optional SSM/Mamba/GDN per-layer recurrent state. Empty for
    /// pure-attention models; populated alongside K/V for hybrid models
    /// like qwen35moe. Each entry is a BufferRange covering the layer's
    /// state region in `ssm_region` (separate buffer to keep KV layout
    /// untouched). The model writes new state at the end of each forward
    /// and reads it back at the start of the next forward.
    pub ssm_region: Option<Region>,
    pub ssm_conv_states: Vec<crate::inference::buffer::BufferRange>,
    pub ssm_gdn_states: Vec<crate::inference::buffer::BufferRange>,
    /// Per-position GDN state snapshots written by a checkpoint verify
    /// forward (the GDN shader emits `K = L` snapshots, slot t = state
    /// after token t). One region; `ssm_gdn_snapshots[layer]` is that
    /// layer's `max_snapshots × gdn_state_floats` slice. Finalize copies
    /// slot `accept_len` into the live `ssm_gdn_states[layer]` — no re-run.
    pub ssm_gdn_snap_region: Option<Region>,
    pub ssm_gdn_snapshots: Vec<crate::inference::buffer::BufferRange>,
    /// Per-layer backup of the full conv1d input window (`[n_padded,
    /// conv_channels]`) from a checkpoint verify, so finalize can extract
    /// the conv state at the accepted position. One region; per-layer slices.
    pub ssm_conv_backup_region: Option<Region>,
    pub ssm_conv_backups: Vec<crate::inference::buffer::BufferRange>,
    /// `L` (= n_draft+1) the snapshot buffers were sized for; also the
    /// `n_padded` channel stride of the conv backups is `(conv_kernel-1)+this`.
    pub ssm_max_snapshots: u32,
    pub ssm_conv_kernel: u32,
    pub ssm_conv_channels: u32,
}

impl KvCache {
    pub fn new(
        device: &Device,
        n_layer: u32,
        head_dim: u32,
        n_head_kv: u32,
        config: KvCacheConfig,
    ) -> Result<Self, Box<dyn Error>> {
        validate_dtype(config.k_dtype, "K")?;
        validate_dtype(config.v_dtype, "V")?;
        validate_head_dim(head_dim, config.k_dtype, "K")?;
        validate_head_dim(head_dim, config.v_dtype, "V")?;

        let max_seq_len = config.max_seq_len as u64;
        let head_dim_u = head_dim as u64;
        let n_head_kv_u = n_head_kv as u64;

        let k_bytes = tensor_bytes(head_dim_u, max_seq_len, n_head_kv_u, config.k_dtype);
        let v_bytes = tensor_bytes(head_dim_u, max_seq_len, n_head_kv_u, config.v_dtype);
        let align = device.limits.min_storage_buffer_offset_alignment.max(1);
        let k_aligned = align_up(k_bytes, align);
        let v_aligned = align_up(v_bytes, align);
        let total = (n_layer as u64) * (k_aligned + v_aligned);

        let region = Region::new(
            device,
            total.max(1),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mut k_layers = Vec::with_capacity(n_layer as usize);
        let mut v_layers = Vec::with_capacity(n_layer as usize);
        let mut cursor = 0u64;
        for _ in 0..n_layer {
            k_layers.push(make_view(
                region.buffer,
                cursor,
                head_dim_u,
                max_seq_len,
                n_head_kv_u,
                config.k_dtype,
            ));
            cursor += k_aligned;
            v_layers.push(make_view(
                region.buffer,
                cursor,
                head_dim_u,
                max_seq_len,
                n_head_kv_u,
                config.v_dtype,
            ));
            cursor += v_aligned;
        }

        Ok(Self {
            config,
            region,
            k_layers,
            v_layers,
            position: 0,
            ssm_region: None,
            ssm_conv_states: Vec::new(),
            ssm_gdn_states: Vec::new(),
            ssm_gdn_snap_region: None,
            ssm_gdn_snapshots: Vec::new(),
            ssm_conv_backup_region: None,
            ssm_conv_backups: Vec::new(),
            ssm_max_snapshots: 0,
            ssm_conv_kernel: 0,
            ssm_conv_channels: 0,
        })
    }

    /// Allocate persistent SSM-block recurrent state for hybrid models
    /// like qwen35moe. `n_ssm_layers` is the number of SSM layers (not
    /// total blocks). Each layer gets a conv state of
    /// `(conv_kernel - 1) * conv_channels` F32 floats plus a GDN state
    /// matrix of `state_size^2 * num_v_heads` F32 floats. State is
    /// zero-initialized via the host pointer.
    pub fn allocate_ssm_state(
        &mut self,
        device: &Device,
        n_ssm_layers: u32,
        conv_state_floats: u32,
        gdn_state_floats: u32,
    ) -> Result<(), Box<dyn Error>> {
        let conv_bytes = (conv_state_floats as u64) * 4;
        let gdn_bytes = (gdn_state_floats as u64) * 4;
        let align = device.limits.min_storage_buffer_offset_alignment.max(1);
        let conv_aligned = align_up(conv_bytes, align);
        let gdn_aligned = align_up(gdn_bytes, align);
        let total = (n_ssm_layers as u64) * (conv_aligned + gdn_aligned);

        let region = Region::new(
            device,
            total.max(1),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Zero-init: each layer's state must start at zero on the first
        // forward. Subsequent forwards write new state to these slots.
        if let Some(host_ptr) = region.host_ptr {
            unsafe {
                std::ptr::write_bytes(host_ptr, 0, total as usize);
            }
        }

        let mut conv_states = Vec::with_capacity(n_ssm_layers as usize);
        let mut gdn_states = Vec::with_capacity(n_ssm_layers as usize);
        let mut cursor = 0u64;
        for _ in 0..n_ssm_layers {
            conv_states.push(crate::inference::buffer::BufferRange {
                buffer: region.buffer,
                offset: cursor,
                size: conv_bytes,
            });
            cursor += conv_aligned;
            gdn_states.push(crate::inference::buffer::BufferRange {
                buffer: region.buffer,
                offset: cursor,
                size: gdn_bytes,
            });
            cursor += gdn_aligned;
        }
        self.ssm_region = Some(region);
        self.ssm_conv_states = conv_states;
        self.ssm_gdn_states = gdn_states;
        Ok(())
    }

    /// Allocate per-position checkpoint buffers for MTP speculative decode:
    /// `max_snapshots` (= n_draft+1) GDN state snapshots per SSM layer plus a
    /// backup of each layer's conv1d input window. Lets a partial-acceptance
    /// step roll the recurrent state back to the accepted position by copying
    /// one snapshot, instead of re-running the main model. Call after
    /// [`allocate_ssm_state`]. No-op (regions stay `None`) when
    /// `max_snapshots == 0` or the model has no SSM state.
    pub fn allocate_ssm_snapshots(
        &mut self,
        device: &Device,
        dims: &crate::models::SsmStateDims,
        max_snapshots: u32,
    ) -> Result<(), Box<dyn Error>> {
        if max_snapshots == 0 || dims.n_ssm_layers == 0 {
            return Ok(());
        }
        let n = dims.n_ssm_layers as u64;
        let align = device.limits.min_storage_buffer_offset_alignment.max(1);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let mem = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        // GDN: max_snapshots × gdn_state_floats per layer.
        let gdn_bytes = (max_snapshots as u64) * (dims.gdn_state_floats as u64) * 4;
        let gdn_aligned = align_up(gdn_bytes, align);
        let gdn_region = Region::new(device, (n * gdn_aligned).max(1), usage, mem)?;
        // Conv: [(conv_kernel-1)+max_snapshots] × conv_channels per layer.
        let n_padded = (dims.conv_kernel - 1 + max_snapshots) as u64;
        let conv_bytes = n_padded * (dims.conv_channels as u64) * 4;
        let conv_aligned = align_up(conv_bytes, align);
        let conv_region = Region::new(device, (n * conv_aligned).max(1), usage, mem)?;

        let mut gdn_snaps = Vec::with_capacity(dims.n_ssm_layers as usize);
        let mut conv_backs = Vec::with_capacity(dims.n_ssm_layers as usize);
        for i in 0..n {
            gdn_snaps.push(crate::inference::buffer::BufferRange {
                buffer: gdn_region.buffer,
                offset: i * gdn_aligned,
                size: gdn_bytes,
            });
            conv_backs.push(crate::inference::buffer::BufferRange {
                buffer: conv_region.buffer,
                offset: i * conv_aligned,
                size: conv_bytes,
            });
        }
        self.ssm_gdn_snap_region = Some(gdn_region);
        self.ssm_gdn_snapshots = gdn_snaps;
        self.ssm_conv_backup_region = Some(conv_region);
        self.ssm_conv_backups = conv_backs;
        self.ssm_max_snapshots = max_snapshots;
        self.ssm_conv_kernel = dims.conv_kernel;
        self.ssm_conv_channels = dims.conv_channels;
        Ok(())
    }

    /// Reset the position counter to 0. Buffer contents stay (will be
    /// overwritten by the next forward pass).
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Roll the write cursor back to `new_pos` after speculative decode
    /// rejected some draft tokens. Symmetric to [`cache_io::advance`].
    /// Buffer contents past `new_pos` are left in place — attention reads
    /// are bounded by the position cursor / `DecodeDyn::kv_len`, so the
    /// stale K/V tail is never read and is overwritten by the next
    /// forward at the same offsets. Never advances (saturates at the
    /// current position).
    ///
    /// NOTE: this only rolls back the *attention* K/V. The SSM/GDN
    /// recurrent state has no per-position undo — pair this with
    /// [`snapshot_ssm`]/[`restore_ssm`] for hybrid models.
    ///
    /// [`cache_io::advance`]: crate::inference::ops::cache_io::advance
    pub fn truncate(&mut self, new_pos: u32) {
        self.position = new_pos.min(self.position);
    }


    pub fn destroy(&mut self, device: &Device) {
        self.region.destroy(device);
        if let Some(mut r) = self.ssm_region.take() {
            r.destroy(device);
        }
        if let Some(mut r) = self.ssm_gdn_snap_region.take() {
            r.destroy(device);
        }
        if let Some(mut r) = self.ssm_conv_backup_region.take() {
            r.destroy(device);
        }
    }
}

fn validate_dtype(ty: GgmlType, side: &str) -> Result<(), Box<dyn Error>> {
    if SUPPORTED_DTYPES.iter().any(|(t, _)| *t == ty) {
        Ok(())
    } else {
        Err(format!("KV cache {side} dtype {ty:?} not supported").into())
    }
}

fn validate_head_dim(head_dim: u32, ty: GgmlType, side: &str) -> Result<(), Box<dyn Error>> {
    let (block_size, _) = ty.block_layout();
    if (head_dim as usize) % block_size != 0 {
        return Err(format!(
            "KV cache {side} dtype {ty:?} requires head_dim ({head_dim}) to be a multiple of block_size {block_size}",
        )
        .into());
    }
    Ok(())
}

/// Bytes needed for one layer's K (or V) tensor of shape
/// `[head_dim, max_seq_len, n_head_kv]` in `dtype`.
fn tensor_bytes(head_dim: u64, max_seq_len: u64, n_head_kv: u64, dtype: GgmlType) -> u64 {
    let (block_size, type_size) = dtype.block_layout();
    let elements = head_dim * max_seq_len * n_head_kv;
    let blocks = elements / block_size as u64;
    blocks * type_size as u64
}

/// Build a TensorView for a single layer's K (or V).
///
/// Layout: **natural ggml** `[head_dim, n_head_kv, max_seq_len]` — innermost
/// is head_dim, then n_head_kv, then max_seq_len. This keeps the prefix
/// `[0, cur_seq_len)` contiguous in memory across all heads (each KV
/// position takes `head_dim * n_head_kv` elements back-to-back), which makes
/// dequant of just the live prefix a single flat dispatch.
///
/// The same memory presents to flash_attn as a permuted view
/// `[head_dim, cur_seq_len, n_head_kv]` with strides
/// `(1, head_dim * n_head_kv, head_dim)` — matching what the model already
/// builds for Q/K/V out of the mul_mm outputs.
fn make_view(
    buffer: vk::Buffer,
    byte_offset: u64,
    head_dim: u64,
    max_seq_len: u64,
    n_head_kv: u64,
    dtype: GgmlType,
) -> TensorView {
    let dims = [head_dim, n_head_kv, max_seq_len, 1];
    let (block_size, type_size) = dtype.block_layout();

    let mut byte_stride = [0u64; 4];
    if block_size > 1 {
        byte_stride[0] = type_size as u64;
        byte_stride[1] = (dims[0] / block_size as u64).max(1) * type_size as u64;
    } else {
        byte_stride[0] = type_size as u64;
        byte_stride[1] = byte_stride[0] * dims[0];
    }
    byte_stride[2] = byte_stride[1] * dims[1];
    byte_stride[3] = byte_stride[2] * dims[2];

    let byte_size = byte_stride[3] * dims[3].max(1);

    let mut element_stride = [0u64; 4];
    let elem_size = byte_stride[0].max(1);
    for i in 0..4 {
        element_stride[i] = byte_stride[i] / elem_size;
    }

    TensorView {
        buffer,
        byte_offset,
        byte_size,
        dims,
        byte_stride,
        element_stride,
        dtype,
    }
}

fn align_up(v: u64, alignment: u64) -> u64 {
    (v + alignment - 1) & !(alignment - 1)
}
