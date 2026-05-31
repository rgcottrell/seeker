//! KV cache. Per-layer K and V buffers persisting across `Engine::forward`
//! calls so prompt prefill happens once and subsequent decode steps run in
//! `O(1)` per token. K and V dtypes are independently configurable from the
//! 9-entry list `{F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1}`.

use std::error::Error;
use std::sync::Arc;

use ash::vk;

use crate::gguf::GgmlType;

use super::device::{Device, DeviceShared};
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
    /// Refcounted device owner so `Drop` can free `region` + the SSM
    /// regions, keeping the logical device alive until it does — regardless
    /// of whether the owning engine drops first.
    device: Arc<DeviceShared>,
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
            device: device.shared(),
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

    /// Reset the write cursor to 0, starting a fresh sequence.
    ///
    /// Attention K/V buffer contents are left in place: they're overwritten
    /// position-by-position by the next forward, and reads are bounded by
    /// `position`. SSM/GDN recurrent state is different — it is *seeded from
    /// the buffer* at the start of every forward and accumulated across the
    /// whole sequence, never overwritten per-position. So a fresh sequence
    /// must zero it, or the new conversation inherits the previous one's
    /// recurrent prior (silent corruption, no error). Mirrors the zero-init in
    /// [`allocate_ssm_state`]. The per-position snapshot/backup buffers are
    /// spec-decode scratch, rewritten within a step, so they need no reset.
    pub fn reset(&mut self) {
        self.position = 0;
        if let Some((host_ptr, size)) = self
            .ssm_region
            .as_ref()
            .and_then(|r| r.host_ptr.map(|p| (p, r.size as usize)))
        {
            // SAFETY: host_ptr maps the whole HOST_VISIBLE|HOST_COHERENT
            // ssm_region of `size` bytes — the same write performed at
            // allocation time.
            unsafe {
                std::ptr::write_bytes(host_ptr, 0, size);
            }
        }
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


}

impl KvCache {
    /// Build a non-owning `KvCache` over slab `byte_offset`s inside a shared
    /// buffer (a [`BatchKvCache`] slot). The `region` is borrowed
    /// ([`Region::borrowed`]) so Drop frees nothing — the `BatchKvCache` owns
    /// the buffer. SSM state is left empty (attention-only; the batched SSM
    /// path lands with qwen35moe in M2). Used to prefill one sequence into its
    /// slab via the existing single-sequence forward path.
    #[allow(clippy::too_many_arguments)]
    fn borrowed_slot(
        config: KvCacheConfig,
        buffer: vk::Buffer,
        host_ptr: Option<*mut u8>,
        buffer_size: u64,
        alignment: u64,
        k_layers: Vec<TensorView>,
        v_layers: Vec<TensorView>,
        position: u32,
        device: Arc<DeviceShared>,
        // Per-slot SSM recurrent state (empty for attention-only models). When
        // present, the borrowed cache points at the owning BatchKvCache's
        // per-sequence state slices so a single-sequence prefill writes its
        // final conv/GDN state where the batched decode reads it. The region is
        // borrowed (Drop frees nothing — the BatchKvCache owns it).
        ssm_conv_states: Vec<crate::inference::buffer::BufferRange>,
        ssm_gdn_states: Vec<crate::inference::buffer::BufferRange>,
        ssm_region: Option<(vk::Buffer, Option<*mut u8>, u64)>,
    ) -> Self {
        let ssm_region = ssm_region
            .map(|(buf, hp, size)| Region::borrowed(buf, hp, size, alignment));
        Self {
            config,
            region: Region::borrowed(buffer, host_ptr, buffer_size, alignment),
            k_layers,
            v_layers,
            position,
            ssm_region,
            ssm_conv_states,
            ssm_gdn_states,
            ssm_gdn_snap_region: None,
            ssm_gdn_snapshots: Vec::new(),
            ssm_conv_backup_region: None,
            ssm_conv_backups: Vec::new(),
            ssm_max_snapshots: 0,
            ssm_conv_kernel: 0,
            ssm_conv_channels: 0,
            device,
        }
    }
}

impl Drop for KvCache {
    fn drop(&mut self) {
        // Free the KV region + any SSM/snapshot regions. The Arc keeps the
        // logical device alive through this; forwards are synchronous (each
        // fence-waits), so the GPU is idle by teardown.
        let dev = self.device.raw();
        self.region.destroy(dev);
        if let Some(mut r) = self.ssm_region.take() {
            r.destroy(dev);
        }
        if let Some(mut r) = self.ssm_gdn_snap_region.take() {
            r.destroy(dev);
        }
        if let Some(mut r) = self.ssm_conv_backup_region.take() {
            r.destroy(dev);
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

/// A KV cache holding `n_slots` independent sequence slabs in one shared buffer
/// per layer, so a batched decode can address all active sequences through the
/// flash-attn batch stride (`nb13`/`nb23`). Each slab has the identical natural
/// `[head_dim, n_head_kv, max_seq_len]` layout as a standalone [`KvCache`]
/// layer; slabs are padded to the storage-buffer alignment so every slot's
/// byte offset is bindable. Attention-only (F32/F16/BF16 caches) for now — the
/// quant-cache and SSM-state slabs land later (M2/M4).
pub struct BatchKvCache {
    pub config: KvCacheConfig,
    pub n_slots: u32,
    pub n_layer: u32,
    head_dim: u32,
    n_head_kv: u32,
    region: Region,
    buffer: vk::Buffer,
    host_ptr: Option<*mut u8>,
    buffer_size: u64,
    alignment: u64,
    /// Byte offset of each layer's K slab-0 / V slab-0.
    k_base: Vec<u64>,
    v_base: Vec<u64>,
    /// Per-slot slab stride in bytes (padded to `alignment`).
    k_slab_stride: u64,
    v_slab_stride: u64,
    /// Current write position (tokens) of each slot.
    pub positions: Vec<u32>,
    /// Per-sequence SSM/GDN recurrent state (hybrid models), allocated by
    /// [`Self::allocate_ssm_state`]. Per layer the GDN state is one contiguous
    /// `B × gdn_state_floats` block (the GDN shader indexes seq as the
    /// outermost state dim → dispatched at `n_seqs = B`); the conv state is
    /// `B × conv_state_floats`, addressed per slot for the per-sequence
    /// conv-input prefix + writeback.
    ssm_region: Option<Region>,
    ssm_conv_base: Vec<u64>,
    ssm_gdn_base: Vec<u64>,
    conv_slab_floats: u32,
    gdn_slab_floats: u32,
    device: Arc<DeviceShared>,
}

impl BatchKvCache {
    pub fn new(
        device: &Device,
        n_layer: u32,
        head_dim: u32,
        n_head_kv: u32,
        n_slots: u32,
        config: KvCacheConfig,
    ) -> Result<Self, Box<dyn Error>> {
        validate_dtype(config.k_dtype, "K")?;
        validate_dtype(config.v_dtype, "V")?;
        validate_head_dim(head_dim, config.k_dtype, "K")?;
        validate_head_dim(head_dim, config.v_dtype, "V")?;
        for (ty, side) in [(config.k_dtype, "K"), (config.v_dtype, "V")] {
            if ty.block_layout().0 != 1 {
                return Err(format!(
                    "BatchKvCache: quant {side} cache dtype {ty:?} not supported yet \
                     (batched decode needs a flat per-element stride; use f16/bf16/f32)"
                )
                .into());
            }
        }

        let max_seq_len = config.max_seq_len as u64;
        let head_dim_u = head_dim as u64;
        let n_head_kv_u = n_head_kv as u64;
        let align = device.limits.min_storage_buffer_offset_alignment.max(1);

        let k_slab = tensor_bytes(head_dim_u, max_seq_len, n_head_kv_u, config.k_dtype);
        let v_slab = tensor_bytes(head_dim_u, max_seq_len, n_head_kv_u, config.v_dtype);
        let k_slab_stride = align_up(k_slab, align);
        let v_slab_stride = align_up(v_slab, align);
        let k_block = k_slab_stride * n_slots as u64;
        let v_block = v_slab_stride * n_slots as u64;

        let mut k_base = Vec::with_capacity(n_layer as usize);
        let mut v_base = Vec::with_capacity(n_layer as usize);
        let mut cursor = 0u64;
        for _ in 0..n_layer {
            let kb = align_up(cursor, align);
            cursor = kb + k_block;
            let vb = align_up(cursor, align);
            cursor = vb + v_block;
            k_base.push(kb);
            v_base.push(vb);
        }
        let total = cursor.max(1);

        let region = Region::new(
            device,
            total,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Zero the whole buffer so no slot ever reads stale K/V on its first
        // forward (mirrors KvCache's implicit zero-on-fresh-alloc reliance).
        if let Some(p) = region.host_ptr {
            unsafe { std::ptr::write_bytes(p, 0, total as usize) };
        }

        Ok(Self {
            config,
            n_slots,
            n_layer,
            head_dim,
            n_head_kv,
            buffer: region.buffer,
            host_ptr: region.host_ptr,
            buffer_size: region.size,
            alignment: region.alignment,
            region,
            k_base,
            v_base,
            k_slab_stride,
            v_slab_stride,
            positions: vec![0; n_slots as usize],
            ssm_region: None,
            ssm_conv_base: Vec::new(),
            ssm_gdn_base: Vec::new(),
            conv_slab_floats: 0,
            gdn_slab_floats: 0,
            device: device.shared(),
        })
    }

    /// Allocate per-sequence SSM recurrent state for a hybrid model. Per layer:
    /// a contiguous `n_slots × gdn_state_floats` GDN block (seq-outermost, so a
    /// single `n_seqs = n_slots` gated-delta-net dispatch reads/writes all
    /// sequences) and an `n_slots × conv_state_floats` conv block (addressed per
    /// slot). Zero-initialized. Layer blocks are storage-aligned.
    pub fn allocate_ssm_state(
        &mut self,
        device: &Device,
        n_ssm_layers: u32,
        conv_state_floats: u32,
        gdn_state_floats: u32,
    ) -> Result<(), Box<dyn Error>> {
        let n = self.n_slots as u64;
        let conv_block = (conv_state_floats as u64) * n * 4;
        let gdn_block = (gdn_state_floats as u64) * n * 4;
        let align = device.limits.min_storage_buffer_offset_alignment.max(1);

        let mut conv_base = Vec::with_capacity(n_ssm_layers as usize);
        let mut gdn_base = Vec::with_capacity(n_ssm_layers as usize);
        let mut cursor = 0u64;
        for _ in 0..n_ssm_layers {
            let cb = align_up(cursor, align);
            cursor = cb + conv_block;
            let gb = align_up(cursor, align);
            cursor = gb + gdn_block;
            conv_base.push(cb);
            gdn_base.push(gb);
        }
        let total = cursor.max(1);

        let region = Region::new(
            device,
            total,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        if let Some(p) = region.host_ptr {
            unsafe { std::ptr::write_bytes(p, 0, total as usize) };
        }
        self.ssm_region = Some(region);
        self.ssm_conv_base = conv_base;
        self.ssm_gdn_base = gdn_base;
        self.conv_slab_floats = conv_state_floats;
        self.gdn_slab_floats = gdn_state_floats;
        Ok(())
    }

    /// The contiguous `n_slots × gdn_state_floats` GDN state block for `layer`
    /// (fed to a single `n_seqs = n_slots` gated-delta-net dispatch).
    pub fn gdn_state_layer(&self, layer: u32) -> crate::inference::buffer::BufferRange {
        let region = self.ssm_region.as_ref().expect("SSM state not allocated");
        crate::inference::buffer::BufferRange {
            buffer: region.buffer,
            offset: self.ssm_gdn_base[layer as usize],
            size: (self.gdn_slab_floats as u64) * (self.n_slots as u64) * 4,
        }
    }

    /// The contiguous `n_slots × conv_state_floats` conv state block for
    /// `layer` (seq-outermost) — for batched conv-input prefix + writeback casts.
    pub fn conv_state_layer(&self, layer: u32) -> crate::inference::buffer::BufferRange {
        let region = self.ssm_region.as_ref().expect("SSM state not allocated");
        crate::inference::buffer::BufferRange {
            buffer: region.buffer,
            offset: self.ssm_conv_base[layer as usize],
            size: (self.conv_slab_floats as u64) * (self.n_slots as u64) * 4,
        }
    }

    /// Sequence `slot`'s conv state for `layer` (the per-sequence conv-input
    /// prefix source and writeback target).
    pub fn conv_state_slot(&self, layer: u32, slot: u32) -> crate::inference::buffer::BufferRange {
        let region = self.ssm_region.as_ref().expect("SSM state not allocated");
        let slab = (self.conv_slab_floats as u64) * 4;
        crate::inference::buffer::BufferRange {
            buffer: region.buffer,
            offset: self.ssm_conv_base[layer as usize] + slot as u64 * slab,
            size: slab,
        }
    }

    /// Sequence `slot`'s GDN state for `layer` — a single-sequence `gdn_floats`
    /// slice of the seq-outermost block. Used to point a borrowed single-slot
    /// prefill cache at this slot's state so the prefill persists its final
    /// recurrent state where the batched decode will pick it up.
    pub fn gdn_state_slot(&self, layer: u32, slot: u32) -> crate::inference::buffer::BufferRange {
        let region = self.ssm_region.as_ref().expect("SSM state not allocated");
        let slab = (self.gdn_slab_floats as u64) * 4;
        crate::inference::buffer::BufferRange {
            buffer: region.buffer,
            offset: self.ssm_gdn_base[layer as usize] + slot as u64 * slab,
            size: slab,
        }
    }

    /// Number of SSM layers with allocated per-slot state (0 if none).
    pub fn n_ssm_layers(&self) -> usize {
        self.ssm_conv_base.len()
    }

    /// Total device memory (K/V slabs + SSM state) backing all slots, in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.region.size + self.ssm_region.as_ref().map_or(0, |r| r.size)
    }

    /// Zero just slot `slot`'s recurrent SSM state (all layers), leaving every
    /// other slab untouched. Use before re-prefilling a reused slab on a
    /// divergent prompt — the recurrent state has no per-position undo, so a
    /// non-extension must start from a fresh (zero) state. The attention K/V
    /// needs no zeroing: the prefill overwrites `[0, len)` and reads are bounded
    /// by the position cursor. (The whole-region [`KvCache::reset`] is unsafe
    /// here — it would clobber the other slabs' live state.)
    pub fn reset_slot(&self, slot: u32) {
        let Some(host) = self.ssm_region.as_ref().and_then(|r| r.host_ptr) else {
            return;
        };
        for layer in 0..self.n_ssm_layers() as u32 {
            for r in [self.conv_state_slot(layer, slot), self.gdn_state_slot(layer, slot)] {
                // SAFETY: `r` is a sub-range of the HOST_VISIBLE|HOST_COHERENT
                // ssm_region that `host` maps from offset 0.
                unsafe {
                    std::ptr::write_bytes(host.add(r.offset as usize), 0, r.size as usize);
                }
            }
        }
    }

    /// Per-slot conv / GDN recurrent-state lengths (floats), for gathering an
    /// active batch's state out of the seq-outermost per-layer blocks.
    pub fn conv_slot_floats(&self) -> u64 {
        self.conv_slab_floats as u64
    }
    pub fn gdn_slot_floats(&self) -> u64 {
        self.gdn_slab_floats as u64
    }

    /// Single-slot natural `[head_dim, n_head_kv, max_seq_len]` K view (slab
    /// `slot`, layer `layer`) — identical layout to `KvCache::k_layers[layer]`.
    pub fn slot_k_view(&self, slot: u32, layer: u32) -> TensorView {
        make_view(
            self.buffer,
            self.k_base[layer as usize] + slot as u64 * self.k_slab_stride,
            self.head_dim as u64,
            self.config.max_seq_len as u64,
            self.n_head_kv as u64,
            self.config.k_dtype,
        )
    }

    pub fn slot_v_view(&self, slot: u32, layer: u32) -> TensorView {
        make_view(
            self.buffer,
            self.v_base[layer as usize] + slot as u64 * self.v_slab_stride,
            self.head_dim as u64,
            self.config.max_seq_len as u64,
            self.n_head_kv as u64,
            self.config.v_dtype,
        )
    }

    /// Batched flash-attn K view for `layer`: permuted `[head_dim, max_seq_len,
    /// n_head_kv, n_slots]` with the slot as the batch dimension (stride = slab
    /// stride). Binds the **whole** per-layer K block (all slots) so the kernel
    /// can read any slab via `DecodeDyn::slot` — a gathered batch addresses
    /// non-contiguous slots. The shader bounds each sequence's KV loop by its
    /// own `kv_len`, so the unused tail past each slot's position is never read.
    pub fn batched_k_attn_view(&self, layer: u32) -> TensorView {
        batched_attn_view(
            self.buffer,
            self.k_base[layer as usize],
            self.k_slab_stride,
            self.head_dim as u64,
            self.config.max_seq_len as u64,
            self.n_head_kv as u64,
            self.n_slots,
            self.config.k_dtype,
        )
    }

    pub fn batched_v_attn_view(&self, layer: u32) -> TensorView {
        batched_attn_view(
            self.buffer,
            self.v_base[layer as usize],
            self.v_slab_stride,
            self.head_dim as u64,
            self.config.max_seq_len as u64,
            self.n_head_kv as u64,
            self.n_slots,
            self.config.v_dtype,
        )
    }

    /// A non-owning single-sequence `KvCache` over slot `slot` (its slabs +
    /// current position). Use it to prefill one sequence into its slab via the
    /// existing single-sequence forward path; afterwards copy its `position`
    /// back into `self.positions[slot]`.
    pub fn slot_kvcache(&self, slot: u32) -> KvCache {
        let k_layers = (0..self.n_layer).map(|l| self.slot_k_view(slot, l)).collect();
        let v_layers = (0..self.n_layer).map(|l| self.slot_v_view(slot, l)).collect();
        // Point the borrowed cache at this slot's per-sequence SSM state so a
        // hybrid prefill persists its final conv/GDN state into the batch slab
        // (the batched decode continues from it). Zero-initialized by
        // allocate_ssm_state, so the prefill still starts from a fresh state.
        let n_ssm = self.n_ssm_layers();
        let (ssm_conv_states, ssm_gdn_states, ssm_region) = if n_ssm > 0 {
            let conv = (0..n_ssm as u32).map(|l| self.conv_state_slot(l, slot)).collect();
            let gdn = (0..n_ssm as u32).map(|l| self.gdn_state_slot(l, slot)).collect();
            let r = self.ssm_region.as_ref().expect("SSM state allocated");
            (conv, gdn, Some((r.buffer, r.host_ptr, r.size)))
        } else {
            (Vec::new(), Vec::new(), None)
        };
        KvCache::borrowed_slot(
            self.config,
            self.buffer,
            self.host_ptr,
            self.buffer_size,
            self.alignment,
            k_layers,
            v_layers,
            self.positions[slot as usize],
            self.device.clone(),
            ssm_conv_states,
            ssm_gdn_states,
            ssm_region,
        )
    }
}

impl Drop for BatchKvCache {
    fn drop(&mut self) {
        let dev = self.device.raw();
        self.region.destroy(dev);
        if let Some(mut r) = self.ssm_region.take() {
            r.destroy(dev);
        }
    }
}

/// Build a batched flash-attn K/V view: permuted `[head_dim, max_seq_len,
/// n_head_kv, B]` with the slot as the (stride `slab_stride` bytes) batch
/// dimension. Assumes a flat per-element stride (block_size == 1).
#[allow(clippy::too_many_arguments)]
fn batched_attn_view(
    buffer: vk::Buffer,
    base: u64,
    slab_stride_bytes: u64,
    head_dim: u64,
    max_seq_len: u64,
    n_head_kv: u64,
    b: u32,
    dtype: GgmlType,
) -> TensorView {
    let ts = dtype.block_layout().1 as u64; // type_size (block_size == 1)
    let element_stride = [1u64, head_dim * n_head_kv, head_dim, slab_stride_bytes / ts];
    let byte_stride = [
        element_stride[0] * ts,
        element_stride[1] * ts,
        element_stride[2] * ts,
        element_stride[3] * ts,
    ];
    TensorView {
        buffer,
        byte_offset: base,
        byte_size: slab_stride_bytes * b as u64,
        dims: [head_dim, max_seq_len, n_head_kv, b as u64],
        byte_stride,
        element_stride,
        dtype,
    }
}
