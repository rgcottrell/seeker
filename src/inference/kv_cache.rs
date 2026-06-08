//! KV cache. Per-layer K and V buffers persisting across `Engine::forward`
//! calls so prompt prefill happens once and subsequent decode steps run in
//! `O(1)` per token. K and V dtypes are independently configurable (the cache
//! may be asymmetric, e.g. K=q8_0 / V=q4_0) from
//! `{F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1}` plus the TurboQuant
//! KV quants `{Turbo2_0, Turbo3_0, Turbo4_0}` (WHT-rotated PolarQuant; require
//! head_dim % 128 == 0).

use std::error::Error;
use std::sync::Arc;

use ash::vk;

use crate::gguf::GgmlType;

use super::buffer::BufferRange;
use super::command::record_copy;
use super::device::{Device, DeviceShared};
use super::memory::Region;
use super::ops::cache_io::per_token_bytes;
use super::weights::TensorView;

#[derive(Debug, Clone, Copy)]
pub struct KvCacheConfig {
    pub k_dtype: GgmlType,
    pub v_dtype: GgmlType,
    pub max_seq_len: u32,
    /// Number of attention (query) heads. Only used to derive the GQA ratio
    /// (`n_head / n_head_kv`) for TurboQuant auto-asymmetric K-protection; `0`
    /// disables it (treated as ratio 1). Non-turbo caches ignore it.
    pub n_head: u32,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            k_dtype: GgmlType::F16,
            v_dtype: GgmlType::F16,
            max_seq_len: 2048,
            n_head: 0,
        }
    }
}

/// A restorable snapshot of a [`KvCache`]'s mutable per-sequence state (write
/// cursor + SSM/GDN recurrent buffer). Produced by [`KvCache::snapshot_state`]
/// and consumed by [`KvCache::restore_state`].
pub struct CacheState {
    position: u32,
    rope_position_lag: u32,
    ssm: Vec<u8>,
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
    // TurboQuant KV quants (WHT-rotated PolarQuant, 128-element blocks).
    // `validate_head_dim` enforces head_dim % 128 == 0 via the generic
    // block-size divisibility check (covers qwen35moe's 256 = 2 blocks/head).
    (GgmlType::Turbo2_0, "turbo2"),
    (GgmlType::Turbo3_0, "turbo3"),
    (GgmlType::Turbo4_0, "turbo4"),
];

/// Whether `ty` is a TurboQuant KV quant (WHT-rotated PolarQuant). Turbo caches
/// store K and V rotated, so the query must be forward-WHT-rotated before
/// attention and the attention output inverse-WHT-rotated afterward (see
/// `inference::ops::turbo_wht`). Gated per side: K-turbo drives the Q rotation,
/// V-turbo drives the output rotation.
pub fn is_turbo(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Turbo2_0 | GgmlType::Turbo3_0 | GgmlType::Turbo4_0
    )
}

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
    /// One buffer per layer, each holding that layer's K then V. Splitting per
    /// layer keeps every allocation under the device's `maxBufferSize` /
    /// `maxMemoryAllocationSize` (~4 GiB on RADV): a full model-max context
    /// (e.g. 256K) needs several GiB of KV total, which overflows a single
    /// buffer. Empty for a borrowed slot view — the owning `BatchKvCache` frees
    /// the buffers, and the per-layer views below carry the real handles.
    regions: Vec<Region>,
    pub k_layers: Vec<TensorView>,
    pub v_layers: Vec<TensorView>,
    /// Number of token positions already written into the cache.
    pub position: u32,
    /// How far the M-RoPE *position cursor* lags behind `position` (the
    /// KV-slot count). Zero for text-only sequences. After an image, the
    /// logical position advances by `max(nx,ny)` while `n_tok = nx*ny` KV
    /// slots are consumed, so this accumulates `Σ (n_tok - max(nx,ny))`.
    /// The rope base for any subsequent forward is `position - rope_position_lag`
    /// while KV writes / `kv_len` keep using `position`. See
    /// `Qwen35MoeModel::forward_impl` and `refresh_replay_inputs`.
    pub rope_position_lag: u32,
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
    /// Refcounted device owner so `Drop` can free the per-layer `regions` +
    /// the SSM regions, keeping the logical device alive until it does —
    /// regardless of whether the owning engine drops first.
    device: Arc<DeviceShared>,
}

/// Detached owner of the per-position SSM checkpoint buffers used by spec-decode
/// verify. The single-sequence path allocates these directly on its `KvCache`
/// ([`KvCache::allocate_ssm_snapshots`]); the serve worker — which runs spec on a
/// *borrowed* single-slot cache — owns one `SsmSnapshotSet` and
/// [`KvCache::attach_ssm_snapshots`]es it to the borrowed view each step (only
/// one verify is ever in flight at a time, so a single shared set suffices).
pub struct SsmSnapshotSet {
    gdn_region: Region,
    conv_region: Region,
    gdn_snaps: Vec<crate::inference::buffer::BufferRange>,
    conv_backs: Vec<crate::inference::buffer::BufferRange>,
    max_snapshots: u32,
    conv_kernel: u32,
    conv_channels: u32,
    /// Owns the device so `Drop` can free the two regions (this is a detached
    /// owner; the regions aren't tracked by any `KvCache`).
    device: Arc<DeviceShared>,
}

impl SsmSnapshotSet {
    /// Allocate the GDN snapshot + conv backup regions for `max_snapshots`
    /// (= n_draft+1) lookahead positions. Only call with `n_ssm_layers > 0` and
    /// `max_snapshots > 0` (the single-seq `allocate_ssm_snapshots` guards both).
    pub fn new(
        device: &Device,
        dims: &crate::models::SsmStateDims,
        max_snapshots: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let (gdn_region, conv_region, gdn_snaps, conv_backs) =
            alloc_ssm_snapshot_regions(device, dims, max_snapshots)?;
        Ok(Self {
            gdn_region,
            conv_region,
            gdn_snaps,
            conv_backs,
            max_snapshots,
            conv_kernel: dims.conv_kernel,
            conv_channels: dims.conv_channels,
            device: device.shared(),
        })
    }
}

impl Drop for SsmSnapshotSet {
    fn drop(&mut self) {
        let dev = self.device.raw();
        self.gdn_region.destroy(dev);
        self.conv_region.destroy(dev);
    }
}

/// Allocate the GDN snapshot + conv backup regions (packed per SSM layer) plus
/// the per-layer `BufferRange` slices for `max_snapshots` lookahead positions.
/// Shared by [`SsmSnapshotSet::new`] (serve's detached owner) and
/// [`KvCache::allocate_ssm_snapshots`] (the owning single-seq cache). Caller
/// guarantees `n_ssm_layers > 0` and `max_snapshots > 0`.
#[allow(clippy::type_complexity)]
fn alloc_ssm_snapshot_regions(
    device: &Device,
    dims: &crate::models::SsmStateDims,
    max_snapshots: u32,
) -> Result<
    (
        Region,
        Region,
        Vec<crate::inference::buffer::BufferRange>,
        Vec<crate::inference::buffer::BufferRange>,
    ),
    Box<dyn Error>,
> {
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
    Ok((gdn_region, conv_region, gdn_snaps, conv_backs))
}

impl KvCache {
    pub fn new(
        device: &Device,
        n_layer: u32,
        head_dim: u32,
        n_head_kv: u32,
        config: KvCacheConfig,
    ) -> Result<Self, Box<dyn Error>> {
        let head_dims = vec![head_dim; n_layer as usize];
        let n_head_kvs = vec![n_head_kv; n_layer as usize];
        Self::new_per_layer(device, &head_dims, &n_head_kvs, config)
    }

    /// As [`new`] but the K/V dimensions vary per layer. Each layer `il` gets
    /// its own buffer sized to `head_dims[il] × n_head_kvs[il] × max_seq_len`,
    /// with the matching natural-contiguous `k_layers`/`v_layers` views, so all
    /// the per-layer view helpers (reshape, permute, slice) work unchanged.
    /// Needed by gemma4, whose interleaved sliding-window / global layers use
    /// different `head_dim` (256 vs 512) and `n_head_kv` (8 vs 1).
    pub fn new_per_layer(
        device: &Device,
        head_dims: &[u32],
        n_head_kvs: &[u32],
        config: KvCacheConfig,
    ) -> Result<Self, Box<dyn Error>> {
        assert_eq!(
            head_dims.len(),
            n_head_kvs.len(),
            "per-layer head_dims / n_head_kvs length mismatch"
        );
        let n_layer = head_dims.len() as u32;
        validate_dtype(config.k_dtype, "K")?;
        validate_dtype(config.v_dtype, "V")?;
        for &hd in head_dims {
            validate_head_dim(hd, config.k_dtype, "K")?;
            validate_head_dim(hd, config.v_dtype, "V")?;
        }

        let max_seq_len = config.max_seq_len as u64;

        // Per-layer K/V dtypes (turbo auto-asymmetric + layer-adaptive may make
        // them differ across layers / sides). Non-turbo configs come back
        // uniform. Each layer's TensorView carries its own dtype, so the write,
        // flash-attn dispatch, and WHT gating all key off it per layer. (Turbo
        // GQA-protection keys off the first layer's n_head_kv; gemma4, the only
        // per-layer-dims arch, is non-turbo so this is uniform anyway.)
        let (k_dtypes, v_dtypes) = resolve_layer_dtypes(
            config.k_dtype,
            config.v_dtype,
            n_layer,
            config.n_head,
            *n_head_kvs.first().unwrap_or(&0),
        );

        let align = device.limits.min_storage_buffer_offset_alignment.max(1);

        // One buffer per layer (K at offset 0, V at `k_aligned`), so no single
        // allocation trips the device's ~4 GiB maxBufferSize /
        // maxMemoryAllocationSize even when a model-max context makes the KV
        // total several GiB.
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mem = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        // Preflight: a KV cache larger than the GPU heap it lives in would OOM
        // the allocation and (on RADV/amdgpu) wedge the device into a
        // device-lost. Total the per-layer K+V bytes up front and fail with an
        // actionable error instead. The 90% margin leaves room for weights /
        // scratch / other allocations sharing the heap. `head_dim × max_seq_len`
        // grows the cache, so an oversized `--ctx-size` is the usual trigger.
        let total_kv: u64 = (0..n_layer as usize)
            .map(|il| {
                let hd = head_dims[il] as u64;
                let nkv = n_head_kvs[il] as u64;
                align_up(tensor_bytes(hd, max_seq_len, nkv, k_dtypes[il]), align)
                    + align_up(tensor_bytes(hd, max_seq_len, nkv, v_dtypes[il]), align)
            })
            .sum();
        if let Some(heap) = crate::inference::memory::heap_size_for_buffer(device, usage, mem) {
            let budget = heap / 10 * 9;
            if total_kv > budget {
                const GIB: f64 = (1u64 << 30) as f64;
                return Err(format!(
                    "KV cache needs {:.1} GiB (max_seq_len={}, {} layers, k={:?} v={:?}) but \
                     the GPU memory heap is only {:.1} GiB — lower --ctx-size, or use a smaller \
                     cache dtype (e.g. --cache-type-k q8_0 --cache-type-v q8_0).",
                    total_kv as f64 / GIB,
                    config.max_seq_len,
                    n_layer,
                    config.k_dtype,
                    config.v_dtype,
                    heap as f64 / GIB,
                )
                .into());
            }
        }

        let mut regions = Vec::with_capacity(n_layer as usize);
        let mut k_layers = Vec::with_capacity(n_layer as usize);
        let mut v_layers = Vec::with_capacity(n_layer as usize);
        for il in 0..n_layer as usize {
            let head_dim_u = head_dims[il] as u64;
            let n_head_kv_u = n_head_kvs[il] as u64;
            let k_aligned = align_up(
                tensor_bytes(head_dim_u, max_seq_len, n_head_kv_u, k_dtypes[il]),
                align,
            );
            let v_aligned = align_up(
                tensor_bytes(head_dim_u, max_seq_len, n_head_kv_u, v_dtypes[il]),
                align,
            );
            let region = Region::new(device, (k_aligned + v_aligned).max(1), usage, mem)?;
            k_layers.push(make_view(
                region.buffer,
                0,
                head_dim_u,
                max_seq_len,
                n_head_kv_u,
                k_dtypes[il],
            ));
            v_layers.push(make_view(
                region.buffer,
                k_aligned,
                head_dim_u,
                max_seq_len,
                n_head_kv_u,
                v_dtypes[il],
            ));
            regions.push(region);
        }

        Ok(Self {
            config,
            regions,
            k_layers,
            v_layers,
            position: 0,
            rope_position_lag: 0,
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
        // Allocate the regions and take ownership onto self, so this cache's
        // `Drop` frees them (the single-seq / owning-cache case).
        let (gdn_region, conv_region, gdn_snaps, conv_backs) =
            alloc_ssm_snapshot_regions(device, dims, max_snapshots)?;
        self.ssm_gdn_snapshots = gdn_snaps;
        self.ssm_conv_backups = conv_backs;
        self.ssm_max_snapshots = max_snapshots;
        self.ssm_conv_kernel = dims.conv_kernel;
        self.ssm_conv_channels = dims.conv_channels;
        self.ssm_gdn_snap_region = Some(gdn_region);
        self.ssm_conv_backup_region = Some(conv_region);
        Ok(())
    }

    /// Point this (borrowed-slot) cache's snapshot *views* at a worker-owned
    /// [`SsmSnapshotSet`] without taking ownership of its regions — the set
    /// outlives the borrow. The `*_region` fields stay `None`, so this cache's
    /// `Drop` frees nothing. Used by `serve`'s single-stream spec step, which
    /// borrows a slot via [`BatchKvCache::slot_kvcache`] (which leaves the
    /// snapshot fields empty) and attaches the shared set before
    /// `decode_speculative`.
    pub fn attach_ssm_snapshots(&mut self, set: &SsmSnapshotSet) {
        self.ssm_gdn_snapshots = set.gdn_snaps.clone();
        self.ssm_conv_backups = set.conv_backs.clone();
        self.ssm_max_snapshots = set.max_snapshots;
        self.ssm_conv_kernel = set.conv_kernel;
        self.ssm_conv_channels = set.conv_channels;
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
        self.rope_position_lag = 0;
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

    /// Snapshot the mutable per-sequence state — the write cursor plus the
    /// SSM/GDN recurrent buffer — so it can be restored later without rebuilding
    /// it. Attention K/V need no snapshot: reads are bounded by `position`, and
    /// the prefix `[0, restored_position)` is never touched by work that runs
    /// past it. Intended as a measurement aid (the bench restores a depth-`d`
    /// state between reps instead of re-prefilling `d` tokens each time); the
    /// caller must ensure no GPU work is in flight (every `forward_*`
    /// fence-waits, so calling right after one is safe).
    pub fn snapshot_state(&self) -> CacheState {
        let ssm = match self
            .ssm_region
            .as_ref()
            .and_then(|r| r.host_ptr.map(|p| (p, r.size as usize)))
        {
            // SAFETY: host_ptr maps the whole HOST_VISIBLE|HOST_COHERENT region
            // of `size` bytes (same mapping reset() writes).
            Some((host_ptr, size)) => unsafe {
                std::slice::from_raw_parts(host_ptr as *const u8, size).to_vec()
            },
            None => Vec::new(),
        };
        CacheState {
            position: self.position,
            rope_position_lag: self.rope_position_lag,
            ssm,
        }
    }

    /// Restore a [`KvCache::snapshot_state`] snapshot. Same no-GPU-work-in-flight
    /// caveat as the snapshot.
    pub fn restore_state(&mut self, s: &CacheState) {
        self.position = s.position;
        self.rope_position_lag = s.rope_position_lag;
        if !s.ssm.is_empty()
            && let Some((host_ptr, size)) = self
                .ssm_region
                .as_ref()
                .and_then(|r| r.host_ptr.map(|p| (p, r.size as usize)))
        {
            let n = s.ssm.len().min(size);
            // SAFETY: copying `n <= size` bytes into the mapped region.
            unsafe {
                std::ptr::copy_nonoverlapping(s.ssm.as_ptr(), host_ptr, n);
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

    /// Total device bytes backing the attention K/V (sum over the per-layer
    /// buffers). Excludes the SSM/recurrent regions. `0` for a borrowed slot.
    pub fn kv_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.size).sum()
    }

    /// Host pointer mapping layer `il`'s K/V buffer, for prompt-cache
    /// (de)serialization. The layer's `k_layers[il]` / `v_layers[il]` views
    /// carry byte offsets relative to this base (K at 0, V after it). `None`
    /// when the cache isn't host-visible or `il` is out of range (e.g. a
    /// borrowed slot, which owns no regions).
    pub fn layer_host_ptr(&self, il: usize) -> Option<*mut u8> {
        self.regions.get(il).and_then(|r| r.host_ptr)
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
        let ssm_region =
            ssm_region.map(|(buf, hp, size)| Region::borrowed(buf, hp, size, alignment));
        Self {
            config,
            // Borrowed slot: the per-layer views carry the BatchKvCache's real
            // buffer handles, so this owns no regions (Drop frees nothing).
            regions: Vec::new(),
            k_layers,
            v_layers,
            position,
            // Borrowed single-sequence slot: text starts with no M-RoPE lag; an
            // image prefill through this path sets it (multimodal-in-scheduler).
            rope_position_lag: 0,
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
        // Free the per-layer KV regions + any SSM/snapshot regions. The Arc
        // keeps the logical device alive through this; forwards are synchronous
        // (each fence-waits), so the GPU is idle by teardown. Empty for a
        // borrowed slot view (frees nothing).
        let dev = self.device.raw();
        for r in &mut self.regions {
            r.destroy(dev);
        }
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

/// Resolve per-layer K/V dtypes from the base `(k, v)` config, applying
/// TurboQuant's K-protection heuristics (ported from the llama-cpp-turboquant
/// fork's `llama-kv-cache.cpp`):
///
/// 1. **Auto-asymmetric** — turbo-K quantization error is amplified by the GQA
///    broadcast (one quantized K head feeds many query heads), catastrophically
///    so at high ratios. When K is turbo, K==V, and `n_head/n_head_kv >= 6`,
///    upgrade K to `Q8_0`. Opt out with `TURBO_AUTO_ASYMMETRIC=0`.
/// 2. **Layer-adaptive "Boundary V"** — `TURBO_LAYER_ADAPTIVE` selects a mode
///    (0=off, 1/2=q8_0 K+V on edge layers, 5/6=turbo4/turbo2 V mix, 7=q8_0 V on
///    first2+last2 else turbo2). Mode 7 auto-enables when V is `turbo2` and
///    `n_layer >= 8`. Opt out with `TURBO_LAYER_ADAPTIVE=0`.
///
/// Returns `(per-layer K dtypes, per-layer V dtypes)`, each length `n_layer`.
/// Non-turbo base configs are returned unchanged (uniform).
fn resolve_layer_dtypes(
    base_k: GgmlType,
    base_v: GgmlType,
    n_layer: u32,
    n_head: u32,
    n_head_kv: u32,
) -> (Vec<GgmlType>, Vec<GgmlType>) {
    let nl = n_layer as usize;
    let mut k = base_k;

    // 1. Auto-asymmetric K upgrade.
    if is_turbo(base_k) && base_k == base_v {
        let gqa = n_head.checked_div(n_head_kv).unwrap_or(1);
        let disabled = std::env::var("TURBO_AUTO_ASYMMETRIC").as_deref() == Ok("0");
        if !disabled && gqa >= 6 {
            tracing::warn!(
                gqa_ratio = gqa,
                n_head,
                n_head_kv,
                "turbo auto-asymmetric: upgrading K {:?} -> Q8_0 (high GQA amplifies turbo-K \
                 error); disable with TURBO_AUTO_ASYMMETRIC=0",
                base_k
            );
            k = GgmlType::Q8_0;
        }
    }

    // 2. Layer-adaptive mode (env override, else auto-enable mode 7 for turbo2-V).
    let mode: i32 = match std::env::var("TURBO_LAYER_ADAPTIVE") {
        Ok(s) => s.trim().parse().unwrap_or(0),
        Err(_) => {
            if base_v == GgmlType::Turbo2_0 && nl >= 8 {
                7
            } else {
                0
            }
        }
    };
    if mode != 0 {
        tracing::info!(mode, "turbo layer-adaptive enabled");
    }

    let k_is_turbo = is_turbo(k);
    let v_is_turbo = is_turbo(base_v);
    let mut kd = vec![k; nl];
    let mut vd = vec![base_v; nl];
    for il in 0..nl {
        match mode {
            1 if k_is_turbo && nl >= 8 && (il < 4 || il >= nl - 4) => {
                kd[il] = GgmlType::Q8_0;
                vd[il] = GgmlType::Q8_0;
            }
            2 if k_is_turbo && nl >= 8 && il >= nl - 8 => {
                kd[il] = GgmlType::Q8_0;
                vd[il] = GgmlType::Q8_0;
            }
            5 if v_is_turbo && nl >= 8 => {
                let boundary = il < 2 || il >= nl - 2;
                vd[il] = if boundary {
                    GgmlType::Turbo4_0
                } else {
                    GgmlType::Turbo2_0
                };
            }
            6 if v_is_turbo && nl >= 8 => {
                vd[il] = if il >= nl - 8 {
                    GgmlType::Turbo4_0
                } else {
                    GgmlType::Turbo2_0
                };
            }
            7 if v_is_turbo && nl >= 8 => {
                let boundary = il < 2 || il >= nl - 2;
                vd[il] = if boundary {
                    GgmlType::Q8_0
                } else {
                    GgmlType::Turbo2_0
                };
            }
            _ => {}
        }
    }
    (kd, vd)
}

fn validate_head_dim(head_dim: u32, ty: GgmlType, side: &str) -> Result<(), Box<dyn Error>> {
    let (block_size, _) = ty.block_layout();
    if !(head_dim as usize).is_multiple_of(block_size) {
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

/// Least common multiple (for slab strides that must satisfy both the device
/// storage-buffer alignment and the quant block byte size).
fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return a.max(b);
    }
    let mut x = a;
    let mut y = b;
    while y != 0 {
        let t = y;
        y = x % y;
        x = t;
    }
    a / x * b // a/gcd*b
}

/// Per-slot slab stride (bytes) for a KV layer-side of the given `dtype`.
/// Padded to the storage-buffer `align` AND to a multiple of the quant block
/// byte size (so block-quant turbo's `elem / QUANT_K` addressing lands on slot
/// boundaries). Shared by [`BatchKvCache::new`] and the prefix-snapshot pool so
/// the two can never drift apart on a future dtype change.
pub(crate) fn slab_stride_for(
    head_dim: u64,
    max_seq_len: u64,
    n_head_kv: u64,
    align: u64,
    dtype: GgmlType,
) -> u64 {
    let (_, type_size) = dtype.block_layout();
    let bytes = tensor_bytes(head_dim, max_seq_len, n_head_kv, dtype);
    align_up(bytes, lcm(align, type_size as u64))
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
    /// One buffer per layer-side (each holds all `n_slots` slabs for that
    /// layer's K, resp. V). Splitting per layer-side keeps every allocation
    /// under the device's ~4 GiB maxBufferSize even at model-max ctx × parallel,
    /// while still placing all slots of a layer-side contiguously so
    /// `batched_attn_view` can stride over slots within one bound buffer.
    k_regions: Vec<Region>,
    v_regions: Vec<Region>,
    alignment: u64,
    /// Per-layer slot slab stride in bytes (padded to `alignment`, and — for
    /// block-quant turbo layers — to a multiple of the block byte size so the
    /// flash-attn block addressing `elem/QUANT_K` lands on slot boundaries).
    /// Per-layer because turbo K-protection (auto-asymmetric / Boundary-V) can
    /// give layers different K/V dtypes (and hence slab sizes).
    k_slab_stride: Vec<u64>,
    v_slab_stride: Vec<u64>,
    /// Per-layer resolved K/V dtypes (see [`resolve_layer_dtypes`]).
    k_dtypes: Vec<GgmlType>,
    v_dtypes: Vec<GgmlType>,
    /// Current write position (tokens) of each slot.
    pub positions: Vec<u32>,
    /// Per-slot M-RoPE position lag — how far the rope cursor trails
    /// `positions[slot]` (the KV-slot count) after an image was prefilled into
    /// that slot (`Σ n_tok − max(nx,ny)`; see [`KvCache::rope_position_lag`]).
    /// Zero for text-only slots, so the batched/unified forward's rope is the
    /// raw position there (unchanged). The scheduler feeds the forward
    /// `positions[slot] − rope_lag[slot]` for the rope while KV writes / kv_len
    /// keep using `positions[slot]`. Reset with the slot.
    pub rope_lag: Vec<u32>,
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

        let max_seq_len = config.max_seq_len as u64;
        let head_dim_u = head_dim as u64;
        let n_head_kv_u = n_head_kv as u64;
        let align = device.limits.min_storage_buffer_offset_alignment.max(1);

        // Per-layer dtypes (turbo auto-asymmetric / Boundary-V). Uniform for
        // non-turbo configs.
        let (k_dtypes, v_dtypes) = resolve_layer_dtypes(
            config.k_dtype,
            config.v_dtype,
            n_layer,
            config.n_head,
            n_head_kv,
        );

        // Per-slot slab stride for `dtype`. Padded to the storage-buffer
        // alignment (so each slot's byte offset is descriptor-bindable for the
        // per-slot prefill path) AND to a multiple of the block byte size (so
        // flash-attn's `elem / QUANT_K` block index lands on slot boundaries for
        // block-quant turbo). For block_size==1 the type size divides the
        // alignment, so this reduces to `align_up(bytes, align)`.
        let slab_stride =
            |dtype: GgmlType| slab_stride_for(head_dim_u, max_seq_len, n_head_kv_u, align, dtype);
        let k_slab_stride: Vec<u64> = k_dtypes.iter().map(|&d| slab_stride(d)).collect();
        let v_slab_stride: Vec<u64> = v_dtypes.iter().map(|&d| slab_stride(d)).collect();

        // One buffer per layer-side, each holding that side's `n_slots` slabs
        // contiguously (slab `slot` at `slot * slab_stride`). Per-layer-side so
        // no single allocation exceeds maxBufferSize even at model-max ctx ×
        // parallel; contiguous slots so `batched_attn_view` strides over slots
        // within the one bound buffer.
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mem = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut k_regions = Vec::with_capacity(n_layer as usize);
        let mut v_regions = Vec::with_capacity(n_layer as usize);
        for il in 0..n_layer as usize {
            let kr = Region::new(
                device,
                (k_slab_stride[il] * n_slots as u64).max(1),
                usage,
                mem,
            )?;
            let vr = Region::new(
                device,
                (v_slab_stride[il] * n_slots as u64).max(1),
                usage,
                mem,
            )?;
            // Zero so no slot reads stale K/V on its first forward (mirrors
            // KvCache's implicit zero-on-fresh-alloc reliance).
            if let Some(p) = kr.host_ptr {
                unsafe { std::ptr::write_bytes(p, 0, kr.size as usize) };
            }
            if let Some(p) = vr.host_ptr {
                unsafe { std::ptr::write_bytes(p, 0, vr.size as usize) };
            }
            k_regions.push(kr);
            v_regions.push(vr);
        }
        let alignment = k_regions.first().map_or(align, |r| r.alignment);

        Ok(Self {
            config,
            n_slots,
            n_layer,
            head_dim,
            n_head_kv,
            k_regions,
            v_regions,
            alignment,
            k_slab_stride,
            v_slab_stride,
            k_dtypes,
            v_dtypes,
            positions: vec![0; n_slots as usize],
            rope_lag: vec![0; n_slots as usize],
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
        let kv: u64 = self
            .k_regions
            .iter()
            .chain(&self.v_regions)
            .map(|r| r.size)
            .sum();
        kv + self.ssm_region.as_ref().map_or(0, |r| r.size)
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
            for r in [
                self.conv_state_slot(layer, slot),
                self.gdn_state_slot(layer, slot),
            ] {
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
    /// Per-layer resolved K / V dtype (turbo K-protection may differ per layer).
    pub fn k_dtype(&self, layer: u32) -> GgmlType {
        self.k_dtypes[layer as usize]
    }
    pub fn v_dtype(&self, layer: u32) -> GgmlType {
        self.v_dtypes[layer as usize]
    }

    pub fn slot_k_view(&self, slot: u32, layer: u32) -> TensorView {
        make_view(
            self.k_regions[layer as usize].buffer,
            slot as u64 * self.k_slab_stride[layer as usize],
            self.head_dim as u64,
            self.config.max_seq_len as u64,
            self.n_head_kv as u64,
            self.k_dtypes[layer as usize],
        )
    }

    pub fn slot_v_view(&self, slot: u32, layer: u32) -> TensorView {
        make_view(
            self.v_regions[layer as usize].buffer,
            slot as u64 * self.v_slab_stride[layer as usize],
            self.head_dim as u64,
            self.config.max_seq_len as u64,
            self.n_head_kv as u64,
            self.v_dtypes[layer as usize],
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
            self.k_regions[layer as usize].buffer,
            0,
            self.k_slab_stride[layer as usize],
            self.head_dim as u64,
            self.config.max_seq_len as u64,
            self.n_head_kv as u64,
            self.n_slots,
            self.k_dtypes[layer as usize],
        )
    }

    pub fn batched_v_attn_view(&self, layer: u32) -> TensorView {
        batched_attn_view(
            self.v_regions[layer as usize].buffer,
            0,
            self.v_slab_stride[layer as usize],
            self.head_dim as u64,
            self.config.max_seq_len as u64,
            self.n_head_kv as u64,
            self.n_slots,
            self.v_dtypes[layer as usize],
        )
    }

    /// A non-owning single-sequence `KvCache` over slot `slot` (its slabs +
    /// current position). Use it to prefill one sequence into its slab via the
    /// existing single-sequence forward path; afterwards copy its `position`
    /// back into `self.positions[slot]`.
    pub fn slot_kvcache(&self, slot: u32) -> KvCache {
        let k_layers = (0..self.n_layer)
            .map(|l| self.slot_k_view(slot, l))
            .collect();
        let v_layers = (0..self.n_layer)
            .map(|l| self.slot_v_view(slot, l))
            .collect();
        // Point the borrowed cache at this slot's per-sequence SSM state so a
        // hybrid prefill persists its final conv/GDN state into the batch slab
        // (the batched decode continues from it). Zero-initialized by
        // allocate_ssm_state, so the prefill still starts from a fresh state.
        let n_ssm = self.n_ssm_layers();
        let (ssm_conv_states, ssm_gdn_states, ssm_region) = if n_ssm > 0 {
            let conv = (0..n_ssm as u32)
                .map(|l| self.conv_state_slot(l, slot))
                .collect();
            let gdn = (0..n_ssm as u32)
                .map(|l| self.gdn_state_slot(l, slot))
                .collect();
            let r = self.ssm_region.as_ref().expect("SSM state allocated");
            (conv, gdn, Some((r.buffer, r.host_ptr, r.size)))
        } else {
            (Vec::new(), Vec::new(), None)
        };
        let mut sc = KvCache::borrowed_slot(
            self.config,
            self.alignment,
            k_layers,
            v_layers,
            self.positions[slot as usize],
            self.device.clone(),
            ssm_conv_states,
            ssm_gdn_states,
            ssm_region,
        );
        // Carry the slot's M-RoPE lag so a single-seq op on the borrowed cache
        // (e.g. an image prefill) continues with the right rope base. The caller
        // copies any updated lag back into `self.rope_lag[slot]`.
        sc.rope_position_lag = self.rope_lag[slot as usize];
        sc
    }

    // ─── Leading-prefix snapshot cache (serve) ───────────────────────
    //
    // The shared leading prefix of a request is prefilled once, then seeded
    // into later divergent requests by copying `KV[0, P)` + the SSM state at
    // P into a fresh slab. Because the SSM/GDN/conv state is position-
    // independent and `KV[0, P)` is contiguous, a GPU→GPU copy + setting the
    // slot's position to P is byte-identical, in-process, to that slab having
    // prefilled `[0, P)` itself (the same determinism pure-extension reuse
    // already relies on). All copies are GPU-side `vkCmdCopyBuffer` — host
    // reads of GPU-written SSM regions return stale bytes despite HOST_COHERENT.

    /// Contiguous K byte range covering tokens `[0, p)` of `slot`'s slab for
    /// `layer`, plus that layer's `per_token_bytes`. Position is the outermost
    /// axis of the `[head_dim, n_head_kv, max_seq_len]` layout, so the prefix is
    /// one flat span — a single `vkCmdCopyBuffer` seeds or captures it.
    pub fn k_prefix_range(&self, slot: u32, layer: u32, p: u32) -> (BufferRange, u64) {
        let il = layer as usize;
        let (block_size, _) = self.k_dtypes[il].block_layout();
        debug_assert_eq!(
            (self.head_dim as u64 * self.n_head_kv as u64) % block_size as u64,
            0,
            "KV per-token elements must be a whole number of quant blocks for a contiguous prefix copy"
        );
        let ptb = per_token_bytes(
            self.head_dim as u64,
            self.n_head_kv as u64,
            self.k_dtypes[il],
        );
        (
            BufferRange {
                buffer: self.k_regions[il].buffer,
                offset: slot as u64 * self.k_slab_stride[il],
                size: p as u64 * ptb,
            },
            ptb,
        )
    }

    /// V-side companion to [`Self::k_prefix_range`].
    pub fn v_prefix_range(&self, slot: u32, layer: u32, p: u32) -> (BufferRange, u64) {
        let il = layer as usize;
        let ptb = per_token_bytes(
            self.head_dim as u64,
            self.n_head_kv as u64,
            self.v_dtypes[il],
        );
        (
            BufferRange {
                buffer: self.v_regions[il].buffer,
                offset: slot as u64 * self.v_slab_stride[il],
                size: p as u64 * ptb,
            },
            ptb,
        )
    }

    /// Allocate a [`PrefixSnapshot`] sized to hold up to `max_cached_len` tokens
    /// of this cache's KV (per-layer dtype/stride, via the shared
    /// [`slab_stride_for`]) plus one sequence's SSM recurrent state.
    pub fn new_prefix_snapshot(
        &self,
        device: &Device,
        max_cached_len: u32,
    ) -> Result<PrefixSnapshot, Box<dyn Error>> {
        let maxl = max_cached_len as u64;
        let align = device.limits.min_storage_buffer_offset_alignment.max(1);
        let head_dim_u = self.head_dim as u64;
        let n_head_kv_u = self.n_head_kv as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mem = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let mut k_regions = Vec::with_capacity(self.n_layer as usize);
        let mut v_regions = Vec::with_capacity(self.n_layer as usize);
        for il in 0..self.n_layer as usize {
            let kb = slab_stride_for(head_dim_u, maxl, n_head_kv_u, align, self.k_dtypes[il]);
            let vb = slab_stride_for(head_dim_u, maxl, n_head_kv_u, align, self.v_dtypes[il]);
            k_regions.push(Region::new(device, kb.max(1), usage, mem)?);
            v_regions.push(Region::new(device, vb.max(1), usage, mem)?);
        }

        let n_ssm = self.n_ssm_layers();
        let conv = self.conv_slab_floats as u64 * 4;
        let gdn = self.gdn_slab_floats as u64 * 4;
        let mut conv_off = Vec::with_capacity(n_ssm);
        let mut gdn_off = Vec::with_capacity(n_ssm);
        let mut cursor = 0u64;
        for _ in 0..n_ssm {
            let co = align_up(cursor, align);
            cursor = co + conv;
            let go = align_up(cursor, align);
            cursor = go + gdn;
            conv_off.push(co);
            gdn_off.push(go);
        }
        let ssm_region = Region::new(device, cursor.max(1), usage, mem)?;

        Ok(PrefixSnapshot {
            k_regions,
            v_regions,
            ssm_region,
            conv_off,
            gdn_off,
            device: self.device.clone(),
        })
    }

    /// Record KV copies between `slot`'s slab and `snap` (`capture` = slab→snap,
    /// else snap→slab) for the prefix `[0, p)`.
    fn record_kv(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        snap: &PrefixSnapshot,
        slot: u32,
        p: u32,
        capture: bool,
    ) {
        for il in 0..self.n_layer {
            let (slab_k, ptb_k) = self.k_prefix_range(slot, il, p);
            let snap_k = BufferRange {
                buffer: snap.k_regions[il as usize].buffer,
                offset: 0,
                size: p as u64 * ptb_k,
            };
            let (src, dst) = if capture {
                (slab_k, snap_k)
            } else {
                (snap_k, slab_k)
            };
            record_copy(device, cmd, src, dst, src.size);

            let (slab_v, ptb_v) = self.v_prefix_range(slot, il, p);
            let snap_v = BufferRange {
                buffer: snap.v_regions[il as usize].buffer,
                offset: 0,
                size: p as u64 * ptb_v,
            };
            let (src, dst) = if capture {
                (slab_v, snap_v)
            } else {
                (snap_v, slab_v)
            };
            record_copy(device, cmd, src, dst, src.size);
        }
    }

    /// Record SSM (conv + GDN, all layers) copies between `slot`'s slab and
    /// `snap` (`capture` = slab→snap, else snap→slab).
    fn record_ssm(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        snap: &PrefixSnapshot,
        slot: u32,
        capture: bool,
    ) {
        let conv = self.conv_slab_floats as u64 * 4;
        let gdn = self.gdn_slab_floats as u64 * 4;
        for layer in 0..self.n_ssm_layers() {
            let slab_conv = self.conv_state_slot(layer as u32, slot);
            let snap_conv = BufferRange {
                buffer: snap.ssm_region.buffer,
                offset: snap.conv_off[layer],
                size: conv,
            };
            let (src, dst) = if capture {
                (slab_conv, snap_conv)
            } else {
                (snap_conv, slab_conv)
            };
            record_copy(device, cmd, src, dst, conv);

            let slab_gdn = self.gdn_state_slot(layer as u32, slot);
            let snap_gdn = BufferRange {
                buffer: snap.ssm_region.buffer,
                offset: snap.gdn_off[layer],
                size: gdn,
            };
            let (src, dst) = if capture {
                (slab_gdn, snap_gdn)
            } else {
                (snap_gdn, slab_gdn)
            };
            record_copy(device, cmd, src, dst, gdn);
        }
    }

    /// Record the copies seeding `dst_slot`'s slab (KV `[0, p)` + SSM state at P)
    /// from `snap`. The caller emits the trailing global barrier before the slab
    /// is read by a forward.
    pub fn record_seed(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        snap: &PrefixSnapshot,
        dst_slot: u32,
        p: u32,
    ) {
        self.record_kv(device, cmd, snap, dst_slot, p, /*capture=*/ false);
        self.record_ssm(device, cmd, snap, dst_slot, /*capture=*/ false);
    }

    /// Record the copies capturing `src_slot`'s prefix `[0, p)` (KV + SSM state
    /// at P) into `snap`. The slab's live SSM state equals state-at-P only when
    /// `positions[src_slot] == p` (a prefill chunk boundary), which the caller
    /// must guarantee.
    pub fn record_capture(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        snap: &PrefixSnapshot,
        src_slot: u32,
        p: u32,
    ) {
        self.record_kv(device, cmd, snap, src_slot, p, /*capture=*/ true);
        self.record_ssm(device, cmd, snap, src_slot, /*capture=*/ true);
    }
}

/// A pool buffer set holding one captured leading prefix: per-layer-side KV
/// "mini-slabs" (sized for up to `max_cached_len` tokens) plus one sequence's
/// SSM recurrent state (conv + GDN per SSM layer). Owns its `Region`s. Lives in
/// `inference` (not `server`) so both [`BatchKvCache`]'s copy helpers and
/// [`crate::inference::Engine`] can address it. Snapshots are byte-identical to
/// a fresh prefill on the same device but NOT across processes/drivers, so they
/// are never persisted to disk.
pub struct PrefixSnapshot {
    k_regions: Vec<Region>,
    v_regions: Vec<Region>,
    ssm_region: Region,
    /// Per-SSM-layer byte offsets of this snapshot's single-sequence conv / GDN
    /// state within `ssm_region`.
    conv_off: Vec<u64>,
    gdn_off: Vec<u64>,
    device: Arc<DeviceShared>,
}

impl PrefixSnapshot {
    /// Total device memory backing this snapshot (KV mini-slabs + SSM state).
    pub fn total_bytes(&self) -> u64 {
        self.k_regions
            .iter()
            .chain(&self.v_regions)
            .map(|r| r.size)
            .sum::<u64>()
            + self.ssm_region.size
    }
}

impl Drop for PrefixSnapshot {
    fn drop(&mut self) {
        let dev = self.device.raw();
        for r in self.k_regions.iter_mut().chain(self.v_regions.iter_mut()) {
            r.destroy(dev);
        }
        self.ssm_region.destroy(dev);
    }
}

impl Drop for BatchKvCache {
    fn drop(&mut self) {
        let dev = self.device.raw();
        for r in self.k_regions.iter_mut().chain(self.v_regions.iter_mut()) {
            r.destroy(dev);
        }
        if let Some(mut r) = self.ssm_region.take() {
            r.destroy(dev);
        }
    }
}

/// Build a batched flash-attn K/V view: permuted `[head_dim, max_seq_len,
/// n_head_kv, B]` with the slot as the (stride `slab_stride` bytes) batch
/// dimension. The within-slab element strides are in true elements (matching
/// `permute_to_attn`); flash-attn reads element-by-element and derives the
/// block via `elem / QUANT_K`. The slot stride is block-granular: a slab spans
/// `slab_stride_bytes / type_size` blocks, i.e. `block_size ×` that many
/// element slots in the `elem / QUANT_K` addressing (slab strides are padded to
/// a multiple of the block byte size, so this divides evenly). For block_size==1
/// this reduces to the flat per-element stride.
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
    let (block_size, type_size) = dtype.block_layout();
    let bs = block_size as u64;
    let ts = type_size as u64;
    let slot_stride = bs * (slab_stride_bytes / ts);
    let element_stride = [1u64, head_dim * n_head_kv, head_dim, slot_stride];
    // FA addresses via `element_stride` (+ `elem/QUANT_K` for the block); the
    // byte strides only feed the descriptor range, so the slot byte distance is
    // the real `slab_stride_bytes`.
    let byte_stride = [
        ts,
        ts * head_dim * n_head_kv,
        ts * head_dim,
        slab_stride_bytes,
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
