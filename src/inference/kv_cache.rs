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
        })
    }

    /// Reset the position counter to 0. Buffer contents stay (will be
    /// overwritten by the next forward pass).
    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn destroy(&mut self, device: &Device) {
        self.region.destroy(device);
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
