//! Vision (mmproj) support: load a CLIP-style vision projector GGUF and parse
//! its `clip.*` metadata into a [`VisionConfig`].
//!
//! Phase 1 only loads + describes the projector; the encoder forward pass and
//! decoder integration land in later phases.

use std::error::Error;

use crate::gguf::{GgufFile, MetadataValue};
use crate::inference::weights::WeightsHandle;

/// The kind of vision projector shipped in an mmproj GGUF
/// (`clip.projector_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectorType {
    /// Qwen3-VL merger (`"qwen3vl_merger"`): deepstack + interpolated pos-embd.
    Qwen3VlMerger,
    /// Qwen2.5-VL merger (`"qwen2.5vl_merger"`): window attention, no deepstack.
    Qwen25VlMerger,
}

impl ProjectorType {
    /// Parse the `clip.projector_type` metadata string. Returns a clear error
    /// for any projector type seeker doesn't (yet) support.
    pub fn parse(s: &str) -> Result<ProjectorType, Box<dyn Error>> {
        match s {
            "qwen3vl_merger" => Ok(ProjectorType::Qwen3VlMerger),
            "qwen2.5vl_merger" => Ok(ProjectorType::Qwen25VlMerger),
            other => Err(format!("unsupported clip.projector_type: {other:?}").into()),
        }
    }
}

/// Parsed `clip.vision.*` configuration for a vision projector.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// The detected projector type.
    pub projector_type: ProjectorType,
    /// Input image size in pixels (`clip.vision.image_size`), if present.
    pub image_size: Option<u32>,
    /// Patch size in pixels (`clip.vision.patch_size`).
    pub patch_size: u32,
    /// Vision transformer embedding dim (`clip.vision.embedding_length`).
    pub n_embd: u32,
    /// Attention head count (`clip.vision.attention.head_count`).
    pub n_head: u32,
    /// Number of transformer blocks (`clip.vision.block_count`).
    pub n_layer: u32,
    /// Feed-forward dim (`clip.vision.feed_forward_length`).
    pub n_ff: u32,
    /// Spatial merge factor (`clip.vision.spatial_merge_size`, default 2).
    pub spatial_merge_size: u32,
    /// Attention layer-norm epsilon (`clip.vision.attention.layer_norm_epsilon`).
    pub eps: f32,
    /// Per-channel image normalization mean (`clip.vision.image_mean`).
    pub image_mean: [f32; 3],
    /// Per-channel image normalization std (`clip.vision.image_std`).
    pub image_std: [f32; 3],
    /// Number of deepstack layers (qwen3vl only; derived from the
    /// `clip.vision.is_deepstack_layers` bool array), if present.
    pub n_deepstack_layers: Option<u32>,
}

/// A loaded vision projector: its parsed config plus uploaded GPU weights.
pub struct VisionModel {
    pub config: VisionConfig,
    pub weights: WeightsHandle,
}

/// Read an integer metadata key, coercing across the GGUF int widths (matching
/// the `read_metadata_u64` pattern used elsewhere). Returns `None` if absent or
/// not an integer.
fn get_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    let v = match gguf.get(key)? {
        MetadataValue::U8(n) => *n as u64,
        MetadataValue::U16(n) => *n as u64,
        MetadataValue::U32(n) => *n as u64,
        MetadataValue::U64(n) => *n,
        MetadataValue::I8(n) if *n >= 0 => *n as u64,
        MetadataValue::I16(n) if *n >= 0 => *n as u64,
        MetadataValue::I32(n) if *n >= 0 => *n as u64,
        MetadataValue::I64(n) if *n >= 0 => *n as u64,
        _ => return None,
    };
    Some(v as u32)
}

/// Read a float metadata key (F32 or F64). Returns `None` if absent or not a
/// float.
fn get_f32(gguf: &GgufFile, key: &str) -> Option<f32> {
    match gguf.get(key)? {
        MetadataValue::F32(v) => Some(*v),
        MetadataValue::F64(v) => Some(*v as f32),
        _ => None,
    }
}

/// Read a string metadata key. Returns `None` if absent or not a string.
fn get_str<'a>(gguf: &'a GgufFile, key: &str) -> Option<&'a str> {
    match gguf.get(key)? {
        MetadataValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// One element of an array as f32 (coercing float/int element types).
fn elem_f32(v: &MetadataValue) -> Option<f32> {
    match v {
        MetadataValue::F32(x) => Some(*x),
        MetadataValue::F64(x) => Some(*x as f32),
        MetadataValue::U8(x) => Some(*x as f32),
        MetadataValue::U16(x) => Some(*x as f32),
        MetadataValue::U32(x) => Some(*x as f32),
        MetadataValue::I8(x) => Some(*x as f32),
        MetadataValue::I16(x) => Some(*x as f32),
        MetadataValue::I32(x) => Some(*x as f32),
        _ => None,
    }
}

/// Read a `[f32; 3]` from an array metadata key, falling back to `default` if
/// absent or malformed.
fn read_f32x3(gguf: &GgufFile, key: &str, default: [f32; 3]) -> [f32; 3] {
    match gguf.get(key) {
        Some(MetadataValue::Array(arr)) if arr.len() >= 3 => {
            let mut out = default;
            for (i, slot) in out.iter_mut().enumerate() {
                if let Some(v) = elem_f32(&arr[i]) {
                    *slot = v;
                }
            }
            out
        }
        _ => default,
    }
}

/// A required integer key, erroring with the key name if absent.
fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32, Box<dyn Error>> {
    get_u32(gguf, key).ok_or_else(|| format!("mmproj missing required metadata key: {key}").into())
}

/// A required float key, erroring with the key name if absent.
fn req_f32(gguf: &GgufFile, key: &str) -> Result<f32, Box<dyn Error>> {
    get_f32(gguf, key).ok_or_else(|| format!("mmproj missing required metadata key: {key}").into())
}

/// Parse a [`VisionConfig`] from an mmproj GGUF's metadata.
pub fn parse_config(gguf: &GgufFile) -> Result<VisionConfig, Box<dyn Error>> {
    let proj_str =
        get_str(gguf, "clip.projector_type").ok_or("mmproj missing clip.projector_type")?;
    let projector_type = ProjectorType::parse(proj_str)?;

    // `is_deepstack_layers` is a bool array marking which blocks are deepstack;
    // llama.cpp derives the count from it. Tolerate its absence (non-qwen3vl).
    let n_deepstack_layers = match gguf.get("clip.vision.is_deepstack_layers") {
        Some(MetadataValue::Array(arr)) => Some(
            arr.iter()
                .filter(|v| matches!(v, MetadataValue::Bool(true)))
                .count() as u32,
        ),
        _ => None,
    };

    Ok(VisionConfig {
        projector_type,
        image_size: get_u32(gguf, "clip.vision.image_size"),
        patch_size: req_u32(gguf, "clip.vision.patch_size")?,
        n_embd: req_u32(gguf, "clip.vision.embedding_length")?,
        n_head: req_u32(gguf, "clip.vision.attention.head_count")?,
        n_layer: req_u32(gguf, "clip.vision.block_count")?,
        n_ff: req_u32(gguf, "clip.vision.feed_forward_length")?,
        spatial_merge_size: get_u32(gguf, "clip.vision.spatial_merge_size").unwrap_or(2),
        eps: req_f32(gguf, "clip.vision.attention.layer_norm_epsilon")?,
        image_mean: read_f32x3(gguf, "clip.vision.image_mean", [0.0; 3]),
        image_std: read_f32x3(gguf, "clip.vision.image_std", [1.0; 3]),
        n_deepstack_layers,
    })
}

/// Load a vision projector from an open mmproj GGUF and its uploaded weights.
/// Parses the config, logs a one-line summary, and returns the model.
pub fn load(gguf: &GgufFile, weights: WeightsHandle) -> Result<VisionModel, Box<dyn Error>> {
    let config = parse_config(gguf)?;
    tracing::info!(
        projector = ?config.projector_type,
        n_embd = config.n_embd,
        n_head = config.n_head,
        n_layer = config.n_layer,
        n_ff = config.n_ff,
        patch_size = config.patch_size,
        spatial_merge_size = config.spatial_merge_size,
        n_deepstack_layers = ?config.n_deepstack_layers,
        tensors = weights.views.len(),
        weight_bytes = weights.total_bytes,
        "loaded vision projector (mmproj)"
    );
    Ok(VisionModel { config, weights })
}
