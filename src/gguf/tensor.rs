use super::error::GgufError;
use super::types::GgmlType;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
    /// Precomputed byte length of the tensor's data.
    pub byte_size: usize,
}

pub(super) fn compute_byte_size(
    name: &str,
    dims: &[u64],
    ggml_type: GgmlType,
) -> Result<usize, GgufError> {
    let (elements_per_block, bytes_per_block) = ggml_type.block_layout();
    let elements: u64 = dims.iter().product();
    if elements % elements_per_block as u64 != 0 {
        return Err(GgufError::BadTensorShape {
            name: name.to_string(),
            elements,
            block: elements_per_block,
        });
    }
    let num_blocks = elements / elements_per_block as u64;
    Ok((num_blocks as usize) * bytes_per_block)
}
