mod error;
mod tensor;
mod types;
mod value;

pub use error::GgufError;
pub use tensor::TensorInfo;
pub use types::{GgmlType, MetadataValueType};
pub use value::MetadataValue;

use std::fs::File;
use std::path::Path;

use indexmap::IndexMap;
use memmap2::Mmap;

const MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

pub struct GgufFile {
    mmap: Mmap,
    version: u32,
    alignment: u64,
    metadata: IndexMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    tensor_data_start: usize,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let parsed = parse_index(&mmap)?;
        Ok(Self {
            mmap,
            version: parsed.version,
            alignment: parsed.alignment,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            tensor_data_start: parsed.tensor_data_start,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Absolute byte offset within the file where the tensor data section begins.
    pub fn data_offset(&self) -> usize {
        self.tensor_data_start
    }

    /// Total file size in bytes (equivalent to the mmap length).
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    pub fn metadata(&self) -> &IndexMap<String, MetadataValue> {
        &self.metadata
    }

    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    pub fn architecture(&self) -> Option<&str> {
        match self.get("general.architecture") {
            Some(MetadataValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor(name)?;
        let start = self.tensor_data_start.checked_add(info.offset as usize)?;
        let end = start.checked_add(info.byte_size)?;
        self.mmap.get(start..end)
    }
}

#[derive(Debug)]
struct Parsed {
    version: u32,
    alignment: u64,
    metadata: IndexMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    tensor_data_start: usize,
}

fn parse_index(data: &[u8]) -> Result<Parsed, GgufError> {
    let mut r = Reader::new(data);

    let magic = r.read_bytes::<4>()?;
    if magic != MAGIC {
        return Err(GgufError::BadMagic(magic));
    }

    let version = r.read_u32()?;
    let (tensor_count, kv_count) = match version {
        1 => (r.read_u32()? as u64, r.read_u32()? as u64),
        2 | 3 => (r.read_u64()?, r.read_u64()?),
        _ => return Err(GgufError::UnsupportedVersion(version)),
    };

    let mut metadata: IndexMap<String, MetadataValue> = IndexMap::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = r.read_string()?;
        let ty = MetadataValueType::from_u32(r.read_u32()?)?;
        let val = r.read_value(ty)?;
        metadata.insert(key, val);
    }

    let alignment = match metadata.get("general.alignment") {
        Some(MetadataValue::U32(v)) => *v as u64,
        _ => DEFAULT_ALIGNMENT,
    };

    let mut tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.read_string()?;
        let n_dims = r.read_u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.read_u64()?);
        }
        let ggml_type = GgmlType::from_u32(r.read_u32()?)?;
        let offset = r.read_u64()?;
        let byte_size = tensor::compute_byte_size(&name, &dims, ggml_type)?;
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
            byte_size,
        });
    }

    let pos = r.position();
    let align = alignment as usize;
    let pad = (align - (pos % align)) % align;
    let tensor_data_start = pos + pad;
    if tensor_data_start > data.len() {
        return Err(GgufError::Truncated {
            at: tensor_data_start,
        });
    }

    Ok(Parsed {
        version,
        alignment,
        metadata,
        tensors,
        tensor_data_start,
    })
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn ensure(&self, n: usize) -> Result<(), GgufError> {
        match self.pos.checked_add(n) {
            Some(end) if end <= self.data.len() => Ok(()),
            _ => Err(GgufError::Truncated { at: self.pos }),
        }
    }

    fn read_bytes<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        self.ensure(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(&self.data[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn read_u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read_bytes()?))
    }
    fn read_u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read_bytes()?))
    }
    fn read_u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read_bytes()?))
    }
    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }
    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.read_bytes()?))
    }
    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.read_bytes()?))
    }
    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.read_bytes()?))
    }
    fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.read_bytes()?))
    }
    fn read_f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.read_bytes()?))
    }
    fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|_| GgufError::BadUtf8 { field: "string" })?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn read_value(&mut self, ty: MetadataValueType) -> Result<MetadataValue, GgufError> {
        Ok(match ty {
            MetadataValueType::U8 => MetadataValue::U8(self.read_u8()?),
            MetadataValueType::I8 => MetadataValue::I8(self.read_i8()?),
            MetadataValueType::U16 => MetadataValue::U16(self.read_u16()?),
            MetadataValueType::I16 => MetadataValue::I16(self.read_i16()?),
            MetadataValueType::U32 => MetadataValue::U32(self.read_u32()?),
            MetadataValueType::I32 => MetadataValue::I32(self.read_i32()?),
            MetadataValueType::F32 => MetadataValue::F32(self.read_f32()?),
            MetadataValueType::Bool => MetadataValue::Bool(self.read_bool()?),
            MetadataValueType::String => MetadataValue::String(self.read_string()?),
            MetadataValueType::U64 => MetadataValue::U64(self.read_u64()?),
            MetadataValueType::I64 => MetadataValue::I64(self.read_i64()?),
            MetadataValueType::F64 => MetadataValue::F64(self.read_f64()?),
            MetadataValueType::Array => {
                let inner_ty = MetadataValueType::from_u32(self.read_u32()?)?;
                let len = self.read_u64()? as usize;
                let mut elems = Vec::with_capacity(len.min(1024));
                for _ in 0..len {
                    elems.push(self.read_value(inner_ty)?);
                }
                MetadataValue::Array(elems)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builder for crafting synthetic GGUF byte streams in tests.
    struct GgufBuilder(Vec<u8>);

    impl GgufBuilder {
        fn new() -> Self {
            Self(Vec::new())
        }
        fn header(mut self, version: u32, tensor_count: u64, kv_count: u64) -> Self {
            self.0.extend_from_slice(&MAGIC);
            self.0.extend_from_slice(&version.to_le_bytes());
            self.0.extend_from_slice(&tensor_count.to_le_bytes());
            self.0.extend_from_slice(&kv_count.to_le_bytes());
            self
        }
        fn string(mut self, s: &str) -> Self {
            self.0.extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.0.extend_from_slice(s.as_bytes());
            self
        }
        fn u32(mut self, v: u32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u64(mut self, v: u64) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn pad_to_alignment(mut self, alignment: usize) -> Self {
            while !self.0.len().is_multiple_of(alignment) {
                self.0.push(0);
            }
            self
        }
        fn raw(mut self, bytes: &[u8]) -> Self {
            self.0.extend_from_slice(bytes);
            self
        }
        fn finish(self) -> Vec<u8> {
            self.0
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes =
            b"NOPE\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let err = parse_index(bytes).unwrap_err();
        assert!(matches!(err, GgufError::BadMagic(_)));
    }

    #[test]
    fn rejects_unknown_version() {
        let bytes = GgufBuilder::new().header(99, 0, 0).finish();
        let err = parse_index(&bytes).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion(99)));
    }

    #[test]
    fn rejects_truncated_header() {
        let err = parse_index(&[0u8; 3]).unwrap_err();
        assert!(matches!(err, GgufError::Truncated { .. }));
    }

    #[test]
    fn parses_empty_v3_header() {
        let bytes = GgufBuilder::new()
            .header(3, 0, 0)
            .pad_to_alignment(32)
            .finish();
        let p = parse_index(&bytes).unwrap();
        assert_eq!(p.version, 3);
        assert_eq!(p.alignment, 32);
        assert!(p.metadata.is_empty());
        assert!(p.tensors.is_empty());
    }

    #[test]
    fn parses_string_and_u32_metadata() {
        // 1 KV: "general.name" = "test"; 1 KV: "general.alignment" = 64
        let bytes = GgufBuilder::new()
            .header(3, 0, 2)
            .string("general.name")
            .u32(MetadataValueType::String as u32)
            .string("test")
            .string("general.alignment")
            .u32(MetadataValueType::U32 as u32)
            .u32(64)
            .pad_to_alignment(64)
            .finish();
        let p = parse_index(&bytes).unwrap();
        assert_eq!(p.alignment, 64);
        assert!(
            matches!(p.metadata.get("general.name"), Some(MetadataValue::String(s)) if s == "test")
        );
    }

    #[test]
    fn parses_nested_array_of_strings() {
        let bytes = GgufBuilder::new()
            .header(3, 0, 1)
            .string("tokenizer.ggml.tokens")
            .u32(MetadataValueType::Array as u32)
            .u32(MetadataValueType::String as u32) // inner type
            .u64(3) // length
            .string("hello")
            .string("world")
            .string("!")
            .pad_to_alignment(32)
            .finish();
        let p = parse_index(&bytes).unwrap();
        let arr = match p.metadata.get("tokenizer.ggml.tokens").unwrap() {
            MetadataValue::Array(a) => a,
            other => panic!("wrong type: {other:?}"),
        };
        assert_eq!(arr.len(), 3);
        assert!(matches!(&arr[0], MetadataValue::String(s) if s == "hello"));
        assert!(matches!(&arr[2], MetadataValue::String(s) if s == "!"));
    }

    #[test]
    fn parses_tensor_info_and_computes_byte_size() {
        // Tensor "x" of shape [4, 2], F32 = 32 bytes total at offset 0.
        let bytes = GgufBuilder::new()
            .header(3, 1, 0)
            .string("x")
            .u32(2) // n_dims
            .u64(4)
            .u64(2)
            .u32(GgmlType::F32 as u32)
            .u64(0)
            .pad_to_alignment(32)
            .raw(&[0u8; 32]) // tensor data
            .finish();
        let p = parse_index(&bytes).unwrap();
        assert_eq!(p.tensors.len(), 1);
        let t = &p.tensors[0];
        assert_eq!(t.name, "x");
        assert_eq!(t.dims, vec![4, 2]);
        assert_eq!(t.ggml_type, GgmlType::F32);
        assert_eq!(t.byte_size, 32);
    }

    #[test]
    fn alignment_padding_positions_tensor_data() {
        // Empty header, no metadata, no tensors. Then 32-byte alignment.
        let bytes = GgufBuilder::new()
            .header(3, 0, 0) // 4 + 4 + 8 + 8 = 24 bytes
            .pad_to_alignment(32) // pad 8 bytes to reach offset 32
            .raw(&[0u8; 64]) // tensor data area
            .finish();
        let p = parse_index(&bytes).unwrap();
        assert_eq!(p.tensor_data_start, 32);
    }

    #[test]
    fn block_layout_q4_k_matches_llama_cpp() {
        // 256 elements per block, 144 bytes per block.
        assert_eq!(GgmlType::Q4_K.block_layout(), (256, 144));
        // A 256-element Q4_K tensor occupies one block = 144 bytes.
        assert_eq!(
            tensor::compute_byte_size("t", &[256], GgmlType::Q4_K).unwrap(),
            144
        );
        // 512 elements = 2 blocks = 288 bytes.
        assert_eq!(
            tensor::compute_byte_size("t", &[2, 256], GgmlType::Q4_K).unwrap(),
            288
        );
    }

    #[test]
    fn rejects_tensor_shape_not_multiple_of_block_size() {
        let err = tensor::compute_byte_size("t", &[100], GgmlType::Q4_K).unwrap_err();
        assert!(matches!(err, GgufError::BadTensorShape { .. }));
    }

    /// Opens the cached SmolLM2 GGUF if it's available locally. Marked
    /// `#[ignore]` so `cargo test` stays hermetic; run with
    /// `cargo test gguf -- --ignored` after `seeker download` has populated
    /// the HF cache.
    #[test]
    #[ignore]
    fn opens_real_smollm2_file() {
        let home = std::env::var("HOME").unwrap();
        let dir = format!(
            "{home}/.cache/huggingface/hub/models--bartowski--SmolLM2-135M-Instruct-GGUF/snapshots"
        );
        let snapshot = std::fs::read_dir(&dir)
            .expect("cached snapshots dir missing")
            .next()
            .expect("no snapshot")
            .expect("dir entry")
            .path();
        let path = snapshot.join("SmolLM2-135M-Instruct-Q4_K_M.gguf");
        let g = GgufFile::open(&path).expect("open gguf");
        assert_eq!(g.version(), 3);
        assert!(g.architecture().is_some(), "architecture metadata present");
        assert!(!g.tensors().is_empty(), "tensor list non-empty");

        // tensor_data must round-trip for at least one tensor and the slice
        // length must match the precomputed byte_size.
        let first = &g.tensors()[0];
        let bytes = g.tensor_data(&first.name).expect("data slice");
        assert_eq!(bytes.len(), first.byte_size);
    }
}
