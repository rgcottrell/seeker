use std::path::{Path, PathBuf};

use clap::Args;
use serde_json::{json, Value as JsonValue};

use crate::commands::download::{resolve_hf, HfResolveArgs};
use crate::gguf::{GgufFile, MetadataValue, TensorInfo};

#[derive(Args)]
pub struct InspectArgs {
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]". (short: -hf, -hfr)
    #[arg(long = "hf-repo", required_unless_present = "model", conflicts_with = "model")]
    hf_repo: Option<String>,

    /// Specific file to inspect within the repo. (short: -hff)
    #[arg(long = "hf-file", requires = "hf_repo", conflicts_with = "model")]
    hf_file: Option<String>,

    /// HF auth token (defaults to HF_TOKEN env / ~/.cache/huggingface/token). (short: -hft)
    #[arg(long = "hf-token", requires = "hf_repo", conflicts_with = "model")]
    hf_token: Option<String>,

    /// Resolve files from the local cache only; never hit the network.
    #[arg(long, requires = "hf_repo", conflicts_with = "model")]
    offline: bool,

    /// Path to a local .gguf model file.
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,

    /// Emit pretty-printed JSON. Large arrays are summarized as `{preview, totalCount}`.
    #[arg(long)]
    json: bool,
}

const ARRAY_HEAD: usize = 8;
const STRING_MAX: usize = 100;

pub async fn run(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let path = match (args.hf_repo, args.model) {
        (Some(repo), None) => {
            resolve_hf(
                &HfResolveArgs {
                    repo,
                    file: args.hf_file,
                    token: args.hf_token,
                    offline: args.offline,
                },
                false,
            )
            .await?
            .main
        }
        (None, Some(model)) => model,
        _ => unreachable!("clap group invariant"),
    };

    let g = GgufFile::open(&path)?;
    if args.json {
        print_json(&path, &g)?;
    } else {
        print_header(&path, &g);
        print_metadata(&g);
        print_tensors(&g);
    }
    Ok(())
}

fn print_json(path: &Path, g: &GgufFile) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let metadata: serde_json::Map<String, JsonValue> = g
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), value_to_json(v)))
        .collect();

    let tensors: Vec<JsonValue> = g.tensors().iter().map(tensor_to_json).collect();

    let root = json!({
        "file": path.display().to_string(),
        "version": g.version(),
        "alignment": g.alignment(),
        "dataOffset": g.data_offset(),
        "fileSize": g.file_size(),
        "metadata": JsonValue::Object(metadata),
        "tensors": tensors,
    });

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, &root)?;
    handle.write_all(b"\n")?;
    Ok(())
}

fn value_to_json(v: &MetadataValue) -> JsonValue {
    match v {
        MetadataValue::U8(n) => JsonValue::from(*n),
        MetadataValue::I8(n) => JsonValue::from(*n),
        MetadataValue::U16(n) => JsonValue::from(*n),
        MetadataValue::I16(n) => JsonValue::from(*n),
        MetadataValue::U32(n) => JsonValue::from(*n),
        MetadataValue::I32(n) => JsonValue::from(*n),
        MetadataValue::U64(n) => JsonValue::from(*n),
        MetadataValue::I64(n) => JsonValue::from(*n),
        MetadataValue::F32(x) => JsonValue::from(f64::from(*x)),
        MetadataValue::F64(x) => JsonValue::from(*x),
        MetadataValue::Bool(b) => JsonValue::from(*b),
        MetadataValue::String(s) => JsonValue::from(s.as_str()),
        MetadataValue::Array(a) => array_to_json(a),
    }
}

/// JSON encoding for metadata arrays. Short arrays stay as plain JSON arrays;
/// arrays longer than `ARRAY_HEAD` become a summary object `{ preview, totalCount }`
/// so that 49k-entry tokenizer arrays don't bury the rest of the output.
fn array_to_json(a: &[MetadataValue]) -> JsonValue {
    if a.len() <= ARRAY_HEAD {
        JsonValue::Array(a.iter().map(value_to_json).collect())
    } else {
        let preview: Vec<JsonValue> = a.iter().take(ARRAY_HEAD).map(value_to_json).collect();
        json!({
            "preview": preview,
            "totalCount": a.len(),
        })
    }
}

fn tensor_to_json(t: &TensorInfo) -> JsonValue {
    json!({
        "byteSize": t.byte_size,
        "dimensions": t.dims,
        "name": t.name,
        "offset": t.offset,
        "type": format!("{:?}", t.ggml_type),
        "typeId": t.ggml_type as u32,
    })
}

fn print_header(path: &Path, g: &GgufFile) {
    println!("file:      {}", path.display());
    println!("version:   {}", g.version());
    println!("alignment: {}", g.alignment());
    println!("metadata:  {} entries", g.metadata().len());
    println!("tensors:   {}", g.tensors().len());
    println!(
        "data offset: {} (file size: {})",
        g.data_offset(),
        g.file_size()
    );
}

fn print_metadata(g: &GgufFile) {
    println!();
    println!("Metadata");
    println!("--------");
    for (k, v) in g.metadata() {
        println!("  {k} = {}", fmt_value(v));
    }
}

fn print_tensors(g: &GgufFile) {
    let tensors = g.tensors();

    let types: Vec<String> = tensors
        .iter()
        .map(|t| format!("{:?}", t.ggml_type))
        .collect();
    let dims: Vec<String> = tensors.iter().map(|t| fmt_dims(&t.dims)).collect();

    let idx_w = tensors.len().saturating_sub(1).to_string().len().max(3);
    let name_w = tensors.iter().map(|t| t.name.len()).max().unwrap_or(0).max(4);
    let type_w = types.iter().map(|s| s.len()).max().unwrap_or(0).max(4);
    let dims_w = dims.iter().map(|s| s.len()).max().unwrap_or(0).max(4);
    let offset_w = tensors
        .iter()
        .map(|t| t.offset.to_string().len())
        .max()
        .unwrap_or(0)
        .max(6);
    let size_w = tensors
        .iter()
        .map(|t| t.byte_size.to_string().len())
        .max()
        .unwrap_or(0)
        .max(4);

    println!();
    println!("Tensors");
    println!("-------");
    println!(
        "  {:<iw$}  {:<nw$}  {:<tw$}  {:<dw$}  {:>ow$}  {:>sw$}",
        "Idx",
        "Name",
        "Type",
        "Dims",
        "Offset",
        "Size",
        iw = idx_w,
        nw = name_w,
        tw = type_w,
        dw = dims_w,
        ow = offset_w,
        sw = size_w,
    );
    for (i, t) in tensors.iter().enumerate() {
        println!(
            "  {:<iw$}  {:<nw$}  {:<tw$}  {:<dw$}  {:>ow$}  {:>sw$}",
            i,
            t.name,
            types[i],
            dims[i],
            t.offset,
            t.byte_size,
            iw = idx_w,
            nw = name_w,
            tw = type_w,
            dw = dims_w,
            ow = offset_w,
            sw = size_w,
        );
    }
}

fn fmt_dims(dims: &[u64]) -> String {
    let parts: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn fmt_value(v: &MetadataValue) -> String {
    match v {
        MetadataValue::U8(n) => format!("{n}"),
        MetadataValue::I8(n) => format!("{n}"),
        MetadataValue::U16(n) => format!("{n}"),
        MetadataValue::I16(n) => format!("{n}"),
        MetadataValue::U32(n) => format!("{n}"),
        MetadataValue::I32(n) => format!("{n}"),
        MetadataValue::U64(n) => format!("{n}"),
        MetadataValue::I64(n) => format!("{n}"),
        MetadataValue::F32(x) => format!("{x:?}"),
        MetadataValue::F64(x) => format!("{x:?}"),
        MetadataValue::Bool(b) => format!("{b}"),
        MetadataValue::String(s) => fmt_string(s),
        MetadataValue::Array(a) => fmt_array(a),
    }
}

fn fmt_string(s: &str) -> String {
    let char_count = s.chars().count();
    if char_count <= STRING_MAX {
        format!("{s:?}")
    } else {
        let prefix: String = s.chars().take(STRING_MAX).collect();
        let repr = format!("{prefix:?}");
        let trimmed = repr.strip_suffix('"').unwrap_or(&repr);
        format!("{trimmed}…\" ({char_count} chars)")
    }
}

fn fmt_array(a: &[MetadataValue]) -> String {
    let head: Vec<String> = a.iter().take(ARRAY_HEAD).map(fmt_value).collect();
    let joined = head.join(", ");
    if a.len() > ARRAY_HEAD {
        let remaining = a.len() - ARRAY_HEAD;
        format!("[{joined}, … {remaining} more]")
    } else {
        format!("[{joined}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_short_string() {
        assert_eq!(fmt_value(&MetadataValue::String("llama".into())), "\"llama\"");
    }

    #[test]
    fn fmt_long_string_truncates_and_reports_char_count() {
        let s = "x".repeat(150);
        let out = fmt_value(&MetadataValue::String(s.clone()));
        assert!(out.starts_with('"'));
        assert!(out.contains("…\" (150 chars)"));
        // Prefix shown should be exactly STRING_MAX chars of the source.
        assert!(out.contains(&"x".repeat(STRING_MAX)));
    }

    #[test]
    fn fmt_short_array() {
        let arr = MetadataValue::Array(vec![
            MetadataValue::U32(1),
            MetadataValue::U32(2),
            MetadataValue::U32(3),
        ]);
        assert_eq!(fmt_value(&arr), "[1, 2, 3]");
    }

    #[test]
    fn fmt_long_array_truncates_to_eight() {
        let arr =
            MetadataValue::Array((0u32..20).map(MetadataValue::U32).collect::<Vec<_>>());
        let out = fmt_value(&arr);
        assert_eq!(out, "[0, 1, 2, 3, 4, 5, 6, 7, … 12 more]");
    }

    #[test]
    fn fmt_array_of_strings() {
        let arr = MetadataValue::Array(vec![
            MetadataValue::String("a".into()),
            MetadataValue::String("b".into()),
        ]);
        assert_eq!(fmt_value(&arr), "[\"a\", \"b\"]");
    }

    #[test]
    fn fmt_floats_use_debug_repr() {
        assert_eq!(fmt_value(&MetadataValue::F32(500000.0)), "500000.0");
        assert_eq!(fmt_value(&MetadataValue::F32(1.0)), "1.0");
    }

    #[test]
    fn fmt_dims_joins_with_commas() {
        assert_eq!(fmt_dims(&[2048, 128256]), "[2048,128256]");
        assert_eq!(fmt_dims(&[32]), "[32]");
    }

    #[test]
    fn json_scalar_values_map_to_native_types() {
        assert_eq!(value_to_json(&MetadataValue::U32(42)), json!(42));
        assert_eq!(value_to_json(&MetadataValue::F32(500000.0)), json!(500000.0));
        assert_eq!(value_to_json(&MetadataValue::Bool(true)), json!(true));
        assert_eq!(
            value_to_json(&MetadataValue::String("llama".into())),
            json!("llama")
        );
    }

    #[test]
    fn json_short_array_stays_plain() {
        let arr = MetadataValue::Array(vec![
            MetadataValue::String("en".into()),
            MetadataValue::String("de".into()),
        ]);
        assert_eq!(value_to_json(&arr), json!(["en", "de"]));
    }

    #[test]
    fn json_long_array_summarized_with_preview_and_count() {
        let arr = MetadataValue::Array((0u32..50).map(MetadataValue::U32).collect());
        let v = value_to_json(&arr);
        let obj = v.as_object().expect("summary object");
        assert_eq!(obj.get("totalCount"), Some(&json!(50)));
        let preview = obj.get("preview").and_then(|p| p.as_array()).expect("preview array");
        assert_eq!(preview.len(), ARRAY_HEAD);
        assert_eq!(preview[0], json!(0));
        assert_eq!(preview[ARRAY_HEAD - 1], json!(ARRAY_HEAD as u32 - 1));
    }

    #[test]
    fn json_exactly_array_head_stays_plain() {
        let arr = MetadataValue::Array((0u32..ARRAY_HEAD as u32).map(MetadataValue::U32).collect());
        let v = value_to_json(&arr);
        let elems = v.as_array().expect("plain array at threshold");
        assert_eq!(elems.len(), ARRAY_HEAD);
    }

    #[test]
    fn json_long_strings_are_untruncated() {
        let s = "x".repeat(500);
        let v = value_to_json(&MetadataValue::String(s.clone()));
        assert_eq!(v, JsonValue::from(s));
    }

    #[test]
    fn json_tensor_has_expected_shape() {
        use crate::gguf::GgmlType;
        let t = TensorInfo {
            name: "blk.15.ffn_norm.weight".into(),
            dims: vec![2048],
            ggml_type: GgmlType::F32,
            offset: 790417536,
            byte_size: 8192,
        };
        assert_eq!(
            tensor_to_json(&t),
            json!({
                "byteSize": 8192,
                "dimensions": [2048],
                "name": "blk.15.ffn_norm.weight",
                "offset": 790417536,
                "type": "F32",
                "typeId": 0,
            })
        );

        let t2 = TensorInfo {
            name: "blk.15.ffn_up.weight".into(),
            dims: vec![2048, 8192],
            ggml_type: GgmlType::Q4_K,
            offset: 790425728,
            byte_size: 9437184,
        };
        assert_eq!(
            tensor_to_json(&t2),
            json!({
                "byteSize": 9437184,
                "dimensions": [2048, 8192],
                "name": "blk.15.ffn_up.weight",
                "offset": 790425728,
                "type": "Q4_K",
                "typeId": 12,
            })
        );
    }
}
