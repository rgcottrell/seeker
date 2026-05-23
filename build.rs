use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One type-tuple specialization of a shader. Declared inline in each .slang
/// file as a comment block:
///
/// ```text
/// //@variants
/// //  f32: A_TYPE=float D_TYPE=float FLOAT_TYPE=float
/// //  f16: A_TYPE=float16_t D_TYPE=float16_t FLOAT_TYPE=float
/// //@end-variants
/// ```
struct Variant {
    /// Variant name (e.g. "f32", "f16"). Empty means "no @variants block in
    /// source" — compile once with no extra -D flags, emit `<STEM>_SPV`.
    name: String,
    defines: Vec<(String, String)>,
}

fn parse_variants(source: &str, path: &Path) -> Vec<Variant> {
    let mut variants = Vec::new();
    let mut in_block = false;
    let mut found_end = false;
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//@variants") {
            if in_block {
                panic!("nested //@variants in {}", path.display());
            }
            in_block = true;
            continue;
        }
        if trimmed.starts_with("//@end-variants") {
            if !in_block {
                panic!("//@end-variants without //@variants in {}", path.display());
            }
            found_end = true;
            in_block = false;
            break;
        }
        if !in_block {
            continue;
        }
        let body = match trimmed.strip_prefix("//") {
            Some(b) => b.trim(),
            None => panic!(
                "non-comment line inside //@variants block in {}",
                path.display()
            ),
        };
        if body.is_empty() {
            continue;
        }
        let (name, rest) = body
            .split_once(':')
            .unwrap_or_else(|| panic!("variant line missing ':' in {}: {body}", path.display()));
        let name = name.trim();
        if name.is_empty() {
            panic!("variant with empty name in {}", path.display());
        }
        let defines: Vec<(String, String)> = rest
            .split_whitespace()
            .map(|kv| {
                let (k, v) = kv.split_once('=').unwrap_or_else(|| {
                    panic!("variant define missing '=' in {}: {kv}", path.display())
                });
                (k.trim().to_string(), v.trim().to_string())
            })
            .collect();
        variants.push(Variant {
            name: name.to_string(),
            defines,
        });
    }
    if in_block && !found_end {
        panic!("unterminated //@variants block in {}", path.display());
    }
    if variants.is_empty() {
        variants.push(Variant {
            name: String::new(),
            defines: Vec::new(),
        });
    }
    variants
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let shaders_dir = Path::new(&manifest_dir).join("shaders");
    let compute_dir = shaders_dir.join("compute");
    let include_dir = shaders_dir.join("include");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", compute_dir.display());
    println!("cargo:rerun-if-changed={}", include_dir.display());

    // `(const_stem, spv_stem, size)` — const_stem becomes `<UPPER>_SPV` in
    // shaders.rs; spv_stem is the filename in OUT_DIR.
    let mut shaders: Vec<(String, String, u64)> = Vec::new();

    let entries = fs::read_dir(&compute_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", compute_dir.display(), e));

    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("slang") {
            continue;
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("non-utf8 filename: {}", path.display()))
            .to_owned();

        println!("cargo:rerun-if-changed={}", path.display());

        let source = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let variants = parse_variants(&source, &path);

        for variant in &variants {
            let (spv_stem, const_stem) = if variant.name.is_empty() {
                (stem.clone(), stem.clone())
            } else {
                (
                    format!("{stem}.{}", variant.name),
                    format!("{stem}_{}", variant.name),
                )
            };
            let spv_path = out_dir.join(format!("{spv_stem}.spv"));

            let mut cmd = Command::new("slangc");
            cmd.arg(&path).arg("-I").arg(&include_dir);
            for (k, v) in &variant.defines {
                cmd.arg(format!("-D{k}={v}"));
            }
            cmd.args([
                "-target", "spirv",
                "-profile", "spirv_1_6",
                "-capability", "vk_mem_model",
                "-capability", "sm_6_4",
                "-capability", "cooperative_matrix",
                "-capability", "cooperative_matrix_2",
                "-O3",
                "-stage", "compute",
                "-entry", "main",
                "-warnings-as-errors", "all",
                "-restrictive-capability-check",
                "-emit-spirv-directly",
                "-fvk-use-entrypoint-name",
                "-o",
            ])
            .arg(&spv_path);

            let status = cmd.status().unwrap_or_else(|e| {
                panic!("failed to spawn slangc for {}: {e}", path.display())
            });

            if !status.success() {
                panic!(
                    "slangc failed for {} variant {:?} (exit {:?})",
                    path.display(),
                    variant.name,
                    status.code()
                );
            }

            let size = fs::metadata(&spv_path)
                .unwrap_or_else(|e| panic!("failed to stat {}: {e}", spv_path.display()))
                .len();

            if size == 0 || size % 4 != 0 {
                panic!(
                    "compiled SPIR-V {} has unexpected size {} (must be non-zero, multiple of 4)",
                    spv_path.display(),
                    size
                );
            }

            shaders.push((const_stem, spv_stem, size));
        }
    }

    shaders.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out = String::new();
    out.push_str("// Auto-generated by build.rs. Do not edit.\n");
    out.push_str("use core::ops::Deref;\n\n");
    out.push_str("#[repr(C, align(4))]\n");
    out.push_str("pub struct Shader<const N: usize>(pub [u8; N]);\n\n");
    out.push_str("impl<const N: usize> Shader<N> {\n");
    out.push_str("    pub const fn as_bytes(&self) -> &[u8] { &self.0 }\n");
    out.push_str("}\n");
    out.push_str("impl<const N: usize> Deref for Shader<N> {\n");
    out.push_str("    type Target = [u8];\n");
    out.push_str("    fn deref(&self) -> &[u8] { &self.0 }\n");
    out.push_str("}\n");
    out.push_str("impl<const N: usize> AsRef<[u8]> for Shader<N> {\n");
    out.push_str("    fn as_ref(&self) -> &[u8] { &self.0 }\n");
    out.push_str("}\n\n");

    for (const_stem, spv_stem, size) in &shaders {
        let upper = const_stem.to_uppercase();
        writeln!(
            out,
            "pub static {upper}_SPV: Shader<{size}> = \
             Shader(*include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{spv_stem}.spv\")));"
        )
        .unwrap();
    }

    fs::write(out_dir.join("shaders.rs"), out).expect("write shaders.rs");
}
