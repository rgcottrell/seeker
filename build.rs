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

/// Point git at the source-controlled hooks in `.githooks/` (the Rust analog of
/// an npm `postinstall` hook install — build scripts always run once on a fresh
/// clone before any cache exists). Best-effort and idempotent: skips outside a
/// git work tree (vendored/release builds with no `.git`), skips if git is
/// missing, and never clobbers an already-set `core.hooksPath`. Bypass any
/// single push with `git push --no-verify`.
fn ensure_git_hooks_path(repo_dir: &str) {
    if !Path::new(repo_dir).join(".git").exists() {
        return;
    }
    // Only configure when unset — respect a deliberate override.
    if let Ok(out) = Command::new("git")
        .args(["-C", repo_dir, "config", "--get", "core.hooksPath"])
        .output()
        && out.status.success()
        && !out.stdout.is_empty()
    {
        return;
    }
    if Command::new("git")
        .args(["-C", repo_dir, "config", "core.hooksPath", ".githooks"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        println!("cargo:warning=enabled git pre-push gate (core.hooksPath=.githooks)");
    }
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    ensure_git_hooks_path(&manifest_dir);
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
            #[rustfmt::skip]
            cmd.args([
                "-target", "spirv",
                "-profile", "spirv_1_6",
                "-capability", "vk_mem_model",
                "-capability", "sm_6_4",
                "-capability", "cooperative_matrix",
                "-capability", "cooperative_matrix_2",
                "-capability", "spvGroupNonUniform",
                "-O3",
                "-stage", "compute",
                "-entry", "main",
                "-warnings-as-errors", "all",
                "-restrictive-capability-check",
                "-emit-spirv-directly",
                "-fvk-use-entrypoint-name",
                // Tight (scalar) buffer layout — required for K-quant
                // structs (e.g. `block_q6_K` is 210 bytes, not 224 as
                // std430 would round to). The device side enables
                // `scalarBlockLayout` via Vulkan12Features.
                "-fvk-use-scalar-layout",
                "-o",
            ])
            .arg(&spv_path);

            let status = cmd
                .status()
                .unwrap_or_else(|e| panic!("failed to spawn slangc for {}: {e}", path.display()));

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

    generate_public_assets(&manifest_dir, &out_dir);
}

/// Walk `<manifest_dir>/public` and generate `<out_dir>/public_assets.rs`: an
/// `Asset` struct plus a `lookup(path) -> Option<Asset>` over forward-slash
/// relative paths, each arm an `include_bytes!` of the source-tree file so the
/// whole tree is baked into the binary. An absent or empty `public/` yields a
/// `lookup` that always returns `None`.
fn generate_public_assets(manifest_dir: &str, out_dir: &Path) {
    let public_dir = Path::new(manifest_dir).join("public");

    // Re-run when the directory's membership changes (a file added/removed).
    // Cargo tolerates a non-existent path here, so a later `mkdir public` still
    // triggers regeneration. Content *edits* are tracked separately by rustc via
    // the generated `include_bytes!`, so they need no build.rs rerun.
    println!("cargo:rerun-if-changed={}", public_dir.display());

    let mut rel_paths: Vec<String> = Vec::new();
    if public_dir.is_dir() {
        collect_assets(&public_dir, &public_dir, &mut rel_paths);
    }
    // Sort for deterministic, byte-reproducible output (matches the shader gen).
    rel_paths.sort();

    let mut out = String::new();
    out.push_str("// Auto-generated by build.rs. Do not edit.\n\n");
    out.push_str("#[derive(Clone, Copy)]\n");
    out.push_str("pub struct Asset {\n");
    out.push_str("    pub bytes: &'static [u8],\n");
    out.push_str("    pub content_type: &'static str,\n");
    out.push_str("}\n\n");
    out.push_str("/// Look up an embedded asset by its forward-slash relative path\n");
    out.push_str("/// (no leading slash), e.g. \"index.html\" or \"assets/app.js\".\n");
    out.push_str("pub fn lookup(path: &str) -> Option<Asset> {\n");
    out.push_str("    let asset = match path {\n");
    for rel in &rel_paths {
        let ct = content_type_for(rel);
        // `{rel:?}` / `{ct:?}` Debug-format the `&str` into correctly-escaped,
        // quoted Rust string literals.
        writeln!(
            out,
            "        {rel:?} => Asset {{ bytes: include_bytes!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/public/\", {rel:?})), content_type: {ct:?} }},"
        )
        .unwrap();
    }
    out.push_str("        _ => return None,\n");
    out.push_str("    };\n");
    out.push_str("    Some(asset)\n");
    out.push_str("}\n");

    fs::write(out_dir.join("public_assets.rs"), out).expect("write public_assets.rs");
}

/// Recursively collect regular files under `dir`, keyed by their path relative
/// to `root` and normalized to forward slashes. Skips dotfiles / dot-dirs
/// (`.gitkeep`, `.DS_Store`, `.git`, …).
fn collect_assets(root: &Path, dir: &Path, out: &mut Vec<String>) {
    // Watch this directory so adding/removing a child re-runs build.rs.
    println!("cargo:rerun-if-changed={}", dir.display());

    let entries =
        fs::read_dir(dir).unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("non-utf8 filename: {}", path.display()));
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_assets(root, &path, out);
            continue;
        }
        if !path.is_file() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let rel = path
            .strip_prefix(root)
            .expect("descendant of root")
            .to_str()
            .unwrap_or_else(|| panic!("non-utf8 path: {}", path.display()))
            .replace(std::path::MAIN_SEPARATOR, "/");
        // A quote/backslash in a filename would break the generated string
        // literal and the POSIX include path; reject such pathological names.
        assert!(
            !rel.contains('"') && !rel.contains('\\'),
            "asset path contains illegal char: {rel}"
        );
        out.push(rel);
    }
}

/// Map a file's extension to a static content-type string at build time.
fn content_type_for(rel: &str) -> &'static str {
    let ext = rel
        .rsplit('.')
        .next()
        .filter(|e| !e.contains('/')) // dotless filename -> no real extension
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match ext.as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" | "map" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        "wasm" => "application/wasm",
        "txt" => "text/plain; charset=utf-8",
        "xml" => "application/xml; charset=utf-8",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}
