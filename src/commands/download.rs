use std::error::Error;
use std::path::{Path, PathBuf};

use clap::Args;
use hf_hub::api::tokio::{ApiBuilder, ApiRepo};
use hf_hub::{Cache, CacheRepo, Repo, RepoType};
use tracing::{debug, info};

#[derive(Args)]
pub struct DownloadArgs {
    /// HF repo id, optionally with a quant suffix: "ORG/NAME[:QUANT]". (short: -hf, -hfr)
    #[arg(long = "hf-repo")]
    hf_repo: String,
    /// Specific file to download. Overrides quant-based selection. (short: -hff)
    #[arg(long = "hf-file")]
    hf_file: Option<String>,
    /// HF auth token (defaults to HF_TOKEN env / ~/.cache/huggingface/token). (short: -hft)
    #[arg(long = "hf-token")]
    hf_token: Option<String>,
    /// Resolve files from the local cache only; never hit the network.
    #[arg(long)]
    offline: bool,
    /// Do not auto-fetch the matching mmproj-*.gguf sidecar.
    #[arg(long = "no-mmproj")]
    no_mmproj: bool,
}

/// Split a string into lowercase tokens on `.` and `-`. Empty tokens dropped.
/// Used to match `:QUANT` arguments against filenames at token boundaries —
/// so `:Q4_K_M` matches `model-Q4_K_M.gguf` but not `model-Q4_K_M_XL.gguf`,
/// and `:UD-Q4_K_XL` (two tokens) matches `model.UD-Q4_K_XL.gguf`.
fn split_tokens(s: &str) -> Vec<String> {
    s.to_ascii_lowercase()
        .split(|c: char| c == '.' || c == '-')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

pub(crate) fn filename_contains_quant(filename: &str, quant: &str) -> bool {
    let file_tokens = split_tokens(filename);
    let quant_tokens = split_tokens(quant);
    if quant_tokens.is_empty() {
        return false;
    }
    file_tokens
        .windows(quant_tokens.len())
        .any(|w| w == quant_tokens.as_slice())
}

fn is_main_gguf(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    lower.ends_with(".gguf") && !lower.contains("mmproj")
}

pub(crate) fn is_mmproj_gguf(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    lower.ends_with(".gguf") && lower.contains("mmproj")
}

/// Extract a quant-like tag (e.g. `Q4_K_M`, `UD-Q4_K_XL`, `Q8_0`, `F16`) from a
/// filename, returning the first matching token. Used to prefer an mmproj
/// sidecar of the same quant as the local main model.
pub(crate) fn quant_tag(filename: &str) -> Option<String> {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    stem.split(['.', '-', '_']).find_map(|tok| {
        let up = tok.to_ascii_uppercase();
        let is_quant = (up.starts_with('Q') && up.chars().any(|c| c.is_ascii_digit()))
            || up == "F16"
            || up == "F32"
            || up == "BF16";
        is_quant.then(|| tok.to_string())
    })
}

/// Directories to scan for a sidecar mmproj, given the main model path. Always
/// the model's own directory; and when the model lives inside an HF cache
/// snapshot (`.../snapshots/<commit>/`), every *sibling* snapshot of the same
/// repo too. HF stores each commit as its own snapshot tree of symlinks into a
/// shared `blobs/` store, so an mmproj added in a different commit is symlinked
/// only under that commit's snapshot even though its blob is right there — scan
/// the siblings so we still find it.
fn sidecar_search_dirs(main: &Path) -> Vec<PathBuf> {
    let Some(dir) = main.parent() else {
        return Vec::new();
    };
    let mut dirs = vec![dir.to_path_buf()];
    if dir.parent().and_then(|p| p.file_name()) == Some(std::ffi::OsStr::new("snapshots")) {
        let snapshots = dir.parent().expect("checked file_name above");
        if let Ok(rd) = std::fs::read_dir(snapshots) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() && p != dir {
                    dirs.push(p);
                }
            }
        }
    }
    dirs
}

/// Scan the main model file's directory (and, for an HF cache snapshot, sibling
/// snapshot directories) for a sidecar mmproj GGUF, preferring one whose quant
/// suffix matches the main file, else the first found. Used for local `-m PATH`
/// runs (the HF path uses [`pick_mmproj`]). Returns `None` (no error) when no
/// mmproj sidecar is found.
pub(crate) fn find_sidecar_mmproj(main: &Path) -> Option<PathBuf> {
    let main_name = main.file_name()?.to_str()?;

    let mut candidates: Vec<PathBuf> = Vec::new();
    for dir in sidecar_search_dirs(main) {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            match path.file_name().and_then(|n| n.to_str()) {
                Some(name) if is_mmproj_gguf(name) => candidates.push(path.clone()),
                _ => {}
            }
        }
    }
    if candidates.is_empty() {
        return None;
    }

    // Prefer a candidate sharing the main file's quant tag.
    if let Some(quant) = quant_tag(main_name) {
        if let Some(p) = candidates.iter().find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| filename_contains_quant(n, &quant))
        }) {
            return Some(p.clone());
        }
    }
    candidates.into_iter().next()
}

fn pick_gguf_matching(files: &[String], quant: &str) -> Option<String> {
    files
        .iter()
        .find(|f| is_main_gguf(f) && filename_contains_quant(f, quant))
        .cloned()
}

fn pick_any_main_gguf(files: &[String]) -> Option<String> {
    files.iter().find(|f| is_main_gguf(f)).cloned()
}

fn pick_mmproj(files: &[String], main_quant: Option<&str>) -> Option<String> {
    let candidates: Vec<&String> = files.iter().filter(|f| is_mmproj_gguf(f)).collect();

    if let Some(q) = main_quant {
        if let Some(m) = candidates.iter().find(|f| filename_contains_quant(f, q)) {
            return Some((*m).clone());
        }
    }
    if let Some(m) = candidates
        .iter()
        .find(|f| filename_contains_quant(f, "f16") || filename_contains_quant(f, "fp16"))
    {
        return Some((*m).clone());
    }
    candidates.first().map(|f| (*f).clone())
}

async fn list_files_online(repo: &ApiRepo) -> Result<Vec<String>, Box<dyn Error>> {
    let info = repo.info().await?;
    Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
}

/// All cached `snapshots/<commit>/` directories for `repo_id`, newest commit
/// first (by directory mtime). Empty when the repo isn't cached.
fn snapshot_dirs(cache: &Cache, repo_id: &str) -> Vec<PathBuf> {
    let folder = cache
        .path()
        .join(Repo::new(repo_id.to_string(), RepoType::Model).folder_name());
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(folder.join("snapshots")) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(_) => Vec::new(),
    };
    // Prefer the most recently fetched commit when a file appears in several.
    dirs.sort_by_key(|p| std::cmp::Reverse(std::fs::metadata(p).and_then(|m| m.modified()).ok()));
    dirs
}

/// Locate `filename` in any cached snapshot of `repo_id` (newest commit first).
/// Used as an offline fallback: `CacheRepo::get` only consults the `refs/main`
/// snapshot, but a file fetched under another commit is physically present
/// (blobs are shared) and just as usable.
fn find_cached_in_snapshots(cache: &Cache, repo_id: &str, filename: &str) -> Option<PathBuf> {
    snapshot_dirs(cache, repo_id)
        .into_iter()
        .map(|d| d.join(filename))
        .find(|p| p.exists())
}

/// List the repo's files from the local cache. Unions filenames across *all*
/// cached snapshot directories rather than just `refs/main`: HF stores each
/// commit as its own `snapshots/<commit>/` tree of symlinks into a shared
/// `blobs/` store, so a file fetched under one commit (e.g. an mmproj sidecar
/// downloaded earlier, or under a now-superseded commit) is physically present
/// and usable even when `refs/main` points at a commit whose snapshot doesn't
/// symlink it. Listing a single snapshot is how an on-disk mmproj goes missing.
fn list_files_offline(cache: &Cache, repo_id: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let dirs = snapshot_dirs(cache, repo_id);
    if dirs.is_empty() {
        let folder = cache
            .path()
            .join(Repo::new(repo_id.to_string(), RepoType::Model).folder_name());
        return Err(format!(
            "offline cache missing for {repo_id}: no snapshots under {}",
            folder.join("snapshots").display()
        )
        .into());
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for dir in &dirs {
        let Ok(rd) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in rd.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if seen.insert(name.to_string()) {
                    out.push(name.to_string());
                }
            }
        }
    }
    Ok(out)
}

async fn fetch_or_resolve(
    cache: &Cache,
    repo_id: &str,
    repo: &ApiRepo,
    cache_repo: &CacheRepo,
    filename: &str,
    offline: bool,
) -> Result<PathBuf, Box<dyn Error>> {
    if offline {
        // `CacheRepo::get` only resolves within the `refs/main` snapshot; fall
        // back to any other cached snapshot that has the file (blobs shared).
        cache_repo
            .get(filename)
            .or_else(|| find_cached_in_snapshots(cache, repo_id, filename))
            .ok_or_else(|| format!("file not in offline cache: {filename}").into())
    } else {
        Ok(repo.get(filename).await?)
    }
}

/// Inputs to the shared HF-resolution helper. Both `download` and `inspect`
/// build one of these from their respective `Args` structs.
pub(crate) struct HfResolveArgs {
    pub repo: String,
    pub file: Option<String>,
    pub token: Option<String>,
    pub offline: bool,
}

/// Result of resolving an HF repo to local paths. `mmproj` is `None` when the
/// caller passed `want_mmproj = false` or when the repo has no sidecar.
pub(crate) struct Resolved {
    pub main: PathBuf,
    pub mmproj: Option<PathBuf>,
}

/// Shared HF resolver used by `download` and `inspect`. Splits `repo` on `:`,
/// lists the repo (online or offline), picks the main `.gguf` by quant rules,
/// and fetches it. When `want_mmproj` is true and the repo ships a matching
/// mmproj sidecar, fetches that too.
pub(crate) async fn resolve_hf(
    args: &HfResolveArgs,
    want_mmproj: bool,
) -> Result<Resolved, Box<dyn Error>> {
    let (repo_id, quant_hint) = match args.repo.split_once(':') {
        Some((r, q)) => (r.to_string(), Some(q.to_string())),
        None => (args.repo.clone(), None),
    };
    if !repo_id.contains('/') {
        return Err("--hf-repo must be in the form ORG/NAME[:QUANT]".into());
    }

    let cache = Cache::from_env();
    let cache_repo = cache.model(repo_id.clone());

    let mut builder = ApiBuilder::from_env();
    if let Some(t) = args.token.clone() {
        builder = builder.with_token(Some(t));
    }
    let api = builder.build()?;
    let api_repo = api.model(repo_id.clone());

    debug!(repo = %repo_id, quant = ?quant_hint, offline = args.offline, "resolving file list");
    let files = if args.offline {
        list_files_offline(&cache, &repo_id)?
    } else {
        list_files_online(&api_repo).await?
    };
    debug!(repo = %repo_id, count = files.len(), "file list resolved");

    // Resolve main file. `effective_quant` is what we pass to the mmproj picker:
    // the user's :QUANT if any, "Q4_K_M" if the default kicked in, else None.
    let (main_file, effective_quant): (String, Option<String>) = match args.file.as_deref() {
        Some(f) => (f.to_string(), quant_hint.clone()),
        None => match quant_hint.as_deref() {
            Some(q) => {
                let f = pick_gguf_matching(&files, q)
                    .ok_or_else(|| format!("no .gguf file matching quant {q} in {repo_id}"))?;
                (f, Some(q.to_string()))
            }
            None => {
                if let Some(f) = pick_gguf_matching(&files, "Q4_K_M") {
                    (f, Some("Q4_K_M".to_string()))
                } else {
                    let f = pick_any_main_gguf(&files)
                        .ok_or_else(|| format!("no .gguf file found in {repo_id}"))?;
                    (f, None)
                }
            }
        },
    };

    info!(file = %main_file, quant = ?effective_quant, "selected main file");
    let main = fetch_or_resolve(&cache, &repo_id, &api_repo, &cache_repo, &main_file, args.offline).await?;

    let mmproj = if want_mmproj {
        if let Some(mmproj) = pick_mmproj(&files, effective_quant.as_deref()) {
            info!(file = %mmproj, "selected mmproj sidecar");
            Some(fetch_or_resolve(&cache, &repo_id, &api_repo, &cache_repo, &mmproj, args.offline).await?)
        } else {
            debug!("no mmproj sidecar found in repo");
            None
        }
    } else {
        None
    };

    Ok(Resolved { main, mmproj })
}

pub async fn run(args: DownloadArgs) -> Result<(), Box<dyn Error>> {
    let resolved = resolve_hf(
        &HfResolveArgs {
            repo: args.hf_repo,
            file: args.hf_file,
            token: args.hf_token,
            offline: args.offline,
        },
        !args.no_mmproj,
    )
    .await?;
    println!("{}", resolved.main.display());
    if let Some(mmproj) = resolved.mmproj {
        println!("{}", mmproj.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn quant_match_basic() {
        assert!(filename_contains_quant("llama-2-7b.Q4_K_M.gguf", "Q4_K_M"));
        assert!(filename_contains_quant("MODEL-Q8_0.gguf", "q8_0"));
        assert!(filename_contains_quant("mmproj-f16.gguf", "f16"));
        assert!(!filename_contains_quant("README.md", "Q4_K_M"));
    }

    #[test]
    fn quant_match_handles_new_multi_token_quants() {
        assert!(filename_contains_quant("model.UD-Q4_K_XL.gguf", "UD-Q4_K_XL"));
        assert!(filename_contains_quant("model-UD-Q4_K_XL.gguf", "ud-q4_k_xl"));
    }

    #[test]
    fn quant_match_respects_token_boundaries() {
        // Q4_K_M_XL is a single token; :Q4_K_M (one token) must not match it.
        assert!(!filename_contains_quant("model.Q4_K_M_XL.gguf", "Q4_K_M"));
    }

    #[test]
    fn pick_gguf_matching_finds_by_quant() {
        let files = s(&["llama.Q4_K_M.gguf", "llama.Q8_0.gguf", "mmproj-f16.gguf"]);
        assert_eq!(
            pick_gguf_matching(&files, "Q8_0"),
            Some("llama.Q8_0.gguf".into())
        );
    }

    #[test]
    fn pick_gguf_matching_finds_unsloth_dynamic_quant() {
        let files = s(&["model.UD-Q4_K_XL.gguf", "model.Q4_K_M.gguf"]);
        assert_eq!(
            pick_gguf_matching(&files, "UD-Q4_K_XL"),
            Some("model.UD-Q4_K_XL.gguf".into())
        );
    }

    #[test]
    fn pick_gguf_matching_returns_none_when_absent() {
        let files = s(&["model.Q4_K_M.gguf"]);
        assert_eq!(pick_gguf_matching(&files, "Q9_K"), None);
    }

    #[test]
    fn pick_gguf_matching_excludes_mmproj() {
        let files = s(&["mmproj-Q4_K_M.gguf", "model.Q4_K_M.gguf"]);
        assert_eq!(
            pick_gguf_matching(&files, "Q4_K_M"),
            Some("model.Q4_K_M.gguf".into())
        );
    }

    #[test]
    fn pick_any_main_gguf_excludes_mmproj() {
        let files = s(&["mmproj-Q4_K_M.gguf", "model.Q4_K_M.gguf"]);
        assert_eq!(pick_any_main_gguf(&files), Some("model.Q4_K_M.gguf".into()));
    }

    #[test]
    fn mmproj_pick_exact_quant_match() {
        let files = s(&["model.Q8_0.gguf", "mmproj-f16.gguf", "mmproj-Q8_0.gguf"]);
        assert_eq!(
            pick_mmproj(&files, Some("Q8_0")),
            Some("mmproj-Q8_0.gguf".into())
        );
    }

    #[test]
    fn mmproj_pick_matches_new_quant_types() {
        let files = s(&["mmproj-UD-Q4_K_XL.gguf", "mmproj-f16.gguf"]);
        assert_eq!(
            pick_mmproj(&files, Some("UD-Q4_K_XL")),
            Some("mmproj-UD-Q4_K_XL.gguf".into())
        );
    }

    #[test]
    fn mmproj_pick_falls_back_to_f16() {
        let files = s(&["mmproj-f16.gguf", "mmproj-Q4_K_M.gguf"]);
        assert_eq!(
            pick_mmproj(&files, Some("Q8_0")),
            Some("mmproj-f16.gguf".into())
        );
    }

    #[test]
    fn mmproj_pick_falls_back_to_first_when_no_quant_match() {
        let files = s(&["mmproj-Q4_K_M.gguf", "mmproj-Q5_K_M.gguf"]);
        assert_eq!(
            pick_mmproj(&files, Some("Q8_0")),
            Some("mmproj-Q4_K_M.gguf".into())
        );
    }

    #[test]
    fn mmproj_pick_none_when_absent() {
        let files = s(&["model.Q4_K_M.gguf", "README.md"]);
        assert_eq!(pick_mmproj(&files, Some("Q4_K_M")), None);
    }

    /// Self-cleaning unique temp directory (no `tempfile` dev-dep).
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let base = std::env::temp_dir().join(format!("seeker-dl-test-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&base);
            std::fs::create_dir_all(&base).unwrap();
            TempDir(base)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Build an HF-style cache under `hub_root`: `refs/main -> main_commit` plus
    /// the given `{commit: [files]}` snapshots, each file a real empty file.
    fn build_hf_cache(hub_root: &Path, repo_id: &str, main_commit: &str, snaps: &[(&str, &[&str])]) {
        let folder = hub_root.join(Repo::new(repo_id.to_string(), RepoType::Model).folder_name());
        std::fs::create_dir_all(folder.join("refs")).unwrap();
        std::fs::write(folder.join("refs").join("main"), main_commit).unwrap();
        for (commit, files) in snaps {
            let snap = folder.join("snapshots").join(commit);
            std::fs::create_dir_all(&snap).unwrap();
            for f in *files {
                std::fs::write(snap.join(f), b"").unwrap();
            }
        }
    }

    // The repro: `refs/main` points at a commit whose snapshot has no mmproj,
    // but an earlier commit's snapshot does. The offline listing must union
    // across snapshots so the sidecar is still found.
    #[test]
    fn offline_list_unions_across_snapshots() {
        let tmp = TempDir::new("union");
        build_hf_cache(
            &tmp.0,
            "unsloth/Demo-GGUF",
            "newcommit",
            &[
                ("newcommit", &["model.UD-Q5_K_XL.gguf"]),
                ("oldcommit", &["model.UD-Q4_K_XL.gguf", "mmproj-BF16.gguf"]),
            ],
        );
        let cache = Cache::new(tmp.0.clone());
        let files = list_files_offline(&cache, "unsloth/Demo-GGUF").unwrap();
        assert!(
            files.iter().any(|f| f == "mmproj-BF16.gguf"),
            "mmproj from non-main snapshot should surface; got {files:?}"
        );
        assert!(pick_mmproj(&files, Some("UD-Q5_K_XL")).is_some());
    }

    #[test]
    fn cached_lookup_finds_file_in_non_main_snapshot() {
        let tmp = TempDir::new("crosssnap");
        build_hf_cache(
            &tmp.0,
            "unsloth/Demo-GGUF",
            "newcommit",
            &[("newcommit", &["model.gguf"]), ("oldcommit", &["mmproj-BF16.gguf"])],
        );
        let cache = Cache::new(tmp.0.clone());
        let found = find_cached_in_snapshots(&cache, "unsloth/Demo-GGUF", "mmproj-BF16.gguf");
        assert!(found.is_some_and(|p| p.exists()));
        assert!(find_cached_in_snapshots(&cache, "unsloth/Demo-GGUF", "nope.gguf").is_none());
    }

    // `-m PATH` into one snapshot must still find an mmproj symlinked only in a
    // sibling snapshot of the same repo.
    #[test]
    fn sidecar_scans_sibling_snapshots() {
        let tmp = TempDir::new("sidecar");
        let folder = tmp
            .0
            .join(Repo::new("unsloth/Demo-GGUF".to_string(), RepoType::Model).folder_name());
        let new_snap = folder.join("snapshots").join("newcommit");
        let old_snap = folder.join("snapshots").join("oldcommit");
        std::fs::create_dir_all(&new_snap).unwrap();
        std::fs::create_dir_all(&old_snap).unwrap();
        let main = new_snap.join("model.UD-Q5_K_XL.gguf");
        std::fs::write(&main, b"").unwrap();
        std::fs::write(old_snap.join("mmproj-BF16.gguf"), b"").unwrap();

        let found = find_sidecar_mmproj(&main);
        assert!(
            found.as_ref().is_some_and(|p| p.ends_with("mmproj-BF16.gguf")),
            "expected sibling-snapshot mmproj, got {found:?}"
        );
    }

    #[test]
    fn sidecar_none_outside_snapshot_layout() {
        let tmp = TempDir::new("nosib");
        let main = tmp.0.join("model.Q4_K_M.gguf");
        std::fs::write(&main, b"").unwrap();
        assert_eq!(find_sidecar_mmproj(&main), None);
    }
}
