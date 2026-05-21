use std::error::Error;
use std::path::PathBuf;

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

fn filename_contains_quant(filename: &str, quant: &str) -> bool {
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

fn is_mmproj_gguf(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    lower.ends_with(".gguf") && lower.contains("mmproj")
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

fn list_files_offline(cache: &Cache, repo_id: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let folder = cache
        .path()
        .join(Repo::new(repo_id.to_string(), RepoType::Model).folder_name());
    let refs_main = folder.join("refs").join("main");
    let commit = std::fs::read_to_string(&refs_main).map_err(|e| {
        format!(
            "offline cache missing for {repo_id}: cannot read {} ({e})",
            refs_main.display()
        )
    })?;
    let snap = folder.join("snapshots").join(commit.trim());
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&snap).map_err(|e| {
        format!(
            "offline cache missing snapshot for {repo_id}: cannot read {} ({e})",
            snap.display()
        )
    })? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            out.push(name.to_string());
        }
    }
    Ok(out)
}

async fn fetch_or_resolve(
    repo: &ApiRepo,
    cache_repo: &CacheRepo,
    filename: &str,
    offline: bool,
) -> Result<PathBuf, Box<dyn Error>> {
    if offline {
        cache_repo
            .get(filename)
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
    let main = fetch_or_resolve(&api_repo, &cache_repo, &main_file, args.offline).await?;

    let mmproj = if want_mmproj {
        if let Some(mmproj) = pick_mmproj(&files, effective_quant.as_deref()) {
            info!(file = %mmproj, "selected mmproj sidecar");
            Some(fetch_or_resolve(&api_repo, &cache_repo, &mmproj, args.offline).await?)
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
}
