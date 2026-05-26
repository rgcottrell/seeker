//! Process-lifetime debug/diagnostic flags. Each `SEEKER_*` env var
//! gets read **once** on first access via `LazyLock`; subsequent
//! reads are an atomic-pointer load + Boolean check (essentially
//! free). This matters because several flags were being checked in
//! the per-barrier / per-layer hot path:
//!
//!   - `SEEKER_BARRIER_PARANOID` — invoked from
//!     `record_compute_barrier(s)`, called 500–1500× per decode
//!     forward. At ~200 ns/`getenv` on glibc that alone was 0.1–0.3 ms
//!     of decode overhead.
//!   - `SEEKER_QWEN_NO_*` / `SEEKER_QWEN_SSM_DUMP` / `SEEKER_QWEN_GDN_SCALE`
//!     — checked per layer / per SSM block / per MoE block inside
//!     `record_forward`. ~290 lookups per forward (and
//!     `SEEKER_QWEN_GDN_SCALE` was `.parse::<f32>()`-ing per SSM call).
//!
//! All `SEEKER_*` flags are diagnostic switches set at process launch;
//! none of them are expected to change mid-run, so caching for the
//! lifetime of the process is sound.

use std::sync::LazyLock;

fn env_is_set(key: &'static str) -> bool {
    std::env::var(key).is_ok()
}

fn env_string(key: &'static str) -> Option<String> {
    std::env::var(key).ok()
}

fn env_u32(key: &'static str) -> Option<u32> {
    std::env::var(key).ok().and_then(|s| s.parse::<u32>().ok())
}

fn env_f32(key: &'static str) -> Option<f32> {
    std::env::var(key).ok().and_then(|s| s.parse::<f32>().ok())
}

// ─── Bool flags (presence-based) ──────────────────────────────────────

/// `record_compute_barrier(s)` widens each barrier to whole-buffer
/// when set — diagnostic for tracking down races, costly for normal
/// runs. Hot path: 500–1500 barriers per decode forward.
pub static BARRIER_PARANOID: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_BARRIER_PARANOID"));

/// `record_inner` matmul gate for the cooperative-matrix prefill
/// kernel (`SEEKER_MM_CM=0` to disable). Default-on.
pub static MM_CM_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("SEEKER_MM_CM").map(|v| v == "0").unwrap_or(false)
});

/// qwen35moe per-block ablation. Used by `record_forward` to bypass
/// the attention/SSM/MoE block at every layer where it would
/// normally run.
pub static QWEN_NO_ATTN: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_NO_ATTN"));
pub static QWEN_NO_SSM:  LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_NO_SSM"));
pub static QWEN_NO_MOE:  LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_NO_MOE"));
pub static QWEN_NO_CONV: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_NO_CONV"));
pub static QWEN_ONLY_RMS: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_ONLY_RMS"));
pub static QWEN_NO_ROUTED: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_NO_ROUTED"));
pub static QWEN_NO_SHARED: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_NO_SHARED"));

// ─── Tap / diff-dump ─────────────────────────────────────────────────

/// Layer-by-layer intermediate dump for diff-comparing seeker against
/// llama.cpp's `cb()` callback. Default off. Diff dumps are an opt-in
/// debugging mode — the cost of caching is not a concern because
/// we'd also be running with `SCRATCH_BYTES = 2 GiB` etc.
pub static QWEN_DIFF_DUMP:   LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_DIFF_DUMP"));
pub static QWEN_DIFF_DIRECT: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_QWEN_DIFF_DIRECT"));

// ─── Parsed configs ──────────────────────────────────────────────────

/// `SEEKER_QWEN_MAX_LAYERS=N` — cap the per-forward layer loop at
/// the first N layers. Diagnostic for bisecting where a numeric
/// issue first appears.
pub static QWEN_MAX_LAYERS: LazyLock<Option<u32>> = LazyLock::new(|| env_u32("SEEKER_QWEN_MAX_LAYERS"));

/// `SEEKER_QWEN_GDN_SCALE=one` — bypass the `1/sqrt(s_v)` GDN output
/// scale (use 1.0 instead). True only when the env var literally
/// equals `"one"`, matching the pre-existing diagnostic.
pub static QWEN_GDN_SCALE_ONE: LazyLock<bool> = LazyLock::new(|| {
    matches!(env_string("SEEKER_QWEN_GDN_SCALE").as_deref(), Some("one"))
});

/// `SEEKER_QWEN_SSM_DUMP=<stage>` — redirect the SSM block's residual
/// contribution to a named intermediate so the lm_head's logits
/// effectively contain that intermediate.
pub static QWEN_SSM_DUMP: LazyLock<Option<String>> = LazyLock::new(|| env_string("SEEKER_QWEN_SSM_DUMP"));

/// `SEEKER_MM_SPLIT_K=<n>` — force/disable lm_head matvec split-K.
/// `0` disables, `1` is a no-op, `n>=2` forces that split factor.
pub static MM_SPLIT_K: LazyLock<Option<u32>> = LazyLock::new(|| env_u32("SEEKER_MM_SPLIT_K"));

/// `SEEKER_CHAT_DEBUG=1` — print the rendered chat prompt to stderr.
/// Not hot-path but cache anyway for consistency.
pub static CHAT_DEBUG: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_CHAT_DEBUG"));

/// `SEEKER_PROFILE_FORWARD=1` — print a per-token `PROF forward:` line
/// showing CPU-side recording time, GPU compute (wait_for_fences)
/// time, and host readback. Use to triage host vs GPU overhead.
pub static PROFILE_FORWARD: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_PROFILE_FORWARD"));
