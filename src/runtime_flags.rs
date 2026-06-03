//! Process-lifetime diagnostic / tuning flags.
//!
//! Two distinct categories live here:
//!
//! 1. **Tuning knobs** (always compiled in): select between functional
//!    code paths or pin kernel parameters — `SEEKER_MM_CM`,
//!    `SEEKER_MM_SPLIT_K`, `SEEKER_FA_*`. These are not debugging
//!    instrumentation; they stay available in every build. Each is read
//!    **once** via `LazyLock` so the per-call cost is a single
//!    atomic-pointer load (the prior `std::env::var(…)` was a getenv +
//!    string-alloc, and several of these sit in per-token / per-op hot
//!    paths).
//!
//! 2. **Debug / profiling flags** (feature-gated): exposed as accessor
//!    functions that read their env var (cached) only when the relevant
//!    Cargo feature is enabled, and otherwise `#[inline(always)]`-return
//!    a constant `false`/`None`. With the feature off the optimizer
//!    constant-folds the call site away, so the surrounding branch is
//!    eliminated and the instrumentation has **zero effect on production
//!    builds**:
//!      - `gpu_debug` gates correctness-debugging flags
//!        (`barrier_paranoid`, the qwen35moe ablations, the tap/diff-dump
//!        toggles, `chat_debug`, `decode_replay_disabled`).
//!      - `profile_gpu` gates the `profile_forward` per-token timing flag.
//!
//! All flags are set at process launch and never change mid-run, so
//! caching for the process lifetime is sound.

use std::sync::LazyLock;

#[allow(dead_code)] // usage varies by feature combination
fn env_is_set(key: &'static str) -> bool {
    std::env::var(key).is_ok()
}

#[allow(dead_code)]
fn env_string(key: &'static str) -> Option<String> {
    std::env::var(key).ok()
}

#[allow(dead_code)]
fn env_u32(key: &'static str) -> Option<u32> {
    std::env::var(key).ok().and_then(|s| s.parse::<u32>().ok())
}

// ─── Tuning knobs (always available — not debugging) ─────────────────

/// `record_inner` matmul gate for the cooperative-matrix prefill
/// kernel (`SEEKER_MM_CM=0` to disable). Default-on.
pub static MM_CM_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("SEEKER_MM_CM")
        .map(|v| v == "0")
        .unwrap_or(false)
});

/// `SEEKER_MM_SPLIT_K=<n>` — force/disable lm_head matvec split-K.
/// `0` disables, `1` is a no-op, `n>=2` forces that split factor.
pub static MM_SPLIT_K: LazyLock<Option<u32>> = LazyLock::new(|| env_u32("SEEKER_MM_SPLIT_K"));

/// Masked (prefill) flash-attention coopmat gate (`SEEKER_FA_CM=0` to fall back
/// to the scalar path). Default-on: the cm1 coopmat kernel is ~5× faster per key
/// than the scalar loop, so a deep-context prefill ubatch (large kv) finishes in
/// one dispatch under the RADV per-dispatch watchdog — the scalar loop is too
/// slow and trips it (device-lost) past ~14k keys. Mirrors llama.cpp, which uses
/// coopmat FA by default for exactly this reason. F16 KV + head_dim ≤ 128 only;
/// other combos fall through to scalar.
pub static FA_CM_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("SEEKER_FA_CM")
        .map(|v| v == "0")
        .unwrap_or(false)
});

/// Vision-tower flash-attention coopmat gate (`SEEKER_FA_CM_VISION=0` to fall
/// back to the scalar split-K path). Default-on: the register-O coopmat cm1
/// kernel encodes the full-res tower ~5× faster than scalar on Strix Halo.
pub static FA_CM_VISION_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("SEEKER_FA_CM_VISION")
        .map(|v| v == "0")
        .unwrap_or(false)
});

/// `SEEKER_FA_SPLIT=0` — disable flash-attention split-K. Read per decode
/// forward (in `pick_k_num`); cache to avoid a per-token getenv.
pub static FA_SPLIT_DISABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("SEEKER_FA_SPLIT").is_ok_and(|v| v == "0"));

/// `SEEKER_FA_SPLIT_KNUM=<n>` — pin the flash-attention split-K factor.
/// Read per decode forward; cache to avoid a per-token getenv + parse.
pub static FA_SPLIT_KNUM: LazyLock<Option<u32>> = LazyLock::new(|| env_u32("SEEKER_FA_SPLIT_KNUM"));

/// `SEEKER_SSM_BATCH=0` — force the per-sequence SSM loop in the unified
/// forward. Default-on: a pure-decode batched step routes the SSM block through
/// `ssm_block_batch` so the in/out projections read their weights once for all
/// B sequences (matvec → one matmul) instead of re-reading per sequence. The
/// batched path is byte-identical to the per-sequence loop at the serve scale
/// (B ≤ 8, below the CoopMat N≥32 gate); this flag is the revert / diff lever.
pub static SSM_BATCH_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("SEEKER_SSM_BATCH")
        .map(|v| v == "0")
        .unwrap_or(false)
});

/// `SEEKER_PROF_STEP=1` — print a per-step `PROF step:` line for the unified
/// forward, splitting CPU command-recording time from GPU compute (fence wait).
/// Gates the persistent batched-decode cmdbuf work: replay only pays off if
/// recording is a meaningful fraction of a (bandwidth-bound) decode step.
/// Always available; cached (one atomic load per step).
pub static PROF_STEP: LazyLock<bool> =
    LazyLock::new(|| std::env::var("SEEKER_PROF_STEP").is_ok_and(|v| v == "1"));

// ─── Debug flags (gpu_debug) ─────────────────────────────────────────

/// Define a presence-based debug flag behind `gpu_debug`. With the
/// feature off the accessor is a `const false` so the call site folds
/// away and the surrounding branch is eliminated from production builds.
macro_rules! gpu_debug_flag {
    ($(#[$m:meta])* $vis:vis fn $name:ident = $env:literal) => {
        $(#[$m])*
        #[cfg(feature = "gpu_debug")]
        #[inline]
        $vis fn $name() -> bool {
            static V: LazyLock<bool> = LazyLock::new(|| env_is_set($env));
            *V
        }
        $(#[$m])*
        #[cfg(not(feature = "gpu_debug"))]
        #[inline(always)]
        $vis fn $name() -> bool { false }
    };
    // Variant: true only when the env var equals a specific value.
    ($(#[$m:meta])* $vis:vis fn $name:ident = $env:literal == $val:literal) => {
        $(#[$m])*
        #[cfg(feature = "gpu_debug")]
        #[inline]
        $vis fn $name() -> bool {
            static V: LazyLock<bool> =
                LazyLock::new(|| matches!(env_string($env).as_deref(), Some($val)));
            *V
        }
        $(#[$m])*
        #[cfg(not(feature = "gpu_debug"))]
        #[inline(always)]
        $vis fn $name() -> bool { false }
    };
}

gpu_debug_flag! {
    /// `record_compute_barrier(s)` widens each barrier to whole-buffer
    /// when set — diagnostic for tracking down races, costly for normal
    /// runs (500–1500 barriers per decode forward).
    pub fn barrier_paranoid = "SEEKER_BARRIER_PARANOID"
}

gpu_debug_flag! {
    /// Layer-by-layer intermediate dump for diff-comparing seeker against
    /// llama.cpp's `cb()` callback.
    pub fn qwen_diff_dump = "SEEKER_QWEN_DIFF_DUMP"
}
gpu_debug_flag! {
    /// Direct-readback variant of the diff dump (register the source
    /// range as the tap, skip the cast dispatch).
    pub fn qwen_diff_direct = "SEEKER_QWEN_DIFF_DIRECT"
}

// qwen35moe per-block ablations — bypass a block at every layer where it
// would normally run, for bisecting numeric issues during bring-up.
gpu_debug_flag! { pub fn qwen_no_attn = "SEEKER_QWEN_NO_ATTN" }
gpu_debug_flag! { pub fn qwen_no_ssm = "SEEKER_QWEN_NO_SSM" }
gpu_debug_flag! { pub fn qwen_no_moe = "SEEKER_QWEN_NO_MOE" }
gpu_debug_flag! { pub fn qwen_no_conv = "SEEKER_QWEN_NO_CONV" }
gpu_debug_flag! { pub fn qwen_only_rms = "SEEKER_QWEN_ONLY_RMS" }
gpu_debug_flag! { pub fn qwen_no_routed = "SEEKER_QWEN_NO_ROUTED" }
gpu_debug_flag! { pub fn qwen_no_shared = "SEEKER_QWEN_NO_SHARED" }

gpu_debug_flag! {
    /// `SEEKER_QWEN_GDN_SCALE=one` — bypass the `1/sqrt(s_v)` GDN output
    /// scale (use 1.0 instead).
    pub fn qwen_gdn_scale_one = "SEEKER_QWEN_GDN_SCALE" == "one"
}

gpu_debug_flag! {
    /// `SEEKER_DECODE_REPLAY=0` forces the legacy record-each-token decode
    /// path — diagnostic only. Default is replay-on.
    pub fn decode_replay_disabled = "SEEKER_DECODE_REPLAY" == "0"
}

gpu_debug_flag! {
    /// `SEEKER_CHAT_DEBUG=1` — print the rendered chat prompt to stderr.
    pub fn chat_debug = "SEEKER_CHAT_DEBUG"
}

/// `SEEKER_QWEN_MAX_LAYERS=N` — cap the per-forward layer loop at the
/// first N layers. Diagnostic for bisecting where a numeric issue first
/// appears. `None` (no cap) in non-`gpu_debug` builds.
#[cfg(feature = "gpu_debug")]
#[inline]
pub fn qwen_max_layers() -> Option<u32> {
    static V: LazyLock<Option<u32>> = LazyLock::new(|| env_u32("SEEKER_QWEN_MAX_LAYERS"));
    *V
}
#[cfg(not(feature = "gpu_debug"))]
#[inline(always)]
pub fn qwen_max_layers() -> Option<u32> {
    None
}

/// `SEEKER_QWEN_SSM_DUMP=<stage>` — redirect the SSM block's residual
/// contribution to a named intermediate so the lm_head's logits
/// effectively contain that intermediate. `None` in non-`gpu_debug` builds.
#[cfg(feature = "gpu_debug")]
#[inline]
pub fn qwen_ssm_dump() -> Option<&'static str> {
    static V: LazyLock<Option<String>> = LazyLock::new(|| env_string("SEEKER_QWEN_SSM_DUMP"));
    V.as_deref()
}
#[cfg(not(feature = "gpu_debug"))]
#[inline(always)]
pub fn qwen_ssm_dump() -> Option<&'static str> {
    None
}

// ─── Profiling flags (profile_gpu) ───────────────────────────────────

/// `SEEKER_PROFILE_FORWARD=1` — print a per-token `PROF forward:` line
/// showing CPU-side recording time, GPU compute (wait_for_fences) time,
/// host readback, and dispatch/barrier counts. `false` (and the entire
/// timing path is compiled out) in non-`profile_gpu` builds.
#[cfg(feature = "profile_gpu")]
#[inline]
pub fn profile_forward() -> bool {
    static V: LazyLock<bool> = LazyLock::new(|| env_is_set("SEEKER_PROFILE_FORWARD"));
    *V
}
#[cfg(not(feature = "profile_gpu"))]
#[inline(always)]
pub fn profile_forward() -> bool {
    false
}
