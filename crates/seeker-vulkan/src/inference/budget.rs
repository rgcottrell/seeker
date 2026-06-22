//! Holistic GPU-memory budgeting — the "fit" analog of llama.cpp's
//! `llama_params_fit`, scoped to a single-device unified-memory APU (Strix
//! Halo: one DDR5 heap, no separate VRAM, so the CPU-offload half of llama.cpp's
//! fit is N/A).
//!
//! Two jobs:
//!
//! 1. **Backstop preflight** ([`kv_preflight`]) — before a KV allocation, check
//!    that it fits the heap it lives in, accounting for everything else already
//!    resident (weights + scratch). With `VK_EXT_memory_budget` this reads the
//!    heap's *live free* bytes, which already exclude resident allocations — so
//!    nothing is double-counted; without it, it falls back to the static heap
//!    size. Fails with an actionable error instead of OOMing the allocation and
//!    wedging the device (RADV/amdgpu device-lost).
//!
//! 2. **Auto-fit context** ([`fit_ctx`]) — when `--ctx-size` is unset, reduce
//!    the model's trained-max default to the largest context that fits the
//!    budget (down to a floor), so a dense model with a large default context
//!    starts instead of failing. Explicit `--ctx-size` skips this (fail-fast).
//!
//! The KV byte projection itself lives in [`super::kv_cache`]
//! (`estimate_kv_bytes` & friends) so the preflight, the fitter, and the actual
//! allocation share one source of truth.

use std::error::Error;

use ash::vk;

use crate::gguf::GgmlType;

use super::device::Device;

const MIB: u64 = 1 << 20;
const GIB: f64 = (1u64 << 30) as f64;

/// Default free-memory margin (MiB) left after every tracked consumer. Covers
/// seeker's own untracked allocations (descriptor pools, command buffers, the
/// reused weight staging buffer, readback) plus a cushion for external pressure
/// the static query can't see. Overridable via `SEEKER_FIT_MARGIN_MIB`.
const DEFAULT_MARGIN_MIB: u32 = 1024;
/// Floor the auto-fit context search will not go below (llama.cpp's `--fit-ctx`
/// default). Overridable via `SEEKER_FIT_MIN_CTX`.
pub const DEFAULT_MIN_CTX: u32 = 4096;
/// Context-search granularity (tokens): the fitter rounds its choice down to a
/// multiple of this so slabs / chunked-prefill land on clean boundaries.
pub const FIT_GRANULARITY: u32 = 256;

/// Whether auto-fit context reduction is enabled (`SEEKER_FIT=0` disables; the
/// backstop preflight still guards every allocation).
pub fn fit_enabled() -> bool {
    !*crate::runtime_flags::FIT_DISABLED
}

/// The free-memory margin (bytes) the fitter leaves.
pub fn fit_margin_bytes() -> u64 {
    (*crate::runtime_flags::FIT_MARGIN_MIB).unwrap_or(DEFAULT_MARGIN_MIB) as u64 * MIB
}

/// The auto-fit context floor.
pub fn fit_min_ctx() -> u32 {
    (*crate::runtime_flags::FIT_MIN_CTX).unwrap_or(DEFAULT_MIN_CTX)
}

/// The memory budget for new allocations on the heap the KV cache lives in.
pub struct HeapBudget {
    /// Nominal heap capacity (`VkMemoryHeap.size`).
    total: u64,
    /// Live free bytes (`VK_EXT_memory_budget`: `heapBudget − heapUsage`), when
    /// the extension is enabled. `None` ⇒ fall back to `total`.
    free: Option<u64>,
    /// Safety fraction applied to the base (serve's `--mem-fraction`; 0.9
    /// elsewhere; 1.0 for the margin-only backstop).
    fraction: f64,
    /// Absolute margin (bytes) to leave free.
    margin: u64,
}

impl HeapBudget {
    fn base(&self) -> u64 {
        self.free.unwrap_or(self.total)
    }

    /// Bytes available for *new* allocations, given the bytes already resident
    /// on the heap. With a live-free reading the residents are already excluded
    /// (do NOT subtract them again — the double-count trap); with the static
    /// fallback they must be subtracted explicitly.
    pub fn usable_for_new(&self, already_resident: u64) -> u64 {
        let base = self.base();
        let cap = ((base as f64) * self.fraction) as u64;
        let cap = cap.min(base.saturating_sub(self.margin));
        match self.free {
            Some(_) => cap,
            None => cap.saturating_sub(already_resident),
        }
    }

    pub fn total(&self) -> u64 {
        self.total
    }
    pub fn free(&self) -> Option<u64> {
        self.free
    }
    pub fn is_live(&self) -> bool {
        self.free.is_some()
    }
}

/// Budget for the heap the KV cache allocates from (`HOST_VISIBLE |
/// HOST_COHERENT` `STORAGE_BUFFER` — the unified DDR5 carveout on Strix Halo).
/// `fraction` is the caller's safety multiplier: serve's `--mem-fraction`, 0.9
/// for chat's fitter, 1.0 for the margin-only backstop.
pub fn kv_heap_budget(device: &Device, fraction: f64) -> HeapBudget {
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let mem = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let idx = super::memory::heap_index_for_buffer(device, usage, mem);
    let total = idx
        .map(|i| device.mem_props.memory_heaps[i].size)
        .unwrap_or(0);
    let free = idx.and_then(|i| heap_free_bytes(device, i));
    HeapBudget {
        total,
        free,
        fraction: fraction.clamp(0.1, 1.0),
        margin: fit_margin_bytes(),
    }
}

/// Live free bytes on heap `heap_idx` via `VK_EXT_memory_budget`
/// (`heapBudget − heapUsage`). `None` when the extension wasn't enabled.
pub fn heap_free_bytes(device: &Device, heap_idx: usize) -> Option<u64> {
    if !device.has_memory_budget {
        return None;
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    // Chain the budget struct manually (ash doesn't generate a `push` for
    // PhysicalDeviceMemoryProperties2). `budget` outlives the query call below;
    // the `&mut` cast to a raw pointer does not hold the borrow.
    let mut props2 = vk::PhysicalDeviceMemoryProperties2 {
        p_next: &mut budget as *mut _ as *mut std::ffi::c_void,
        ..Default::default()
    };
    unsafe {
        device
            .instance
            .get_physical_device_memory_properties2(device.physical, &mut props2)
    };
    let b = budget.heap_budget[heap_idx];
    let u = budget.heap_usage[heap_idx];
    Some(b.saturating_sub(u))
}

/// Backstop preflight for a KV allocation of `kv_bytes` (already `× n_slots`)
/// on the heap the cache lives in. Margin-only (`fraction = 1.0`): the precise
/// fitters (chat's [`fit_ctx`], serve's slot resolver) run first and are
/// stricter; this only fires on a genuine near-overflow that would OOM/wedge.
/// No-op if the heap can't be resolved (the allocator then fails precisely) or
/// the allocation fits.
#[allow(clippy::too_many_arguments)]
pub fn kv_preflight(
    device: &Device,
    kv_bytes: u64,
    max_seq_len: u32,
    n_layer: u32,
    n_slots: u32,
    k: GgmlType,
    v: GgmlType,
) -> Result<(), Box<dyn Error>> {
    let budget = kv_heap_budget(device, 1.0);
    if budget.total() == 0 {
        return Ok(()); // heap unresolved — defer to the precise allocator error
    }
    let usable = budget.usable_for_new(0);
    if kv_bytes <= usable {
        return Ok(());
    }
    let max_ctx = ((max_seq_len as u64) * usable)
        .checked_div(kv_bytes)
        .unwrap_or(0) as u32;
    let basis = if budget.is_live() {
        "free GPU memory"
    } else {
        "GPU heap"
    };
    let slots = if n_slots > 1 {
        format!(", {n_slots} slots")
    } else {
        String::new()
    };
    Err(format!(
        "KV cache needs {:.1} GiB (max_seq_len={max_seq_len}, {n_layer} layers{slots}, \
         k={k:?} v={v:?}) but only {:.1} GiB of {basis} is usable — lower --ctx-size \
         (≈ {max_ctx} fits), or use a smaller cache dtype (e.g. --cache-type-k q8_0 \
         --cache-type-v q8_0).",
        kv_bytes as f64 / GIB,
        usable as f64 / GIB,
    )
    .into())
}

/// Even the floor context doesn't fit the budget.
#[derive(Debug)]
pub struct FitInfeasible {
    pub floor: u32,
    pub need: u64,
    pub usable: u64,
}

/// Largest `ctx ≤ requested` (a multiple of [`FIT_GRANULARITY`], `≥ floor`)
/// whose projected ctx-dependent bytes fit `usable`. `cost_at_ctx(ctx)` returns
/// the bytes that scale with context — KV (`× n_slots`) + scratch(ctx);
/// ctx-independent costs (SSM, prefix pool) must already be subtracted from
/// `usable` by the caller. `spec_headroom` is the extra lookahead the cache
/// physically holds (added inside `cost_at_ctx` by the caller's closure). The
/// cost is monotone non-decreasing in ctx, so a binary search is exact.
pub fn fit_ctx(
    requested: u32,
    floor: u32,
    usable: u64,
    mut cost_at_ctx: impl FnMut(u32) -> u64,
) -> Result<u32, FitInfeasible> {
    let gran = FIT_GRANULARITY.max(1);
    let floor = floor.max(1).min(requested);

    // Fits as-is?
    if cost_at_ctx(requested) <= usable {
        return Ok(requested);
    }
    // Floor feasibility.
    let need_floor = cost_at_ctx(floor);
    if need_floor > usable {
        return Err(FitInfeasible {
            floor,
            need: need_floor,
            usable,
        });
    }
    // Binary search the largest feasible ctx in [floor, requested]
    // (cost is monotone non-decreasing). Invariant: lo feasible, hi infeasible.
    let mut lo = floor;
    let mut hi = requested;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if cost_at_ctx(mid) <= usable {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Round down to a granule but never below the (feasible) floor; rounding
    // down stays feasible by monotonicity.
    Ok((lo / gran * gran).max(floor))
}

/// Every major GPU-memory consumer for one configuration, for the startup
/// breakdown report (the `llama_memory_breakdown_print` analog).
pub struct MemoryProjection {
    pub weights: u64,
    pub scratch: u64,
    pub kv: u64,
    pub ssm: u64,
    pub prefix_pool: u64,
}

impl MemoryProjection {
    pub fn total(&self) -> u64 {
        self.weights + self.scratch + self.kv + self.ssm + self.prefix_pool
    }
}

/// Log the memory breakdown at startup. `chosen_ctx < requested_ctx` signals an
/// auto-reduction.
pub fn log_breakdown(
    proj: &MemoryProjection,
    budget: &HeapBudget,
    requested_ctx: u32,
    chosen_ctx: u32,
    n_slots: u32,
) {
    let g = |b: u64| format!("{:.2} GiB", b as f64 / GIB);
    tracing::info!(
        heap_total = %g(budget.total()),
        heap_free = budget.free().map(g).unwrap_or_else(|| "n/a (static)".into()),
        usable = %g(budget.usable_for_new(proj.weights + proj.scratch)),
        weights = %g(proj.weights),
        scratch = %g(proj.scratch),
        kv = %g(proj.kv),
        ssm = %g(proj.ssm),
        prefix_pool = %g(proj.prefix_pool),
        total = %g(proj.total()),
        n_slots,
        ctx_requested = requested_ctx,
        ctx_chosen = chosen_ctx,
        auto_reduced = chosen_ctx < requested_ctx,
        "gpu memory breakdown",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // A linear cost model: `fixed + per_tok * ctx`. Exercises the common case.
    fn linear(fixed: u64, per_tok: u64) -> impl FnMut(u32) -> u64 {
        move |ctx: u32| fixed + per_tok * ctx as u64
    }

    #[test]
    fn fit_ctx_already_fits_returns_requested() {
        let got = fit_ctx(8192, 4096, 1_000_000_000, linear(0, 1000)).unwrap();
        assert_eq!(got, 8192);
    }

    #[test]
    fn fit_ctx_halves_when_2x_over() {
        // per_tok=1000, requested=8192 → needs 8_192_000; budget ~half.
        let usable = 4_096_000;
        let got = fit_ctx(8192, 256, usable, linear(0, 1000)).unwrap();
        assert!(got <= 4096, "got {got}");
        assert!(got >= 4096 - FIT_GRANULARITY, "got {got}");
        // Chosen must fit; the next granule must not.
        assert!(1000 * got as u64 <= usable);
        assert!(1000 * (got + FIT_GRANULARITY) as u64 > usable);
        // Rounded to a granule.
        assert_eq!(got % FIT_GRANULARITY, 0);
    }

    #[test]
    fn fit_ctx_infeasible_at_floor() {
        // Even floor=4096 needs 4_096_000 > usable=1_000_000.
        let err = fit_ctx(8192, 4096, 1_000_000, linear(0, 1000)).unwrap_err();
        assert_eq!(err.floor, 4096);
        assert!(err.need > err.usable);
    }

    #[test]
    fn fit_ctx_converges_on_nonlinear_cost() {
        // Simulate a future SWA windowed cache: cost is a step function (flat
        // past a window), still monotone non-decreasing. Binary search must
        // still find the largest fitting ctx.
        let usable = 5_000;
        let cost = |ctx: u32| -> u64 {
            let capped = ctx.min(2048) as u64; // window cap
            100 + capped * 3
        };
        let got = fit_ctx(65536, 256, usable, cost).unwrap();
        assert!(cost(got) <= usable);
        assert_eq!(got % FIT_GRANULARITY, 0);
    }

    #[test]
    fn usable_for_new_static_subtracts_residents() {
        let b = HeapBudget {
            total: 100,
            free: None,
            fraction: 1.0,
            margin: 0,
        };
        assert_eq!(b.usable_for_new(30), 70);
    }

    #[test]
    fn usable_for_new_live_does_not_double_count() {
        // free already excludes residents → resident arg ignored.
        let b = HeapBudget {
            total: 100,
            free: Some(40),
            fraction: 1.0,
            margin: 0,
        };
        assert_eq!(b.usable_for_new(30), 40);
    }

    #[test]
    fn usable_for_new_margin_and_fraction_stricter_wins() {
        let b = HeapBudget {
            total: 1000,
            free: None,
            fraction: 0.9,
            margin: 50,
        };
        // min(900, 950) = 900, then static subtract 0.
        assert_eq!(b.usable_for_new(0), 900);
        let b2 = HeapBudget {
            total: 1000,
            free: None,
            fraction: 0.99,
            margin: 50,
        };
        // min(990, 950) = 950.
        assert_eq!(b2.usable_for_new(0), 950);
    }
}
