//! Per-block GPU timestamp profiling — feature `profile_gpu` only.
//!
//! When the feature is OFF (the default), this module reduces to a
//! tiny `BlockClass` stub so the public `DispatchContext::mark(…)`
//! signature compiles. Everything else (the query pool, the per-mark
//! cursor, the host readback, the printing) lives behind `cfg(feature
//! = "profile_gpu")`. Call sites in the model code call `ctx.mark(…)`
//! unconditionally; the empty-body variant is `#[inline(always)]` so
//! release builds emit no instructions.
//!
//! Mark style: a single `mark(class)` call serves as both the
//! end-of-previous-region and start-of-next-region boundary. The host
//! pass walks consecutive `(t[i], t[i+1])` pairs and attributes the
//! duration to `marks[i]`'s class, then aggregates by class for the
//! printed line.
//!
//! Stage: `BOTTOM_OF_PIPE_BIT` is the "after this command's last
//! pipeline stage completes" boundary — defensible default, doesn't
//! introduce ALL_COMMANDS-style serialisation on RADV.

/// Block class for the marks emitted at high-level boundaries inside
/// `record_forward`. Defined unconditionally so the `mark(…)` no-op
/// signature compiles even without the `profile_gpu` feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockClass {
    Embed,
    Attn,
    Ssm,
    MoE,
    Epilogue,
    LmHead,
    Sampler,
}

impl BlockClass {
    pub fn label(self) -> &'static str {
        match self {
            BlockClass::Embed => "embed",
            BlockClass::Attn => "attn",
            BlockClass::Ssm => "ssm",
            BlockClass::MoE => "moe",
            BlockClass::Epilogue => "epilogue",
            BlockClass::LmHead => "lm_head",
            BlockClass::Sampler => "sampler",
        }
    }
}

#[cfg(feature = "profile_gpu")]
mod inner {
    use super::BlockClass;
    use ash::vk;
    use std::error::Error;

    use crate::inference::device::Device;

    /// Fixed capacity — 256 timestamps × 8 bytes = 2 KiB. The
    /// qwen35moe forward currently uses ~80 marks; well within budget.
    pub const POOL_SIZE: u32 = 256;

    pub struct ProfileRecorder {
        pub query_pool: vk::QueryPool,
        pub marks: Vec<BlockClass>,
        pub next: u32,
        pub ns_per_tick: f64,
    }

    impl ProfileRecorder {
        pub fn new(device: &Device) -> Result<Self, Box<dyn Error>> {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(POOL_SIZE);
            let query_pool = unsafe { device.device.create_query_pool(&info, None) }?;
            // `timestampPeriod` is the conversion factor between
            // timestamp ticks and nanoseconds, as a `float`. RDNA
            // typically reports a value like 1.0 (1 tick = 1 ns) but
            // we always honour whatever the device reports.
            let ns_per_tick = device.limits.timestamp_period as f64;
            Ok(Self {
                query_pool,
                marks: Vec::with_capacity(POOL_SIZE as usize),
                next: 0,
                ns_per_tick,
            })
        }

        /// Clear the pool and our parallel `marks` vector. Called at
        /// the top of every forward, right after `begin_command_buffer`.
        pub fn reset(&mut self, device: &Device, cmd: vk::CommandBuffer) {
            unsafe {
                device
                    .device
                    .cmd_reset_query_pool(cmd, self.query_pool, 0, POOL_SIZE);
            }
            self.marks.clear();
            self.next = 0;
        }

        /// Emit a `vkCmdWriteTimestamp` at the current cursor and
        /// remember the class. Drops the mark silently on overflow (we
        /// log once at engine init if the pool is too small).
        pub fn mark(&mut self, device: &Device, cmd: vk::CommandBuffer, class: BlockClass) {
            if self.next >= POOL_SIZE {
                return;
            }
            unsafe {
                device.device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    self.next,
                );
            }
            self.marks.push(class);
            self.next += 1;
        }

        /// After fence wait, copy the pool's u64 ticks to host, convert
        /// to milliseconds, sum durations by class, and print a single
        /// `PROF blocks: …` line. Returns the per-class totals in
        /// milliseconds so the caller can include them in any other
        /// PROF output.
        pub fn readback_and_print(&self, device: &Device) {
            if self.next == 0 {
                return;
            }
            let mut ticks = [0u64; POOL_SIZE as usize];
            let _ = unsafe {
                device.device.get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut ticks[..self.next as usize],
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            };
            // Pair consecutive marks: marks[i] owns the duration
            // (ticks[i+1] - ticks[i]). Anything after the final mark
            // is un-tracked.
            let n = self.next as usize;
            let mut totals_ns: [u64; 7] = [0; 7];
            for i in 0..n.saturating_sub(1) {
                let dt = ticks[i + 1].wrapping_sub(ticks[i]) as f64 * self.ns_per_tick;
                let slot = self.marks[i] as usize;
                totals_ns[slot] = totals_ns[slot].saturating_add(dt as u64);
            }
            let kinds = [
                BlockClass::Embed,
                BlockClass::Attn,
                BlockClass::Ssm,
                BlockClass::MoE,
                BlockClass::Epilogue,
                BlockClass::LmHead,
                BlockClass::Sampler,
            ];
            let mut out = String::from("PROF blocks: ");
            for (k, kind) in kinds.iter().enumerate() {
                let ms = totals_ns[*kind as usize] as f64 / 1_000_000.0;
                if k > 0 {
                    out.push(' ');
                }
                out.push_str(&format!("{}={ms:.2}ms", kind.label()));
            }
            eprintln!("{out}");
        }

        pub fn destroy(&mut self, device: &Device) {
            unsafe { device.device.destroy_query_pool(self.query_pool, None) };
        }
    }
}

#[cfg(feature = "profile_gpu")]
pub use inner::ProfileRecorder;
