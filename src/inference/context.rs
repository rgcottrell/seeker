//! Dispatch context — the handle a [`crate::models::Model`]'s
//! `record_forward` writes into. Owns the active command buffer, the scratch
//! region, the descriptor pool, and a borrow of the pipeline cache + device.

use std::error::Error;

use ash::vk;

use crate::gguf::GgmlType;

use super::buffer::BufferRange;
use super::descriptor::DescriptorAllocator;
use super::device::Device;
use super::memory::Region;
use super::pipeline::PipelineCache;
use super::weights::{TensorView, WeightsHandle};

/// Prefill submission-splitting budget (llama.cpp-style). When present on a
/// [`DispatchContext`], [`DispatchContext::maybe_flush`] ends + submits the
/// current command buffer and starts a fresh one once the recorded weight-bytes
/// (or dispatch count) since the last flush crosses the budget — so no single
/// submission can exceed the GPU's TDR watchdog on a large prefill. `None` on
/// decode-replay and other single-submit paths (see `inference::Engine`).
pub struct FlushState {
    /// The engine fence, reused for each intermediate flush submit.
    pub fence: vk::Fence,
    /// Flush once `bytes_since_flush` reaches this (bytes of weights read).
    pub budget_bytes: u64,
    /// Flush once `nodes_since_flush` reaches this (dispatch-count fallback for
    /// non-matmul stretches like flash-attn).
    pub node_budget: u32,
    pub bytes_since_flush: u64,
    pub nodes_since_flush: u32,
}

pub struct DispatchContext<'a> {
    pub device: &'a Device,
    pub weights: &'a WeightsHandle,
    pub scratch: &'a mut Region,
    pub pipelines: &'a mut PipelineCache,
    pub descriptors: &'a DescriptorAllocator,
    pub cmd: vk::CommandBuffer,
    /// Prefill submission-splitting budget, or `None` to never flush mid-forward
    /// (decode replay / record-for-replay / single-submit debug paths).
    pub flush: Option<FlushState>,
    /// Per-forward dynamic-params slot. Reserved at the top of every
    /// forward (always the first scratch alloc, so the offset is stable
    /// across calls — required for the persistent-decode-cmdbuf
    /// optimization, where the recorded cmdbuf binds this range and
    /// subsequent replays only host-write new values into the slot).
    /// Holds [`super::decode_dyn::DecodeDyn`].
    pub decode_dyn: BufferRange,
    /// Offsets captured during the first decode recording so the host
    /// can re-populate the same slots between subsequent submits of
    /// the cached decode command buffer. Model fills `token_buf` and
    /// `positions_buf`; sampler fills `sampler_output` and (if
    /// penalties recorded) `penalty_pairs`. None during prefill or
    /// when the recording isn't going to be cached.
    pub replay_plan: Option<super::decode_dyn::ReplayPlan>,
    /// Optional list of `(name, range)` snapshots the model can push to.
    /// The engine reads these back after submit, alongside the main logits.
    /// Used for layer-by-layer diff dumps vs llama.cpp's `cb()` callback.
    /// Present only under the `gpu_debug` feature; in default builds the
    /// field doesn't exist and [`tap`](DispatchContext::tap) is an
    /// `#[inline(always)]` no-op.
    #[cfg(feature = "gpu_debug")]
    pub taps: Vec<(String, BufferRange)>,
    /// Bumped by every `bind_and_dispatch` — when `SEEKER_PROFILE_FORWARD=1`
    /// is set, `forward_sampled` prints this alongside its timing
    /// breakdown to give a per-token count of `vkCmdDispatch` calls.
    /// Present only under the `profile_gpu` feature so the per-dispatch
    /// increment vanishes from default builds.
    #[cfg(feature = "profile_gpu")]
    pub n_dispatches: u32,
    /// GPU-timestamp recorder for per-block profiling. Present only
    /// when the `profile_gpu` Cargo feature is enabled; in default
    /// builds the field doesn't exist and the [`mark`] method
    /// (defined below) is an `#[inline(always)]` no-op so call sites
    /// emit no instructions.
    ///
    /// [`mark`]: DispatchContext::mark
    #[cfg(feature = "profile_gpu")]
    pub profile: Option<&'a mut super::profile::ProfileRecorder>,
    /// Release-capable per-op buffer dump (NOT gpu_debug-gated). When set,
    /// [`dump`](DispatchContext::dump) records a transfer copy of a tensor into
    /// `dump.buffer` during recording; the host reads them back after submit.
    /// Used to bisect release-only nondeterminism (run the same forward twice,
    /// find the first record whose bytes differ → the offending op). `None` on
    /// every production path, so `dump()` is a cheap early-return.
    pub dump: Option<DumpState>,
}

/// A persistent host-visible sink for [`DispatchContext::dump`]. The owning
/// debug method allocates `buffer` (host-visible, TRANSFER_DST) and keeps its
/// backing memory alive across the submit, then reads `records` back via
/// `host_ptr`.
pub struct DumpState {
    pub buffer: vk::Buffer,
    pub host_ptr: *mut u8,
    pub capacity: u64,
    pub cursor: u64,
    /// `(label, byte_offset, byte_size)` per recorded dump, in record order.
    pub records: Vec<(String, u64, u64)>,
}

impl<'a> DispatchContext<'a> {
    /// Boundary marker — call once at the start of each high-level
    /// block (attention/SSM/MoE/lm_head/etc.). The next `mark` closes
    /// the previous region; consecutive timestamps form the
    /// duration-per-class summary printed at end of forward.
    ///
    /// With the `profile_gpu` feature OFF (default), this is an
    /// inlined empty function — every call site elides to nothing.
    #[cfg(feature = "profile_gpu")]
    pub fn mark(&mut self, kind: super::profile::BlockClass) {
        if let Some(p) = self.profile.as_deref_mut() {
            p.mark(self.device, self.cmd, kind);
        }
    }

    #[cfg(not(feature = "profile_gpu"))]
    #[inline(always)]
    pub fn mark(&mut self, _kind: super::profile::BlockClass) {}

    /// Release-capable dump: when `self.dump` is set, record a transfer copy of
    /// `t`'s bytes into the dump buffer (after a global barrier so the copy
    /// observes `t`'s compute writes) and register `(label, offset, size)` for
    /// host readback after submit. No-op (early return) when dump is unset —
    /// the production path. Used to bisect release-only nondeterminism by
    /// comparing the recorded bytes across two runs of the same input.
    pub fn dump(&mut self, label: &str, t: TensorView) {
        self.dump_range(
            label,
            BufferRange {
                buffer: t.buffer,
                offset: t.byte_offset,
                size: t.byte_size,
            },
        );
    }

    /// [`dump`](Self::dump) for a raw [`BufferRange`] (e.g. scratch slots that
    /// aren't `TensorView`s, like MoE routing `ids`).
    pub fn dump_range(&mut self, label: &str, r: BufferRange) {
        let (dump_buf, offset) = match self.dump.as_mut() {
            Some(d) => {
                if d.cursor + r.size > d.capacity {
                    return; // out of dump space — silently skip the tail
                }
                let off = d.cursor;
                d.records.push((label.to_string(), off, r.size));
                d.cursor += r.size;
                (d.buffer, off)
            }
            None => return,
        };
        super::command::record_global_barrier(self.device, self.cmd);
        let copy = vk::BufferCopy::default()
            .src_offset(r.offset)
            .dst_offset(offset)
            .size(r.size);
        unsafe {
            self.device.device.cmd_copy_buffer(
                self.cmd,
                r.buffer,
                dump_buf,
                std::slice::from_ref(&copy),
            );
        }
    }
}

impl<'a> DispatchContext<'a> {
    /// Snapshot the scratch bump cursor. Pair with [`scratch_restore`] to
    /// reclaim any slots allocated since the checkpoint — useful for
    /// scoping per-layer scratch within a forward pass.
    ///
    /// Safe to reuse the freed scratch range immediately on the CPU side
    /// because the GPU executes the recorded command buffer in order, with
    /// our barriers, so by the time later dispatches run the earlier ones
    /// have already finished reading their scratch slots.
    pub fn scratch_checkpoint(&self) -> u64 {
        self.scratch.cursor
    }

    /// Restore the scratch bump cursor to a previous checkpoint, freeing
    /// every slot allocated since then.
    pub fn scratch_restore(&mut self, cursor: u64) {
        self.scratch.cursor = cursor;
    }

    /// Account `weight_bytes` of weight read toward the prefill flush budget.
    /// Called by the matmul recorders (the weight `TensorView`'s `byte_size`).
    /// No-op when `flush` is `None`.
    pub fn account_matmul(&mut self, weight_bytes: u64) {
        if let Some(f) = self.flush.as_mut() {
            f.bytes_since_flush += weight_bytes;
        }
    }

    /// Prefill submission-splitting: bump the dispatch count and, if the recorded
    /// weight-bytes or dispatch count since the last flush has crossed the budget,
    /// end + submit the current command buffer, wait the fence, then reset and
    /// re-begin the same `cmd` handle so recording continues into a fresh
    /// submission. Called after every recorded dispatch (the universal
    /// `bind_and_dispatch` hook). No-op when `flush` is `None`.
    ///
    /// Correctness across the cut: the fence-wait makes every prior dispatch
    /// *complete* before the next cmdbuf runs, so cross-cmdbuf reads see finished
    /// values (the fence subsumes the in-cmdbuf barrier). Live state survives —
    /// weights/KV are persistent buffers and the residual / in-flight per-layer
    /// scratch keep their offsets (we do NOT reset scratch or move its cursor).
    pub fn maybe_flush(&mut self) -> Result<(), Box<dyn Error>> {
        // Decide within a scoped borrow of `self.flush`, so `self.device`/`self.cmd`
        // are free to use for the submit afterwards.
        let fence = {
            let Some(f) = self.flush.as_mut() else {
                return Ok(());
            };
            f.nodes_since_flush += 1;
            if f.bytes_since_flush < f.budget_bytes && f.nodes_since_flush < f.node_budget {
                return Ok(());
            }
            f.bytes_since_flush = 0;
            f.nodes_since_flush = 0;
            f.fence
        };
        let dev = &self.device.device;
        let cmd = self.cmd;
        let queue = self.device.queue;
        unsafe {
            dev.end_command_buffer(cmd)?;
            dev.reset_fences(&[fence])?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            dev.queue_submit(queue, &[submit], fence)?;
            dev.wait_for_fences(&[fence], true, u64::MAX)?;
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(cmd, &begin)?;
        }
        Ok(())
    }

    /// Reserve a `bytes`-byte slot in scratch and return its `BufferRange`.
    /// Cursor advances; the slot is valid only until the next forward pass.
    pub fn alloc_scratch(&mut self, bytes: u64) -> Result<BufferRange, Box<dyn Error>> {
        let off = self.scratch.alloc(bytes)?;
        Ok(BufferRange {
            buffer: self.scratch.buffer,
            offset: off,
            size: bytes,
        })
    }

    /// Copy `src` (any F32 tensor view) into a fresh contiguous F32 scratch
    /// slot and register it as a tap under `name`. The engine will read it
    /// back after submit and print a sum. No-op unless `SEEKER_QWEN_DIFF_DUMP`
    /// is set.
    ///
    /// With the `gpu_debug` feature OFF (default) this whole method — and
    /// the engine's tap-readback path and `taps` field — is compiled out;
    /// the variant below is an `#[inline(always)]` no-op so model call
    /// sites stay unconditional and emit no instructions in production.
    ///
    /// **Synchronization:** the tap cast READS from `src`. Without an
    /// explicit barrier, a later dispatch that WRITES to `src` (e.g. an
    /// in-place residual add) can race with this read because Vulkan
    /// dispatches in the same command buffer may overlap unless an
    /// explicit memory barrier orders them. We issue a barrier on `src`'s
    /// range right after the cast so any future writes to that memory
    /// wait for this cast to complete its read.
    #[cfg(feature = "gpu_debug")]
    pub fn tap(&mut self, name: &str, src: TensorView) -> Result<(), Box<dyn Error>> {
        if !crate::runtime_flags::qwen_diff_dump() {
            return Ok(());
        }
        debug_assert_eq!(src.dtype, GgmlType::F32, "tap only supports F32 tensors");
        let n_elements: u64 = src.dims.iter().product();
        // Direct readback mode: register src's range AS the tap range. The
        // engine reads bytes directly at src.byte_offset after fence wait,
        // without a cast dispatch. This rules out the cast as a source of
        // discrepancy.
        if crate::runtime_flags::qwen_diff_direct() {
            self.taps.push((
                name.to_string(),
                BufferRange {
                    buffer: src.buffer,
                    offset: src.byte_offset,
                    size: n_elements * 4,
                },
            ));
            return Ok(());
        }
        let dst = self.alloc_tensor(src.dims, GgmlType::F32)?;
        crate::inference::ops::cast::record_cast(self, src, dst)?;
        self.taps.push((
            name.to_string(),
            BufferRange {
                buffer: dst.buffer,
                offset: dst.byte_offset,
                size: n_elements * 4,
            },
        ));
        Ok(())
    }

    /// No-op tap for production builds (`gpu_debug` off). Call sites in
    /// model code stay unconditional; the optimizer elides them.
    #[cfg(not(feature = "gpu_debug"))]
    #[inline(always)]
    pub fn tap(&mut self, _name: &str, _src: TensorView) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    /// Reserve scratch space for a tensor with logical shape `dims` and the
    /// given `dtype`, contiguous layout (ggml convention).
    pub fn alloc_tensor(
        &mut self,
        dims: [u64; 4],
        dtype: GgmlType,
    ) -> Result<TensorView, Box<dyn Error>> {
        let (block_size, type_size) = dtype.block_layout();
        let mut byte_stride = [0u64; 4];
        byte_stride[0] = type_size as u64 / (block_size as u64).max(1);
        if block_size > 1 {
            byte_stride[0] = type_size as u64;
            byte_stride[1] = (dims[0] / block_size as u64).max(1) * type_size as u64;
        } else {
            byte_stride[1] = byte_stride[0] * dims[0];
        }
        byte_stride[2] = byte_stride[1] * dims[1];
        byte_stride[3] = byte_stride[2] * dims[2];
        let byte_size = byte_stride[3] * dims[3].max(1);

        let mut element_stride = [0u64; 4];
        let element_size = byte_stride[0].max(1);
        for i in 0..4 {
            element_stride[i] = byte_stride[i] / element_size;
        }

        let offset = self.scratch.alloc(byte_size)?;
        Ok(TensorView {
            buffer: self.scratch.buffer,
            byte_offset: offset,
            byte_size,
            dims,
            byte_stride,
            element_stride,
            dtype,
        })
    }
}
