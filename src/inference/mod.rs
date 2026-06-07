//! Vulkan inference runtime — architecture-agnostic.
//!
//! Owns the Vulkan device, GPU memory regions, pipeline cache, descriptor
//! pool, and the per-op dispatch recorders in [`ops`]. Knows nothing about
//! LLaMA, Qwen, or any specific transformer arch — those live in
//! `crate::models::*` and use [`context::DispatchContext`] + the [`ops`]
//! helpers to record their forward pass.

pub mod buffer;
pub mod command;
pub mod context;
pub mod decode_dyn;
pub mod descriptor;
pub mod device;
pub mod kv_cache;
pub mod memory;
pub mod ops;
pub mod pipeline;
pub mod profile;
pub mod sample;
pub mod weights;

use std::error::Error;

use ash::vk;

use crate::gguf::GgufFile;

use buffer::BufferRange;
use context::DispatchContext;
use descriptor::DescriptorAllocator;
use device::Device;
use memory::Region;
use pipeline::PipelineCache;
use weights::WeightsHandle;

/// Top-level runtime. Built once, then used to upload weights and run
/// multiple forward passes against any [`crate::models::Model`].
pub struct Engine {
    pub device: Device,
    pub pipelines: PipelineCache,
    pub descriptors: DescriptorAllocator,
    pub scratch: Region,
    /// Small HOST_CACHED staging buffer for result readback. The GPU-write-heavy
    /// `scratch` is HOST_COHERENT-only (write-combined): fast for GPU writes, but
    /// the CPU reads it uncached (~185 MB/s on this UMA APU). Copying the few
    /// KB/MB of results into this cached buffer first (a cheap device-side copy)
    /// makes the host read cache-speed — matters for the stochastic spec verify's
    /// full-vocab logit readback. Grown on demand; see [`Engine::run_spec_record`].
    pub readback: Region,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    /// Secondary command buffer used to cache the decode forward graph
    /// for replay. Cleared on prefill / sampler-config change / k_num
    /// boundary; once populated, subsequent decode tokens host-update
    /// the input scratch slots and resubmit this cmdbuf without
    /// re-recording.
    pub decode_command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    /// Per-block GPU-timestamp recorder. Only present in builds with
    /// the `profile_gpu` feature; non-profiling builds skip this field
    /// entirely (no query-pool allocation, no host readback path).
    #[cfg(feature = "profile_gpu")]
    pub profile: profile::ProfileRecorder,
    /// Current cached decode recording. `None` when invalid (after
    /// prefill, sampler-config change, or k_num boundary crossing).
    pub decode_cache: Option<DecodeCache>,
    /// Scratch cursor immediately after the decode recording finishes.
    /// While `decode_cache` is `Some`, subsequent decode replays skip
    /// `scratch.reset()` so the bindings captured in
    /// `decode_command_buffer` stay valid. Prefill bumps the cache and
    /// resumes the normal reset-each-call behavior.
    pub decode_scratch_cursor: u64,
    /// Physical micro-batch size (llama.cpp `n_ubatch`). Prefill is split
    /// into `≤ n_ubatch`-token passes so the per-pass scratch working set
    /// stays bounded regardless of prompt length. `0` ⇒ unbounded (legacy
    /// single-pass behavior). See [`Engine::forward_sampled`].
    pub n_ubatch: u32,
    /// Logical batch size (llama.cpp `n_batch`). Validation-only in this
    /// single-sequence engine — `n_ubatch` is the memory-relevant knob.
    pub n_batch: u32,
}

/// State captured by the first decode recording so subsequent replays
/// know which scratch slots to refresh and when to invalidate.
#[derive(Debug, Clone)]
pub struct DecodeCache {
    /// `Sampler::config().graph_hash()` at recording time.
    pub sampler_config_hash: u64,
    /// `Model::decode_shape_key(kv, shader_core_count)` at recording time.
    pub shape_key: u64,
    /// Captured `kv_len` / `k_num` / `blocks_per_split` so we can re-
    /// stamp DecodeDyn between replays without re-running the heuristic.
    pub kv_len: u32,
    pub k_num: u32,
    pub blocks_per_split: u32,
    pub plan: decode_dyn::ReplayPlan,
    pub model_constants: decode_dyn::ModelReplayConstants,
}

/// Placeholder scratch size used between `Engine::new` and the first
/// `allocate_scratch`. Kept tiny — the real region is sized from the model.
const SCRATCH_PLACEHOLDER_BYTES: u64 = 1 << 20;

/// Initial size of the HOST_CACHED readback staging buffer (grown on demand to
/// the largest single readback, e.g. a `[vocab, n+1]` logit block).
const READBACK_PLACEHOLDER_BYTES: u64 = 1 << 20;

/// Allocate a HOST_CACHED (fast CPU read) staging buffer for result readback,
/// falling back to plain HOST_COHERENT (write-combined) if the device exposes no
/// cached host-visible memory type (e.g. the CPU `llvmpipe` fallback).
fn new_readback_region(device: &Device, bytes: u64) -> Result<Region, Box<dyn Error>> {
    let usage = vk::BufferUsageFlags::TRANSFER_DST;
    let cached = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED;
    Region::new(device, bytes, usage, cached).or_else(|_| {
        Region::new(
            device,
            bytes,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    })
}

/// Result of one MTP speculative-decode step ([`Engine::decode_speculative`]).
pub struct SpecStepOut {
    /// Tokens accepted this step (`accept_len + 1`, always ≥ 1): the matched
    /// draft prefix plus one bonus/corrective token. All are committed to the
    /// KV cache and `cache.position` has advanced past them.
    pub emitted: Vec<u32>,
    /// Number of draft tokens accepted (`emitted.len() - 1`) — for logging /
    /// acceptance-rate tracking.
    pub accepted: u32,
    /// Pre-`output_norm` hidden state at the last committed position — seeds
    /// the next step's MTP drafting.
    pub h_last: Vec<f32>,
    /// The last committed token — fed as input to the next step's verify.
    pub last_token: u32,
}

impl Engine {
    pub fn new(n_ubatch: u32, n_batch: u32) -> Result<Self, Box<dyn Error>> {
        if n_ubatch != 0 && n_ubatch > n_batch {
            return Err(format!("n_ubatch ({n_ubatch}) must be <= n_batch ({n_batch})").into());
        }
        let device = Device::new()?;
        let pipelines = PipelineCache::new();
        let descriptors = DescriptorAllocator::new(&device)?;

        // Scratch is host-visible + device-local. On Apple Silicon (unified
        // memory) this maps trivially; on discrete GPUs with BAR/ReBAR this
        // also works. If neither is available we'd need a staging path.
        // INDIRECT_BUFFER lets us point `vkCmdDispatchIndirect` at slots
        // inside scratch (flash_attn split-K grid lives here so the
        // recorded cmdbuf is replay-stable).
        //
        // This is a placeholder; callers must call `allocate_scratch` (sized
        // from the model + n_ubatch) before running forwards. We allocate a
        // tiny one up front so `scratch` is always a valid Region.
        let scratch = Region::new(
            &device,
            SCRATCH_PLACEHOLDER_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let readback = new_readback_region(&device, READBACK_PLACEHOLDER_BYTES)?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.device.create_command_pool(&pool_info, None) }?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);
        let cmd_bufs = unsafe { device.device.allocate_command_buffers(&alloc_info) }?;
        let command_buffer = cmd_bufs[0];
        let decode_command_buffer = cmd_bufs[1];

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.device.create_fence(&fence_info, None) }?;

        #[cfg(feature = "profile_gpu")]
        let profile = profile::ProfileRecorder::new(&device)?;

        Ok(Self {
            device,
            pipelines,
            descriptors,
            scratch,
            readback,
            command_pool,
            command_buffer,
            decode_command_buffer,
            fence,
            #[cfg(feature = "profile_gpu")]
            profile,
            decode_cache: None,
            decode_scratch_cursor: 0,
            n_ubatch,
            n_batch,
        })
    }

    /// (Re)allocate the scratch region to `bytes`, replacing the placeholder
    /// from `new`. Sized by the caller from [`crate::models::Model::scratch_bytes_estimate`]
    /// so the compute buffer fits the worst-case forward at the configured
    /// `n_ubatch` / context — llama.cpp-style worst-case reservation. Must be
    /// called once after the model is opened, before any forward.
    pub fn allocate_scratch(&mut self, bytes: u64) -> Result<(), Box<dyn Error>> {
        // No work can be in flight (scratch is replaced wholesale, and any
        // cached decode recording binds the old buffer).
        unsafe { self.device.device.device_wait_idle()? };
        self.decode_cache = None;
        self.scratch.destroy(&self.device.device);
        let bytes = bytes.max(SCRATCH_PLACEHOLDER_BYTES);
        self.scratch = Region::new(
            &self.device,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        tracing::info!(
            scratch_mib = bytes / (1 << 20),
            "scratch region sized for model + n_ubatch",
        );
        Ok(())
    }

    /// Upload every tensor in `gguf` into a new dedicated weights region.
    pub fn upload_weights(&self, gguf: &GgufFile) -> Result<WeightsHandle, Box<dyn Error>> {
        // The device-local upload path stages copies through the engine's
        // command pool / queue / fence (no forward pass is in flight yet).
        weights::upload(
            &self.device,
            gguf,
            self.command_pool,
            self.device.queue,
            self.fence,
        )
    }

    /// Allocate a KV cache sized for the given architecture. Caller picks
    /// dtypes (independently for K and V) and `max_seq_len`.
    pub fn allocate_kv_cache(
        &self,
        n_layer: u32,
        head_dim: u32,
        n_head_kv: u32,
        config: kv_cache::KvCacheConfig,
    ) -> Result<kv_cache::KvCache, Box<dyn Error>> {
        kv_cache::KvCache::new(&self.device, n_layer, head_dim, n_head_kv, config)
    }

    /// Allocate a KV cache whose K/V dims vary per layer (gemma4's interleaved
    /// sliding-window / global attention). `head_dims[il]` / `n_head_kvs[il]`
    /// size layer `il`'s buffer.
    pub fn allocate_kv_cache_per_layer(
        &self,
        head_dims: &[u32],
        n_head_kvs: &[u32],
        config: kv_cache::KvCacheConfig,
    ) -> Result<kv_cache::KvCache, Box<dyn Error>> {
        kv_cache::KvCache::new_per_layer(&self.device, head_dims, n_head_kvs, config)
    }

    /// Run a forward pass: the closure records dispatches into the
    /// `DispatchContext` and returns the `BufferRange` containing the final
    /// logits (vocab_size F32s). The engine handles begin/end/submit/wait
    /// and reads the logits back as a `Vec<f32>`.
    pub fn forward<F>(
        &mut self,
        weights: &WeightsHandle,
        record: F,
    ) -> Result<Vec<f32>, Box<dyn Error>>
    where
        F: FnOnce(&mut DispatchContext) -> Result<BufferRange, Box<dyn Error>>,
    {
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;

        // Reserve the DecodeDyn slot before the model records anything else
        // so its offset is stable (= first alloc after reset). Bindings into
        // this slot must remain valid across replays when the
        // persistent-decode-cmdbuf path lands.
        let decode_dyn_range = {
            let off = self.scratch.alloc(decode_dyn::DecodeDyn::SIZE)?;
            BufferRange {
                buffer: self.scratch.buffer,
                offset: off,
                size: decode_dyn::DecodeDyn::SIZE,
            }
        };

        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.reset(&self.device, self.command_buffer);

        #[cfg(feature = "gpu_debug")]
        let taps;
        let logits_range = {
            let mut ctx = DispatchContext {
                flush: None,
                device: &self.device,
                weights,
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd: self.command_buffer,
                #[cfg(feature = "gpu_debug")]
                taps: Vec::new(),
                #[cfg(feature = "profile_gpu")]
                n_dispatches: 0,
                decode_dyn: decode_dyn_range,
                replay_plan: None,
                dump: None,
                #[cfg(feature = "profile_gpu")]
                profile: Some(&mut self.profile),
            };
            let r = record(&mut ctx)?;
            #[cfg(feature = "gpu_debug")]
            {
                taps = ctx.taps;
            }
            r
        };
        // Closing timestamp — the last marked region (typically LmHead
        // for the bulk `forward` path, since `sampler.record_chain`
        // isn't called here) needs a trailing `t[i+1]` so its duration
        // shows up in the readback pair-walk.
        #[cfg(feature = "profile_gpu")]
        self.profile.mark(
            &self.device,
            self.command_buffer,
            profile::BlockClass::Epilogue,
        );

        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.readback_and_print(&self.device);

        // Read logits back from scratch's host pointer.
        let host_ptr = self.scratch.host_ptr.ok_or(
            "scratch region is not host-visible — readback path requires a staging buffer",
        )?;
        if logits_range.size % 4 != 0 {
            return Err(format!("logits size {} not 4-byte aligned", logits_range.size).into());
        }
        let count = (logits_range.size / 4) as usize;
        let mut out = vec![0f32; count];
        // SAFETY: logits_range refers to a region inside self.scratch (its
        // buffer is self.scratch.buffer) and is fully within its bounds.
        unsafe {
            let src = host_ptr.add(logits_range.offset as usize) as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), count);
        }

        // Print sums for any taps the model recorded. Used for layer-by-layer
        // diff dumps vs llama.cpp's `cb()` callback. Output is one line per
        // tap: `TAP <name> n=<count> sum=<value> max_abs=<value>`. The whole
        // readback path is compiled out without the `gpu_debug` feature.
        #[cfg(feature = "gpu_debug")]
        for (name, range) in &taps {
            if range.size % 4 != 0 {
                eprintln!(
                    "TAP {name}: size {} not 4-byte aligned, skipping",
                    range.size
                );
                continue;
            }
            let n = (range.size / 4) as usize;
            let mut buf = vec![0f32; n];
            unsafe {
                let src = host_ptr.add(range.offset as usize) as *const f32;
                std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), n);
            }
            let sum: f32 = buf.iter().sum();
            let max_abs: f32 = buf.iter().map(|x| x.abs()).fold(0.0, f32::max);
            let head: Vec<String> = buf.iter().take(5).map(|v| format!("{v:.4}")).collect();
            println!(
                "TAP {name} n={n} off={} sum={sum:.6} max_abs={max_abs:.6} head=[{}]",
                range.offset,
                head.join(", ")
            );
        }
        Ok(out)
    }

    /// Batched **decode** step: advance `B = tokens.len()` independent
    /// sequences by one token each in a single forward + per-sequence sample.
    /// `tokens[s]` / `positions[s]` / `samplers[s]` belong to slot `s` of
    /// `batch`. Returns the `B` sampled tokens. The model's
    /// `record_forward_batch` produces `[vocab, B]` logits; each column is
    /// sampled by its own sampler chain and read back.
    pub fn forward_batch_decode(
        &mut self,
        model: &dyn crate::models::Model,
        batch: &mut kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
        samplers: &mut [&mut sample::Sampler],
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        let b = tokens.len();
        if b == 0 {
            return Err("forward_batch_decode: empty batch".into());
        }
        if positions.len() != b || samplers.len() != b || slots.len() != b {
            return Err(
                "forward_batch_decode: tokens/positions/slots/samplers length mismatch".into(),
            );
        }
        // The batched graph shape differs from the single-sequence decode
        // cmdbuf; drop any cached recording (replay is single-sequence only).
        self.decode_cache = None;

        self.scratch.reset();
        self.descriptors.reset(&self.device)?;
        let decode_dyn_range = {
            let off = self.scratch.alloc(decode_dyn::DecodeDyn::SIZE)?;
            BufferRange {
                buffer: self.scratch.buffer,
                offset: off,
                size: decode_dyn::DecodeDyn::SIZE,
            }
        };

        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.reset(&self.device, self.command_buffer);

        let token_ranges: Vec<BufferRange> = {
            let mut ctx = DispatchContext {
                flush: None,
                device: &self.device,
                weights: model.weights(),
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd: self.command_buffer,
                #[cfg(feature = "gpu_debug")]
                taps: Vec::new(),
                #[cfg(feature = "profile_gpu")]
                n_dispatches: 0,
                decode_dyn: decode_dyn_range,
                replay_plan: None,
                dump: None,
                #[cfg(feature = "profile_gpu")]
                profile: Some(&mut self.profile),
            };
            let logits = model.record_forward_batch(&mut ctx, batch, tokens, positions, slots)?;
            let vocab = logits.dims[0];
            let elem = logits.byte_stride[0];
            let mut ranges = Vec::with_capacity(b);
            for (s, sampler) in samplers.iter_mut().enumerate() {
                // Column s of the [vocab, B] logits = sequence s's logits.
                let col = weights::TensorView {
                    buffer: logits.buffer,
                    byte_offset: logits.byte_offset + (s as u64) * vocab * elem,
                    byte_size: vocab * elem,
                    dims: [vocab, 1, 1, 1],
                    byte_stride: [elem, vocab * elem, vocab * elem, vocab * elem],
                    element_stride: [1, vocab, vocab, vocab],
                    dtype: logits.dtype,
                };
                ranges.push(sampler.record_chain(&mut ctx, col)?);
            }
            ranges
        };
        #[cfg(feature = "profile_gpu")]
        self.profile.mark(
            &self.device,
            self.command_buffer,
            profile::BlockClass::Epilogue,
        );

        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.readback_and_print(&self.device);

        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible — readback requires mapped memory")?;
        let mut out = Vec::with_capacity(b);
        for (sampler, r) in samplers.iter_mut().zip(token_ranges.iter()) {
            let token = unsafe { std::ptr::read(host_ptr.add(r.offset as usize) as *const u32) };
            sampler.accept(token);
            out.push(token);
        }
        Ok(out)
    }

    /// Unified varlen forward (M5): `B` sequences contributing `seq_lens[s]`
    /// tokens each (prefill chunk or 1 for decode), packed flat in
    /// `tokens`/`positions` (`N_total = sum`). Runs one
    /// `record_forward_unified` and samples each sequence's last-token column
    /// (`[vocab, B]`). Returns one token per sequence; the caller decides which
    /// to keep (a mid-prefill chunk's token is discarded). The samplers/`slots`
    /// arrays are per-sequence (length `B`), not per-token.
    #[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
    pub fn forward_unified(
        &mut self,
        model: &dyn crate::models::Model,
        batch: &mut kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        seq_lens: &[u32],
        slots: &[u32],
        samplers: &mut [&mut sample::Sampler],
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        let b = seq_lens.len();
        let n_total = tokens.len();
        if b == 0 || n_total == 0 {
            return Err("forward_unified: empty batch".into());
        }
        if positions.len() != n_total || samplers.len() != b || slots.len() != b {
            return Err(
                "forward_unified: tokens/positions/seq_lens/slots/samplers length mismatch".into(),
            );
        }
        // Varlen graph shape differs from the single-sequence decode cmdbuf.
        self.decode_cache = None;

        // Per-step CPU-record vs GPU-compute split (gates the persistent
        // batched-decode cmdbuf — replay only helps if recording is a real
        // fraction of the step). One cheap `Instant::now` per step.
        let t_rec0 = std::time::Instant::now();
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;
        let decode_dyn_range = {
            let off = self.scratch.alloc(decode_dyn::DecodeDyn::SIZE)?;
            BufferRange {
                buffer: self.scratch.buffer,
                offset: off,
                size: decode_dyn::DecodeDyn::SIZE,
            }
        };

        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.reset(&self.device, self.command_buffer);

        let token_ranges: Vec<BufferRange> = {
            let mut ctx = DispatchContext {
                flush: Self::build_flush(self.fence, model.weights()),
                device: &self.device,
                weights: model.weights(),
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd: self.command_buffer,
                #[cfg(feature = "gpu_debug")]
                taps: Vec::new(),
                #[cfg(feature = "profile_gpu")]
                n_dispatches: 0,
                decode_dyn: decode_dyn_range,
                replay_plan: None,
                dump: None,
                #[cfg(feature = "profile_gpu")]
                profile: Some(&mut self.profile),
            };
            let logits = model
                .record_forward_unified(&mut ctx, batch, tokens, positions, seq_lens, slots)?;
            let vocab = logits.dims[0];
            let elem = logits.byte_stride[0];
            let mut ranges = Vec::with_capacity(b);
            for (s, sampler) in samplers.iter_mut().enumerate() {
                let col = weights::TensorView {
                    buffer: logits.buffer,
                    byte_offset: logits.byte_offset + (s as u64) * vocab * elem,
                    byte_size: vocab * elem,
                    dims: [vocab, 1, 1, 1],
                    byte_stride: [elem, vocab * elem, vocab * elem, vocab * elem],
                    element_stride: [1, vocab, vocab, vocab],
                    dtype: logits.dtype,
                };
                ranges.push(sampler.record_chain(&mut ctx, col)?);
            }
            ranges
        };
        #[cfg(feature = "profile_gpu")]
        self.profile.mark(
            &self.device,
            self.command_buffer,
            profile::BlockClass::Epilogue,
        );

        let (t_record, t_gpu) = unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            let t_record = t_rec0.elapsed();
            let t_gpu0 = std::time::Instant::now();
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
            (t_record, t_gpu0.elapsed())
        };
        if *crate::runtime_flags::PROF_STEP {
            let (r, g) = (
                t_record.as_secs_f64() * 1000.0,
                t_gpu.as_secs_f64() * 1000.0,
            );
            eprintln!(
                "PROF step: b={} n_tok={} record={r:.2}ms gpu={g:.2}ms record_frac={:.0}%",
                b,
                tokens.len(),
                100.0 * r / (r + g).max(1e-9),
            );
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.readback_and_print(&self.device);

        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible — readback requires mapped memory")?;
        let mut out = Vec::with_capacity(b);
        for (sampler, r) in samplers.iter_mut().zip(token_ranges.iter()) {
            let token = unsafe { std::ptr::read(host_ptr.add(r.offset as usize) as *const u32) };
            sampler.accept(token);
            out.push(token);
        }
        Ok(out)
    }

    /// Seed `dst_slot`'s slab from a captured leading-prefix `snap`: a one-time
    /// fenced transfer cmdbuf that GPU→GPU copies `KV[0, p)` + the SSM state at
    /// P into the slab. The caller then sets `positions[dst_slot] = p` and
    /// prefills only the divergent suffix `[p, len)`. Byte-identical, in-process,
    /// to the slab having prefilled `[0, p)` itself (SSM state is position-
    /// independent, `KV[0, p)` is contiguous). Invalidates the decode replay
    /// cache (it binds specific slab buffers).
    pub fn seed_slab(
        &mut self,
        batch: &kv_cache::BatchKvCache,
        snap: &kv_cache::PrefixSnapshot,
        dst_slot: u32,
        p: u32,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        batch.record_seed(&self.device, self.command_buffer, snap, dst_slot, p);
        command::record_global_barrier(&self.device, self.command_buffer);
        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        // The cached decode cmdbuf binds specific slab buffers; a seed changes
        // what the next forward must read, so force a fresh recording.
        self.decode_cache = None;
        Ok(())
    }

    /// Capture `src_slot`'s leading prefix `[0, p)` (KV + the SSM state at P)
    /// into `snap`. The slab's live SSM state equals state-at-P only when
    /// `positions[src_slot] == p` (a prefill chunk boundary) — the caller must
    /// guarantee that. A read of the slab; the active slab state is unchanged,
    /// so the decode cache stays valid.
    pub fn capture_prefix(
        &mut self,
        batch: &kv_cache::BatchKvCache,
        snap: &kv_cache::PrefixSnapshot,
        src_slot: u32,
        p: u32,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        batch.record_capture(&self.device, self.command_buffer, snap, src_slot, p);
        command::record_global_barrier(&self.device, self.command_buffer);
        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        Ok(())
    }

    /// Debug: run [`crate::models::Model::record_forward_unified`] with the
    /// release-capable per-op dump enabled, and return `(label, hash)` for each
    /// dumped buffer (FNV-1a over the raw bytes). Run the same input twice and
    /// diff the hashes to find the first op whose output is nondeterministic
    /// across runs — the tool for bisecting the qwen unified release race.
    /// No sampling; the logits are discarded.
    pub fn forward_unified_dump(
        &mut self,
        model: &dyn crate::models::Model,
        batch: &mut kv_cache::BatchKvCache,
        tokens: &[u32],
        positions: &[u32],
        seq_lens: &[u32],
        slots: &[u32],
    ) -> Result<Vec<(String, u64)>, Box<dyn Error>> {
        // Host-visible sink for the dumped buffers (kept alive across submit).
        let cap: u64 = 512 << 20;
        let dump_region = Region::new(
            &self.device,
            cap,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let dump_host = dump_region.host_ptr.ok_or("dump region not host-visible")?;

        self.decode_cache = None;
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;
        let decode_dyn_range = {
            let off = self.scratch.alloc(decode_dyn::DecodeDyn::SIZE)?;
            BufferRange {
                buffer: self.scratch.buffer,
                offset: off,
                size: decode_dyn::DecodeDyn::SIZE,
            }
        };

        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.reset(&self.device, self.command_buffer);

        let records = {
            let mut ctx = DispatchContext {
                flush: None,
                device: &self.device,
                weights: model.weights(),
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd: self.command_buffer,
                #[cfg(feature = "gpu_debug")]
                taps: Vec::new(),
                #[cfg(feature = "profile_gpu")]
                n_dispatches: 0,
                decode_dyn: decode_dyn_range,
                replay_plan: None,
                dump: Some(crate::inference::context::DumpState {
                    buffer: dump_region.buffer,
                    host_ptr: dump_host,
                    capacity: cap,
                    cursor: 0,
                    records: Vec::new(),
                }),
                #[cfg(feature = "profile_gpu")]
                profile: Some(&mut self.profile),
            };
            let _ = model
                .record_forward_unified(&mut ctx, batch, tokens, positions, seq_lens, slots)?;
            ctx.dump.take().expect("dump set").records
        };

        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        // FNV-1a over each record's raw bytes.
        let mut out = Vec::with_capacity(records.len());
        for (label, off, size) in records {
            let bytes =
                unsafe { std::slice::from_raw_parts(dump_host.add(off as usize), size as usize) };
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            out.push((label, h));
        }
        Ok(out)
    }

    /// Run a forward pass and sample a token, all on the GPU.
    ///
    /// Three execution paths, picked from `tokens.len()` and the
    /// validity of the cached decode recording:
    ///
    /// 1. **Prefill (L > 1)**: scratch reset, fresh record on the main
    ///    cmdbuf, single submit. Invalidates any cached decode
    ///    recording (graph shape differs).
    /// 2. **Decode-record (L = 1, cache stale)**: scratch reset, fresh
    ///    record on the decode cmdbuf, captures the input-slot offsets
    ///    + flash_attn grid params into `decode_cache`, submit.
    /// 3. **Decode-replay (L = 1, cache valid)**: skip scratch reset,
    ///    skip recording, host-update the input slots (decode_dyn,
    ///    token_buf, positions_buf, penalty pairs) at the captured
    ///    offsets, resubmit the cached cmdbuf. Drops the ~2.4 ms of
    ///    per-token CPU recording cost.
    ///
    /// The cache is invalidated whenever the sampler's
    /// `SamplerConfig::graph_hash()` changes or the model's
    /// `decode_grid(kv, …)` returns a different `(k_num, blocks_per_split)`
    /// than the one recorded (the wg count is baked into the cmdbuf via
    /// `cmd_update_buffer`).
    pub fn forward_sampled(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        tokens: &[u32],
        position_offset: u32,
        sampler: &mut sample::Sampler,
    ) -> Result<u32, Box<dyn Error>> {
        let prof = crate::runtime_flags::profile_forward();
        let t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        #[cfg(feature = "profile_gpu")]
        if prof {
            crate::inference::command::BARRIER_COMPUTE_COUNT.with(|c| c.set(0));
            crate::inference::command::BARRIER_GLOBAL_COUNT.with(|c| c.set(0));
        }

        let l = tokens.len() as u32;
        if l == 0 {
            return Err("forward_sampled called with empty token list".into());
        }
        // Chunked prefill: a prompt longer than the effective ubatch is fed in
        // sequential passes so the per-pass scratch working set stays bounded
        // regardless of prompt length. Deep-context attention is kept under the
        // RADV watchdog by the coopmat flash-attn kernel (fast enough for a full
        // ubatch in one dispatch), not by shrinking the chunk. Only the final
        // chunk computes logits + samples. An effective ubatch of 0 disables
        // chunking (legacy single-pass).
        let ub = self.effective_ubatch();
        if ub != 0 && l > ub {
            return self.forward_sampled_chunked(model, cache, tokens, position_offset, sampler);
        }
        let is_decode = l == 1;
        let kv_after = position_offset + l;
        let core_count = self.device.shader_core_count;
        let want_grid = if is_decode {
            model.decode_grid(kv_after, core_count)
        } else {
            None
        };
        let want_config_hash = sampler.config().graph_hash();
        // `SEEKER_DECODE_REPLAY=0` (requires the `gpu_debug` feature) forces
        // the legacy record-each-token path — diagnostic only. Default is
        // replay-on; the cached decode cmdbuf saves the ~2.5 ms/token CPU
        // recording cost. `decode_replay_disabled()` constant-folds to
        // `false` in production builds, eliminating the per-token getenv.
        let allow_replay = !crate::runtime_flags::decode_replay_disabled();
        let can_replay = is_decode
            && allow_replay
            && self.decode_cache.as_ref().is_some_and(|c| {
                c.sampler_config_hash == want_config_hash
                    && Some((c.k_num, c.blocks_per_split)) == want_grid
            });

        if can_replay {
            return self.forward_sampled_replay(
                model,
                cache,
                tokens,
                position_offset,
                sampler,
                t0,
                kv_after,
            );
        }
        self.forward_sampled_record(
            model,
            cache,
            tokens,
            position_offset,
            sampler,
            t0,
            is_decode,
            want_grid,
            want_config_hash,
        )
    }

    #[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
    fn forward_sampled_replay(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        tokens: &[u32],
        position_offset: u32,
        sampler: &mut sample::Sampler,
        t0: Option<std::time::Instant>,
        kv_after: u32,
    ) -> Result<u32, Box<dyn Error>> {
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible — replay requires mapped memory")?;
        let cache_state = self
            .decode_cache
            .as_ref()
            .expect("forward_sampled_replay called without a cached recording");
        let plan = cache_state.plan.clone();
        let mc = cache_state.model_constants;
        let kv_len = kv_after;
        let k_num = cache_state.k_num;
        let blocks_per_split = cache_state.blocks_per_split;

        // Update DecodeDyn fields owned by the engine. The sampler
        // refresh below covers uniform_rng + penalty_count; the model
        // doesn't touch decode_dyn directly during replay.
        decode_dyn::write_field(
            host_ptr,
            plan.decode_dyn_offset,
            decode_dyn::OFFSET_KV_LEN,
            kv_len,
        );
        decode_dyn::write_field(
            host_ptr,
            plan.decode_dyn_offset,
            decode_dyn::OFFSET_K_NUM,
            k_num,
        );
        decode_dyn::write_field(
            host_ptr,
            plan.decode_dyn_offset,
            decode_dyn::OFFSET_BLOCKS_PER_SPLIT,
            blocks_per_split,
        );
        let rope_d_offset = position_offset * mc.rope_d_offset_per_position;
        decode_dyn::write_field(
            host_ptr,
            plan.decode_dyn_offset,
            decode_dyn::OFFSET_ROPE_D_OFFSET,
            rope_d_offset,
        );
        let v_cache_d_offset = position_offset * mc.v_cache_d_offset_per_position;
        decode_dyn::write_field(
            host_ptr,
            plan.decode_dyn_offset,
            decode_dyn::OFFSET_V_CACHE_D_OFFSET,
            v_cache_d_offset,
        );

        model.refresh_replay_inputs(host_ptr, &plan, tokens, position_offset)?;
        sampler.refresh_replay_inputs(host_ptr, &plan)?;

        let t_record = t0.map(|t| t.elapsed());

        unsafe {
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.decode_command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        let t_wait = t0.map(|t| t.elapsed());
        #[cfg(feature = "profile_gpu")]
        self.profile.readback_and_print(&self.device);

        let sampler_output_offset = plan
            .sampler_output_offset
            .ok_or("replay plan missing sampler_output_offset")?;
        let token = unsafe {
            let src = host_ptr.add(sampler_output_offset as usize) as *const u32;
            std::ptr::read(src)
        };

        crate::inference::ops::cache_io::advance(cache, tokens.len() as u32);
        sampler.accept(token);

        if let (Some(rec), Some(wait)) = (t_record, t_wait) {
            let total = t0.unwrap().elapsed();
            #[cfg(feature = "profile_gpu")]
            let counts = {
                let bc = crate::inference::command::BARRIER_COMPUTE_COUNT.with(|c| c.get());
                let bg = crate::inference::command::BARRIER_GLOBAL_COUNT.with(|c| c.get());
                format!("barriers=(compute={bc} global={bg}) ")
            };
            #[cfg(not(feature = "profile_gpu"))]
            let counts = "";
            eprintln!(
                "PROF forward[replay]: {counts}record={:.2}ms gpu_wait={:.2}ms readback={:.2}ms total={:.2}ms",
                rec.as_secs_f64() * 1000.0,
                (wait - rec).as_secs_f64() * 1000.0,
                (total - wait).as_secs_f64() * 1000.0,
                total.as_secs_f64() * 1000.0,
            );
        }
        Ok(token)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_sampled_record(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        tokens: &[u32],
        position_offset: u32,
        sampler: &mut sample::Sampler,
        t0: Option<std::time::Instant>,
        is_decode: bool,
        want_grid: Option<(u32, u32)>,
        want_config_hash: u64,
    ) -> Result<u32, Box<dyn Error>> {
        // Want to cache the decode recording? Only if the model opts
        // in (replay_constants() returns Some) AND the sampler chain's
        // graph is stable across replays (no L>1 mask alloc).
        let cache_recording = is_decode && model.replay_constants().is_some();
        let cmd = if cache_recording {
            self.decode_command_buffer
        } else {
            self.command_buffer
        };

        // Either path resets everything: a stale `decode_cache` would
        // bind to scratch ranges we're about to recycle.
        self.decode_cache = None;
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;

        let decode_dyn_range = {
            let off = self.scratch.alloc(decode_dyn::DecodeDyn::SIZE)?;
            BufferRange {
                buffer: self.scratch.buffer,
                offset: off,
                size: decode_dyn::DecodeDyn::SIZE,
            }
        };

        unsafe {
            self.device
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            // A cached decode recording is resubmitted across many tokens
            // by `forward_sampled_replay` without re-recording, so it must
            // NOT carry ONE_TIME_SUBMIT (which promises a single submit and
            // trips the validation layer on every replay). Replays are
            // serial and fence-synchronized, so empty flags suffice — no
            // SIMULTANEOUS_USE. The non-cached path submits once: keep the
            // ONE_TIME_SUBMIT hint so the driver can optimize.
            let flags = if cache_recording {
                vk::CommandBufferUsageFlags::empty()
            } else {
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            };
            let begin = vk::CommandBufferBeginInfo::default().flags(flags);
            self.device.device.begin_command_buffer(cmd, &begin)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.reset(&self.device, cmd);

        let replay_plan_init = if cache_recording {
            Some(decode_dyn::ReplayPlan {
                decode_dyn_offset: decode_dyn_range.offset,
                ..Default::default()
            })
        } else {
            None
        };

        let captured_plan;
        #[cfg(feature = "gpu_debug")]
        let taps;
        #[cfg(feature = "profile_gpu")]
        let n_dispatches;
        let token_range = {
            let mut ctx = DispatchContext {
                flush: if cache_recording {
                    None
                } else {
                    Self::build_flush(self.fence, model.weights())
                },
                device: &self.device,
                weights: model.weights(),
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd,
                #[cfg(feature = "gpu_debug")]
                taps: Vec::new(),
                #[cfg(feature = "profile_gpu")]
                n_dispatches: 0,
                decode_dyn: decode_dyn_range,
                replay_plan: replay_plan_init,
                dump: None,
                #[cfg(feature = "profile_gpu")]
                profile: Some(&mut self.profile),
            };
            let logits = model
                .record_forward(
                    &mut ctx,
                    cache,
                    tokens,
                    position_offset,
                    /*compute_logits=*/ true,
                )?
                .ok_or("record_forward(compute_logits=true) returned no logits")?;
            let r = sampler.record_chain(&mut ctx, logits)?;
            #[cfg(feature = "gpu_debug")]
            {
                taps = ctx.taps;
            }
            #[cfg(feature = "profile_gpu")]
            {
                n_dispatches = ctx.n_dispatches;
            }
            captured_plan = ctx.replay_plan;
            r
        };
        #[cfg(feature = "profile_gpu")]
        self.profile
            .mark(&self.device, cmd, profile::BlockClass::Sampler);
        let t_record = t0.map(|t| t.elapsed());

        unsafe {
            self.device.device.end_command_buffer(cmd)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        let t_wait = t0.map(|t| t.elapsed());
        #[cfg(feature = "profile_gpu")]
        self.profile.readback_and_print(&self.device);

        if token_range.size < 4 {
            return Err(format!("sampler output too small: {} bytes", token_range.size).into());
        }
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region is not host-visible — readback requires host-visible scratch")?;
        let token = unsafe {
            let src = host_ptr.add(token_range.offset as usize) as *const u32;
            std::ptr::read(src)
        };
        #[cfg(feature = "gpu_debug")]
        for (name, range) in &taps {
            if range.size % 4 != 0 {
                eprintln!(
                    "TAP {name}: size {} not 4-byte aligned, skipping",
                    range.size
                );
                continue;
            }
            let n = (range.size / 4) as usize;
            let mut buf = vec![0f32; n];
            unsafe {
                let src = host_ptr.add(range.offset as usize) as *const f32;
                std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), n);
            }
            let sum: f32 = buf.iter().sum();
            let max_abs: f32 = buf.iter().map(|x| x.abs()).fold(0.0, f32::max);
            let head: Vec<String> = buf.iter().take(5).map(|v| format!("{v:.4}")).collect();
            println!(
                "TAP {name} n={n} off={} sum={sum:.6} max_abs={max_abs:.6} head=[{}]",
                range.offset,
                head.join(", ")
            );
        }
        sampler.accept(token);

        // Stash the recording so subsequent matching decodes can replay.
        if cache_recording
            && let (Some((k_num, blocks_per_split)), Some(mc), Some(plan)) =
                (want_grid, model.replay_constants(), captured_plan)
            && plan.token_buf_offset.is_some()
            && plan.positions_buf_offset.is_some()
            && plan.sampler_output_offset.is_some()
        {
            let kv_after = position_offset + tokens.len() as u32;
            self.decode_cache = Some(DecodeCache {
                sampler_config_hash: want_config_hash,
                shape_key: 0,
                kv_len: kv_after,
                k_num,
                blocks_per_split,
                plan,
                model_constants: mc,
            });
            self.decode_scratch_cursor = self.scratch.cursor;
        }

        if let (Some(rec), Some(wait)) = (t_record, t_wait) {
            let total = t0.unwrap().elapsed();
            #[cfg(feature = "profile_gpu")]
            let counts = {
                let bc = crate::inference::command::BARRIER_COMPUTE_COUNT.with(|c| c.get());
                let bg = crate::inference::command::BARRIER_GLOBAL_COUNT.with(|c| c.get());
                format!("dispatches={n_dispatches} barriers=(compute={bc} global={bg}) ")
            };
            #[cfg(not(feature = "profile_gpu"))]
            let counts = "";
            eprintln!(
                "PROF forward: {counts}record={:.2}ms gpu_wait={:.2}ms readback={:.2}ms total={:.2}ms",
                rec.as_secs_f64() * 1000.0,
                (wait - rec).as_secs_f64() * 1000.0,
                (total - wait).as_secs_f64() * 1000.0,
                total.as_secs_f64() * 1000.0,
            );
        }
        Ok(token)
    }

    /// Prefill a multimodal prompt (one image) and sample the first token.
    /// Splices the vision-tower embeddings into the decoder residual at the
    /// `<|image_pad|>` rows and uses the qwen-vl M-RoPE positions (via
    /// [`crate::models::Model::record_forward_image_chunk`]).
    ///
    /// **Chunked**, like text prefill: the prompt is fed in `≤ n_ubatch`-token
    /// passes (each its own submit), so a large image (thousands of pad tokens)
    /// doesn't run as one oversized forward — which exceeds the GPU's compute
    /// watchdog and loses the device. Every chunk but the last is KV-only; the
    /// final chunk samples. The image-pad columns are split across whichever
    /// chunks they fall in; the M-RoPE positions are built globally so the
    /// image's 2D cursor stays continuous across chunk boundaries (numerically
    /// identical to a single full-prompt pass). `n_ubatch == 0` ⇒ single pass.
    ///
    /// After the whole prefill, `cache.rope_position_lag` is advanced once by
    /// `n_tok - max(nx,ny)` (the image consumed `n_tok` KV slots but advanced the
    /// logical cursor by only `max(nx,ny)`); subsequent decode forwards read it.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_image_sampled(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        tokens: &[u32],
        image_embeddings: &[f32],
        image_start: usize,
        image_nx: usize,
        image_ny: usize,
        sampler: &mut sample::Sampler,
    ) -> Result<u32, Box<dyn Error>> {
        if tokens.is_empty() {
            return Err("forward_image_sampled called with empty token list".into());
        }
        let prompt_pos0 = cache.position;
        let n = tokens.len();
        let eub = self.effective_ubatch();
        let ub = if eub == 0 { n } else { eub as usize };

        let mut token = 0u32;
        let mut start = 0usize;
        while start < n {
            let end = (start + ub).min(n);
            let is_final = end == n;
            let out = self.record_image_chunk(
                model,
                cache,
                &tokens[start..end],
                start,
                image_embeddings,
                image_start,
                image_nx,
                image_ny,
                prompt_pos0,
                if is_final { Some(&mut *sampler) } else { None },
            )?;
            if let Some(t) = out {
                token = t;
            }
            start = end;
        }
        // The image consumed `n_tok = nx*ny` KV slots but advanced the logical
        // M-RoPE cursor by only `max(nx,ny)`; subsequent (decode) forwards read
        // this lag to keep their positions continuous past the image. gemma4 uses
        // sequential 1D positions (image advances the cursor 1:1), so no lag.
        if model.image_uses_mrope() {
            let n_tok = image_nx * image_ny;
            cache.rope_position_lag += (n_tok - image_nx.max(image_ny)) as u32;
        }
        Ok(token)
    }

    /// Prefill an audio-containing prompt — the `gemma4ua` analog of
    /// [`Self::forward_image_sampled`]. The `<|audio|>`-placeholder columns
    /// (`audio_start..audio_start + n_audio_tok`) are overwritten by the audio
    /// encoder's projected frames (see [`crate::audio::encoder`]).
    ///
    /// Audio tokens use plain sequential 1D positions (no M-RoPE), so we feed the
    /// clip as a **1×N media grid** (`nx = n_audio_tok`, `ny = 1`) through the
    /// shared, validated image-prefill path: the residual splice and chunked
    /// submission are modality-agnostic, and the post-prefill rope-lag term
    /// `n_tok − max(nx, ny) = n_audio_tok − n_audio_tok = 0` regardless of
    /// `image_uses_mrope`, so no lag is ever introduced. The encoder's
    /// `proj_dim == n_embd` requirement is enforced by the splice length check in
    /// the model's `record_forward_image_chunk`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_audio_sampled(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        tokens: &[u32],
        audio_embeddings: &[f32],
        audio_start: usize,
        n_audio_tok: usize,
        sampler: &mut sample::Sampler,
    ) -> Result<u32, Box<dyn Error>> {
        self.forward_image_sampled(
            model,
            cache,
            tokens,
            audio_embeddings,
            audio_start,
            n_audio_tok,
            1,
            sampler,
        )
    }

    /// Record + submit ONE chunk of an image prefill (see
    /// [`Self::forward_image_sampled`]). `sampler.is_some()` ⇒ final chunk:
    /// record the sampler chain, read back and return the sampled token. `None`
    /// ⇒ KV-only chunk (no epilogue/sampler/readback), returns `None`.
    #[allow(clippy::too_many_arguments)]
    fn record_image_chunk(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        chunk_tokens: &[u32],
        chunk_global_start: usize,
        image_embeddings: &[f32],
        image_global_start: usize,
        image_nx: usize,
        image_ny: usize,
        prompt_pos0: u32,
        mut sampler: Option<&mut sample::Sampler>,
    ) -> Result<Option<u32>, Box<dyn Error>> {
        let compute_logits = sampler.is_some();

        // A prefill changes the graph shape — invalidate any cached decode
        // recording so a stale binding can't survive into the next decode.
        self.decode_cache = None;
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;

        let decode_dyn_range = {
            let off = self.scratch.alloc(decode_dyn::DecodeDyn::SIZE)?;
            BufferRange {
                buffer: self.scratch.buffer,
                offset: off,
                size: decode_dyn::DecodeDyn::SIZE,
            }
        };

        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.reset(&self.device, self.command_buffer);

        #[cfg(feature = "gpu_debug")]
        let taps: Vec<(String, BufferRange)>;
        let token_range = {
            let mut ctx = DispatchContext {
                flush: Self::build_flush(self.fence, model.weights()),
                device: &self.device,
                weights: model.weights(),
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd: self.command_buffer,
                #[cfg(feature = "gpu_debug")]
                taps: Vec::new(),
                #[cfg(feature = "profile_gpu")]
                n_dispatches: 0,
                decode_dyn: decode_dyn_range,
                replay_plan: None,
                dump: None,
                #[cfg(feature = "profile_gpu")]
                profile: Some(&mut self.profile),
            };
            let logits = model.record_forward_image_chunk(
                &mut ctx,
                cache,
                chunk_tokens,
                chunk_global_start,
                image_embeddings,
                image_global_start,
                image_nx,
                image_ny,
                prompt_pos0,
                compute_logits,
            )?;
            let tr = if let Some(s) = sampler.as_deref_mut() {
                let logits =
                    logits.ok_or("record_forward_image_chunk: no logits on final chunk")?;
                Some(s.record_chain(&mut ctx, logits)?)
            } else {
                None
            };
            #[cfg(feature = "gpu_debug")]
            {
                taps = ctx.taps;
            }
            tr
        };

        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.readback_and_print(&self.device);

        #[cfg(feature = "gpu_debug")]
        if let Some(host_ptr) = self.scratch.host_ptr {
            for (name, range) in &taps {
                if range.size % 4 != 0 {
                    eprintln!(
                        "TAP {name}: size {} not 4-byte aligned, skipping",
                        range.size
                    );
                    continue;
                }
                let n = (range.size / 4) as usize;
                let mut buf = vec![0f32; n];
                unsafe {
                    let src = host_ptr.add(range.offset as usize) as *const f32;
                    std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), n);
                }
                let sum: f32 = buf.iter().sum();
                let max_abs: f32 = buf.iter().map(|x| x.abs()).fold(0.0, f32::max);
                let head: Vec<String> = buf.iter().take(5).map(|v| format!("{v:.4}")).collect();
                println!(
                    "TAP {name} n={n} off={} sum={sum:.6} max_abs={max_abs:.6} head=[{}]",
                    range.offset,
                    head.join(", ")
                );
            }
        }

        // `record_forward_image_chunk` advanced `cache.position`; the caller
        // sets `rope_position_lag` once after all chunks.
        let Some(token_range) = token_range else {
            return Ok(None);
        };
        if token_range.size < 4 {
            return Err(format!("sampler output too small: {} bytes", token_range.size).into());
        }
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region is not host-visible — readback requires host-visible scratch")?;
        let token = unsafe {
            let src = host_ptr.add(token_range.offset as usize) as *const u32;
            std::ptr::read(src)
        };
        if let Some(s) = sampler {
            s.accept(token);
        }
        Ok(Some(token))
    }

    /// Build the prefill submission-splitting budget ([`context::FlushState`])
    /// for a forward that records into a freshly-begun command buffer, or `None`
    /// when flushing is disabled (`SEEKER_PREFILL_FLUSH=0`). Budget =
    /// `min(100 MB, total_weight_bytes / 40)` (overridable via
    /// `SEEKER_PREFILL_FLUSH_MB`) — a *bytes-of-weight-read* budget, which is a
    /// time budget in the one unit ~independent of quant. Associated (no `&self`
    /// borrow) so it can be evaluated inline in a `DispatchContext` literal
    /// alongside `&mut self.scratch`; `fence` is `Copy`.
    fn build_flush(
        fence: vk::Fence,
        weights: &weights::WeightsHandle,
    ) -> Option<context::FlushState> {
        if *crate::runtime_flags::PREFILL_FLUSH_DISABLED {
            return None;
        }
        let budget_bytes = match *crate::runtime_flags::PREFILL_FLUSH_MB {
            Some(mb) => (mb as u64) * 1024 * 1024,
            None => (100u64 * 1024 * 1024).min((weights.total_bytes / 40).max(1)),
        };
        Some(context::FlushState {
            fence,
            budget_bytes: budget_bytes.max(1),
            node_budget: 100,
            bytes_since_flush: 0,
            nodes_since_flush: 0,
        })
    }

    /// The prefill micro-batch: the configured [`Self::n_ubatch`] (0 =
    /// single-pass). The GPU TDR watchdog is handled by mid-forward submission
    /// splitting ([`build_flush`](Self::build_flush)), not by clamping the
    /// micro-batch — token-chunking now bounds only the per-chunk scratch.
    fn effective_ubatch(&self) -> u32 {
        self.n_ubatch
    }

    /// Prefill a prompt longer than the effective ubatch by feeding it in
    /// sequential chunks. Every chunk but the last is KV-only (no
    /// logits / sampler — see [`Engine::forward_kv_only`]); the final chunk
    /// runs the normal sampling path and returns the next token.
    ///
    /// Each chunk passes `position_offset = cache.position`, which the model
    /// advances by the chunk length inside `record_forward`. Because the KV
    /// writes, RoPE positions, causal mask, and persistent SSM/conv/GDN state
    /// are all keyed off the absolute position, feeding ordered chunks is
    /// numerically identical to a single full-prompt pass — only the peak
    /// scratch differs (bounded to one chunk instead of the whole prompt).
    fn forward_sampled_chunked(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        tokens: &[u32],
        position_offset: u32,
        sampler: &mut sample::Sampler,
    ) -> Result<u32, Box<dyn Error>> {
        if cache.position != position_offset {
            return Err(format!(
                "forward_sampled_chunked: cache.position {} != position_offset {position_offset}",
                cache.position
            )
            .into());
        }
        let ub = self.effective_ubatch() as usize;
        let n = tokens.len();
        debug_assert!(ub > 0 && n > ub, "chunked path entered with n={n} ub={ub}");
        let mut start = 0usize;
        loop {
            let end = (start + ub).min(n);
            let chunk = &tokens[start..end];
            let pos = cache.position; // advanced by each record_forward
            if end == n {
                // Final chunk: sample. Recurse — `chunk.len() <= n_ubatch`, so
                // this never re-enters the chunking branch; and when the
                // remainder is a single token it correctly takes the decode
                // path and (re)builds the persistent-decode replay cache.
                return self.forward_sampled(model, cache, chunk, pos, sampler);
            }
            self.forward_kv_only(model, cache, chunk, pos)?;
            start = end;
        }
    }

    /// Record + submit a KV-only prefill pass for one ubatch chunk: runs the
    /// model with `compute_logits = false` (no final norm / lm_head / sampler /
    /// readback), populating the KV + recurrent state for `tokens` at
    /// `position_offset`. Used by [`Engine::forward_sampled_chunked`] for every
    /// chunk except the last, and by `bench --dump-logits` for long prompts.
    ///
    /// Each call is its own submit + fence-wait: scratch is reused across
    /// chunks, so the next chunk must not record until this chunk's GPU work
    /// (including the persistent conv/GDN state copy-back) has completed.
    pub(crate) fn forward_kv_only(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        tokens: &[u32],
        position_offset: u32,
    ) -> Result<(), Box<dyn Error>> {
        // A stale cached decode recording would bind scratch ranges we're
        // about to recycle.
        self.decode_cache = None;
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;

        // Reserve the DecodeDyn slot first (stable offset), exactly like the
        // record path — flash_attn reads `DecodeDyn::kv_len` from it, so it is
        // required even though no sampler runs.
        let decode_dyn_range = {
            let off = self.scratch.alloc(decode_dyn::DecodeDyn::SIZE)?;
            BufferRange {
                buffer: self.scratch.buffer,
                offset: off,
                size: decode_dyn::DecodeDyn::SIZE,
            }
        };

        let cmd = self.command_buffer;
        unsafe {
            self.device
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.device.begin_command_buffer(cmd, &begin)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.reset(&self.device, cmd);

        {
            let mut ctx = DispatchContext {
                flush: Self::build_flush(self.fence, model.weights()),
                device: &self.device,
                weights: model.weights(),
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd,
                #[cfg(feature = "gpu_debug")]
                taps: Vec::new(),
                #[cfg(feature = "profile_gpu")]
                n_dispatches: 0,
                decode_dyn: decode_dyn_range,
                replay_plan: None,
                dump: None,
                #[cfg(feature = "profile_gpu")]
                profile: Some(&mut self.profile),
            };
            // KV-only: compute_logits = false ⇒ no epilogue, no sampler, no
            // readback. Returns None.
            let _ = model.record_forward(
                &mut ctx,
                cache,
                tokens,
                position_offset,
                /*compute_logits=*/ false,
            )?;
        }

        unsafe {
            self.device.device.end_command_buffer(cmd)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.readback_and_print(&self.device);
        Ok(())
    }

    /// Write F32 data into a scratch slot via the mapped host pointer. Used
    /// for inputs that originate on the CPU side (token id positions, etc.).
    pub fn write_scratch_f32(
        &self,
        range: BufferRange,
        data: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible")?;
        let bytes = std::mem::size_of_val(data);
        if bytes as u64 > range.size {
            return Err(format!("write_scratch_f32: {bytes} > range.size {}", range.size).into());
        }
        unsafe {
            let dst = host_ptr.add(range.offset as usize) as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }

    /// Record one closure into a fresh command buffer, submit, wait, and
    /// read back the requested scratch ranges as `Vec<f32>` each. Used by
    /// the MTP speculative path, which needs fine-grained control over
    /// multiple forwards per token-step with host decisions in between.
    ///
    /// Resets scratch + descriptors (so it invalidates any cached decode
    /// recording) and reserves the `DecodeDyn` slot first (offset stable),
    /// exactly like [`forward`]. The closure records dispatches and returns
    /// the ranges to read back (logits, hidden states, …).
    /// Grow the HOST_CACHED readback staging buffer to hold at least `bytes`.
    /// Safe to replace wholesale here: `run_spec_record` waits on its fence each
    /// call, so no in-flight command buffer references the old region.
    fn ensure_readback(&mut self, bytes: u64) -> Result<(), Box<dyn Error>> {
        if self.readback.size >= bytes {
            return Ok(());
        }
        let new = new_readback_region(&self.device, bytes.max(self.readback.size * 2))?;
        let mut old = std::mem::replace(&mut self.readback, new);
        old.destroy(&self.device.device);
        Ok(())
    }

    fn run_spec_record<F>(
        &mut self,
        weights: &WeightsHandle,
        record: F,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>>
    where
        F: FnOnce(&mut DispatchContext) -> Result<Vec<BufferRange>, Box<dyn Error>>,
    {
        self.decode_cache = None;
        self.scratch.reset();
        self.descriptors.reset(&self.device)?;

        let decode_dyn_range = {
            let off = self.scratch.alloc(decode_dyn::DecodeDyn::SIZE)?;
            BufferRange {
                buffer: self.scratch.buffer,
                offset: off,
                size: decode_dyn::DecodeDyn::SIZE,
            }
        };

        unsafe {
            self.device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.reset(&self.device, self.command_buffer);

        let ranges = {
            let mut ctx = DispatchContext {
                flush: None,
                device: &self.device,
                weights,
                scratch: &mut self.scratch,
                pipelines: &mut self.pipelines,
                descriptors: &self.descriptors,
                cmd: self.command_buffer,
                #[cfg(feature = "gpu_debug")]
                taps: Vec::new(),
                #[cfg(feature = "profile_gpu")]
                n_dispatches: 0,
                decode_dyn: decode_dyn_range,
                replay_plan: None,
                dump: None,
                #[cfg(feature = "profile_gpu")]
                profile: Some(&mut self.profile),
            };
            record(&mut ctx)?
        };

        // Stage results into the HOST_CACHED readback buffer so the host read
        // below is cache-speed, not write-combined (~185 MB/s on this UMA APU) —
        // matters for the stochastic spec verify's full-vocab logit readback.
        // Pack the ranges contiguously and copy each (after a compute→transfer
        // barrier) before submitting; the host then reads the cached copy.
        let packed: Vec<u64> = {
            let total: u64 = ranges.iter().map(|r| r.size).sum();
            self.ensure_readback(total.max(1))?;
            command::record_global_barrier(&self.device, self.command_buffer);
            let mut offs = Vec::with_capacity(ranges.len());
            let mut dst = 0u64;
            for r in &ranges {
                let copy = vk::BufferCopy::default()
                    .src_offset(r.offset)
                    .dst_offset(dst)
                    .size(r.size);
                unsafe {
                    self.device.device.cmd_copy_buffer(
                        self.command_buffer,
                        r.buffer,
                        self.readback.buffer,
                        std::slice::from_ref(&copy),
                    );
                }
                offs.push(dst);
                dst += r.size;
            }
            offs
        };

        unsafe {
            self.device.device.end_command_buffer(self.command_buffer)?;
            self.device.device.reset_fences(&[self.fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.device
                .device
                .queue_submit(self.device.queue, &[submit], self.fence)?;
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        #[cfg(feature = "profile_gpu")]
        self.profile.readback_and_print(&self.device);

        let host_ptr = self
            .readback
            .host_ptr
            .ok_or("readback region is not host-visible")?;
        let mut out = Vec::with_capacity(ranges.len());
        for (i, r) in ranges.iter().enumerate() {
            if r.size % 4 != 0 {
                return Err(
                    format!("spec readback range size {} not 4-byte aligned", r.size).into(),
                );
            }
            let n = (r.size / 4) as usize;
            let mut buf = vec![0f32; n];
            // SAFETY: the result was copied to `packed[i]` in the cached readback
            // buffer (sized by ensure_readback); offset+size are in bounds.
            unsafe {
                let src = host_ptr.add(packed[i] as usize) as *const f32;
                std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), n);
            }
            out.push(buf);
        }
        Ok(out)
    }

    /// Run one MTP speculative-decode step: draft `n_max` tokens with the
    /// model's NextN head, verify them with a single batched main-model
    /// forward, accept the longest prefix via lossless host sample-and-
    /// compare, and roll back the KV + SSM state for rejected drafts.
    ///
    /// Preconditions: `cache.position == position`; `last_token` is the
    /// most recently sampled token (not yet in the cache); `h_last` is the
    /// main model's pre-`output_norm` hidden state at position
    /// `position - 1` (length `n_embd`). The model must
    /// [`support MTP spec`](crate::models::Model::supports_mtp_spec).
    ///
    /// On return, all `emitted` tokens are committed and `cache.position`
    /// has advanced to `position + emitted.len()`.
    #[allow(clippy::too_many_arguments)] // high-arity by nature (dims/buffers/flags)
    pub fn decode_speculative(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        last_token: u32,
        h_last: &[f32],
        position: u32,
        sampler: &mut sample::Sampler,
        n_max: u32,
    ) -> Result<SpecStepOut, Box<dyn Error>> {
        let weights = model.weights();
        let hidden = h_last.len();
        let vocab = model.vocab_size() as usize;
        let n = n_max.clamp(1, 8) as usize;

        // ── 1. Draft N tokens with the MTP head (greedy argmax). Each step
        //       is its own tiny submit so we can argmax on the host and feed
        //       the recurrence forward.
        let dbg = std::env::var("SEEKER_SPEC_DEBUG").is_ok();
        let t_draft = std::time::Instant::now();
        let mut drafts: Vec<u32> = Vec::with_capacity(n);
        let mut h: Vec<f32> = h_last.to_vec();
        let mut tok = last_token;
        for k in 0..n as u32 {
            // Absolute MTP position: draft 0 is at position-1 (consuming the
            // last committed hidden + token), so it attends over the full
            // seeded/committed MTP KV [0, position-1).
            let mtp_pos = (position + k).saturating_sub(1);
            let outs = self.run_spec_record(weights, |ctx| {
                let d = model.record_mtp_draft(ctx, cache, &h, tok, mtp_pos)?;
                Ok(vec![d.draft_token, d.block_out.range()])
            })?;
            // outs[0] is the 4-byte u32 token id read back as f32 bits.
            let draft_k = outs[0][0].to_bits();
            drafts.push(draft_k);
            h = outs[1].clone();
            tok = draft_k;
        }
        let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;

        // ── 2. Verify: one batched main forward over [last_token, drafts…],
        //       in CHECKPOINT mode so each SSM layer emits per-position
        //       recurrent-state snapshots (instead of writing the live
        //       state). Partial acceptance then rolls back by committing one
        //       snapshot in finalize — no re-run.
        let verify_tokens: Vec<u32> = std::iter::once(last_token)
            .chain(drafts.iter().copied())
            .collect();
        // Greedy + no penalties → argmax each verify position on the GPU and
        // read back only N+1 token ids (4 bytes each), avoiding the ~5 MB
        // write-combined logits readback. Lossless and identical to the
        // non-spec greedy path. Stochastic / penalty configs take the host
        // sample-and-compare path (reads back full logits).
        let greedy_verify = sampler.config().is_greedy() && !sampler.config().any_penalty();
        let t_verify = std::time::Instant::now();
        let (emitted, residual): (Vec<u32>, Vec<f32>) = if greedy_verify {
            let vocab_u = vocab as u64;
            let outs = self.run_spec_record(weights, |ctx| {
                let o =
                    model.record_forward_full(ctx, cache, &verify_tokens, position, true, true)?;
                let logits = o.logits.expect("record_forward_full computes logits");
                let mut ranges = Vec::with_capacity(n + 2);
                for i in 0..=n {
                    let col = crate::inference::weights::TensorView {
                        buffer: logits.buffer,
                        byte_offset: logits.byte_offset + i as u64 * vocab_u * 4,
                        byte_size: vocab_u * 4,
                        dims: [vocab_u, 1, 1, 1],
                        byte_stride: [4, vocab_u * 4, vocab_u * 4, vocab_u * 4],
                        element_stride: [1, vocab_u, vocab_u, vocab_u],
                        dtype: crate::gguf::GgmlType::F32,
                    };
                    ranges.push(crate::inference::ops::sampler::record_greedy(ctx, col)?);
                }
                ranges.push(o.residual.range());
                Ok(ranges)
            })?;
            let main_argmax: Vec<u32> = (0..=n).map(|i| outs[i][0].to_bits()).collect();
            // Greedy sample-and-compare on host (no logits needed).
            let mut emitted = Vec::with_capacity(n + 1);
            for i in 0..=n {
                let s = main_argmax[i];
                sampler.accept(s);
                emitted.push(s);
                if i < n && s == drafts[i] {
                    continue;
                }
                break;
            }
            (emitted, outs[n + 1].clone())
        } else {
            let outs = self.run_spec_record(weights, |ctx| {
                let o =
                    model.record_forward_full(ctx, cache, &verify_tokens, position, true, true)?;
                Ok(vec![
                    o.logits
                        .expect("record_forward_full computes logits")
                        .range(),
                    o.residual.range(),
                ])
            })?;
            let all_logits = &outs[0]; // [vocab, n+1] (row i at i*vocab)
            let logit_rows: Vec<Vec<f32>> = (0..=n)
                .map(|i| all_logits[i * vocab..(i + 1) * vocab].to_vec())
                .collect();
            (
                sampler.sample_and_compare(&logit_rows, &drafts),
                outs[1].clone(),
            )
        };
        let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
        let residual = &residual; // [n_embd, n+1] contiguous (col i at i*hidden)

        let accept_len = emitted.len() - 1; // # accepted drafts
        let last_committed = *emitted.last().unwrap();

        if dbg {
            eprintln!(
                "SPEC dbg: pos={position} last_token={last_token} drafts={drafts:?} emitted={emitted:?}",
            );
        }

        // ── 4. Commit. Truncate the main K/V to the accepted length (stale
        //       draft K/V is never read), then FINALIZE the SSM state by
        //       committing the per-position snapshot at the accepted
        //       position — no re-run. The verify advanced cache.position
        //       through all N+1 tokens; roll it back to new_pos.
        let new_pos = position + accept_len as u32 + 1;
        cache.truncate(new_pos);
        let h_last_out = residual[accept_len * hidden..(accept_len + 1) * hidden].to_vec();
        let t_fin = std::time::Instant::now();
        let needs_finalize = !cache.ssm_gdn_snapshots.is_empty();
        if needs_finalize {
            self.run_spec_record(weights, |ctx| {
                model.record_ssm_finalize(ctx, cache, accept_len as u32)?;
                Ok(vec![])
            })?;
        }
        debug_assert_eq!(
            cache.position, new_pos,
            "spec step left cache.position wrong"
        );
        if dbg {
            let fin_ms = t_fin.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "SPEC time: draft={draft_ms:.1}ms (×{n}) verify={verify_ms:.1}ms finalize={fin_ms:.1}ms accepted={accept_len}/{n}",
            );
        }

        Ok(SpecStepOut {
            emitted,
            accepted: accept_len as u32,
            h_last: h_last_out,
            last_token: last_committed,
        })
    }

    /// Run [`Model::record_forward_full`] and read back the logits plus the
    /// per-position pre-`output_norm` hidden states. With `full_logits=false`
    /// the returned logits is just the final position's vocab row
    /// (`[vocab]`); otherwise all `L` rows concatenated (`[vocab*L]`). The
    /// second return is the full residual `[n_embd*L]` (position-major); the
    /// caller slices the last column for `h_last` and the prefix for the MTP
    /// seed. Advances `cache.position` by `tokens.len()`.
    pub fn forward_full_readback(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        tokens: &[u32],
        position: u32,
        full_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let weights = model.weights();
        let outs = self.run_spec_record(weights, |ctx| {
            let o = model.record_forward_full(ctx, cache, tokens, position, full_logits, false)?;
            Ok(vec![
                o.logits
                    .expect("record_forward_full computes logits")
                    .range(),
                o.residual.range(),
            ])
        })?;
        let logits = outs[0].clone();
        let residual = outs[1].clone();
        Ok((logits, residual))
    }

    /// Populate the MTP draft head's KV cache for `tokens.len()` positions
    /// from the given main-model hidden states (see
    /// [`Model::record_mtp_seed`]). Used to seed the head from the prompt
    /// after prefill so drafting attends to real prior context.
    pub fn run_mtp_seed(
        &mut self,
        model: &dyn crate::models::Model,
        cache: &mut kv_cache::KvCache,
        hiddens: &[f32],
        tokens: &[u32],
        position_offset: u32,
    ) -> Result<(), Box<dyn Error>> {
        let weights = model.weights();
        self.run_spec_record(weights, |ctx| {
            model.record_mtp_seed(ctx, cache, hiddens, tokens, position_offset)?;
            Ok(vec![])
        })?;
        Ok(())
    }

    /// Write u32 data into a scratch slot (e.g. token ids for get_rows).
    pub fn write_scratch_u32(
        &self,
        range: BufferRange,
        data: &[u32],
    ) -> Result<(), Box<dyn Error>> {
        let host_ptr = self
            .scratch
            .host_ptr
            .ok_or("scratch region not host-visible")?;
        let bytes = std::mem::size_of_val(data);
        if bytes as u64 > range.size {
            return Err(format!("write_scratch_u32: {bytes} > range.size {}", range.size).into());
        }
        unsafe {
            let dst = host_ptr.add(range.offset as usize) as *mut u32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device.device_wait_idle();
            self.device.device.destroy_fence(self.fence, None);
            self.device
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        self.scratch.destroy(&self.device.device);
        self.readback.destroy(&self.device.device);
        self.descriptors.destroy(&self.device);
        self.pipelines.destroy(&self.device);
    }
}
