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

pub struct DispatchContext<'a> {
    pub device: &'a Device,
    pub weights: &'a WeightsHandle,
    pub scratch: &'a mut Region,
    pub pipelines: &'a mut PipelineCache,
    pub descriptors: &'a DescriptorAllocator,
    pub cmd: vk::CommandBuffer,
    /// Optional list of `(name, range)` snapshots the model can push to.
    /// The engine reads these back after submit, alongside the main logits.
    /// Used for layer-by-layer diff dumps vs llama.cpp's `cb()` callback.
    pub taps: Vec<(String, BufferRange)>,
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
    /// back after submit and print a sum. No-op if `SEEKER_QWEN_DIFF_DUMP`
    /// env var is not set — keep call sites unconditional so the dump
    /// instrumentation isn't litter when off.
    ///
    /// **Synchronization:** the tap cast READS from `src`. Without an
    /// explicit barrier, a later dispatch that WRITES to `src` (e.g. an
    /// in-place residual add) can race with this read because Vulkan
    /// dispatches in the same command buffer may overlap unless an
    /// explicit memory barrier orders them. We issue a barrier on `src`'s
    /// range right after the cast so any future writes to that memory
    /// wait for this cast to complete its read.
    pub fn tap(&mut self, name: &str, src: TensorView) -> Result<(), Box<dyn Error>> {
        // Fast path: cached LazyLock — first read does the env-var
        // lookup, subsequent reads are a single atomic-pointer load.
        // (Each forward has ~150 taps in the qwen35moe path; before
        // caching, that was 150 getenv syscalls per forward.)
        if !*crate::runtime_flags::QWEN_DIFF_DUMP {
            return Ok(());
        }
        debug_assert_eq!(src.dtype, GgmlType::F32, "tap only supports F32 tensors");
        let n_elements: u64 = src.dims.iter().product();
        // Direct readback mode: register src's range AS the tap range. The
        // engine reads bytes directly at src.byte_offset after fence wait,
        // without a cast dispatch. This rules out the cast as a source of
        // discrepancy.
        if *crate::runtime_flags::QWEN_DIFF_DIRECT {
            self.taps.push((name.to_string(), BufferRange {
                buffer: src.buffer,
                offset: src.byte_offset,
                size: n_elements * 4,
            }));
            return Ok(());
        }
        let dst = self.alloc_tensor(src.dims, GgmlType::F32)?;
        crate::inference::ops::cast::record_cast(self, src, dst)?;
        self.taps.push((name.to_string(), BufferRange {
            buffer: dst.buffer,
            offset: dst.byte_offset,
            size: n_elements * 4,
        }));
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
