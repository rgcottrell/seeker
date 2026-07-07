//! The Qwen3-Embedding forward pass on the Strix Halo NPU.
//!
//! Same token-major, `b_col_maj`-weight op chain as `examples/layer0.rs` (input
//! RMSNorm·attn_norm → Q/K/V → per-head Q/K RMSNorm + NEOX RoPE → GQA causal
//! attention → O-proj + residual → SwiGLU FFN + residual), looped over all layers.
//! The kernels are fixed-shape for a token block of [`L_PAD`]; a prompt of
//! `L ≤ L_PAD` tokens runs in that block and the first `L` positions are returned
//! (causal attention makes the pad positions irrelevant to the real ones).
//!
//! ## Precision (the hybrid)
//! The **matmuls run on the NPU with f32 accumulation** (f32-output GEMMs — bf16
//! inputs, f32 accumulate; the host rounds to bf16 only as the next kernel's input).
//! This is essential: the NPU's bf16-*output* GEMM accumulates far worse and, over 28
//! layers, collapses the embedding (cosine 0.78 vs Vulkan). With f32-output GEMMs it
//! is 0.9985, and doing the cheap **RMSNorm / RoPE / softmax / SiLU in f32 on the
//! host** (the bf16 LUT kernels are the remaining drift) reaches **0.99994**. This is
//! exactly FastFlowLM's split (NPU matmul + host-fp32 norms/softmax). The residual
//! stream is kept in f32 across layers (Vulkan does too). `SEEKER_NPU_ONCHIP_OPS=1`
//! moves those four op-classes back onto the NPU (bf16 LUT) — the lower-accuracy
//! fully-on-NPU path, kept for future f32-precision NPU-kernel work and measurement.
//!
//! Activations round-trip to the host between ops (each op is its own xclbin, created
//! and dropped per call — the XDNA2 NPU caps concurrent hardware contexts). Keeping
//! activations resident and caching contexts on-NPU are M5 perf levers.
use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

use half::{bf16, f16};
use seeker_core::gguf::{GgmlType, GgufFile};

use crate::npu::{Buffer, Context};

/// Largest token block the AIE kernels are built for (the GEMM M%512 constraint).
pub const L_PAD: usize = 512;
const HEAD_DIM: usize = 128;
// NB: every RMSNorm runs on the aie2p `rms_norm.cc` kernel, which hardcodes
// eps = 1e-5 (Qwen3 config is 1e-6; the difference is negligible against a
// mean-square of ~O(1) over the normed dimension).
const MASK_NEG: f32 = -1e4;

/// A token-block **bucket**: the fixed kernel shapes a prompt runs in. The forward
/// pads a prompt of `l` tokens up to the smallest bucket's `lpad ≥ l`, so the GEMMs
/// are sized to the real length instead of always the 512 max. For short prompts the
/// 128-bucket cuts the NPU GEMM FLOPs several-fold — dramatically for attention, whose
/// key dim collapses 1024→128 and value width 256→128. The on-chip op path (its
/// norm/softmax/silu xclbins are 512-only) always uses [`B512`].
struct Shape {
    lpad: usize,       // token block = weight-GEMM M
    keys: usize,       // padded key count = QKᵀ N and ·V K
    vpad: usize,       // padded value width = ·V N
    qkv: &'static str, // fused Q|K|V proj (N = q_dim + 2·kv_dim), one dispatch for 3
    wo: &'static str,
    // gate/up are NOT fused: the fused N=2·n_ff=6144 GEMM trips an AIE DMA-stride limit
    // at M=512 (B512). Left as two dispatches.
    gate_up: &'static str,
    down: &'static str,
    qkt: &'static str, // QKᵀ (b_col_maj), M = gqa·lpad
    av: &'static str,  // ·V (row-major B), M = gqa·lpad
}

/// Full 512-token block (KEYS padded to the 1024 softmax tile, VPAD to the N%256 rule).
const B512: Shape = Shape {
    lpad: 512,
    keys: 1024,
    vpad: 256,
    qkv: "gemm_512x1024x4096_bcm",
    wo: "gemm_512x2048x1024_bcm",
    gate_up: "gemm_512x1024x3072_bcm",
    down: "gemm_512x3072x1024_bcm",
    qkt: "gemm_1024x128x1024_bcm",
    av: "gemm_1024x1024x256",
};
/// 128-token block for short prompts (m=16 weight tiles, m=32/n=16 attention; keys and
/// value width need no padding at this size).
const B128: Shape = Shape {
    lpad: 128,
    keys: 128,
    vpad: 128,
    qkv: "gemm_128x1024x4096_bcm",
    wo: "gemm_128x2048x1024_bcm",
    gate_up: "gemm_128x1024x3072_bcm",
    down: "gemm_128x3072x1024_bcm",
    qkt: "gemm_256x128x128_bcm",
    av: "gemm_256x128x128",
};
/// 256-token block (m=32 weights; attention M=gqa·256=512, keys=256 and vpad=128 both
/// tile-aligned so no padding). Covers the 128 < l ≤ 256 range.
const B256: Shape = Shape {
    lpad: 256,
    keys: 256,
    vpad: 128,
    qkv: "gemm_256x1024x4096_bcm",
    wo: "gemm_256x2048x1024_bcm",
    gate_up: "gemm_256x1024x3072_bcm",
    down: "gemm_256x3072x1024_bcm",
    qkt: "gemm_512x128x256_bcm",
    av: "gemm_512x256x128",
};

/// f32 → bf16 bits, branchless round-to-nearest-even. For the finite activations here
/// this is numerically identical to `bf16::from_f32(x).to_bits()` but far cheaper (no
/// NaN/inf/subnormal branches) — and these conversions are the bulk of the host time.
#[inline]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_bf16_bits(x)).collect()
}
/// bf16 bits → f32 — an exact left-shift (bf16 is the high 16 bits of f32).
fn deq(v: &[u16]) -> Vec<f32> {
    v.iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}
/// True iff env var `k` is set to a truthy value (so `FOO=0`/`false` disables it,
/// not merely-present).
fn env_host(k: &str) -> bool {
    std::env::var(k)
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}
/// eps used by the host-f32 op fallbacks (matches the GGUF; the NPU rms_norm.cc uses 1e-5).
const HOST_EPS: f32 = 1e-6;

/// A loaded xclbin: its device [`Context`] and the synced instruction buffer, reused
/// across every dispatch of that kernel (the xclbin load + instr sync happen once).
///
/// Field order matters: Buffers (`instr`, `bos`) are allocated from `ctx`, so they're
/// declared before it to be dropped first (Rust drops fields in declaration order) —
/// freed while their Context is still alive, not after it's destroyed.
///
/// `bos` are the reusable operand/output BOs. Every kernel here is fixed-shape, so a
/// given stem is always dispatched with the same operand sizes → the BOs are allocated
/// once (on the first dispatch) and reused, avoiding a pinned-memory alloc/free of
/// ~3–4 BOs on each of the ~1000 dispatches a forward issues (the dominant per-dispatch
/// overhead on this unified-memory APU, where the DMA sync itself is nearly free).
struct Loaded {
    instr: Buffer,
    bos: RefCell<Vec<Buffer>>, // [input0.., output]; empty until the first dispatch
    ninstr: u32,
    ctx: Context,
}

/// The XDNA2 NPU caps concurrent hardware contexts at 16 (`CREATE_HWCTX` EINVAL past
/// that), so the cache is bounded and evicts least-recently-used. A single forward uses
/// 6 distinct GEMM xclbins (one bucket: qkv, wo, gate_up, down, QKᵀ, ·V) — always cached
/// for the whole 28-layer loop, so the BO reuse never thrashes within a forward. The
/// three buckets total 18 xclbins, so a process that mixes all three prompt sizes evicts
/// across forwards (6 reloads on the first layer after a bucket switch), but a single-size
/// workload stays fully warm. The on-chip path's ~18 distinct kernels evict a few per layer.
const CTX_CACHE_CAP: usize = 15;

/// Runs fixed-shape AIE kernels by xclbin name, caching loaded Contexts (M5: avoids
/// reloading the xclbin + re-syncing instructions on every one of the ~1000 dispatches
/// a forward issues). `Loaded` is stored by value (no `Rc`) to keep the embedder
/// `Send`. (Keeping *activations* resident across ops isn't viable here — the hybrid
/// does the norm/rope/softmax/SiLU in f32 on the host, which needs the data host-side;
/// resident *weights* are the tractable variant, see [`Self::run_f32out_w`].)
struct KernelRunner {
    dir: PathBuf,
    cache: RefCell<HashMap<String, Loaded>>,
    lru: RefCell<Vec<String>>, // front = least-recently-used
    // (count, total dispatch time) per kernel stem, for SEEKER_NPU_TIMING.
    timing: RefCell<HashMap<String, (u32, std::time::Duration)>>,
    // Resident weight BOs, keyed by (kernel key, weight ptr): a GEMM weight is
    // immutable across the whole run, so it's uploaded to its own BO once and reused
    // instead of re-copied into the shared input BO on every dispatch (the weight is
    // ~4–6 MB and re-copied 28× per forward). Keyed by ptr so wk/wv, which share a
    // kernel, get distinct BOs. Never evicted (the resident set is the model weights).
    wbos: RefCell<HashMap<(String, usize), Buffer>>,
}

impl KernelRunner {
    fn new() -> Self {
        let dir = std::env::var("SEEKER_NPU_KERNEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels"));
        Self {
            dir,
            cache: RefCell::new(HashMap::new()),
            lru: RefCell::new(Vec::new()),
            timing: RefCell::new(HashMap::new()),
            wbos: RefCell::new(HashMap::new()),
        }
    }

    /// Total NPU-dispatch time + per-stem breakdown (only populated under SEEKER_NPU_TIMING).
    fn timing_report(&self) -> (std::time::Duration, Vec<(String, u32, std::time::Duration)>) {
        let t = self.timing.borrow();
        let total = t.values().map(|(_, d)| *d).sum();
        let mut rows: Vec<_> = t.iter().map(|(s, (c, d))| (s.clone(), *c, *d)).collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.2));
        (total, rows)
    }

    /// Cache key for a kernel: `subdir/stem` (stems can repeat across subdirs).
    fn key(subdir: &str, stem: &str) -> String {
        format!("{subdir}/{stem}")
    }

    /// Ensure the kernel is loaded (loading + evicting LRU if at the HW-context cap on a
    /// miss) and mark it most-recently-used.
    fn ensure(&self, subdir: &str, stem: &str) -> Result<(), Box<dyn Error>> {
        let key = Self::key(subdir, stem);
        if self.cache.borrow().contains_key(&key) {
            let mut lru = self.lru.borrow_mut();
            lru.retain(|s| s != &key);
            lru.push(key);
            return Ok(());
        }
        // Miss: evict LRU entries (dropping their Context frees the HW slot) so the new
        // load stays within the cap, THEN create the new Context.
        while self.cache.borrow().len() >= CTX_CACHE_CAP {
            let victim = self.lru.borrow_mut().remove(0);
            self.cache.borrow_mut().remove(&victim);
        }
        let base = self.dir.join(subdir).join("build").join(stem);
        let ctx = Context::new(&base.with_extension("xclbin"), "MLIR_AIE")?;
        let insts = std::fs::read(base.with_extension("insts.bin"))?;
        let mut instr = ctx.alloc_instr(insts.len())?;
        instr.as_mut_bytes().copy_from_slice(&insts);
        instr.sync_to_device()?;
        let ninstr = insts.len() as u32;
        self.cache.borrow_mut().insert(
            key.clone(),
            Loaded {
                instr,
                bos: RefCell::new(Vec::new()),
                ninstr,
                ctx,
            },
        );
        self.lru.borrow_mut().push(key);
        Ok(())
    }

    /// Copy `inputs` into the kernel's pooled input BOs (allocating the pool — one BO
    /// per input + an `out_bytes` output — on the first dispatch, reusing it after),
    /// bind in order (inputs.., output), run, and sync the output back. The output is
    /// left in the last pooled BO for the caller to read.
    fn dispatch(k: &Loaded, inputs: &[&[u16]], out_bytes: usize) -> Result<(), Box<dyn Error>> {
        let mut pool = k.bos.borrow_mut();
        if pool.is_empty() {
            for inp in inputs {
                pool.push(k.ctx.alloc_data(inp.len() * 2)?);
            }
            pool.push(k.ctx.alloc_data(out_bytes)?);
        }
        for (bo, inp) in pool.iter_mut().zip(inputs) {
            bo.as_mut_slice::<u16>().copy_from_slice(inp);
            bo.sync_to_device()?;
        }
        let last = pool.len() - 1;
        pool[last].sync_to_device()?; // DPU ABI: sync the (write-only) output before the run
        let refs: Vec<&Buffer> = pool.iter().collect();
        k.ctx.run(&k.instr, k.ninstr, &refs)?;
        drop(refs);
        pool[last].sync_from_device()?;
        Ok(())
    }

    /// Run one kernel with a bf16 output, returning the output bf16 bits.
    fn run(
        &self,
        subdir: &str,
        stem: &str,
        inputs: &[&[u16]],
        out_elems: usize,
    ) -> Result<Vec<u16>, Box<dyn Error>> {
        self.ensure(subdir, stem)?;
        let t0 = std::time::Instant::now();
        let cache = self.cache.borrow();
        let k = &cache[&Self::key(subdir, stem)];
        Self::dispatch(k, inputs, out_elems * 2)?;
        self.tick(stem, t0.elapsed());
        Ok(k.bos.borrow().last().unwrap().as_slice::<u16>().to_vec())
    }

    fn tick(&self, stem: &str, dur: std::time::Duration) {
        if env_host("SEEKER_NPU_TIMING") {
            let mut t = self.timing.borrow_mut();
            let e = t
                .entry(stem.to_string())
                .or_insert((0, std::time::Duration::ZERO));
            e.0 += 1;
            e.1 += dur;
        }
    }

    /// Like [`run`](Self::run) but the output BO is f32 (4 bytes/elem) — for the
    /// f32-output GEMMs, whose f32 accumulation is essential to accuracy.
    fn run_f32out(
        &self,
        subdir: &str,
        stem: &str,
        inputs: &[&[u16]],
        out_elems: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        self.ensure(subdir, stem)?;
        let t0 = std::time::Instant::now();
        let cache = self.cache.borrow();
        let k = &cache[&Self::key(subdir, stem)];
        Self::dispatch(k, inputs, out_elems * 4)?;
        self.tick(stem, t0.elapsed());
        Ok(k.bos.borrow().last().unwrap().as_slice::<f32>().to_vec())
    }

    /// f32-output GEMM with a RESIDENT weight: the activation `act` is copied per call,
    /// but the immutable `weight` lives in its own BO (uploaded once, keyed by ptr), so
    /// it isn't re-copied into an input BO every dispatch. The kernel binds
    /// [act, weight, out]; the per-kernel pool holds just [act, out].
    fn run_f32out_w(
        &self,
        subdir: &str,
        stem: &str,
        act: &[u16],
        weight: &[u16],
        out_elems: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        self.ensure(subdir, stem)?;
        let key = Self::key(subdir, stem);
        let wkey = (key.clone(), weight.as_ptr() as usize);
        let t0 = std::time::Instant::now();
        let cache = self.cache.borrow();
        let k = &cache[&key];
        // Upload the weight to its resident BO once (copy + sync happen a single time).
        if !self.wbos.borrow().contains_key(&wkey) {
            let mut wbo = k.ctx.alloc_data(weight.len() * 2)?;
            wbo.as_mut_slice::<u16>().copy_from_slice(weight);
            wbo.sync_to_device()?;
            self.wbos.borrow_mut().insert(wkey.clone(), wbo);
        }
        let wbos = self.wbos.borrow();
        let wbo = &wbos[&wkey];
        {
            let mut pool = k.bos.borrow_mut(); // [act, out]
            if pool.is_empty() {
                pool.push(k.ctx.alloc_data(act.len() * 2)?);
                pool.push(k.ctx.alloc_data(out_elems * 4)?);
            }
            pool[0].as_mut_slice::<u16>().copy_from_slice(act);
            pool[0].sync_to_device()?;
            pool[1].sync_to_device()?; // sync the write-only output before the run
            let refs: Vec<&Buffer> = vec![&pool[0], wbo, &pool[1]];
            k.ctx.run(&k.instr, k.ninstr, &refs)?;
            drop(refs);
            pool[1].sync_from_device()?;
        }
        self.tick(stem, t0.elapsed());
        Ok(k.bos.borrow()[1].as_slice::<f32>().to_vec())
    }
}

/// Per-layer weights: matmul operands as bf16 bits (fed as `b_col_maj` B, i.e. GGUF
/// `[out][in]` order), norm weights as f32 (broadcast/tiled per use).
struct Layer {
    attn_norm: Vec<f32>,
    // Q|K|V fused along the output dim (GGUF stores each as [out][in] row-major, so a
    // flat concat is exactly the [q_dim+2·kv_dim][n_embd] fused weight) → one GEMM.
    wqkv: Vec<u16>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: Vec<u16>,
    ffn_norm: Vec<f32>,
    gate: Vec<u16>,
    up: Vec<u16>,
    down: Vec<u16>,
}

/// Model config + weights for the on-NPU Qwen3-Embedding forward.
pub struct Qwen3Forward {
    pub n_embd: usize,
    n_head: usize,
    n_kv: usize,
    q_dim: usize,
    kv_dim: usize,
    n_ff: usize,
    rope_base: f32,
    vocab: usize,
    token_embd: Vec<f32>, // [vocab][n_embd], row id contiguous
    layers: Vec<Layer>,
    kernels: KernelRunner,
}

fn f16_bits(gguf: &GgufFile, name: &str) -> Result<Vec<u16>, Box<dyn Error>> {
    let info = gguf.tensor(name).ok_or(format!("missing {name}"))?;
    if info.ggml_type != GgmlType::F16 {
        return Err(format!(
            "{name} must be F16 for the NPU backend, got {:?}",
            info.ggml_type
        )
        .into());
    }
    Ok(gguf
        .tensor_data(name)
        .ok_or("no data")?
        .chunks_exact(2)
        .map(|b| bf16::from_f32(f16::from_le_bytes([b[0], b[1]]).to_f32()).to_bits())
        .collect())
}

fn f16_f32(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let info = gguf.tensor(name).ok_or(format!("missing {name}"))?;
    if info.ggml_type != GgmlType::F16 {
        return Err(format!("{name} must be F16, got {:?}", info.ggml_type).into());
    }
    Ok(gguf
        .tensor_data(name)
        .ok_or("no data")?
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

fn f32_vec(gguf: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let info = gguf.tensor(name).ok_or(format!("missing {name}"))?;
    if info.ggml_type != GgmlType::F32 {
        return Err(format!("{name} must be F32, got {:?}", info.ggml_type).into());
    }
    let raw = gguf.tensor_data(name).ok_or(format!("missing {name}"))?;
    if raw.len() != len * 4 {
        return Err(format!("{name} expected F32[{len}], got {} bytes", raw.len()).into());
    }
    Ok(raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

impl Qwen3Forward {
    /// Read config + dequantize all weights to bf16 from the GGUF.
    pub fn load(gguf: &GgufFile) -> Result<Self, Box<dyn Error>> {
        let arch = gguf.architecture().unwrap_or("qwen3").to_string();
        let mu = |k: &str| gguf.meta_u32(&format!("{arch}.{k}"));
        let n_layers = mu("block_count").ok_or("missing block_count")? as usize;
        let n_embd = mu("embedding_length").ok_or("missing embedding_length")? as usize;
        let n_head = mu("attention.head_count").ok_or("missing head_count")? as usize;
        let n_kv = mu("attention.head_count_kv").unwrap_or(n_head as u32) as usize;
        let n_ff = mu("feed_forward_length").ok_or("missing feed_forward_length")? as usize;
        let te = gguf
            .tensor("token_embd.weight")
            .ok_or("missing token_embd")?;
        let vocab = te.dims[1] as usize;
        let q_dim = gguf.tensor("blk.0.attn_q.weight").ok_or("missing wq")?.dims[1] as usize;
        let kv_dim = gguf.tensor("blk.0.attn_v.weight").ok_or("missing wv")?.dims[1] as usize;
        let head_dim = q_dim / n_head;
        // The xclbins are fixed-shape (Qwen3-Embedding-0.6B). Reject any model whose
        // dims differ — the kernels would under/over-size the operand/output BOs.
        let want = [
            ("head_dim", head_dim, HEAD_DIM),
            ("n_embd", n_embd, 1024),
            ("q_dim", q_dim, 2048),
            ("kv_dim", kv_dim, 1024),
            ("n_ff", n_ff, 3072),
        ];
        for (name, got, exp) in want {
            if got != exp {
                return Err(format!(
                    "NPU backend has fixed-shape kernels for Qwen3-Embedding-0.6B \
                     ({name}={exp}); this model has {name}={got}"
                )
                .into());
            }
        }
        let rope_base = gguf
            .meta_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(1e6);

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            // Q|K|V concatenated along the output dim → one fused proj GEMM. GGUF stores
            // each weight as [out][in] row-major, so flat concatenation is exactly the
            // [q_dim + 2·kv_dim][n_embd] operand the fused kernel wants.
            let mut wqkv = f16_bits(gguf, &format!("{p}.attn_q.weight"))?;
            wqkv.extend(f16_bits(gguf, &format!("{p}.attn_k.weight"))?);
            wqkv.extend(f16_bits(gguf, &format!("{p}.attn_v.weight"))?);
            layers.push(Layer {
                attn_norm: f32_vec(gguf, &format!("{p}.attn_norm.weight"), n_embd)?,
                wqkv,
                q_norm: f32_vec(gguf, &format!("{p}.attn_q_norm.weight"), HEAD_DIM)?,
                k_norm: f32_vec(gguf, &format!("{p}.attn_k_norm.weight"), HEAD_DIM)?,
                wo: f16_bits(gguf, &format!("{p}.attn_output.weight"))?,
                ffn_norm: f32_vec(gguf, &format!("{p}.ffn_norm.weight"), n_embd)?,
                gate: f16_bits(gguf, &format!("{p}.ffn_gate.weight"))?,
                up: f16_bits(gguf, &format!("{p}.ffn_up.weight"))?,
                down: f16_bits(gguf, &format!("{p}.ffn_down.weight"))?,
            });
        }

        Ok(Self {
            n_embd,
            n_head,
            n_kv,
            q_dim,
            kv_dim,
            n_ff,
            rope_base,
            vocab,
            token_embd: f16_f32(gguf, "token_embd.weight")?,
            layers,
            kernels: KernelRunner::new(),
        })
    }

    /// NEOX cos/sin tables broadcast to token-major `[L_PAD][n_heads·128]`, optionally
    /// pre-multiplied by `scale` (used to fold the attention scale into q).
    fn rope_tables(&self, n_heads: usize, scale: f32, lpad: usize) -> (Vec<u16>, Vec<u16>) {
        let d = n_heads * HEAD_DIM;
        let (mut cos, mut sin) = (vec![0.0f32; lpad * d], vec![0.0f32; lpad * d]);
        for t in 0..lpad {
            for i in 0..d {
                let j = (i % HEAD_DIM) % (HEAD_DIM / 2);
                let theta = t as f32 * self.rope_base.powf(-2.0 * j as f32 / HEAD_DIM as f32);
                cos[t * d + i] = theta.cos() * scale;
                sin[t * d + i] = theta.sin() * scale;
            }
        }
        (bits(&cos), bits(&sin))
    }

    fn rms_mul_name(n: usize) -> (String, String, String) {
        (
            format!("rmsnorm_128_{n}"),
            format!("eltwise_mul_bf16_{n}"),
            format!("eltwise_add_bf16_{n}"),
        )
    }

    /// `b_col_maj` GEMM: out[m][n] = Σ_k A[m][k]·B[n][k], **f32 output**. Inputs are
    /// bf16 bits; the f32 accumulation + f32 output is what holds accuracy (the NPU's
    /// bf16-*output* GEMM accumulates far worse — it was the dominant precision sink
    /// over 28 layers). `SEEKER_NPU_HOST_GEMM` swaps in a host f32-accumulate reference.
    fn gemm_bcm(
        &self,
        stem: &str,
        a: &[u16],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if env_host("SEEKER_NPU_HOST_GEMM") {
            let (af, bf) = (deq(a), deq(b));
            let mut out = vec![0.0f32; m * n];
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        acc += af[mi * k + ki] * bf[ni * k + ki];
                    }
                    out[mi * n + ni] = acc;
                }
            }
            return Ok(out);
        }
        self.kernels.run_f32out("gemm", stem, &[a, b], m * n)
    }

    /// [`gemm_bcm`](Self::gemm_bcm) for a GEMM whose `b` is a persistent layer weight.
    /// With `SEEKER_NPU_RESIDENT_WEIGHTS` set, `b` feeds a resident BO (uploaded once)
    /// instead of being re-copied every dispatch — the ~4–6 MB weight is otherwise
    /// re-copied 28× per forward. This is a SUSTAINED-workload lever (serve/batch): it
    /// trades a one-time upload of the whole weight set (~the model size, on the first
    /// forward) for ~9% off every later forward, so it only pays back past ~5 forwards.
    /// Default off, so the one-shot CLI keeps its faster cold path.
    fn gemm_bcm_w(
        &self,
        stem: &str,
        a: &[u16],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if env_host("SEEKER_NPU_HOST_GEMM") {
            let (af, bf) = (deq(a), deq(b));
            let mut out = vec![0.0f32; m * n];
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        acc += af[mi * k + ki] * bf[ni * k + ki];
                    }
                    out[mi * n + ni] = acc;
                }
            }
            return Ok(out);
        }
        if env_host("SEEKER_NPU_RESIDENT_WEIGHTS") {
            self.kernels.run_f32out_w("gemm", stem, a, b, m * n)
        } else {
            self.kernels.run_f32out("gemm", stem, &[a, b], m * n)
        }
    }

    /// Row-major-B GEMM: out[m][n] = Σ_k A[m][k]·B[k][n] (the ·V layout), f32 output.
    fn gemm_rm(
        &self,
        stem: &str,
        a: &[u16],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if env_host("SEEKER_NPU_HOST_GEMM") {
            let (af, bf) = (deq(a), deq(b));
            let mut out = vec![0.0f32; m * n];
            for mi in 0..m {
                for ki in 0..k {
                    let av = af[mi * k + ki];
                    for ni in 0..n {
                        out[mi * n + ni] += av * bf[ki * n + ni];
                    }
                }
            }
            return Ok(out);
        }
        self.kernels.run_f32out("gemm", stem, &[a, b], m * n)
    }

    /// Whole-token RMSNorm(x)·weight → bf16 bits (input to the next GEMM). f32 on the
    /// host by default (the hybrid: the bf16 LUT rms_norm kernel is too lossy over many
    /// layers); `SEEKER_NPU_ONCHIP_OPS` runs it on the NPU instead (for future
    /// f32-precision NPU-kernel work / measurement).
    fn norm_mul(
        &self,
        x: &[f32],
        weight: &[f32],
        l: usize,
        lpad: usize,
    ) -> Result<Vec<u16>, Box<dyn Error>> {
        let w = weight.len();
        let n = lpad * w;
        if env_host("SEEKER_NPU_ONCHIP_OPS") {
            let xn = self
                .kernels
                .run("norm", "rmsnorm_1024_524288", &[&bits(x)], n)?;
            let wt = tile(weight, w, lpad);
            return self
                .kernels
                .run("eltwise", "eltwise_mul_bf16_524288", &[&xn, &bits(&wt)], n);
        }
        // Only the first `l` (real) token rows matter; the pad rows feed nothing real
        // (causal attention + discarded output), so leave them zero.
        let mut out = vec![0.0f32; n];
        for t in 0..l {
            let col = &x[t * w..(t + 1) * w];
            let ms = col.iter().map(|v| v * v).sum::<f32>() / w as f32;
            let inv = 1.0 / (ms + HOST_EPS).sqrt();
            for i in 0..w {
                out[t * w + i] = col[i] * inv * weight[i];
            }
        }
        Ok(bits(&out))
    }

    /// Per-head RMSNorm(128)·norm_w + NEOX RoPE on the NPU (cos/sin may carry scale).
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope(
        &self,
        proj: &[u16],
        n_heads: usize,
        norm_w: &[f32],
        cos: &[u16],
        sin: &[u16],
        l: usize,
        lpad: usize,
    ) -> Result<Vec<u16>, Box<dyn Error>> {
        let n = lpad * n_heads * HEAD_DIM;
        if env_host("SEEKER_NPU_ONCHIP_OPS") {
            let (rms, mul, add) = Self::rms_mul_name(n);
            let normed = self.kernels.run("norm", &rms, &[proj], n)?;
            let wt: Vec<f32> = (0..n).map(|i| norm_w[i % HEAD_DIM]).collect();
            let normed = self
                .kernels
                .run("eltwise", &mul, &[&normed, &bits(&wt)], n)?;
            let t1 = self.kernels.run("eltwise", &mul, &[&normed, cos], n)?;
            let rh = bits(&rot_half(&deq(&normed)));
            let t2 = self.kernels.run("eltwise", &mul, &[&rh, sin], n)?;
            return self.kernels.run("eltwise", &add, &[&t1, &t2], n);
        }
        // Host f32 (default): per-head rmsnorm·norm_w + NEOX rope (cos/sin carry any scale).
        let (pf, cf, sf) = (deq(proj), deq(cos), deq(sin));
        let d = n_heads * HEAD_DIM;
        let mut out = vec![0.0f32; n];
        for t in 0..l {
            for h in 0..n_heads {
                let b = t * d + h * HEAD_DIM;
                let head = &pf[b..b + HEAD_DIM];
                let ms = head.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32;
                let inv = 1.0 / (ms + HOST_EPS).sqrt();
                let nd: Vec<f32> = (0..HEAD_DIM).map(|i| head[i] * inv * norm_w[i]).collect();
                for j in 0..HEAD_DIM / 2 {
                    let (c, s) = (cf[b + j], sf[b + j]);
                    out[b + j] = nd[j] * c - nd[j + HEAD_DIM / 2] * s;
                    out[b + j + HEAD_DIM / 2] = nd[j] * s + nd[j + HEAD_DIM / 2] * c;
                }
            }
        }
        Ok(bits(&out))
    }

    /// GQA causal attention → attn_out[lpad][q_dim] as bf16 bits (the O-proj GEMM's
    /// input — converted in the gather so the caller needs no `bits(&attn)`). q is
    /// pre-scaled; k is bf16 bits (post-rope), v is the f32 value projection. `sh` is
    /// the token-block bucket (kernel shapes + key/value padding).
    fn attention(
        &self,
        q: &[u16],
        k: &[u16],
        v: &[f32],
        l: usize,
        sh: &Shape,
    ) -> Result<Vec<u16>, Box<dyn Error>> {
        let (n_head, n_kv, q_dim, kv_dim) = (self.n_head, self.n_kv, self.q_dim, self.kv_dim);
        let gqa = n_head / n_kv;
        let (lpad, keys, vpad) = (sh.lpad, sh.keys, sh.vpad);
        // GQA batching: the `gqa` Q-heads sharing a KV head share K and V, so stack them
        // along M and do ONE dense GEMM per KV head (M = gqa·lpad) instead of one per
        // Q-head — halving the QKᵀ / ·V dispatch count.
        //
        // Each KV head is fully independent, so its softmax reads the QKᵀ GEMM output
        // directly and writes a per-pair [mb][keys] probs buffer — never a global
        // [n_head][lpad][keys] one. That drops the split (GEMM rows → global scores) and
        // stack (global probs → GEMM rows) copies of the two big buffers per layer.
        //
        // The QKᵀ (·) and ·V GEMMs stay in two SEPARATE phases (all QKᵀ, then all ·V)
        // rather than a per-pair QKᵀ→·V pipeline: interleaving them forces an XDNA2
        // hardware-context swap between the two xclbins on every pair (16/layer vs 1),
        // which costs far more than the host copies it would save.
        let mb = gqa * lpad;
        // q and k arrive as bf16 bits (post-rope, pre-scaled) and feed the QKᵀ GEMM as
        // bf16 — so we reshuffle their u16 words straight into the batched operands with
        // NO deq/re-bits round-trip. v is the f32 value projection, converted per element
        // as it's gathered. Only the first `l` token rows are real; the operand buffers
        // are allocated full (zeroed) so the fixed-shape kernels see a padded input, but
        // we only touch the real rows — the pad rows stay 0 (Q·Kᵀ 0, causally masked; the
        // pad output rows are never read).
        let onchip = env_host("SEEKER_NPU_ONCHIP_OPS");
        // Causal + pad-column mask for one KV pair: row = sub·lpad + tq, so the mask is
        // the same for every pair (depends only on tq = row % lpad) — build it once.
        let mask_bits = onchip.then(|| {
            let mut mask = vec![0.0f32; mb * keys];
            for sub in 0..gqa {
                for tq in 0..lpad {
                    let row = (sub * lpad + tq) * keys;
                    for tk in 0..keys {
                        if tk > tq || tk >= lpad {
                            mask[row + tk] = MASK_NEG;
                        }
                    }
                }
            }
            bits(&mask)
        });
        // Phase 1: QKᵀ + softmax → per-pair probs (the QKᵀ xclbin stays resident; the
        // default softmax is host, so no context swap interleaves the GEMMs).
        let mut e = vec![0.0f32; keys];
        let mut pair_probs: Vec<Vec<u16>> = Vec::with_capacity(n_kv);
        for kv in 0..n_kv {
            // Stack the gqa Q-heads' q rows (they share this pair's K).
            let mut q_pair = vec![0u16; mb * HEAD_DIM]; // bf16 bits; rows≥l = 0
            let mut k_kv = vec![0u16; keys * HEAD_DIM]; // bf16 bits, [keys][128] bcm
            for t in 0..l {
                let ko = t * kv_dim + kv * HEAD_DIM;
                k_kv[t * HEAD_DIM..(t + 1) * HEAD_DIM].copy_from_slice(&k[ko..ko + HEAD_DIM]);
                for sub in 0..gqa {
                    let qo = t * q_dim + (gqa * kv + sub) * HEAD_DIM;
                    let row = sub * lpad + t;
                    q_pair[row * HEAD_DIM..(row + 1) * HEAD_DIM]
                        .copy_from_slice(&q[qo..qo + HEAD_DIM]);
                }
            }
            let s = self.gemm_bcm(sh.qkt, &q_pair, &k_kv, mb, HEAD_DIM, keys)?;
            // Softmax over keys per row. Rows [sub·lpad .. (sub+1)·lpad) are Q-head
            // (gqa·kv + sub); the pair layout already matches the ·V GEMM's M.
            let probs_pair = if let Some(ref mask_bits) = mask_bits {
                let masked = self.kernels.run(
                    "eltwise",
                    "eltwise_add_bf16_1048576",
                    &[&bits(&s), mask_bits],
                    mb * keys,
                )?;
                self.kernels
                    .run("norm", "softmax_1048576", &[&masked], mb * keys)?
            } else {
                // Host f32 (default): causal softmax per row with accurate exp, written
                // straight to bf16. Only the first `l` (real) query rows per sub-block are
                // computed — the pad rows feed nothing real and their probs stay 0.
                let mut p = vec![0u16; mb * keys];
                for sub in 0..gqa {
                    for tq in 0..l {
                        let row = (sub * lpad + tq) * keys;
                        let mx = s[row..=row + tq].iter().cloned().fold(f32::MIN, f32::max);
                        let mut den = 0.0f32;
                        for tk in 0..=tq {
                            e[tk] = (s[row + tk] - mx).exp();
                            den += e[tk];
                        }
                        for tk in 0..=tq {
                            p[row + tk] = f32_to_bf16_bits(e[tk] / den);
                        }
                    }
                }
                p
            };
            pair_probs.push(probs_pair);
        }
        // Phase 2: ·V (the ·V xclbin now stays resident across all pairs).
        let mut attn_out = vec![0u16; lpad * q_dim]; // bf16 bits (O-proj GEMM input)
        for (kv, probs_pair) in pair_probs.iter().enumerate() {
            let mut v_kv = vec![0u16; keys * vpad]; // bf16 bits, [keys][vpad]; rows≥l = 0
            for t in 0..l {
                let vo = t * kv_dim + kv * HEAD_DIM;
                for (dst, &sv) in v_kv[t * vpad..t * vpad + HEAD_DIM]
                    .iter_mut()
                    .zip(&v[vo..vo + HEAD_DIM])
                {
                    *dst = f32_to_bf16_bits(sv);
                }
            }
            let o = self.gemm_rm(sh.av, probs_pair, &v_kv, mb, keys, vpad)?;
            for sub in 0..gqa {
                let h = gqa * kv + sub;
                for t in 0..l {
                    let ao = t * q_dim + h * HEAD_DIM;
                    let orow = (sub * lpad + t) * vpad;
                    for (dst, &ov) in attn_out[ao..ao + HEAD_DIM]
                        .iter_mut()
                        .zip(&o[orow..orow + HEAD_DIM])
                    {
                        *dst = f32_to_bf16_bits(ov);
                    }
                }
            }
        }
        Ok(attn_out)
    }

    /// Run the full forward for `tokens` (≤ L_PAD) and return the pre-output_norm
    /// residual as `[n_embd * tokens.len()]`, token-major.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let l = tokens.len();
        if l == 0 || l > L_PAD {
            return Err(format!("NPU backend handles 1..={L_PAD} tokens, got {l}").into());
        }
        let t_fwd = std::time::Instant::now();
        if env_host("SEEKER_NPU_TIMING") {
            // Per-forward report: clear the cumulative counters so each forward stands
            // alone (else timing_report's cross-forward sum exceeds this forward's total,
            // underflowing the `total - npu` host split on the 2nd forward).
            self.kernels.timing.borrow_mut().clear();
        }
        let (n_embd, q_dim, kv_dim, n_ff) = (self.n_embd, self.q_dim, self.kv_dim, self.n_ff);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        // Pick the smallest token-block bucket that fits the prompt: short prompts run
        // the 128/256-block kernels (far fewer GEMM FLOPs) instead of always the 512 max.
        // The on-chip op and fused-FFN xclbins are 512-only, so those paths stay B512.
        let full_only = env_host("SEEKER_NPU_ONCHIP_OPS") || env_host("SEEKER_NPU_FUSED_FFN");
        let sh: &Shape = if full_only || l > B256.lpad {
            &B512
        } else if l <= B128.lpad {
            &B128
        } else {
            &B256
        };
        let lpad = sh.lpad;

        // get_rows (host): seed the token block; pad positions get token 0.
        let mut x = vec![0.0f32; lpad * n_embd];
        for t in 0..lpad {
            let id = if t < l { tokens[t] as usize } else { 0 };
            if id >= self.vocab {
                return Err(format!("token id {id} >= vocab {}", self.vocab).into());
            }
            x[t * n_embd..(t + 1) * n_embd]
                .copy_from_slice(&self.token_embd[id * n_embd..(id + 1) * n_embd]);
        }

        let (cos_q, sin_q) = self.rope_tables(self.n_head, scale, lpad);
        let (cos_k, sin_k) = self.rope_tables(self.n_kv, 1.0, lpad);

        // The residual stream stays f32 across layers (Vulkan keeps it f32 too):
        // bf16 here loses the small per-layer increments once the residual grows
        // large over 28 layers (bf16's step at magnitude ~100 is ~0.5), collapsing
        // accuracy. bf16 is used only as GEMM/kernel *input* precision; the +proj
        // and +down residual adds are done in f32 on the host.
        let mut resid = x;
        // Diagnostic knob: cap layers (compare partial forwards to the host f32 ref).
        let max = std::env::var("SEEKER_NPU_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(self.layers.len());
        for ly in self.layers.iter().take(max) {
            // ---- attention block (GEMMs output f32; bf16 only as kernel inputs) ----
            let xnw = self.norm_mul(&resid, &ly.attn_norm, l, lpad)?;
            // Fused Q|K|V projection in one dispatch (the three shared xnw). The output
            // rows are [q(q_dim) | k(kv_dim) | v(kv_dim)]; split them back into contiguous
            // q/k/v (only the real `l` rows) for the per-head norm/rope + attention.
            let qkv = self.gemm_bcm_w(sh.qkv, &xnw, &ly.wqkv, lpad, n_embd, q_dim + 2 * kv_dim)?;
            let (mut q, mut k, mut v) = (
                vec![0.0f32; lpad * q_dim],
                vec![0.0f32; lpad * kv_dim],
                vec![0.0f32; lpad * kv_dim],
            );
            let stride = q_dim + 2 * kv_dim;
            for t in 0..l {
                let b = t * stride;
                q[t * q_dim..(t + 1) * q_dim].copy_from_slice(&qkv[b..b + q_dim]);
                k[t * kv_dim..(t + 1) * kv_dim]
                    .copy_from_slice(&qkv[b + q_dim..b + q_dim + kv_dim]);
                v[t * kv_dim..(t + 1) * kv_dim]
                    .copy_from_slice(&qkv[b + q_dim + kv_dim..b + stride]);
            }
            let q_roped =
                self.qk_norm_rope(&bits(&q), self.n_head, &ly.q_norm, &cos_q, &sin_q, l, lpad)?;
            let k_roped =
                self.qk_norm_rope(&bits(&k), self.n_kv, &ly.k_norm, &cos_k, &sin_k, l, lpad)?;
            let attn = self.attention(&q_roped, &k_roped, &v, l, sh)?;
            let proj = self.gemm_bcm_w(sh.wo, &attn, &ly.wo, lpad, q_dim, n_embd)?;
            for (r, p) in resid.iter_mut().zip(&proj) {
                *r += p;
            }

            // ---- FFN block ----
            let xn2w = self.norm_mul(&resid, &ly.ffn_norm, l, lpad)?;
            let g = self.gemm_bcm_w(sh.gate_up, &xn2w, &ly.gate, lpad, n_embd, n_ff)?;
            let u = self.gemm_bcm_w(sh.gate_up, &xn2w, &ly.up, lpad, n_embd, n_ff)?;
            // SwiGLU: silu(gate) * up. Default = host f32 (the whole product, so it isn't
            // rounded to bf16 early and the default path needs no eltwise/activation
            // xclbins — only the GEMMs). On-chip path uses the bf16 silu + mul kernels.
            let hidden = if env_host("SEEKER_NPU_FUSED_FFN") {
                // Fused SwiGLU: silu(gate)*up in one NPU dispatch (silu_mul.cc). MEASURED
                // NET-NEGATIVE as a *separate* dispatch (+87ms NPU vs −38ms host over 28
                // layers = ~+30ms): the FFN is compute-bound and the host swiglu is cheap,
                // so a round-trip to the NPU doesn't pay. Default off; a real win needs
                // *on-chip* fusion (fold silu*mul into the gate/up GEMM, no round-trip).
                self.kernels.run(
                    "fused",
                    "swiglu_1572864",
                    &[&bits(&g), &bits(&u)],
                    lpad * n_ff,
                )?
            } else if env_host("SEEKER_NPU_ONCHIP_OPS") {
                let gs =
                    self.kernels
                        .run("activation", "silu_1572864", &[&bits(&g)], lpad * n_ff)?;
                self.kernels.run(
                    "eltwise",
                    "eltwise_mul_bf16_1572864",
                    &[&gs, &bits(&u)],
                    lpad * n_ff,
                )?
            } else {
                // Host f32 SwiGLU straight to bf16, first `l` rows only (pad rows of
                // gate/up are already 0 → product 0).
                let mut h = vec![0u16; lpad * n_ff];
                for i in 0..l * n_ff {
                    let (gv, uv) = (g[i], u[i]);
                    h[i] = f32_to_bf16_bits((gv / (1.0 + (-gv).exp())) * uv);
                }
                h
            };
            let dn = self.gemm_bcm_w(sh.down, &hidden, &ly.down, lpad, n_ff, n_embd)?;
            for (r, d) in resid.iter_mut().zip(&dn) {
                *r += d;
            }
        }

        // Return the real tokens' residual (token-major), already f32.
        resid.truncate(l * n_embd);
        if env_host("SEEKER_NPU_TIMING") {
            let total = t_fwd.elapsed();
            let (npu, rows) = self.kernels.timing_report();
            eprintln!(
                "── forward {:.0}ms: NPU dispatch {:.0}ms ({:.0}%), host {:.0}ms ({:.0}%) ──",
                total.as_secs_f64() * 1e3,
                npu.as_secs_f64() * 1e3,
                npu.as_secs_f64() / total.as_secs_f64() * 100.0,
                (total - npu).as_secs_f64() * 1e3,
                (total - npu).as_secs_f64() / total.as_secs_f64() * 100.0,
            );
            for (stem, count, dur) in rows {
                eprintln!(
                    "   {:>5}× {:>7.0}ms  {stem}",
                    count,
                    dur.as_secs_f64() * 1e3
                );
            }
        }
        Ok(resid)
    }
}

/// rot_half within each 128-block: out = [−hi, lo].
fn rot_half(x: &[f32]) -> Vec<f32> {
    let mut r = vec![0.0f32; x.len()];
    for blk in 0..(x.len() / HEAD_DIM) {
        let b = blk * HEAD_DIM;
        for i in 0..HEAD_DIM / 2 {
            r[b + i] = -x[b + i + HEAD_DIM / 2];
            r[b + i + HEAD_DIM / 2] = x[b + i];
        }
    }
    r
}

/// Broadcast an `[width]` weight to token-major `[rows][width]`.
fn tile(w: &[f32], width: usize, rows: usize) -> Vec<f32> {
    (0..rows * width).map(|i| w[i % width]).collect()
}
