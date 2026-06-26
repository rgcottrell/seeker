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

/// Token block width the AIE kernels are built for (the GEMM M%512 constraint).
pub const L_PAD: usize = 512;
const HEAD_DIM: usize = 128;
// NB: every RMSNorm runs on the aie2p `rms_norm.cc` kernel, which hardcodes
// eps = 1e-5 (Qwen3 config is 1e-6; the difference is negligible against a
// mean-square of ~O(1) over the normed dimension).
const KEYS: usize = 1024; // L_PAD padded up to the softmax tile width
const VPAD: usize = 256; // head_dim padded up to the GEMM N%256 rule
const MASK_NEG: f32 = -1e4;

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}
fn deq(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
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
/// Field order matters: the `instr` Buffer is allocated from `ctx`, so it's declared
/// first to be dropped first (Rust drops fields in declaration order) — the buffer is
/// freed while its Context is still alive, not after the Context is destroyed.
struct Loaded {
    instr: Buffer,
    ninstr: u32,
    ctx: Context,
}

/// The XDNA2 NPU caps concurrent hardware contexts at 16 (`CREATE_HWCTX` EINVAL past
/// that), so the cache is bounded and evicts least-recently-used. The hybrid forward
/// uses only 7 distinct GEMM xclbins, well under the cap → nothing is evicted; the
/// on-chip path's ~18 distinct kernels evict a few per layer.
const CTX_CACHE_CAP: usize = 15;

/// Runs fixed-shape AIE kernels by xclbin name, caching loaded Contexts (M5: avoids
/// reloading the xclbin + re-syncing instructions on every one of the ~1000 dispatches
/// a forward issues). `Loaded` is stored by value (no `Rc`) to keep the embedder
/// `Send`. Keeping activations resident across ops is the remaining lever.
struct KernelRunner {
    dir: PathBuf,
    cache: RefCell<HashMap<String, Loaded>>,
    lru: RefCell<Vec<String>>, // front = least-recently-used
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
        }
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
        self.cache
            .borrow_mut()
            .insert(key.clone(), Loaded { instr, ninstr, ctx });
        self.lru.borrow_mut().push(key);
        Ok(())
    }

    /// Alloc a BO per bf16-bit input + an output BO of `out_bytes`, bind in order
    /// (inputs.., output), run, return the output BO.
    fn dispatch(k: &Loaded, inputs: &[&[u16]], out_bytes: usize) -> Result<Buffer, Box<dyn Error>> {
        let mut bos: Vec<Buffer> = Vec::with_capacity(inputs.len());
        for inp in inputs {
            let mut b = k.ctx.alloc_data(inp.len() * 2)?;
            b.as_mut_slice::<u16>().copy_from_slice(inp);
            b.sync_to_device()?;
            bos.push(b);
        }
        let mut out = k.ctx.alloc_data(out_bytes)?;
        out.as_mut_bytes().fill(0);
        out.sync_to_device()?;
        let refs: Vec<&Buffer> = bos.iter().chain(std::iter::once(&out)).collect();
        k.ctx.run(&k.instr, k.ninstr, &refs)?;
        drop(refs);
        out.sync_from_device()?;
        Ok(out)
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
        let cache = self.cache.borrow();
        let out = Self::dispatch(&cache[&Self::key(subdir, stem)], inputs, out_elems * 2)?;
        Ok(out.as_slice::<u16>().to_vec())
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
        let cache = self.cache.borrow();
        let out = Self::dispatch(&cache[&Self::key(subdir, stem)], inputs, out_elems * 4)?;
        Ok(out.as_slice::<f32>().to_vec())
    }
}

/// Per-layer weights: matmul operands as bf16 bits (fed as `b_col_maj` B, i.e. GGUF
/// `[out][in]` order), norm weights as f32 (broadcast/tiled per use).
struct Layer {
    attn_norm: Vec<f32>,
    wq: Vec<u16>,
    wk: Vec<u16>,
    wv: Vec<u16>,
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
            layers.push(Layer {
                attn_norm: f32_vec(gguf, &format!("{p}.attn_norm.weight"), n_embd)?,
                wq: f16_bits(gguf, &format!("{p}.attn_q.weight"))?,
                wk: f16_bits(gguf, &format!("{p}.attn_k.weight"))?,
                wv: f16_bits(gguf, &format!("{p}.attn_v.weight"))?,
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
    fn rope_tables(&self, n_heads: usize, scale: f32) -> (Vec<u16>, Vec<u16>) {
        let d = n_heads * HEAD_DIM;
        let (mut cos, mut sin) = (vec![0.0f32; L_PAD * d], vec![0.0f32; L_PAD * d]);
        for t in 0..L_PAD {
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
    fn norm_mul(&self, x: &[f32], weight: &[f32]) -> Result<Vec<u16>, Box<dyn Error>> {
        let w = weight.len();
        let n = L_PAD * w;
        if env_host("SEEKER_NPU_ONCHIP_OPS") {
            let xn = self
                .kernels
                .run("norm", "rmsnorm_1024_524288", &[&bits(x)], n)?;
            let wt = tile(weight, w, L_PAD);
            return self
                .kernels
                .run("eltwise", "eltwise_mul_bf16_524288", &[&xn, &bits(&wt)], n);
        }
        let mut out = vec![0.0f32; n];
        for t in 0..L_PAD {
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
    fn qk_norm_rope(
        &self,
        proj: &[u16],
        n_heads: usize,
        norm_w: &[f32],
        cos: &[u16],
        sin: &[u16],
    ) -> Result<Vec<u16>, Box<dyn Error>> {
        let n = L_PAD * n_heads * HEAD_DIM;
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
        for t in 0..L_PAD {
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

    /// GQA causal attention → attn_out[L_PAD][q_dim] f32. q is pre-scaled; k is bf16
    /// bits (post-rope), v is the f32 value projection.
    fn attention(&self, q: &[u16], k: &[u16], v: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let (n_head, n_kv, q_dim, kv_dim) = (self.n_head, self.n_kv, self.q_dim, self.kv_dim);
        let gqa = n_head / n_kv;
        let (qf, kf, vf) = (deq(q), deq(k), v);
        let mut scores = vec![0.0f32; n_head * L_PAD * KEYS];
        for h in 0..n_head {
            let kv = h / gqa;
            let (mut q_h, mut k_pad) = (
                vec![0.0f32; L_PAD * HEAD_DIM],
                vec![0.0f32; KEYS * HEAD_DIM],
            );
            for t in 0..L_PAD {
                let (qo, ko) = (t * q_dim + h * HEAD_DIM, t * kv_dim + kv * HEAD_DIM);
                q_h[t * HEAD_DIM..(t + 1) * HEAD_DIM].copy_from_slice(&qf[qo..qo + HEAD_DIM]);
                k_pad[t * HEAD_DIM..(t + 1) * HEAD_DIM].copy_from_slice(&kf[ko..ko + HEAD_DIM]);
            }
            let s = self.gemm_bcm(
                "gemm_512x128x1024_bcm",
                &bits(&q_h),
                &bits(&k_pad),
                L_PAD,
                HEAD_DIM,
                KEYS,
            )?;
            scores[h * L_PAD * KEYS..(h + 1) * L_PAD * KEYS].copy_from_slice(&s);
        }
        let n_sc = n_head * L_PAD * KEYS;
        let probs = if env_host("SEEKER_NPU_ONCHIP_OPS") {
            // causal + pad-column mask (broadcast across heads), batched NPU softmax.
            let mut mask = vec![0.0f32; n_sc];
            for h in 0..n_head {
                for tq in 0..L_PAD {
                    let row = (h * L_PAD + tq) * KEYS;
                    for tk in 0..KEYS {
                        if tk > tq || tk >= L_PAD {
                            mask[row + tk] = MASK_NEG;
                        }
                    }
                }
            }
            let scores = self.kernels.run(
                "eltwise",
                "eltwise_add_bf16_8388608",
                &[&bits(&scores), &bits(&mask)],
                n_sc,
            )?;
            self.kernels
                .run("norm", "softmax_8388608", &[&scores], n_sc)?
        } else {
            // Host f32 (default): causal softmax per row with accurate exp.
            let sf = &scores;
            let mut p = vec![0.0f32; n_sc];
            for h in 0..n_head {
                for tq in 0..L_PAD {
                    let row = (h * L_PAD + tq) * KEYS;
                    let last = tq.min(L_PAD - 1);
                    let mx = sf[row..=row + last]
                        .iter()
                        .cloned()
                        .fold(f32::MIN, f32::max);
                    let mut den = 0.0f32;
                    for tk in 0..=last {
                        let e = (sf[row + tk] - mx).exp();
                        p[row + tk] = e;
                        den += e;
                    }
                    for tk in 0..=last {
                        p[row + tk] /= den;
                    }
                }
            }
            bits(&p)
        };
        let mut attn_out = vec![0.0f32; L_PAD * q_dim];
        for h in 0..n_head {
            let kv = h / gqa;
            let mut v_pad = vec![0.0f32; KEYS * VPAD];
            for t in 0..L_PAD {
                let vo = t * kv_dim + kv * HEAD_DIM;
                v_pad[t * VPAD..t * VPAD + HEAD_DIM].copy_from_slice(&vf[vo..vo + HEAD_DIM]);
            }
            let o = self.gemm_rm(
                "gemm_512x1024x256",
                &probs[h * L_PAD * KEYS..(h + 1) * L_PAD * KEYS],
                &bits(&v_pad),
                L_PAD,
                KEYS,
                VPAD,
            )?;
            for t in 0..L_PAD {
                let ao = t * q_dim + h * HEAD_DIM;
                attn_out[ao..ao + HEAD_DIM].copy_from_slice(&o[t * VPAD..t * VPAD + HEAD_DIM]);
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
        let (n_embd, q_dim, kv_dim, n_ff) = (self.n_embd, self.q_dim, self.kv_dim, self.n_ff);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        // get_rows (host): seed the token block; pad positions get token 0.
        let mut x = vec![0.0f32; L_PAD * n_embd];
        for t in 0..L_PAD {
            let id = if t < l { tokens[t] as usize } else { 0 };
            if id >= self.vocab {
                return Err(format!("token id {id} >= vocab {}", self.vocab).into());
            }
            x[t * n_embd..(t + 1) * n_embd]
                .copy_from_slice(&self.token_embd[id * n_embd..(id + 1) * n_embd]);
        }

        let (cos_q, sin_q) = self.rope_tables(self.n_head, scale);
        let (cos_k, sin_k) = self.rope_tables(self.n_kv, 1.0);

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
            let xnw = self.norm_mul(&resid, &ly.attn_norm)?;
            let q = self.gemm_bcm("gemm_512x1024x2048_bcm", &xnw, &ly.wq, L_PAD, n_embd, q_dim)?;
            let k = self.gemm_bcm(
                "gemm_512x1024x1024_bcm",
                &xnw,
                &ly.wk,
                L_PAD,
                n_embd,
                kv_dim,
            )?;
            let v = self.gemm_bcm(
                "gemm_512x1024x1024_bcm",
                &xnw,
                &ly.wv,
                L_PAD,
                n_embd,
                kv_dim,
            )?;
            let q_roped = self.qk_norm_rope(&bits(&q), self.n_head, &ly.q_norm, &cos_q, &sin_q)?;
            let k_roped = self.qk_norm_rope(&bits(&k), self.n_kv, &ly.k_norm, &cos_k, &sin_k)?;
            let attn = self.attention(&q_roped, &k_roped, &v)?;
            let proj = self.gemm_bcm(
                "gemm_512x2048x1024_bcm",
                &bits(&attn),
                &ly.wo,
                L_PAD,
                q_dim,
                n_embd,
            )?;
            for (r, p) in resid.iter_mut().zip(&proj) {
                *r += p;
            }

            // ---- FFN block ----
            let xn2w = self.norm_mul(&resid, &ly.ffn_norm)?;
            let g = self.gemm_bcm(
                "gemm_512x1024x3072_bcm",
                &xn2w,
                &ly.gate,
                L_PAD,
                n_embd,
                n_ff,
            )?;
            let u = self.gemm_bcm("gemm_512x1024x3072_bcm", &xn2w, &ly.up, L_PAD, n_embd, n_ff)?;
            // SwiGLU: silu(gate) * up. Default = host f32 (the whole product, so it isn't
            // rounded to bf16 early and the default path needs no eltwise/activation
            // xclbins — only the GEMMs). On-chip path uses the bf16 silu + mul kernels.
            let hidden = if env_host("SEEKER_NPU_ONCHIP_OPS") {
                let gs =
                    self.kernels
                        .run("activation", "silu_1572864", &[&bits(&g)], L_PAD * n_ff)?;
                self.kernels.run(
                    "eltwise",
                    "eltwise_mul_bf16_1572864",
                    &[&gs, &bits(&u)],
                    L_PAD * n_ff,
                )?
            } else {
                bits(
                    &g.iter()
                        .zip(&u)
                        .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
                        .collect::<Vec<_>>(),
                )
            };
            let dn = self.gemm_bcm(
                "gemm_512x3072x1024_bcm",
                &hidden,
                &ly.down,
                L_PAD,
                n_ff,
                n_embd,
            )?;
            for (r, d) in resid.iter_mut().zip(&dn) {
                *r += d;
            }
        }

        // Return the real tokens' residual (token-major), already f32.
        resid.truncate(l * n_embd);
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
