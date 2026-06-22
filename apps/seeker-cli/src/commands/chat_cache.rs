//! `--prompt-cache`: freeze a chat session's live cache to a file and thaw it
//! on a later run, so the conversation resumes without re-prefilling.
//!
//! The snapshot is byte-exact: the attention K/V for positions `[0, position)`
//! of every layer, the full SSM/GDN recurrent region (hybrid models), the token
//! ids, and the conversation messages, behind a header that validates the cache
//! belongs to the same model + cache config. Restoring reproduces the in-memory
//! state the process had at save time, so the next turn behaves exactly as if
//! the process had never exited (the freeze/thaw introduces no new state).
//!
//! On any mismatch / corruption the loader returns an `Err`; the caller treats
//! that as "ignore the cache and start fresh" rather than failing the session.

use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::chat_template::ChatMessage;
use crate::gguf::GgmlType;
use crate::inference::kv_cache::KvCache;

const MAGIC: &[u8; 4] = b"SKC1";
const VERSION: u32 = 1;

/// Save the live cache + tokens + messages to `path`. `arch` is the GGUF
/// architecture string, stored for validation on load.
pub fn save(
    path: &Path,
    arch: &str,
    cache: &KvCache,
    tokens: &[u32],
    messages: &[ChatMessage],
) -> Result<(), Box<dyn Error>> {
    let k0 = cache
        .k_layers
        .first()
        .ok_or("prompt-cache: cache has no layers")?;
    let v0 = &cache.v_layers[0];
    let position = cache.position;
    let k_stride = k0.byte_stride[2]; // bytes per token position, all heads
    let v_stride = v0.byte_stride[2];
    let msg_json = serde_json::to_vec(messages)?;
    let (has_ssm, ssm_size) = match &cache.ssm_region {
        Some(r) => (1u8, r.size),
        None => (0u8, 0u64),
    };

    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    write_bytes(&mut f, arch.as_bytes())?;
    for v in [
        cache.k_layers.len() as u32,
        k0.dims[0] as u32, // head_dim
        k0.dims[1] as u32, // n_head_kv
        cache.config.k_dtype as u32,
        cache.config.v_dtype as u32,
        cache.config.max_seq_len,
        position,
    ] {
        f.write_all(&v.to_le_bytes())?;
    }
    f.write_all(&k_stride.to_le_bytes())?;
    f.write_all(&v_stride.to_le_bytes())?;
    f.write_all(&[has_ssm])?;
    f.write_all(&ssm_size.to_le_bytes())?;
    f.write_all(&(tokens.len() as u32).to_le_bytes())?;
    for &t in tokens {
        f.write_all(&t.to_le_bytes())?;
    }
    write_bytes(&mut f, &msg_json)?;

    // Payload: per-layer K then V live bytes, then the SSM region.
    let k_live = position as usize * k_stride as usize;
    let v_live = position as usize * v_stride as usize;
    for (il, (k, v)) in cache.k_layers.iter().zip(&cache.v_layers).enumerate() {
        // SAFETY: layer `il`'s K/V live in one HOST_VISIBLE|HOST_COHERENT buffer
        // mapped at `host`; each view's byte_offset + live span lies within it
        // (K at 0, V after it), and the GPU's writes are coherent (the last
        // forward's fence was already awaited).
        let host = cache
            .layer_host_ptr(il)
            .ok_or("prompt-cache: KV region not host-visible")?;
        f.write_all(unsafe { live_slice(host, k.byte_offset, k_live) })?;
        f.write_all(unsafe { live_slice(host, v.byte_offset, v_live) })?;
    }
    if let Some(r) = &cache.ssm_region {
        let p = r
            .host_ptr
            .ok_or("prompt-cache: SSM region not host-visible")?;
        f.write_all(unsafe { std::slice::from_raw_parts(p, r.size as usize) })?;
    }
    f.flush()?;
    Ok(())
}

/// Load a snapshot from `path` into `cache`, returning the restored tokens and
/// messages. `Ok(None)` means there is no cache file yet (a normal first run).
/// `Err` means the file is missing-but-unreadable, corrupt, or for a different
/// model/config — the caller should warn and start fresh.
#[allow(clippy::type_complexity)]
pub fn load(
    path: &Path,
    arch: &str,
    cache: &mut KvCache,
) -> Result<Option<(Vec<u32>, Vec<ChatMessage>)>, Box<dyn Error>> {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let mut r = Reader::new(&data);
    if r.take(4)? != MAGIC {
        return Err("prompt-cache: not a seeker prompt-cache file".into());
    }
    if r.u32()? != VERSION {
        return Err("prompt-cache: unsupported file version".into());
    }
    let file_arch = std::str::from_utf8(r.bytes()?)?;
    let n_layer = r.u32()?;
    let head_dim = r.u32()?;
    let n_head_kv = r.u32()?;
    let k_dtype = r.u32()?;
    let v_dtype = r.u32()?;
    let max_seq_len = r.u32()?;
    let position = r.u32()?;
    let k_stride = r.u64()?;
    let v_stride = r.u64()?;
    let has_ssm = r.take(1)?[0] == 1;
    let ssm_size = r.u64()?;

    // Validate against the live model/cache; any mismatch → reject (the caller
    // starts fresh). This guards against pointing --prompt-cache at a file
    // written for a different model, quant, or --ctx-size.
    let k0 = &cache.k_layers[0];
    let v0 = &cache.v_layers[0];
    let ok = file_arch == arch
        && n_layer as usize == cache.k_layers.len()
        && head_dim == k0.dims[0] as u32
        && n_head_kv == k0.dims[1] as u32
        && k_dtype == cache.config.k_dtype as u32
        && v_dtype == cache.config.v_dtype as u32
        && max_seq_len == cache.config.max_seq_len
        && k_stride == k0.byte_stride[2]
        && v_stride == v0.byte_stride[2]
        && has_ssm == cache.ssm_region.is_some()
        && position <= cache.config.max_seq_len;
    if !ok {
        return Err(format!(
            "prompt-cache: file does not match this model/config (arch {file_arch:?}, \
             {n_layer} layers, ctx {max_seq_len}) — ignoring"
        )
        .into());
    }
    // Round-trip the dtypes through from_u32 so an unknown value is rejected.
    GgmlType::from_u32(k_dtype)?;
    GgmlType::from_u32(v_dtype)?;

    let n_tokens = r.u32()? as usize;
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(r.u32()?);
    }
    let messages: Vec<ChatMessage> = serde_json::from_slice(r.bytes()?)?;

    // Copy the payload back into the live per-layer cache buffers.
    let k_live = position as usize * k_stride as usize;
    let v_live = position as usize * v_stride as usize;
    for (il, (k, v)) in cache.k_layers.iter().zip(&cache.v_layers).enumerate() {
        // SAFETY: layer `il`'s buffer is mapped at `host`; spans validated above
        // to lie within each view's allocation.
        let host = cache
            .layer_host_ptr(il)
            .ok_or("prompt-cache: KV region not host-visible")?;
        let kb = r.take(k_live)?;
        unsafe {
            std::ptr::copy_nonoverlapping(kb.as_ptr(), host.add(k.byte_offset as usize), k_live)
        };
        let vb = r.take(v_live)?;
        unsafe {
            std::ptr::copy_nonoverlapping(vb.as_ptr(), host.add(v.byte_offset as usize), v_live)
        };
    }
    if has_ssm {
        let sr = cache.ssm_region.as_ref().unwrap();
        if ssm_size != sr.size {
            return Err("prompt-cache: SSM region size mismatch — ignoring".into());
        }
        let p = sr
            .host_ptr
            .ok_or("prompt-cache: SSM region not host-visible")?;
        let sb = r.take(ssm_size as usize)?;
        unsafe { std::ptr::copy_nonoverlapping(sb.as_ptr(), p, ssm_size as usize) };
    }
    cache.position = position;
    Ok(Some((tokens, messages)))
}

/// SAFETY: caller guarantees `[offset, offset+len)` is within the region the
/// `host` base maps and is initialized (live KV the GPU has written).
unsafe fn live_slice<'a>(host: *mut u8, offset: u64, len: usize) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(host.add(offset as usize), len) }
}

fn write_bytes<W: Write>(w: &mut W, b: &[u8]) -> io::Result<()> {
    w.write_all(&(b.len() as u32).to_le_bytes())?;
    w.write_all(b)
}

/// Little-endian, bounds-checked cursor over the snapshot bytes.
struct Reader<'a> {
    d: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], Box<dyn Error>> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or("prompt-cache: length overflow")?;
        if end > self.d.len() {
            return Err("prompt-cache: file truncated".into());
        }
        let s = &self.d[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, Box<dyn Error>> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Box<dyn Error>> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Result<&'a [u8], Box<dyn Error>> {
        let n = self.u32()? as usize;
        self.take(n)
    }
}

#[cfg(test)]
mod tests {
    use super::Reader;

    #[test]
    fn reader_parses_and_rejects_truncation() {
        let mut data = Vec::new();
        data.extend_from_slice(b"SKC1");
        data.extend_from_slice(&7u32.to_le_bytes());
        data.extend_from_slice(&(2u32).to_le_bytes()); // len-prefixed bytes
        data.extend_from_slice(b"hi");
        let mut r = Reader::new(&data);
        assert_eq!(r.take(4).unwrap(), b"SKC1");
        assert_eq!(r.u32().unwrap(), 7);
        assert_eq!(r.bytes().unwrap(), b"hi");
        // Past the end → error, not panic.
        assert!(r.u32().is_err());

        // A length prefix that overruns the buffer is rejected.
        let bad = [4u32.to_le_bytes().as_slice(), b"ab"].concat();
        assert!(Reader::new(&bad).bytes().is_err());
    }
}
