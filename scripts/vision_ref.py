#!/usr/bin/env python3
"""Qwen3-VL (mmproj) vision reference harness for seeker-rs.

Phase-2 Slice-1 scope: this script is the *numerical reference* for the CPU
preprocessing implemented in `src/vision/preprocess.rs`. It

  * reads the mmproj GGUF metadata + tensors via llama.cpp's `gguf-py`,
  * implements `smart_resize` + preprocessing **identically** to the Rust side
    (same rounding, same corner-aligned bilinear resize, same planar layout),
  * can dump the preprocessed pixel-tensor stats (shape, first/last values,
    checksum) for a given image, and
  * loads the ViT/merger/deepstack weights and asserts their shapes against the
    known Qwen3-VL architecture, leaving STUBS for the later forward pass.

It does NOT implement the ViT forward yet — only preprocessing + token-count +
weight-shape assertions, so later slices can fill in `vit_forward_stub`.

Reference: llama.cpp `src/tools/mtmd/mtmd-image.cpp`
  (`calc_size_preserved_ratio` @156, `resize_bilinear` @212) and `clip.cpp`
  (qwen3vl token-count @3162/3185, planar `inp_raw` layout @3403).

Usage:
    # Validation-gate path (smart_resize + token count; ZERO third-party deps —
    # use this to reproduce the resized-dims/token-count agreement):
    python3 scripts/vision_ref.py --smart-resize 200x150

    # Full path (reads GGUF + decodes image; needs numpy + pillow + gguf-py):
    python3 scripts/vision_ref.py --mmproj <mmproj.gguf> --image <img> [--dump-prep]
    python3 scripts/vision_ref.py --make-test-image /tmp/viz_test.png

Dependencies:
  * `--smart-resize WxH` needs only the Python stdlib.
  * the full `--mmproj/--image` path additionally needs numpy, pillow, and
    llama.cpp's gguf-py (auto-located; override with --gguf-py). The script
    degrades gracefully and tells you exactly which dep is missing.
"""

import argparse
import math
import os
import sys

# --- llama.cpp qwen3vl preprocessing defaults (no GGUF min/max-pixel keys) ---
# clip.cpp QWEN3VL case: set_limit_image_tokens(8, 4096); clip-model.h:
# patch_area = patch_size^2 * n_merge^2; image_{min,max}_pixels =
# {min,max}_tokens * patch_area.
QWEN3VL_DEFAULT_MIN_TOKENS = 8
QWEN3VL_DEFAULT_MAX_TOKENS = 4096

DEFAULT_GGUF_PY_CANDIDATES = [
    "/home/bob/tools/llama.cpp/src/gguf-py",
    "/home/bob/tools/llama.cpp/gguf-py",
]


def _import_gguf(explicit):
    paths = []
    if explicit:
        paths.append(explicit)
    paths.extend(DEFAULT_GGUF_PY_CANDIDATES)
    for p in paths:
        if os.path.isdir(p):
            sys.path.insert(0, p)
            try:
                import gguf  # noqa: F401

                return gguf
            except Exception:  # pragma: no cover - fall through to next
                sys.path.pop(0)
                continue
    try:
        import gguf  # noqa: F401

        return gguf
    except Exception as e:  # pragma: no cover
        sys.exit(
            f"error: could not import gguf-py (tried {paths}); pip install gguf "
            f"or pass --gguf-py. ({e})"
        )


# ----------------------------- preprocessing -------------------------------- #


def _round_by_factor(x, f):
    # C++ std::round / Rust f32::round are round-half-AWAY-from-zero; Python's
    # built-in round() is round-half-to-even, so emulate away-from-zero to be
    # byte-identical to llama.cpp and the Rust port.
    q = x / f
    return int(math.floor(q + 0.5) if q >= 0 else math.ceil(q - 0.5)) * f


def _ceil_by_factor(x, f):
    return int(math.ceil(x / f)) * f


def _floor_by_factor(x, f):
    return int(math.floor(x / f)) * f


def smart_resize(w, h, align, min_pixels, max_pixels):
    """Port of calc_size_preserved_ratio (mtmd-image.cpp:156). Returns (w,h)."""
    assert align > 0
    h_bar = max(align, _round_by_factor(h, align))
    w_bar = max(align, _round_by_factor(w, align))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(align, _floor_by_factor(h / beta, align))
        w_bar = max(align, _floor_by_factor(w / beta, align))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = _ceil_by_factor(h * beta, align)
        w_bar = _ceil_by_factor(w * beta, align)
    return w_bar, h_bar


def resize_bilinear_u8(src, src_w, src_h, dst_w, dst_h):
    """Corner-aligned bilinear on interleaved RGB8 (mtmd-image.cpp:212).

    src: numpy uint8 array shape (src_h, src_w, 3). Returns (dst_h, dst_w, 3).

    IMPORTANT: llama.cpp computes the whole interpolation in C++ `float`
    (32-bit) and then `std::lround`s to u8. At pixels whose interpolated value
    lands on a .5 boundary, f64 vs f32 arithmetic flips the result by 1 LSB, so
    we stay in float32 to be byte-identical to C++ (and to the Rust port, which
    also uses f32).
    """
    import numpy as np

    f32 = np.float32
    dst_w = max(1, dst_w)
    dst_h = max(1, dst_h)
    x_ratio = f32(src_w - 1) / f32(dst_w - 1) if dst_w > 1 else f32(0.0)
    y_ratio = f32(src_h - 1) / f32(dst_h - 1) if dst_h > 1 else f32(0.0)

    xs = (np.arange(dst_w, dtype=f32) * x_ratio).astype(f32)
    ys = (np.arange(dst_h, dtype=f32) * y_ratio).astype(f32)
    x0 = np.minimum(xs.astype(np.int64), src_w - 1)
    y0 = np.minimum(ys.astype(np.int64), src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    xf = (xs - x0.astype(f32))[None, :, None].astype(f32)
    yf = (ys - y0.astype(f32))[:, None, None].astype(f32)

    s = src.astype(f32)
    a = s[y0[:, None], x0[None, :]]
    b = s[y0[:, None], x1[None, :]]
    c = s[y1[:, None], x0[None, :]]
    d = s[y1[:, None], x1[None, :]]
    top = (a + (b - a) * xf).astype(f32)
    bot = (c + (d - c) * xf).astype(f32)
    out = (top + (bot - top) * yf).astype(f32)
    # std::lround == round-half-away-from-zero (values are >= 0 here).
    out = np.floor(out.astype(np.float64) + 0.5)
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(rgb, cfg):
    """rgb: numpy uint8 (H,W,3). cfg: dict. Returns dict of preprocessed data."""
    import numpy as np

    src_h, src_w = rgb.shape[0], rgb.shape[1]
    align = cfg["patch_size"] * cfg["spatial_merge_size"]
    rw, rh = smart_resize(src_w, src_h, align, cfg["min_pixels"], cfg["max_pixels"])
    resized = resize_bilinear_u8(rgb, src_w, src_h, rw, rh)  # (rh, rw, 3)

    mean = np.array(cfg["image_mean"], dtype=np.float32)
    std = np.array(cfg["image_std"], dtype=np.float32)
    norm = (resized.astype(np.float32) / 255.0 - mean) / std  # (rh, rw, 3)

    # Planar / channel-major to match llama.cpp inp_raw (clip.cpp:3403):
    # pixels[c*H*W + y*W + x]. norm is (H,W,C) -> transpose to (C,H,W) -> flatten.
    planar = np.transpose(norm, (2, 0, 1)).reshape(-1).astype(np.float32)

    patch = cfg["patch_size"]
    merge = cfg["spatial_merge_size"]
    grid_w, grid_h = rw // patch, rh // patch
    n_tokens = (rw // (patch * merge)) * (rh // (patch * merge))
    return {
        "pixels": planar,
        "resized_w": rw,
        "resized_h": rh,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "n_tokens": n_tokens,
    }


# ------------------------------- gguf config -------------------------------- #


def field_value(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        return f.contents()
    except Exception:
        return default


def build_config(reader):
    patch_size = int(field_value(reader, "clip.vision.patch_size", 16))
    merge = int(field_value(reader, "clip.vision.spatial_merge_size", 2))
    mean = field_value(reader, "clip.vision.image_mean", [0.5, 0.5, 0.5])
    std = field_value(reader, "clip.vision.image_std", [0.5, 0.5, 0.5])
    mean = [float(x) for x in mean]
    std = [float(x) for x in std]

    min_px = field_value(reader, "clip.vision.image_min_pixels")
    max_px = field_value(reader, "clip.vision.image_max_pixels")
    if min_px is None or max_px is None:
        patch_area = patch_size * patch_size * merge * merge
        min_px = QWEN3VL_DEFAULT_MIN_TOKENS * patch_area
        max_px = QWEN3VL_DEFAULT_MAX_TOKENS * patch_area
        px_src = "qwen3vl default (set_limit_image_tokens(8,4096))"
    else:
        min_px, max_px = int(min_px), int(max_px)
        px_src = "GGUF clip.vision.image_{min,max}_pixels"

    return {
        "projector_type": field_value(reader, "clip.projector_type"),
        "patch_size": patch_size,
        "spatial_merge_size": merge,
        "image_mean": mean,
        "image_std": std,
        "min_pixels": int(min_px),
        "max_pixels": int(max_px),
        "pixels_source": px_src,
        "n_embd": int(field_value(reader, "clip.vision.embedding_length", 1152)),
        "n_layer": int(field_value(reader, "clip.vision.block_count", 27)),
        "n_head": int(field_value(reader, "clip.vision.attention.head_count", 16)),
        "n_ff": int(field_value(reader, "clip.vision.feed_forward_length", 4304)),
        "image_size": int(field_value(reader, "clip.vision.image_size", 768)),
        "eps": float(
            field_value(reader, "clip.vision.attention.layer_norm_epsilon", 1e-6)
        ),
    }


# ------------------------- weight loading + shape asserts ------------------- #


def load_and_assert_weights(reader, cfg):
    """Load tensors, assert key shapes vs the Qwen3-VL architecture.

    gguf-py reports tensor.shape in *reversed* (numpy) order vs the logical
    [out, in] GGUF order. We assert against the numpy shapes seen in the file.
    Returns {name: tensor} for the asserted tensors (handy for the later
    forward stub).
    """
    byname = {t.name: t for t in reader.tensors}
    n_embd = cfg["n_embd"]
    n_ff = cfg["n_ff"]

    def shp(name):
        t = byname.get(name)
        return None if t is None else tuple(int(x) for x in t.shape)

    def want(name, expected):
        got = shp(name)
        assert got is not None, f"missing tensor {name}"
        assert got == expected, f"{name}: got {got}, expected {expected}"

    ps = cfg["patch_size"]
    # Patch embed: dual conv2d [patch, patch, 3, n_embd] (numpy order) + bias.
    want("v.patch_embd.weight", (ps, ps, 3, n_embd))
    want("v.patch_embd.weight.1", (ps, ps, 3, n_embd))
    want("v.patch_embd.bias", (n_embd,))
    # Learned pos-embd: [n_embd, 2304] = 48*48 base grid.
    want("v.position_embd.weight", (n_embd, 2304))
    # post-LN affine + merger.
    want("v.post_ln.weight", (n_embd,))
    want("v.post_ln.bias", (n_embd,))
    want("mm.0.weight", (n_embd * 4, 2048))  # 4608 -> 2048
    want("mm.0.bias", (2048,))
    want("mm.1.weight", (2048, 2048))
    want("mm.1.bias", (2048,))
    # One representative ViT block (separate Q/K/V each [n_embd, n_embd]).
    for proj in ("attn_q", "attn_k", "attn_v", "attn_out"):
        want(f"v.blk.0.{proj}.weight", (n_embd, n_embd))
    want("v.blk.0.ffn_up.weight", (n_embd, n_ff))
    want("v.blk.0.ffn_down.weight", (n_ff, n_embd))
    # Deepstack side-channels (qwen3vl): 3 of them.
    for i in range(3):
        want(f"v.deepstack.{i}.norm.weight", (n_embd * 4,))
        want(f"v.deepstack.{i}.fc1.weight", (n_embd * 4, 2048))
        want(f"v.deepstack.{i}.fc2.weight", (2048, 2048))

    return byname


def vit_forward_stub(weights, prep, cfg):  # pragma: no cover - intentional stub
    """STUB for the later ViT forward (patch-embed -> blocks -> merger).

    Later slices will implement: dual conv2d patch-embed (+bias) -> 2x2 merge ->
    add interpolated learned pos-embd -> 27 ViT blocks (LN-affine, separate QKV +
    vision 2D-RoPE theta=10000, full attn, GELU FFN) -> post-LN -> merger
    (mm.0 -> GELU -> mm.1) -> [2048, n_tokens], plus deepstack at blocks 7/15/23.
    Not implemented in this slice.
    """
    raise NotImplementedError(
        "ViT forward is a later-slice deliverable; preprocessing-only here."
    )


# --------------------------------- helpers ---------------------------------- #


def make_test_image(path, w=200, h=150):
    """Deterministic gradient PNG (same formula as the Rust unit test)."""
    import numpy as np
    from PIL import Image

    arr = np.zeros((h, w, 3), dtype=np.uint8)
    xs = np.arange(w, dtype=np.int64)
    ys = np.arange(h, dtype=np.int64)
    arr[:, :, 0] = (xs % 256)[None, :]
    arr[:, :, 1] = (ys % 256)[:, None]
    arr[:, :, 2] = (xs[None, :] + ys[:, None]) % 256
    Image.fromarray(arr, "RGB").save(path)
    return path


def stats(name, a):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    head = ", ".join(f"{v:.6f}" for v in a.reshape(-1)[:5])
    tail = ", ".join(f"{v:.6f}" for v in a.reshape(-1)[-5:])
    print(
        f"  {name}: shape={a.shape} sum={a.sum():.6f} mean={a.mean():.6f} "
        f"min={a.min():.6f} max={a.max():.6f}"
    )
    print(f"    first5=[{head}]")
    print(f"    last5 =[{tail}]")


# ----------------------------------- main ----------------------------------- #


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mmproj", help="path to mmproj GGUF")
    ap.add_argument("--image", help="path to an input image")
    ap.add_argument(
        "--dump-prep",
        action="store_true",
        help="print preprocessed pixel-tensor stats + checksum",
    )
    ap.add_argument(
        "--dump-bin",
        metavar="PATH",
        help="also write the planar f32 pixel tensor (little-endian) here",
    )
    ap.add_argument(
        "--make-test-image",
        metavar="PATH",
        help="write a deterministic 200x150 gradient PNG and exit",
    )
    ap.add_argument("--gguf-py", help="explicit path to llama.cpp gguf-py dir")
    ap.add_argument(
        "--smart-resize",
        metavar="WxH",
        help="compute smart_resize + n_tokens for the given input size using "
        "the QWEN3-VL defaults (stdlib only; no numpy/pillow/gguf needed). "
        "Override the defaults with --patch/--merge/--min-pixels/--max-pixels.",
    )
    ap.add_argument("--patch", type=int, default=16, help="patch_size (default 16)")
    ap.add_argument(
        "--merge", type=int, default=2, help="spatial_merge_size (default 2)"
    )
    ap.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="override min_pixels (default: qwen3vl 8*patch_area)",
    )
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="override max_pixels (default: qwen3vl 4096*patch_area)",
    )
    args = ap.parse_args()

    # --- stdlib-only validation-gate path: smart_resize + token count ---
    if args.smart_resize:
        try:
            w_str, h_str = args.smart_resize.lower().split("x")
            w, h = int(w_str), int(h_str)
        except Exception:
            ap.error("--smart-resize expects WxH, e.g. 200x150")
        patch, merge = args.patch, args.merge
        align = patch * merge
        patch_area = patch * patch * merge * merge
        min_px = (
            args.min_pixels
            if args.min_pixels is not None
            else QWEN3VL_DEFAULT_MIN_TOKENS * patch_area
        )
        max_px = (
            args.max_pixels
            if args.max_pixels is not None
            else QWEN3VL_DEFAULT_MAX_TOKENS * patch_area
        )
        rw, rh = smart_resize(w, h, align, min_px, max_px)
        n_tokens = (rw // align) * (rh // align)
        print(f"input         = {w}x{h}")
        print(f"align         = {align} (patch {patch} * merge {merge})")
        print(f"min/max pixels= {min_px}/{max_px}")
        print(f"resized (WxH) = {rw}x{rh}")
        print(f"grid (patches)= {rw // patch}x{rh // patch}")
        print(f"n_tokens      = {n_tokens}")
        return

    if args.make_test_image:
        p = make_test_image(args.make_test_image)
        print(f"wrote deterministic test image: {p}")
        return

    if not args.mmproj:
        ap.error("--mmproj is required (unless --make-test-image or --smart-resize)")

    gguf = _import_gguf(args.gguf_py)
    import numpy as np

    reader = gguf.GGUFReader(args.mmproj)
    cfg = build_config(reader)

    print("=== mmproj vision config ===")
    print(f"  projector_type     = {cfg['projector_type']}")
    print(f"  patch_size         = {cfg['patch_size']}")
    print(f"  spatial_merge_size = {cfg['spatial_merge_size']}")
    print(f"  align (patch*merge)= {cfg['patch_size'] * cfg['spatial_merge_size']}")
    print(
        f"  n_embd/n_layer/n_head/n_ff = "
        f"{cfg['n_embd']}/{cfg['n_layer']}/{cfg['n_head']}/{cfg['n_ff']}"
    )
    print(f"  image_mean         = {cfg['image_mean']}")
    print(f"  image_std          = {cfg['image_std']}")
    print(
        f"  min/max pixels     = {cfg['min_pixels']}/{cfg['max_pixels']}  "
        f"[{cfg['pixels_source']}]"
    )

    print("=== weight shape assertions ===")
    load_and_assert_weights(reader, cfg)
    print("  all asserted tensor shapes OK")

    if not args.image:
        print("(no --image given; skipping preprocessing)")
        return

    from PIL import Image

    img = Image.open(args.image).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    src_w, src_h = img.width, img.height
    prep = preprocess(rgb, cfg)

    print("=== preprocessing ===")
    print(f"  input              = {src_w}x{src_h}")
    print(f"  resized (WxH)      = {prep['resized_w']}x{prep['resized_h']}")
    print(f"  grid (WxH patches) = {prep['grid_w']}x{prep['grid_h']}")
    print(f"  n_tokens           = {prep['n_tokens']}")

    if args.dump_bin:
        prep["pixels"].astype("<f4").tofile(args.dump_bin)
        print(f"  wrote planar f32   -> {args.dump_bin} ({prep['pixels'].size} floats)")

    if args.dump_prep:
        print("=== preprocessed tensor (planar RGB, channel-major) ===")
        stats("pixels", prep["pixels"])
        chk = float(np.sum(prep["pixels"].astype(np.float64)))
        print(f"  checksum(sum f64)  = {chk:.6f}")


if __name__ == "__main__":
    main()
