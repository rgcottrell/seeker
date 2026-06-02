# Multistage build for `seeker serve`. Stage 1 builds the release binary against
# glibc with slangc-compiled shaders baked in; stage 2 is a slim Debian runtime
# carrying the Vulkan loader + Mesa RADV so the binary can drive an AMD GPU
# passed in at `podman run`.
#
# seeker is GPU-only (Vulkan via Slang) and `ash` dlopen()s libvulkan at runtime,
# so a static-musl/scratch image can't work — the runtime needs the loader + an
# ICD present. Shaders are baked into the binary by build.rs, so no shader files
# ship in the runtime layer.

# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM rust:bookworm AS builder
ARG SLANG_VERSION=2026.8

# openssl-sys (hf-hub) links system OpenSSL; curl+jq fetch the slang release.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl-dev pkg-config curl jq ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# build.rs compiles the .slang shaders with slangc — fetch the standalone
# linux-x86_64 release for the pinned version. The selector matches the arch
# immediately followed by .tar.gz so it skips the `-debug-info` (no slangc) and
# `-glibc-*` variants and lands on the generic release. Symlink the real slangc
# into /usr/local/bin (already on PATH); glibc resolves the symlink so slangc's
# $ORIGIN-relative RPATH still finds its sibling libs.
RUN set -eux; \
    url=$(curl -fsSL "https://api.github.com/repos/shader-slang/slang/releases/tags/v${SLANG_VERSION}" \
        | jq -r '.assets[] | select(.name | test("linux-x86_64\\.tar\\.gz$")) | .browser_download_url'); \
    test -n "$url" && test "$url" != "null"; \
    curl -fsSL "$url" -o /tmp/slang.tar.gz; \
    mkdir -p /opt/slang; tar -xzf /tmp/slang.tar.gz -C /opt/slang; \
    real=$(find /opt/slang -name slangc -type f | head -1); \
    test -n "$real"; \
    ln -s "$real" /usr/local/bin/slangc; \
    slangc -v

WORKDIR /build
COPY . .
RUN cargo build --release --bin seeker

# ── Stage 2: runtime ────────────────────────────────────────────────────────
# Ubuntu 26.04 carries Mesa 26.0.3, whose RADV recognizes recent AMD parts
# (e.g. Strix Halo / gfx1151). Debian bookworm's Mesa (~22.3) is too old and
# fails device init with `amdgpu: unknown (family_id, …)`. The glibc-2.36 binary
# from the bookworm builder runs fine on Ubuntu's newer glibc (forward-compatible).
FROM ubuntu:26.04 AS runtime

# libvulkan1 = Vulkan loader; mesa-vulkan-drivers = RADV ICD for AMD;
# ca-certificates pulls openssl/libssl (the binary's libssl.so.3 runtime dep) and
# provides the CA bundle for HTTPS (HF downloads / TLS).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/seeker /usr/local/bin/seeker

EXPOSE 11434
ENTRYPOINT ["seeker"]
CMD ["serve", "--host", "0.0.0.0"]
