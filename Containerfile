# Multistage build for `seeker serve`. Stage 1 builds the release binary;
# stage 2 is a slim runtime with the Vulkan loader + Mesa RADV. seeker is
# GPU-only and dlopen()s libvulkan at runtime, so it needs the loader + an ICD
# present; shaders are baked into the binary by build.rs.

# ── Stage 1: build ──────────────────────────────────────────────────────────
# Builder Rust version comes from the justfile, which extracts it from
# rust-toolchain.toml (the single source of truth). Build via `just build` so the
# arg is supplied; a bare `podman build` must pass --build-arg RUST_VERSION=<ver>.
ARG RUST_VERSION
FROM rust:${RUST_VERSION}-bookworm AS builder
ARG SLANG_VERSION=2026.8

# openssl-sys (hf-hub) links system OpenSSL; curl+jq fetch the slang release.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl-dev pkg-config curl jq ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# build.rs needs slangc. The selector matches the arch immediately before
# .tar.gz to skip the -debug-info and -glibc-* variants. Symlink onto PATH;
# the $ORIGIN-relative RPATH still finds slangc's sibling libs.
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
# fails device init.
FROM ubuntu:26.04 AS runtime

# libvulkan1 = Vulkan loader; mesa-vulkan-drivers = RADV ICD; ca-certificates
# = libssl runtime dep + CA bundle for HTTPS (HF downloads).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/seeker /usr/local/bin/seeker

EXPOSE 11434
ENTRYPOINT ["seeker"]
# A usable invocation needs a model, GPU device passthrough, and a port mapping
# — too much to bake in here. Drive it via the justfile (`just serve` / `just
# chat`), which supply the full `serve`/`chat` command and args. Bare runs just
# print help.
CMD ["--help"]
