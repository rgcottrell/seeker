#!/usr/bin/env bash
#
# Pre-commit / CI-parity checks. The tree must be warning-clean on all three:
#   1. formatting   (cargo fmt --check)
#   2. clippy       (cargo clippy ... -D warnings  — also covers rustc warnings)
#   3. build        (cargo build  — codegen sanity)
#
# Run before every commit. `.github/workflows/ci.yml` runs this same script.
#
#   scripts/check.sh         verify only (non-zero exit if anything is dirty/warns)
#   scripts/check.sh --fix   auto-format + apply clippy fixes first, then verify
#
# Building and clippy need `slangc` on PATH (build.rs compiles the .slang
# shaders); running the test suite additionally needs a Vulkan device, so tests
# are intentionally not part of this gate.
set -euo pipefail

cd "$(dirname "$0")/.."

fix=0
[ "${1:-}" = "--fix" ] && fix=1

run() { echo "+ $*" >&2; "$@"; }

if [ "$fix" = "1" ]; then
  run cargo fmt
  run cargo clippy --all-targets --all-features --fix --allow-dirty --allow-staged
fi

run cargo fmt --check
run cargo clippy --all-targets --all-features -- -D warnings
run cargo build --all-targets --all-features

echo "OK: fmt, clippy, and build are clean." >&2
