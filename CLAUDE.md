# seeker-rs

GPU LLM inference engine in Rust — Vulkan compute via Slang shaders.

## Commit gate

The tree must stay warning-clean on **formatting, clippy, and the build**.
Before every commit, run:

```sh
scripts/check.sh          # verify: cargo fmt --check, clippy -D warnings, build
scripts/check.sh --fix    # auto-format + apply clippy fixes, then verify
```

Do not commit unless `scripts/check.sh` passes. CI
(`.github/workflows/ci.yml`) runs the same checks on every push / PR.

### Git hook (pre-push)

A source-controlled `pre-push` hook (`.githooks/pre-push`) runs the gate
automatically before any push, so WIP commits stay frictionless but a branch is
verified before it leaves your machine. Enable it once per clone:

```sh
scripts/install-hooks.sh   # sets core.hooksPath -> .githooks
```

Bypass a single push with `git push --no-verify`.

## Build dependency

`build.rs` compiles the `.slang` compute shaders with `slangc`, which ships
with the Vulkan SDK (the tree is built against slang 2026.8). `slangc` must be
on `PATH`. Building and clippy need only `slangc`; running the tests / actual
inference additionally needs a Vulkan device.
