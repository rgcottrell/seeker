#!/usr/bin/env bash
#
# One-time per clone: point git at the source-controlled hooks in .githooks/.
# Git does not track .git/hooks, so each clone must opt in once. core.hooksPath
# is stored in the repo config (shared across all worktrees of this repo).
#
# build.rs normally does this automatically on the first build; this script is
# the manual fallback (e.g. pushing a docs-only change without ever building).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
echo "core.hooksPath -> .githooks (pre-push gate active for this repo)."
