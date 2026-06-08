#!/usr/bin/env bash
# Clean B=1 losslessness: at --parallel 1 (no batching at all), spec output
# must equal non-spec output for each prompt, greedy. Uses chat completions
# (proper template) so the continuation is well-defined.
set -u
BIN=./target/release/seeker
MODEL=/models/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
PORT=11455
CTX=4096
NTOK=80
OUT=/tmp/serve_spec_b1
mkdir -p "$OUT"
PROMPT_A="Explain in three sentences why the sky appears blue during the day."
PROMPT_B="Write a short haiku about a mountain stream in winter."

SRV=
cleanup() { [[ -n "$SRV" ]] && kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
trap cleanup EXIT

start() { # $1 n_max
  pkill -x seeker 2>/dev/null; sleep 1
  SEEKER_SPEC_DEBUG=1 "$BIN" serve -m "$MODEL" --port "$PORT" --no-mmproj \
    --parallel 1 --ctx-size "$CTX" --temp 0 --spec-draft-n-max "$1" \
    >"$OUT/srv_$1.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "DIED($1)"; tail -15 "$OUT/srv_$1.log"; return 1; }
    sleep 1
  done; echo "timeout($1)"; return 1
}
chat() { # $1 outfile  $2 prompt
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$(jq -Rs . <<<"$2")}],\"max_tokens\":$NTOK,\"temperature\":0,\"stream\":false}" \
    >"$OUT/$1.json" || { echo "chat() request failed for: $2" >&2; exit 1; }
  jq -er '.choices[0].message.content' "$OUT/$1.json" >"$OUT/$1.txt" \
    || { echo "chat() no content (error response) for: $2" >&2; cat "$OUT/$1.json" >&2; exit 1; }
}

start 0 || exit 1; chat ns_A "$PROMPT_A"; chat ns_B "$PROMPT_B"; kill "$SRV"; wait "$SRV" 2>/dev/null; SRV=
start 2 || exit 1; chat sp_A "$PROMPT_A"; chat sp_B "$PROMPT_B"; kill "$SRV"; wait "$SRV" 2>/dev/null; SRV=

echo "=== B=1 LOSSLESS (parallel 1, chat, greedy) ==="
pass=1
for p in A B; do
  if diff -q "$OUT/sp_$p.txt" "$OUT/ns_$p.txt" >/dev/null 2>&1; then echo "PASS  prompt $p (spec == nospec)";
  else echo "FAIL  prompt $p"; echo "  spec:"; sed 's/^/    /' "$OUT/sp_$p.txt"; echo "  nospec:"; sed 's/^/    /' "$OUT/ns_$p.txt"; pass=0; fi
done
echo; echo "spec acceptance:"; grep "SPEC batched" "$OUT/srv_2.log" | tail -10
[[ $pass -eq 1 ]] && echo "ALL PASS" || echo "FAILED"
