#!/usr/bin/env bash
# Validate concurrent batched speculative decoding in `seeker serve`:
#   - lossless: spec output (temp=0 greedy) == non-spec output, per prompt
#   - concurrency-independent: a prompt's output is the same solo vs in a 2-batch
# Greedy (temp 0) makes every path byte-comparable.
set -u

BIN=./target/release/seeker
MODEL=/models/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
PORT=11455
CTX=4096
NTOK=48
OUT=/tmp/serve_spec_val
mkdir -p "$OUT"

PROMPT_A="Explain in three sentences why the sky appears blue during the day."
PROMPT_B="Write a short haiku about a mountain stream in winter."

cleanup() { [[ -n "${SRV:-}" ]] && kill "$SRV" 2>/dev/null; wait "${SRV:-}" 2>/dev/null; }
trap cleanup EXIT

start_server() { # $1 = n_max
  pkill -x seeker 2>/dev/null; sleep 1
  SEEKER_SPEC_DEBUG=1 "$BIN" serve -m "$MODEL" --port "$PORT" --no-mmproj \
    --parallel 2 --ctx-size "$CTX" --temp 0 \
    --spec-draft-n-max "$1" >"$OUT/server_$1.log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$SRV" 2>/dev/null; then echo "SERVER DIED (n_max=$1):"; tail -20 "$OUT/server_$1.log"; return 1; fi
    sleep 1
  done
  echo "server readiness timeout"; tail -20 "$OUT/server_$1.log"; return 1
}

# completion → choices[0].text, into $1=outfile, run in background; echoes PID
req() { # $1 outfile  $2 prompt
  curl -sf "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":$(jq -Rs . <<<"$2"),\"max_tokens\":$NTOK,\"temperature\":0,\"stream\":false}" \
    | jq -er '.choices[0].text' >"$1" || { echo "req() failed for: $2" >&2; exit 1; }
}

run_suite() { # $1 = tag (nospec|spec)
  local tag=$1
  req "$OUT/${tag}_A_solo.txt" "$PROMPT_A";
  req "$OUT/${tag}_B_solo.txt" "$PROMPT_B"
  # concurrent: fire both at once so they batch
  req "$OUT/${tag}_A_conc.txt" "$PROMPT_A" & local pa=$!
  req "$OUT/${tag}_B_conc.txt" "$PROMPT_B" & local pb=$!
  wait "$pa" "$pb"
}

echo "=== non-spec (n_max=0) ==="
start_server 0 || exit 1
run_suite nospec
kill "$SRV"; wait "$SRV" 2>/dev/null; SRV=

echo "=== spec (n_max=2) ==="
start_server 2 || exit 1
run_suite spec
kill "$SRV"; wait "$SRV" 2>/dev/null; SRV=

echo
echo "=== RESULTS ==="
pass=1
chk() { # $1 label  $2 fileX  $3 fileY
  if diff -q "$2" "$3" >/dev/null 2>&1; then echo "PASS  $1"; else echo "FAIL  $1"; pass=0;
    echo "  --- $2:"; sed 's/^/      /' "$2"; echo "  --- $3:"; sed 's/^/      /' "$3"; fi
}
chk "lossless A (spec solo == nospec solo)" "$OUT/spec_A_solo.txt" "$OUT/nospec_A_solo.txt"
chk "lossless B (spec solo == nospec solo)" "$OUT/spec_B_solo.txt" "$OUT/nospec_B_solo.txt"
chk "concurrency-indep A (spec conc == spec solo)" "$OUT/spec_A_conc.txt" "$OUT/spec_A_solo.txt"
chk "concurrency-indep B (spec conc == spec solo)" "$OUT/spec_B_conc.txt" "$OUT/spec_B_solo.txt"
chk "nospec concurrency-indep A" "$OUT/nospec_A_conc.txt" "$OUT/nospec_A_solo.txt"
chk "nospec concurrency-indep B" "$OUT/nospec_B_conc.txt" "$OUT/nospec_B_solo.txt"
echo
echo "spec acceptance (from server log):"
grep "SPEC batched" "$OUT/server_2.log" | tail -8
[[ $pass -eq 1 ]] && echo "ALL PASS" || echo "SOME FAILED"
