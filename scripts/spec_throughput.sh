#!/usr/bin/env bash
# Concurrent throughput: 2 simultaneous N-token generations, wall-clock to both
# done, spec (n_max=2) vs nospec. SEEKER_SPEC_DEBUG prints per-step phase timing.
set -u
BIN=./target/release/seeker
MODEL=/models/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
PORT=11457; CTX=4096; NTOK=200
OUT=/tmp/spec_tput; mkdir -p "$OUT"
PA="Write a detailed, factual paragraph explaining how photosynthesis converts sunlight into chemical energy in plants."
PB="Describe step by step how a four-stroke internal combustion engine works, covering each stroke."
SRV=
trap '[[ -n "$SRV" ]] && kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null' EXIT
start() { pkill -x seeker 2>/dev/null; sleep 1
  SEEKER_SPEC_DEBUG=1 "$BIN" serve -m "$MODEL" --port "$PORT" --no-mmproj --parallel 2 --ctx-size "$CTX" --temp 0 --spec-draft-n-max "$1" >"$OUT/srv_$1.log" 2>&1 &
  SRV=$!; for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0; kill -0 "$SRV" 2>/dev/null || { echo DIED; tail -12 "$OUT/srv_$1.log"; return 1; }; sleep 1; done; echo timeout; return 1; }
gen() { curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":$(jq -Rs . <<<"$1")}],\"max_tokens\":$NTOK,\"temperature\":0,\"stream\":false}" \
  | jq -er '.usage.completion_tokens' >"$2" || { echo "gen() failed for: $1" >&2; exit 1; }; }
measure() { local t0 t1; t0=$(date +%s.%N)
  gen "$PA" "$OUT/$1_a.tok" & local pa=$!
  gen "$PB" "$OUT/$1_b.tok" & local pb=$!
  wait "$pa" "$pb"; t1=$(date +%s.%N)
  echo "$(echo "$t1 - $t0" | bc) $(cat "$OUT/$1_a.tok") $(cat "$OUT/$1_b.tok")"; }
for nm in 0 2; do
  start "$nm" || exit 1
  read -r secs ta tb < <(measure "$nm")
  tot=$((ta + tb)); tps=$(echo "scale=1; $tot / $secs" | bc)
  echo "n_max=$nm : ${secs}s for $tot tokens (A=$ta B=$tb) => ${tps} tok/s aggregate"
  kill "$SRV"; wait "$SRV" 2>/dev/null; SRV=
done
echo "=== per-phase timing (n_max=2, sampled) ==="
grep "SPEC batched" "$OUT/srv_2.log" | tail -12
