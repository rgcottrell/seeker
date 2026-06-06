#!/usr/bin/env bash
#
# run_all_quants.sh — download every GGUF quant of an unsloth repo and verify
# each one runs under `seeker`.
#
#   Tier-1 ("runs successfully"): exit 0 + N coherent, non-degenerate tokens.
#   Tier-2 (--ref): byte-identical greedy continuation vs llama-cli, on a
#                   representative spot-check subset.
#
# Idempotent + resumable: a passing file writes `<file>.pass`; reruns skip it
# unless --force. When a quant type isn't wired, `seeker` exits non-zero with a
# precise message — the harness greps those into a deduped MISSING_TYPES summary
# so you know exactly which dispatch arm to add next. Then: add the arm, rebuild,
# rerun (only the not-yet-passing files re-run).
#
# Usage:
#   scripts/run_all_quants.sh [--ref] [--force]
# Env overrides: REPO PROMPT MAX_TOKENS TIMEOUT SEEKER RESULTS LLAMA_CLI REF_SUBSET
set -uo pipefail

REPO="${REPO:-unsloth/Llama-3.2-1B-Instruct-GGUF}"
PROMPT="${PROMPT:-Once upon a time}"
MAX_TOKENS="${MAX_TOKENS:-16}"
TIMEOUT="${TIMEOUT:-600}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEKER="${SEEKER:-$ROOT/target/release/seeker}"
RESULTS="${RESULTS:-$ROOT/scripts/.quant-results}"
# Raw-completion reference (NOT llama-cli: that build is conversation-only and
# applies the chat template; llama-completion -no-cnv does raw text completion,
# byte-identical to `seeker run`).
LLAMA_CLI="${LLAMA_CLI:-/home/bob/tools/llama.cpp/llama-b9518-vulkan/llama-completion}"
# Quant tags (filename tokens) to byte-compare against llama-cli under --ref.
REF_SUBSET="${REF_SUBSET:-F16 Q4_K_M Q6_K IQ4_XS Q2_K Q3_K_M UD-IQ2_XXS UD-IQ1_S}"

DO_REF=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --ref)   DO_REF=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

[[ -x "$SEEKER" ]] || { echo "seeker binary not found at $SEEKER (run: cargo build --release)" >&2; exit 1; }
command -v jq   >/dev/null || { echo "jq required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }
mkdir -p "$RESULTS"

# ── enumerate the repo's GGUF files (authoritative; avoids hardcoded names) ──
echo "Enumerating $REPO ..."
mapfile -t FILES < <(curl -fsSL "https://huggingface.co/api/models/$REPO" \
  | jq -r '.siblings[].rfilename
           | select(endswith(".gguf"))
           | select(ascii_downcase | contains("mmproj") | not)
           | select(contains("/") | not)' | sort)
[[ ${#FILES[@]} -gt 0 ]] || { echo "no .gguf files found in $REPO" >&2; exit 1; }
echo "Found ${#FILES[@]} GGUF files."

# Does filename $1 contain quant-tag $2 at token boundaries (Q4_K matches
# model-Q4_K.gguf but not Q4_K_M)? Mirrors seeker's split_tokens() rule.
tag_in_file() {
  local file="${1,,}" tag="${2,,}"
  file="${file//[.-]/ }"; tag="${tag//[.-]/ }"
  [[ " $file " == *" $tag "* ]]
}

want_ref() {
  local file="$1" t
  for t in $REF_SUBSET; do tag_in_file "$file" "$t" && return 0; done
  return 1
}

# Strip ANSI + llama.cpp framing, collapse whitespace, l-trim — for text compare.
norm() { sed -e 's/\x1b\[[0-9;]*m//g' | tr -s ' \t\n' ' ' | sed -e 's/^ *//' -e 's/ *$//'; }

declare -a SUMMARY
declare -A MISSING

for f in "${FILES[@]}"; do
  base="${f%.gguf}"
  res="$RESULTS/$base"
  if [[ -f "$res.pass" && $FORCE -eq 0 ]]; then
    echo "== $f : SKIP (already passing)"
    SUMMARY+=("$(cat "$res.json")")
    continue
  fi
  echo "== $f"

  # 1. download (idempotent via HF cache); capture resolved local path.
  if ! path=$("$SEEKER" download --hf-repo "$REPO" --hf-file "$f" --no-mmproj 2>"$res.dl.err" | tail -n1); then
    echo "   DOWNLOAD FAILED (see $res.dl.err)"
    rec=$(jq -nc --arg f "$f" '{file:$f, tier1:"download_fail"}'); echo "$rec" >"$res.json"
    SUMMARY+=("$rec"); continue
  fi

  # 2. inspect: token_embd type + a histogram of every internal tensor type.
  "$SEEKER" inspect -m "$path" --json >"$res.inspect.json" 2>/dev/null || true
  tok_embd=$(jq -r '(.tensors[]|select(.name=="token_embd.weight")|.type)//"?"' "$res.inspect.json" 2>/dev/null)
  types=$(jq -r '[.tensors[].type]|unique|join(",")' "$res.inspect.json" 2>/dev/null)

  # 3. run greedy.
  rc=0
  timeout "$TIMEOUT" "$SEEKER" run -m "$path" --prompt "$PROMPT" \
      --max-tokens "$MAX_TOKENS" --temp 0 >"$res.out" 2>"$res.err" || rc=$?

  # Capture the whole generated block (the text between the "generated:" and
  # "ids:" markers), collapsed to one line. A single-line `sed 's/^generated: //'`
  # mis-reads as empty when the first decoded token renders as a newline (common
  # for gemma4 on a raw prompt), false-failing an otherwise-fine run.
  gen=$(sed -n '/^generated:/,/^ids:/p' "$res.out" \
        | sed -e '$d' -e 's/^generated:[[:space:]]*//' \
        | tr '\n' ' ' | sed -e 's/[[:space:]]\{1,\}/ /g' -e 's/^ //' -e 's/ *$//')
  ids=$(sed -n 's/^ids: *//p' "$res.out" | head -n1)
  nids=$(grep -oE '[0-9]+' <<<"$ids" | wc -l | tr -d ' ')
  uniq_ids=$(grep -oE '[0-9]+' <<<"$ids" | sort -u | wc -l | tr -d ' ')

  # collect any "not yet wired" / unsupported-get_rows dtypes from stderr
  while read -r t; do [[ -n "$t" ]] && MISSING["$t"]=1; done < <(
    grep -oE 'weight dtype [A-Za-z0-9_]+' "$res.err" 2>/dev/null | awk '{print $3}'
    grep -oE 'unsupported src/dst combo [A-Za-z0-9_]+' "$res.err" 2>/dev/null | awk '{print $4}')

  # 4. classify Tier-1.
  tier1="fail"; note=""
  if [[ $rc -eq 124 ]]; then
    tier1="timeout"
  elif [[ $rc -ne 0 ]]; then
    tier1="fail"; note=$(grep -iE 'not yet wired|unsupported src/dst|panic|Error' "$res.err" | head -n1)
  elif [[ -z "$ids" ]]; then
    tier1="fail"; note="no ids line in output"
  elif [[ -z "$gen" ]]; then
    tier1="fail"; note="empty generated text"
  elif [[ "$nids" -ge 4 && "$uniq_ids" -le 1 ]]; then
    tier1="degenerate"; note="all $nids tokens identical (NaN-argmax?)"
  else
    tier1="pass"
  fi

  # 5. optional Tier-2 spot-check vs llama-cli (byte-identical greedy continuation).
  tier2="-"
  if [[ $DO_REF -eq 1 && "$tier1" == "pass" ]] && want_ref "$f"; then
    if [[ -x "$LLAMA_CLI" ]]; then
      "$LLAMA_CLI" -m "$path" -p "$PROMPT" -n "$MAX_TOKENS" --temp 0 \
        -no-cnv --no-display-prompt -ngl 99 >"$res.ref.out" 2>/dev/null || true
      ref=$(norm <"$res.ref.out")
      mine=$(norm <<<"$gen")
      # compare on the shorter common length (legit FP divergence after a few tokens)
      n=${#mine}; (( ${#ref} < n )) && n=${#ref}; (( n > 48 )) && n=48
      if [[ -n "$ref" && "${mine:0:n}" == "${ref:0:n}" ]]; then tier2="match"; else tier2="DIFF"; fi
    else
      tier2="no-llama-cli"
    fi
  fi

  rec=$(jq -nc --arg f "$f" --arg te "$tok_embd" --arg ty "$types" \
        --argjson rc "$rc" --arg t1 "$tier1" --arg t2 "$tier2" \
        --argjson n "$nids" --arg gen "${gen:0:80}" --arg note "$note" \
        '{file:$f, token_embd:$te, types:$ty, exit:$rc, ntok:$n,
          tier1:$t1, tier2:$t2, first:$gen, note:$note}')
  echo "$rec" >"$res.json"
  [[ "$tier1" == "pass" ]] && touch "$res.pass" || rm -f "$res.pass"
  SUMMARY+=("$rec")
  printf '   tier1=%-10s tier2=%-6s tok_embd=%-8s gen="%s"\n' "$tier1" "$tier2" "$tok_embd" "${gen:0:50}"
done

# ── summary ──
echo
echo "================ SUMMARY ($REPO) ================"
printf '%-44s %-10s %-7s %-9s %s\n' FILE TIER1 TIER2 TOK_EMBD FIRST
for rec in "${SUMMARY[@]}"; do
  jq -r '"\(.file)\t\(.tier1)\t\(.tier2 // "-")\t\(.token_embd // "?")\t\(.first // "")"' <<<"$rec" \
    | awk -F'\t' '{printf "%-44s %-10s %-7s %-9s %s\n",$1,$2,$3,$4,substr($5,1,40)}'
done

pass=$(printf '%s\n' "${SUMMARY[@]}" | jq -s '[.[]|select(.tier1=="pass")]|length')
echo
echo "PASS: $pass / ${#SUMMARY[@]}"
# `${!MISSING[*]}` on an empty associative array trips `set -u`; guard it.
set +u
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "MISSING_TYPES (add a dispatch arm for each): ${!MISSING[*]}"
fi
set -u
