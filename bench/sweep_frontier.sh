#!/bin/bash
# sweep_frontier.sh — the LFM3-1.7B sweep behind the `frontier` profile.
#
# Runs on the Jetson with the STT/TTS services resident (so RAM numbers are
# realistic). Stages:
#   quants  — every downloaded LFM3-1.7B quant, no speculation, reasoning 256
#   spec    — ngram-mod speculative decoding on the selected best quant
#   reason  — reasoning off / 0 / 512 on the selected best quant
#
# Usage:
#   bench/sweep_frontier.sh quants
#   BEST=Q8_0 bench/sweep_frontier.sh spec
#   BEST=Q8_0 bench/sweep_frontier.sh reason
#
# Env: MODELS (default /var/lib/super-soaker/models), REPS (default 3),
#      REASONING_BUDGET (default 256), LLAMA_BIN, OUT_DIR.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${MODELS:-/var/lib/super-soaker/models}"
STAGE="${1:-quants}"
REPS="${REPS:-3}"
RB="${REASONING_BUDGET:-256}"
BEST="${BEST:-}"
MSG='[budget reached — answer briefly in spoken form]'
QUANTS="${QUANTS:-QAD-Q4_0 Q4_K_M Q5_K_M Q6_K Q8_0 F16}"

run() { # label model [extra args...]
    local label="$1" model="$2"; shift 2
    CTX="${CTX:-16384}" bash "$HERE/profile_battery.sh" \
        "$model" LFM3-1.7B "$label" "$REPS" -- \
        --reasoning-budget "$RB" --reasoning-budget-message "$MSG" "$@"
}

case "$STAGE" in
quants)
    for q in $QUANTS; do
        [ -f "$M/LFM3-1.7B-$q.gguf" ] || { echo "skip $q (missing)"; continue; }
        run "lfm3-$q-rb$RB" "$M/LFM3-1.7B-$q.gguf"
    done
    ;;
spec)
    [ -n "$BEST" ] || { echo "set BEST=<quant tag>" >&2; exit 2; }
    # LFM3 has no published DSpark/DFlash drafter (LiquidAI ships DSpark only
    # for LFM2.5), so ngram-mod is the only available speculative mode.
    run "lfm3-$BEST-ngram-n7" "$M/LFM3-1.7B-$BEST.gguf" \
        --spec-type ngram-mod --spec-ngram-mod-n-max 7
    ;;
reason)
    [ -n "$BEST" ] || { echo "set BEST=<quant tag>" >&2; exit 2; }
    # Reasoning budget must be BELOW the caller's max_tokens or the budget
    # wrap-up never fires and the answer is truncated by max_tokens instead.
    # The frozen battery uses max_tokens=160 (tools) / 200 (quality), so the
    # useful budget range on this battery tops out around 192.
    for r in ${REASON_LIST:-off 0 64 96 128 160 192 256}; do
        case "$r" in
        off) args=(--reasoning off) ;;
        *)   args=(--reasoning-budget "$r" --reasoning-budget-message "$MSG") ;;
        esac
        CTX="${CTX:-16384}" bash "$HERE/profile_battery.sh" \
            "$M/LFM3-1.7B-$BEST.gguf" LFM3-1.7B "lfm3-$BEST-rb$r" "$REPS" -- "${args[@]}"
    done
    ;;
*)
    echo "unknown stage: $STAGE (quants|spec|reason)" >&2; exit 2 ;;
esac
