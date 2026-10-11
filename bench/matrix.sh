#!/bin/bash
# Benchmark matrix driver — runs INSIDE the dev container (models at /models).
#
# Usage:
#   bash /app/bench/matrix.sh quants        # target quant sweep (baseline + spec)
#   bash /app/bench/matrix.sh draft         # draft-quant / n-max sweep (best target)
#   bash /app/bench/matrix.sh flags         # cache-reuse / fa / prio / thinking
#   bash /app/bench/matrix.sh llama-bench   # raw llama-bench pp/tg sweep
#   bash /app/bench/matrix.sh one <name> <llama-server args...>  # single config
#
# Results land in $RESULTS_DIR (default /results).
set -uo pipefail

MODE="${1:-quants}"
shift || true

RESULTS_DIR="${RESULTS_DIR:-/results}"
MODELS="${MODELS:-/models}"
PORT="${PORT:-8080}"
CTX="${CTX:-32768}"
mkdir -p "$RESULTS_DIR"

# quant sweep targets, fastest/smallest first
QUANTS=(Q4_0 QAD-Q4_0 Q4_K_M Q5_K_M Q6_K Q8_0)
TARGET="${TARGET:-Q4_K_M}"        # overridden per sweep
DRAFT="${DRAFT:-Q8_0}"            # DSpark draft quant

wait_health() {
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

start_server() {
    echo "[matrix] starting llama-server: $*"
    llama-server "$@" >"$RESULTS_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    if ! wait_health; then
        echo "[matrix] ERROR: server failed to become healthy"
        tail -30 "$RESULTS_DIR/server.log"
        stop_server
        return 1
    fi
    echo "[matrix] healthy (pid $SERVER_PID)"
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null
    wait "${SERVER_PID:-}" 2>/dev/null
    SERVER_PID=""
    sleep 2
}

run_suite() {
    local label="$1"
    echo "[matrix] === $label ==="
    python3 /app/bench/bench_openai.py --base-url "http://127.0.0.1:${PORT}" \
        --model LFM2.5-2.6B --label "$label" --out-dir "$RESULTS_DIR"
    python3 /app/bench/quality_check.py --base-url "http://127.0.0.1:${PORT}" \
        --model LFM2.5-2.6B --label "$label" --out-dir "$RESULTS_DIR"
}

common_flags() {
    echo -m "$1" -ngl 99 -c "$CTX" -np 1 -fa on --host 127.0.0.1 --port "$PORT" \
         --jinja --alias LFM2.5-2.6B --metrics
}

case "$MODE" in
one)
    name="$1"; shift
    start_server "$@" || exit 1
    run_suite "$name"
    stop_server
    ;;
quants)
    for q in "${QUANTS[@]}"; do
        m="$MODELS/LFM2.5-2.6B-${q}.gguf"
        [ -f "$m" ] || { echo "[matrix] skip $q (no file)"; continue; }
        start_server $(common_flags "$m") --cache-reuse 256 || continue
        run_suite "${q}-nospec"
        stop_server
        if [ -f "$MODELS/LFM2.5-2.6B-DSpark-${DRAFT}.gguf" ]; then
            start_server $(common_flags "$m") --cache-reuse 256 \
                -md "$MODELS/LFM2.5-2.6B-DSpark-${DRAFT}.gguf" \
                --spec-type draft-dspark --spec-draft-n-max 10 --spec-draft-n-min 0 || continue
            run_suite "${q}-spec-draft${DRAFT}"
            stop_server
        fi
    done
    ;;
draft)
    m="$MODELS/LFM2.5-2.6B-${TARGET}.gguf"
    for dq in Q4_K_M Q8_0 F16; do
        d="$MODELS/LFM2.5-2.6B-DSpark-${dq}.gguf"
        [ -f "$d" ] || { echo "[matrix] skip draft $dq"; continue; }
        for nmax in 4 9 10; do
            start_server $(common_flags "$m") --cache-reuse 256 \
                -md "$d" --spec-type draft-dspark \
                --spec-draft-n-max "$nmax" --spec-draft-n-min 0 || continue
            run_suite "${TARGET}-draft${dq}-nmax${nmax}"
            stop_server
        done
    done
    ;;
flags)
    m="$MODELS/LFM2.5-2.6B-${TARGET}.gguf"
    d="$MODELS/LFM2.5-2.6B-DSpark-${DRAFT}.gguf"
    spec=(-md "$d" --spec-type draft-dspark --spec-draft-n-max 10 --spec-draft-n-min 0)
    [ -f "$d" ] || spec=()

    # cache-reuse
    for cr in 0 256 512; do
        start_server $(common_flags "$m") --cache-reuse "$cr" "${spec[@]}" || continue
        run_suite "${TARGET}-cache${cr}"
        stop_server
    done
    # flash-attn: REQUIRED for DSpark (dflash requires ctx_other / FA),
    # so only testable without a draft model
    if [ ${#spec[@]} -eq 0 ]; then
        start_server -m "$m" -ngl 99 -c "$CTX" -np 1 -fa off --host 127.0.0.1 --port "$PORT" \
            --jinja --alias LFM2.5-2.6B --metrics --cache-reuse 256 || true
        run_suite "${TARGET}-faoff"
        stop_server
    else
        echo "[matrix] skipping fa-off: DSpark requires flash attention"
    fi
    # prio levels
    for prio in -1 0 1 2 3; do
        start_server $(common_flags "$m") --cache-reuse 256 --prio "$prio" "${spec[@]}" || continue
        run_suite "${TARGET}-prio${prio}"
        stop_server
    done
    # reasoning budgets
    for rb in off 0 128 256 512; do
        case "$rb" in
            off) extra=(-rea off) ;;
            *)   extra=(--reasoning-budget "$rb") ;;
        esac
        start_server $(common_flags "$m") --cache-reuse 256 "${spec[@]}" "${extra[@]}" || continue
        run_suite "${TARGET}-reason${rb}"
        stop_server
    done
    ;;
llama-bench)
    for q in "${QUANTS[@]}"; do
        m="$MODELS/LFM2.5-2.6B-${q}.gguf"
        [ -f "$m" ] || continue
        echo "[matrix] llama-bench $q"
        llama-bench -m "$m" -ngl 99 -fa 1 -p 512,2048 -n 128 -r 3 \
            | tee "$RESULTS_DIR/llama-bench-${q}.txt"
    done
    ;;
*)
    echo "unknown mode: $MODE"; exit 1
    ;;
esac
