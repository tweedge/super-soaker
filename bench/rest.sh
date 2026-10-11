#!/bin/bash
# Remaining flag sweeps after the chain (fa-off is impossible with DSpark:
# dflash/dspark requires flash attention). Target: QAD-Q4_0 + DSpark-Q8_0.
set -uo pipefail
RESULTS_DIR="${RESULTS_DIR:-/results}"
PORT="${PORT:-8080}"
M=/models/LFM2.5-2.6B-QAD-Q4_0.gguf
D=/models/LFM2.5-2.6B-DSpark-Q8_0.gguf
mkdir -p "$RESULTS_DIR"

wait_health() {
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

run_cfg() {
    local name="$1"; shift
    echo "[rest] === $name ==="
    llama-server -m "$M" -ngl 99 -c 32768 -np 1 -fa on --host 127.0.0.1 \
        --port "$PORT" --jinja --alias LFM2.5-2.6B --metrics --cache-reuse 256 \
        -md "$D" --spec-type draft-dspark --spec-draft-n-max 9 --spec-draft-n-min 0 \
        "$@" >"$RESULTS_DIR/server.log" 2>&1 &
    local pid=$!
    if ! wait_health; then
        echo "[rest] ERROR: $name failed to become healthy"
        tail -5 "$RESULTS_DIR/server.log"
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        return 1
    fi
    python3 /app/bench/bench_openai.py --base-url "http://127.0.0.1:${PORT}" \
        --model LFM2.5-2.6B --label "$name" --out-dir "$RESULTS_DIR"
    python3 /app/bench/quality_check.py --base-url "http://127.0.0.1:${PORT}" \
        --model LFM2.5-2.6B --label "$name" --out-dir "$RESULTS_DIR"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    sleep 2
}

# priority sweep
for prio in -1 0 1 2 3; do
    run_cfg "QAD-prio${prio}" --prio "$prio"
done

# polling (busy-wait instead of sleeps — may cut draft/launch latency)
run_cfg "QAD-poll" --poll 1 --poll-batch 1

# reasoning budget sweep (speed AND quality)
run_cfg "QAD-reasonoff" -rea off
for rb in 0 64 128 256 512; do
    run_cfg "QAD-reason${rb}" --reasoning-budget "$rb"
done

echo "[rest] complete"
