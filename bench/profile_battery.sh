#!/bin/bash
# profile_battery.sh — run the frozen HA battery against an arbitrary
# llama-server configuration, with the super-soaker serving flags, and record
# per-rep JSON plus a RAM/swap/power sample.
#
# This exists so model/quant/spec comparisons are apples-to-apples: it mirrors
# services/2-llm-llama.cpp-super-soaker-managed/serve.sh's flags exactly
# (ngl 99, np 1, fa on, b/ub, q8_0 KV, cache-reuse, jinja, metrics) and only
# varies what the caller asks for.
#
# Usage:
#   bench/profile_battery.sh MODEL_PATH ALIAS LABEL [REPS] -- [extra llama-server args]
#
# Env (defaults match serve.sh):
#   PORT=8080 CTX=16384 N_BATCH=1024 UBATCH=256 KV_TYPE=q8_0 CACHE_REUSE=256
#   THREADS=$(nproc) LLAMA_BIN=/opt/super-soaker/llama.cpp/bin/llama-server
#   OUT_DIR=bench/results BATTERY=bench/ha_battery.json RAISE_NICE=0
#
# Example (quant sweep, no speculation):
#   CTX=16384 bench/profile_battery.sh \
#       /var/lib/super-soaker/models/LFM3-1.7B-Q6_K.gguf LFM3-1.7B Q6_K 3 -- \
#       --reasoning-budget 256 --reasoning-budget-message "[budget reached]"
#
# Example (ngram-mod speculation):
#   ... 3 -- --reasoning-budget 256 --spec-type ngram-mod --spec-ngram-mod-n-max 7
#
# The server is started, warmed (one throwaway completion), then the battery is
# run REPS times. Memory is sampled as MemTotal-MemFree-Cached (Jetson iGPU
# kernel allocations are invisible to RSS) plus the server's VmSwap.
set -uo pipefail

MODEL="${1:?model path}"; ALIAS="${2:?alias}"; LABEL="${3:?label}"; REPS="${4:-1}"
shift 4 || true
[ "${1:-}" = "--" ] && shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"
N_BATCH="${N_BATCH:-1024}"
UBATCH="${UBATCH:-256}"
KV_TYPE="${KV_TYPE:-q8_0}"
CACHE_REUSE="${CACHE_REUSE:-256}"
THREADS="${THREADS:-$(nproc)}"
LLAMA_BIN="${LLAMA_BIN:-/opt/super-soaker/llama.cpp/bin/llama-server}"
OUT_DIR="${OUT_DIR:-$HERE/results}"
BATTERY="${BATTERY:-$HERE/ha_battery.json}"
BASE="http://127.0.0.1:${PORT}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/server-${LABEL}.log"

if [ ! -f "$MODEL" ]; then echo "model not found: $MODEL" >&2; exit 1; fi
if [ ! -x "$LLAMA_BIN" ]; then echo "llama-server not found: $LLAMA_BIN" >&2; exit 1; fi

mem_used_kb() {
    awk '/^MemTotal:/{t=$2} /^MemFree:/{f=$2} /^Cached:/{c=$2} END{print t-f-c}' /proc/meminfo
}

# refuse to run if something already owns the port: a stale server makes the
# whole measurement meaningless (and the new process just dies on bind).
if curl -sf "${BASE}/health" >/dev/null 2>&1; then
    echo "[battery] ERROR: a server is already answering on ${BASE}; stop it first" >&2
    echo "[battery]        (pkill -f 'llama-server.*--port ${PORT}')" >&2
    exit 1
fi

CMD=("$LLAMA_BIN" -m "$MODEL" -ngl 99 -c "$CTX" -np 1 -t "$THREADS" \
     -fa on -b "$N_BATCH" -ub "$UBATCH" -ctk "$KV_TYPE" -ctv "$KV_TYPE" \
     --host 127.0.0.1 --port "$PORT" --alias "$ALIAS" --jinja \
     --cache-reuse "$CACHE_REUSE" --metrics "$@")

echo "[battery] $LABEL"
echo "[battery] cmd: ${CMD[*]}"
"${CMD[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

for i in $(seq 1 600); do
    if curl -sf "${BASE}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[battery] ERROR: server exited during startup" >&2
        tail -30 "$LOG" >&2; exit 1
    fi
    sleep 1
done
if ! curl -sf "${BASE}/health" >/dev/null 2>&1; then
    echo "[battery] ERROR: server not healthy after 600 s" >&2; tail -30 "$LOG" >&2; exit 1
fi
echo "[battery] healthy (pid $SERVER_PID)"
# the health check above can pass against a *pre-existing* server if ours lost
# the bind race; verify our pid still owns the socket before trusting results
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[battery] ERROR: server exited after startup (port race)" >&2
    tail -20 "$LOG" >&2; exit 1
fi

# let the allocator settle, then baseline memory
sleep 3
BASE_MEM_KB=$(mem_used_kb)

# one throwaway completion so CUDA graphs/allocator are warm before run 1
curl -sf -o /dev/null -X POST "${BASE}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8,\"temperature\":0.1}" \
    || { echo "[battery] ERROR: warmup request failed" >&2; exit 1; }

PEAK_MEM_KB=$BASE_MEM_KB
echo "$PEAK_MEM_KB" > /var/tmp/.battery-peak-mem
( while kill -0 "$SERVER_PID" 2>/dev/null; do
      v=$(mem_used_kb)
      [ "$v" -gt "$(cat /var/tmp/.battery-peak-mem 2>/dev/null || echo 0)" ] && echo "$v" > /var/tmp/.battery-peak-mem
      # watchdog: never let the box thrash into zram (see AGENTS.md)
      avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      if [ "${avail:-999999}" -lt 80000 ]; then
          echo "[battery] WATCHDOG: MemAvailable ${avail} kB < 80 MB — killing server" >&2
          kill "$SERVER_PID" 2>/dev/null
          echo "watchdog" > /var/tmp/.battery-watchdog-fired
          break
      fi
      sleep 0.2
  done ) &
SAMPLER=$!
rm -f /var/tmp/.battery-watchdog-fired

for r in $(seq 1 "$REPS"); do
    echo "[battery] rep $r/$REPS"
    python3 "$HERE/ha_benchmark.py" --base-url "$BASE" --model "$ALIAS" \
        --label "${LABEL}-r${r}" --battery "$BATTERY" \
        --out "$OUT_DIR/ha_${LABEL}-r${r}.json" \
        || echo "[battery] WARN: rep $r exited nonzero" >&2
done

kill "$SAMPLER" 2>/dev/null
[ -f /var/tmp/.battery-peak-mem ] && PEAK_MEM_KB=$(cat /var/tmp/.battery-peak-mem)
SWAP_KB=$(awk '/^VmSwap:/{print $2}' "/proc/${SERVER_PID}/status" 2>/dev/null || echo 0)
FREE_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
cat > "$OUT_DIR/mem_${LABEL}.json" <<EOF
{"label":"$LABEL","model":"$MODEL","alias":"$ALIAS",
 "ctx":$CTX,"reps":$REPS,
 "baseline_used_mb":$((BASE_MEM_KB/1024)),
 "peak_used_mb":$((PEAK_MEM_KB/1024)),
 "server_vmswap_kb":${SWAP_KB:-0},
 "mem_available_kb":${FREE_KB:-0}}
EOF
echo "[battery] memory: baseline $((BASE_MEM_KB/1024)) MB, peak $((PEAK_MEM_KB/1024)) MB, server VmSwap ${SWAP_KB:-0} kB"
echo "[battery] results -> $OUT_DIR/ha_${LABEL}-r*.json"
echo "[battery] done"
