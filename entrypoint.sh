#!/bin/bash
# super-soaker entrypoint: configure + launch llama-server, self-test, serve.
set -uo pipefail

log() { echo "[super-soaker] $*"; }

# ---- configuration (env) ----------------------------------------------------
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
MODEL_PATH="${MODEL_PATH:-/models/LFM2.5-2.6B-Q4_K_M.gguf}"
DRAFT_PATH="${DRAFT_PATH:-/models/LFM2.5-2.6B-DSpark-Q8_0.gguf}"
SPEC="${SPEC:-auto}"              # auto|on|off
CTX="${CTX:-32768}"
THREADS="${THREADS:-$(nproc)}"
N_BATCH="${N_BATCH:-1024}"        # logical batch (prefill chunk)
UBATCH="${UBATCH:-256}"           # physical batch — bounds CUDA workspace peak
KV_TYPE="${KV_TYPE:-q8_0}"        # KV cache type (f16 = 0.5GiB @32k, q8_0 = 0.25GiB)
REASONING_BUDGET="${REASONING_BUDGET:-128}"   # empty/"off"|"auto"|N (0 = turbo voice)
CACHE_REUSE="${CACHE_REUSE:-256}"
PRIO="${PRIO:-2}"                  # 2 = high (prio sweep: all levels equal on this box)
ALIAS="${ALIAS:-LFM2.5-2.6B}"
API_KEY="${API_KEY:-}"
API_KEY_FILE="${API_KEY_FILE:-}"
SELF_TEST="${SELF_TEST:-on}"
SELF_TEST_STRICT="${SELF_TEST_STRICT:-off}"
JETSON_CLOCKS="${JETSON_CLOCKS:-off}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-}"

if [ ! -f "$MODEL_PATH" ]; then
    log "ERROR: model not found: $MODEL_PATH"; exit 1
fi

if [ -n "$API_KEY_FILE" ] && [ -f "$API_KEY_FILE" ]; then
    API_KEY="$(cat "$API_KEY_FILE")"
fi

# Optional: lock clocks for stable max performance (needs privileged container
# or host-side `jetson_clocks`). Best-effort inside the container.
if [ "$JETSON_CLOCKS" = "on" ]; then
    if jetson_clocks >/dev/null 2>&1; then
        log "jetson_clocks applied"
    else
        log "jetson_clocks not available in container (run it on the host for max perf)"
    fi
fi

# ---- assemble llama-server command -----------------------------------------
CMD=(llama-server
    -m "$MODEL_PATH"
    -ngl 99
    -c "$CTX"
    -np 1                     # single slot: required for LFM2 short-conv state (issue #29002)
    -t "$THREADS"
    -fa on
    -b "$N_BATCH"
    -ub "$UBATCH"
    -ctk "$KV_TYPE"
    -ctv "$KV_TYPE"
    --host "$HOST"
    --port "$PORT"
    --alias "$ALIAS"
    --jinja                   # chat template + tool calling
    --cache-reuse "$CACHE_REUSE"
    --metrics
)

USE_SPEC=0
case "$SPEC" in
    on)  USE_SPEC=1 ;;
    off) USE_SPEC=0 ;;
    auto)
        if [ -n "$DRAFT_PATH" ] && [ -f "$DRAFT_PATH" ]; then USE_SPEC=1; else USE_SPEC=0; fi ;;
esac
if [ "$USE_SPEC" = "1" ]; then
    if [ ! -f "$DRAFT_PATH" ]; then
        log "ERROR: SPEC=$SPEC but draft not found: $DRAFT_PATH"; exit 1
    fi
    CMD+=(-md "$DRAFT_PATH"
          --spec-type draft-dspark
          --spec-draft-n-max 9
          --spec-draft-n-min 0)
    log "speculative decoding: ON (draft: $DRAFT_PATH)"
else
    log "speculative decoding: OFF"
fi

if [ -n "$REASONING_BUDGET" ]; then
    case "$REASONING_BUDGET" in
        off) CMD+=(-rea off) ;;
        auto) CMD+=(-rea auto) ;;
        on) CMD+=(-rea on) ;;
        *) CMD+=(--reasoning-budget "$REASONING_BUDGET") ;;
    esac
fi

if [ -n "$PRIO" ]; then CMD+=(--prio "$PRIO"); fi
if [ -n "$API_KEY" ]; then CMD+=(--api-key "$API_KEY"); fi
if [ -n "$SLOT_SAVE_PATH" ]; then CMD+=(--slot-save-path "$SLOT_SAVE_PATH"); fi
if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    CMD+=("${EXTRA_ARR[@]}")
fi

log "model:  $MODEL_PATH"
log "ctx:    $CTX  threads: $THREADS  port: $PORT"
log "cmd:    ${CMD[*]}"

# ---- launch -----------------------------------------------------------------
"${CMD[@]}" &
SERVER_PID=$!
trap 'log "shutting down"; kill $SERVER_PID 2>/dev/null; exit 0' TERM INT

# wait for /health
log "waiting for server health..."
for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "server healthy after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        log "ERROR: llama-server exited during startup"; exit 1
    fi
    sleep 1
done

# ---- self-test --------------------------------------------------------------
if [ "$SELF_TEST" = "on" ]; then
    log "running end-to-end self-test..."
    if python3 /app/selftest.py --base-url "http://127.0.0.1:${PORT}" \
            --model "$ALIAS" --out /tmp/selftest.json; then
        log "self-test PASSED"
    else
        log "self-test FAILED"
        if [ "$SELF_TEST_STRICT" = "on" ]; then
            log "SELF_TEST_STRICT=on -> exiting"
            kill $SERVER_PID 2>/dev/null
            exit 1
        fi
    fi
else
    log "self-test disabled (SELF_TEST=$SELF_TEST)"
fi

log "ready — OpenAI-compatible API on http://${HOST}:${PORT}/v1"
wait $SERVER_PID
