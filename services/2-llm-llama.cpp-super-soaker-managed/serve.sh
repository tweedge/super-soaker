#!/bin/bash
# super-soaker LLM service (native): configure + launch llama-server, self-test, serve.
# Model: any LFM2.5 profile selected by LLM_MODEL (see "model selection" below);
# every profile pairs a target GGUF with its DSpark-Q8_0 speculative drafter.
set -uo pipefail

log() { echo "[super-soaker-llm] $*"; }

# ---- configuration (env) ----------------------------------------------------
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
SPEC="${SPEC:-auto}"              # auto|on|off
THREADS="${THREADS:-$(nproc)}"
N_BATCH="${N_BATCH:-1024}"        # logical batch (prefill chunk)
UBATCH="${UBATCH:-256}"           # physical batch — bounds CUDA workspace peak
KV_TYPE="${KV_TYPE:-q8_0}"        # KV cache type (f16 = 0.5GiB @32k, q8_0 = 0.25GiB)
CACHE_REUSE="${CACHE_REUSE:-256}"
PRIO="${PRIO:-2}"                  # 2 = high (prio sweep: all levels equal on this box)
API_KEY="${API_KEY:-}"
API_KEY_FILE="${API_KEY_FILE:-}"
SELF_TEST="${SELF_TEST:-on}"
SELF_TEST_STRICT="${SELF_TEST_STRICT:-off}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-}"
LLAMA_BIN="${LLAMA_BIN:-/opt/super-soaker/llama.cpp/bin/llama-server}"
SELFTEST="${SELFTEST:-/opt/super-soaker/services/2-llm-llama.cpp-super-soaker-managed/selftest.py}"

# ---- model selection --------------------------------------------------------
# LLM_MODEL picks the serving variant (see README "Model choice"):
#   lfm2.5-1.2b-chat         LFM2.5-1.2B-Instruct QAD-Q4_0 + its DSpark draft
#                            (default; fastest chat profile — benchmark-
#                            measured ~2x lower end-to-end latency)
#   lfm2.5-2.6b-reasoning    LFM2.5-2.6B QAD-Q4_0 + DSpark draft, reasoning
#                            budget 128 (the agentic profile: reliable facts,
#                            never over-triggers, ~2s slower than the 1.2B)
#   lfm2.5-8b-a1b-reasoning  LFM2.5-8B-A1B (MoE 8.3B/1.5B active), Unsloth
#                            UD-IQ2_M + DSpark-Q8_0 draft, reasoning budget
#                            128 (best tool/quality accuracy on this box;
#                            fits only at IQ2-class quants — see README)
# Explicit env overrides (MODEL_PATH, DRAFT_PATH, CTX, REASONING_BUDGET) win.
LLM_MODEL="${LLM_MODEL:-lfm2.5-1.2b-chat}"
M=/var/lib/super-soaker/models
case "$LLM_MODEL" in
  lfm2.5-1.2b-chat)
    MODEL_PATH="${MODEL_PATH:-$M/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf}"
    DRAFT_PATH="${DRAFT_PATH:-$M/LFM2.5-1.2B-Instruct-DSpark-Q8_0.gguf}"
    CTX="${CTX:-8192}"
    REASONING_BUDGET="${REASONING_BUDGET:-0}"
    ;;
  lfm2.5-2.6b-reasoning)
    MODEL_PATH="${MODEL_PATH:-$M/LFM2.5-2.6B-QAD-Q4_0.gguf}"
    DRAFT_PATH="${DRAFT_PATH:-$M/LFM2.5-2.6B-DSpark-Q8_0.gguf}"
    CTX="${CTX:-32768}"
    REASONING_BUDGET="${REASONING_BUDGET:-128}"
    ;;
  lfm2.5-8b-a1b-reasoning)
    MODEL_PATH="${MODEL_PATH:-$M/LFM2.5-8B-A1B-UD-IQ2_M.gguf}"
    DRAFT_PATH="${DRAFT_PATH:-$M/LFM2.5-8B-A1B-DSpark-Q8_0.gguf}"
    CTX="${CTX:-32768}"
    REASONING_BUDGET="${REASONING_BUDGET:-128}"
    ;;
  *) echo "[super-soaker-llm] ERROR: unknown LLM_MODEL '$LLM_MODEL'" >&2; exit 1 ;;
esac
ALIAS="${ALIAS:-$(basename "$MODEL_PATH" .gguf)}"

if [ ! -x "$LLAMA_BIN" ]; then
    log "ERROR: llama-server not found: $LLAMA_BIN"; exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    log "ERROR: model not found: $MODEL_PATH"; exit 1
fi

if [ -n "$API_KEY_FILE" ] && [ -f "$API_KEY_FILE" ]; then
    API_KEY="$(cat "$API_KEY_FILE")"
fi

# ---- assemble llama-server command -----------------------------------------
CMD=("$LLAMA_BIN"
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
    if python3 "$SELFTEST" --base-url "http://127.0.0.1:${PORT}" \
            --model "$ALIAS" --out /tmp/super-soaker-llm-selftest.json; then
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