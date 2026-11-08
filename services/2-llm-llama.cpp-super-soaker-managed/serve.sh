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
# required wrap-up, injected before the end-of-thinking tag when the reasoning
# budget force-closes thinking (llama-server --reasoning-budget-message); a
# bare mid-thought cut otherwise dumps the ragged draft tail into the reply
REASONING_BUDGET_MESSAGE="${REASONING_BUDGET_MESSAGE:-[budget reached — answer briefly in spoken form]}"

# ---- model selection --------------------------------------------------------
# LLM_MODEL picks the serving variant (see README "Model choice"):
#   fast      LFM2.5-1.2B-Instruct Q8_0 + its DSpark draft, no reasoning
#             (lowest latency — benchmark-measured ~2x lower end-to-end)
#   balanced  LFM2.5-2.6B QAD-Q4_0 + DSpark draft, reasoning budget 256
#             (default; the agentic profile: reliable facts, never over-
#             triggers, ~2s slower than the 1.2B)
#   thorough  LFM2.5-8B-A1B (MoE 8.3B/1.5B active), Unsloth UD-IQ2_M +
#             DSpark-Q8_0 draft, reasoning budget 256 (best tool/quality
#             accuracy on this box; fits only at IQ2-class quants — README)
#   frontier  LFM3-1.7B (tri-hybrid: conv + sliding-window + global attention)
#             LiquidAI QAD-Q4_0, no speculative drafter (LiquidAI publishes no
#             LFM3 DSpark), reasoning budget 160 (best accuracy/speed on the
#             frozen battery — PERFORMANCE.md). Needs llama.cpp >= b11960; the
#             repo pin (v0.9.0) provides it.
# Explicit env overrides (MODEL_PATH, DRAFT_PATH, CTX, REASONING_BUDGET) win.
LLM_MODEL="${LLM_MODEL:-balanced}"
# legacy profile names (pre-rename) still resolve so old env files keep working
case "$LLM_MODEL" in
  lfm2.5-1.2b-chat)        LLM_MODEL=fast ;;
  lfm2.5-2.6b-reasoning)   LLM_MODEL=balanced ;;
  lfm2.5-8b-a1b-reasoning) LLM_MODEL=thorough ;;
esac
M=/var/lib/super-soaker/models
case "$LLM_MODEL" in
  frontier)
    # LFM3-1.7B ships no DSpark/DFlash drafter, so DRAFT_PATH is deliberately
    # empty and speculation uses the drafter-free ngram-mod mode below.
    # QAD-Q4_0 measured best on the
    # frozen battery (accuracy and speed) among QAD-Q4_0/Q4_K_M/Q5_K_M/Q6_K/
    # Q8_0; F16 does not fit with STT+TTS resident. REASONING_BUDGET=160 keeps
    # the thinking pass inside the battery's answer budget — LFM3 always thinks
    # first, and a budget at or above the caller's max_tokens truncates the
    # answer instead of landing the wrap-up. See PERFORMANCE.md.
    MODEL_PATH="${MODEL_PATH:-$M/LFM3-1.7B-QAD-Q4_0.gguf}"
    DRAFT_PATH="${DRAFT_PATH:-}"
    CTX="${CTX:-16384}"
    REASONING_BUDGET="${REASONING_BUDGET:-160}"
    # LFM3's short-conv/recurrent state makes llama.cpp disable mid-prompt
    # --cache-reuse ("not supported by this context"); plain LCP prefix reuse
    # still works, which is what the HA warm path relies on (measured: a
    # follow-up turn reused 749 of 1008 prompt tokens, 238 ms prefill).
    #
    # ngram-mod is the only speculative mode available without a drafter. It is
    # quality-neutral on the frozen battery and a large speed win on HA's
    # templated traffic: steady-state battery mean 75-97 tok/s vs 39.6 without
    # it, replies ~1.6-2.3 s vs ~3.5 s, novel-prompt decode unchanged at ~42.
    # n_max 7 matches the dense-model tuning (notes/05). See PERFORMANCE.md.
    SPEC_TYPE="${SPEC_TYPE:-ngram-mod}"
    SPEC_N_MAX="${SPEC_N_MAX:-7}"
    ;;
  fast)
    # Q8_0 target (fidelity insurance — notes/07 proved battery accuracy is
    # identical at QAD-Q4_0/Q8_0/F16; cap is the model). Spec config is the
    # measured plateau top: plain draft-dspark n_max 9 (native drafter,
    # acc 2.99/step — long blocks pay at this acceptance; ngram/p-min/
    # p-split/backend-sampling all measured neutral-or-worse here).
    MODEL_PATH="${MODEL_PATH:-$M/LFM2.5-1.2B-Instruct-Q8_0.gguf}"
    DRAFT_PATH="${DRAFT_PATH:-$M/LFM2.5-1.2B-Instruct-DSpark-Q8_0.gguf}"
    # Same 16384 ceiling as the other profiles (2026-10-17): the KV saving is
    # small at 1.2B but a uniform cap keeps every profile inside the RAM budget.
    CTX="${CTX:-16384}"
    REASONING_BUDGET="${REASONING_BUDGET:-0}"
    ;;
  balanced)
    MODEL_PATH="${MODEL_PATH:-$M/LFM2.5-2.6B-QAD-Q4_0.gguf}"
    DRAFT_PATH="${DRAFT_PATH:-$M/LFM2.5-2.6B-DSpark-Q8_0.gguf}"
    # 16384, not 32768: measured -352 MB of nvmap (3.30 -> 2.94 GB) on the
    # 8 GB box. HA prompts are ~2.5k tokens and its longest observed tool
    # loop reached ~3.9k, so 16k keeps >4x headroom. The 32k KV was what
    # left no room when a TTS synthesis allocated during an HA tool loop
    # and the kernel OOM-killed llama-server (2026-10-17).
    CTX="${CTX:-16384}"
    REASONING_BUDGET="${REASONING_BUDGET:-256}"
    # optimized spec stack (measured 2026-10-13, notes/05): dense verify is
    # cheap so n_max=7 (not 9) wins and p-min truncation HURTS (unlike MoE);
    # ngram-mod adds free accepted tokens on templated HA traffic.
    SPEC_TYPE="${SPEC_TYPE:-draft-dspark,ngram-mod}"
    SPEC_N_MAX="${SPEC_N_MAX:-7}"
    SPEC_EXTRA="${SPEC_EXTRA:-}"
    ;;
  thorough)
    MODEL_PATH="${MODEL_PATH:-$M/LFM2.5-8B-A1B-UD-IQ2_M.gguf}"
    DRAFT_PATH="${DRAFT_PATH:-$M/LFM2.5-8B-A1B-DSpark-Q8_0.gguf}"
    # 16384, not 32768: the 8B-A1B is the tightest profile on this box
    # (measured 7.04/7.62 GB peak at 32k). The same 16k ceiling used by
    # balanced gives it KV headroom without touching quant or spec settings.
    CTX="${CTX:-16384}"
    REASONING_BUDGET="${REASONING_BUDGET:-256}"
    # optimized spec stack (measured 2026-10-13, notes/04):
    # n_max=5 (short blocks win at MoE verify-traffic saturation),
    # p-min 0.4 (DSpark confidence truncation), ngram-mod (free accepted
    # tokens on templated HA tool-JSON/phrasing — no cost on novel speech).
    SPEC_TYPE="${SPEC_TYPE:-draft-dspark,ngram-mod}"
    SPEC_N_MAX="${SPEC_N_MAX:-5}"
    SPEC_P_MIN="${SPEC_P_MIN:-0.4}"
    SPEC_EXTRA="${SPEC_EXTRA:---spec-draft-p-min ${SPEC_P_MIN}}"
    ;;
  *) echo "[super-soaker-llm] ERROR: unknown LLM_MODEL '$LLM_MODEL' (valid: fast | balanced | thorough | frontier)" >&2; exit 1 ;;
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

# Speculative decoding comes in two shapes:
#   * draft-model modes (draft-dspark, draft-dflash, ...) need -md DRAFT_PATH
#   * drafter-free modes (ngram-mod, ngram-cache) run without any draft file
# LFM3 has no published DSpark/DFlash drafter, so the `frontier` profile uses
# ngram-mod. --spec-draft-n-max only exists for the draft-model path; the
# ngram modes size their own blocks (--spec-ngram-mod-n-max).
case "${SPEC_TYPE:-draft-dspark}" in
    *draft-*) SPEC_NEEDS_DRAFT=1 ;;   # draft-dspark / draft-dflash / draft-...
    *)        SPEC_NEEDS_DRAFT=0 ;;   # ngram-mod / ngram-cache (drafter-free)
esac
USE_SPEC=0
case "$SPEC" in
    on)  USE_SPEC=1 ;;
    off) USE_SPEC=0 ;;
    auto)
        if [ "$SPEC_NEEDS_DRAFT" = "1" ] && [ -n "$DRAFT_PATH" ] && [ -f "$DRAFT_PATH" ]; then
            USE_SPEC=1
        elif [ "$SPEC_NEEDS_DRAFT" = "0" ] && [ -n "$SPEC_TYPE" ]; then
            USE_SPEC=1
        else
            USE_SPEC=0
        fi ;;
esac
if [ "$USE_SPEC" = "1" ]; then
    if [ "$SPEC_NEEDS_DRAFT" = "1" ]; then
        if [ ! -f "$DRAFT_PATH" ]; then
            log "ERROR: SPEC=$SPEC but draft not found: $DRAFT_PATH"; exit 1
        fi
        CMD+=(-md "$DRAFT_PATH"
              --spec-type "${SPEC_TYPE:-draft-dspark}"
              --spec-draft-n-max "${SPEC_N_MAX:-9}"
              --spec-draft-n-min 0)
    else
        CMD+=(--spec-type "${SPEC_TYPE}")
        if [ -n "${SPEC_N_MAX:-}" ]; then
            CMD+=(--spec-ngram-mod-n-max "$SPEC_N_MAX")
        fi
    fi
    if [ -n "${SPEC_EXTRA:-}" ]; then
        # shellcheck disable=SC2206
        SPEC_EXTRA_ARR=($SPEC_EXTRA)
        CMD+=("${SPEC_EXTRA_ARR[@]}")
    fi
    log "speculative decoding: ON (type: ${SPEC_TYPE:-draft-dspark}${DRAFT_PATH:+, draft: $DRAFT_PATH})"
else
    log "speculative decoding: OFF"
fi

if [ -n "$REASONING_BUDGET" ]; then
    case "$REASONING_BUDGET" in
        off) CMD+=(-rea off) ;;
        auto) CMD+=(-rea auto) ;;
        on) CMD+=(-rea on) ;;
        *)
            CMD+=(--reasoning-budget "$REASONING_BUDGET")
            if [ "$REASONING_BUDGET" -gt 0 ] 2>/dev/null; then
                CMD+=(--reasoning-budget-message "$REASONING_BUDGET_MESSAGE")
            fi
            ;;
    esac
fi

if [ -n "${TEMPLATE_FILE:-}" ]; then
    if [ -f "$TEMPLATE_FILE" ]; then
        CMD+=(--chat-template-file "$TEMPLATE_FILE")
        log "chat template override: $TEMPLATE_FILE"
    else
        log "ERROR: TEMPLATE_FILE not found: $TEMPLATE_FILE"; exit 1
    fi
fi
if [ -n "${REASONING_FORMAT:-}" ]; then CMD+=(--reasoning-format "$REASONING_FORMAT"); fi

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