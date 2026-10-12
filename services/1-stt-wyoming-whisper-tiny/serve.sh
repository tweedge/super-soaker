#!/bin/bash
# super-soaker STT service (native): wyoming-faster-whisper (tiny/en) on CUDA.
set -uo pipefail

log() { echo "[super-soaker-stt] $*"; }

PORT="${PORT:-10300}"
HOST="${HOST:-0.0.0.0}"
MODEL="${MODEL:-tiny}"
LANGUAGE="${LANGUAGE:-en}"
COMPUTE_TYPE="${COMPUTE_TYPE:-int8}"
BEAM_SIZE="${BEAM_SIZE:-5}"
DATA_DIR="${DATA_DIR:-/var/lib/super-soaker/stt}"
VENV="${VENV:-/opt/super-soaker/venv-stt}"
SELF_TEST="${SELF_TEST:-on}"
SELF_TEST_STRICT="${SELF_TEST_STRICT:-off}"
SELFTEST="${SELFTEST:-/opt/super-soaker/services/1-stt-wyoming-whisper-tiny/selftest.py}"
WARMUP="${WARMUP:-/opt/super-soaker/services/1-stt-wyoming-whisper-tiny/warmup.py}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
CT2_LIB="${CT2_LIB:-/opt/super-soaker/ctranslate2/lib}"
export LD_LIBRARY_PATH="${CT2_LIB}:${LD_LIBRARY_PATH:-}"

mkdir -p "$DATA_DIR"

ARGS=(
    --model "${MODEL}"
    --device cuda
    --compute-type "${COMPUTE_TYPE}"
    --beam-size "${BEAM_SIZE}"
    --uri "tcp://${HOST}:${PORT}"
    --data-dir "$DATA_DIR"
    --download-dir "$DATA_DIR"
)
if [ -n "$LANGUAGE" ]; then
    ARGS+=(--language "$LANGUAGE")
fi

log "model=$MODEL compute=$COMPUTE_TYPE beam=$BEAM_SIZE port=$PORT"
"$VENV/bin/python3" -m wyoming_faster_whisper "${ARGS[@]}" &
SERVER_PID=$!
trap 'log "shutting down"; kill $SERVER_PID 2>/dev/null; exit 0' TERM INT

log "waiting for Wyoming server on port $PORT..."
TIMEOUT=120
ELAPSED=0
while ! (echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: server exited before becoming ready"; exit 1
    fi
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        log "ERROR: server not ready within ${TIMEOUT}s"; exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
log "server ready — warming CUDA with $WARMUP_RUNS transcription(s)..."
"$VENV/bin/python3" "$WARMUP" --host 127.0.0.1 --port "$PORT" --runs "$WARMUP_RUNS"

if [ "$SELF_TEST" = "on" ]; then
    log "running end-to-end self-test..."
    if "$VENV/bin/python3" "$SELFTEST" --host 127.0.0.1 --port "$PORT"; then
        log "self-test PASSED"
    else
        log "self-test FAILED"
        if [ "$SELF_TEST_STRICT" = "on" ]; then
            kill $SERVER_PID 2>/dev/null
            exit 1
        fi
    fi
fi

log "ready — Wyoming STT on tcp://${HOST}:${PORT}"
wait $SERVER_PID