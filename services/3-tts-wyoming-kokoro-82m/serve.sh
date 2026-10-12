#!/bin/bash
# super-soaker TTS service (native): Kokoro-82M via ONNX Runtime (CUDA), Wyoming.
set -uo pipefail

log() { echo "[super-soaker-tts] $*"; }

export WYOMING_PORT="${WYOMING_PORT:-10210}"
export KOKORO_SPEED="${KOKORO_SPEED:-1.0}"
export KOKORO_QUANTIZATION="${KOKORO_QUANTIZATION:-fp16}"
export KOKORO_VOICE="${KOKORO_VOICE:-af_heart}"
export KOKORO_GPU_MEM_MB="${KOKORO_GPU_MEM_MB:-1024}"
export KOKORO_MODEL_DIR="${KOKORO_MODEL_DIR:-/var/lib/super-soaker/tts}"
VENV="${VENV:-/opt/super-soaker/venv-tts}"
SELF_TEST="${SELF_TEST:-on}"
SELF_TEST_STRICT="${SELF_TEST_STRICT:-off}"
SELFTEST="${SELFTEST:-/opt/super-soaker/services/3-tts-wyoming-kokoro-82m/selftest.py}"

log "quantization=$KOKORO_QUANTIZATION voice=$KOKORO_VOICE gpu_mem=${KOKORO_GPU_MEM_MB}MB port=$WYOMING_PORT"
"$VENV/bin/python3" /opt/super-soaker/services/3-tts-wyoming-kokoro-82m/kokoro_wyoming.py &
SERVER_PID=$!
trap 'log "shutting down"; kill $SERVER_PID 2>/dev/null; exit 0' TERM INT

log "waiting for Wyoming server on port $WYOMING_PORT..."
TIMEOUT=120
ELAPSED=0
while ! (echo > /dev/tcp/127.0.0.1/"$WYOMING_PORT") 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: server exited before becoming ready"; exit 1
    fi
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        log "ERROR: server not ready within ${TIMEOUT}s"; exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ "$SELF_TEST" = "on" ]; then
    log "running end-to-end self-test..."
    if "$VENV/bin/python3" "$SELFTEST" --host 127.0.0.1 --port "$WYOMING_PORT"; then
        log "self-test PASSED"
    else
        log "self-test FAILED"
        if [ "$SELF_TEST_STRICT" = "on" ]; then
            kill $SERVER_PID 2>/dev/null
            exit 1
        fi
    fi
fi

log "ready — Wyoming TTS on tcp://0.0.0.0:$WYOMING_PORT"
wait $SERVER_PID