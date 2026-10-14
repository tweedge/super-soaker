#!/usr/bin/env bash
# =============================================================================
# super-soaker bootstrap.sh — provision a bare NVIDIA Jetson (L4T r36.x) into a
# running three-service voice stack:
#
#   1-stt-wyoming-whisper-tiny      Wyoming STT  tcp/10300
#   2-llm-llama.cpp-super-soaker-managed
#                                   OpenAI API   tcp/8080  (model from --model)
#   3-tts-wyoming-kokoro-82m        Wyoming TTS  tcp/10210
#
# Everything is built locally at pinned versions and installed natively (no
# Docker): one shared toolchain in /opt/super-soaker, models in
# /var/lib/super-soaker, three systemd services that start on boot, power
# mode set to MAXN_SUPER (clocks boost under load, scale down at idle —
# nothing is pinned).
#
# Usage:  sudo ./bootstrap.sh [--skip-build] [--no-reboot] [--force-jobs]
#   --skip-build   reuse wheels/binaries already in /opt/super-soaker/cache
#   --no-reboot    do not reboot at the end (services start anyway)
#   --force-jobs   allow JOBS>2 (dangerous on 8 GB devices — see below)
#
#   --model {fast|balanced|thorough}      default balanced
#                                         fast      = LFM2.5-1.2B-Instruct, no
#                                                     reasoning (lowest latency)
#                                         balanced  = LFM2.5-2.6B + reasoning
#                                         thorough  = LFM2.5-8B-A1B MoE + reasoning
#   --reasoning-budget {N|off|on|auto}    thinking token budget; N caps
#                                         thinking at N tokens (0 = off).
#                                         Defaults per model: 0 for fast,
#                                         128 for balanced and thorough.
#
# Build parallelism is RAM-critical: the compilers (especially nvcc for ONNX
# Runtime) peak at several GB per thread on an 8 GB Orin Nano Super. JOBS=1 is
# the safe default and the build takes a WHILE (1-2 h); JOBS=2 is the tested
# maximum here. JOBS>2 is refused unless --force-jobs is given and is only
# sensible on 16 GB devices (e.g. Jetson Orin NX).
#
# This script is for NEW nodes only. It is safe to re-run (idempotent).
# Optional desktop removal lives in clean.sh and is NOT invoked from here.
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="/opt/super-soaker"
CACHE_DIR="${INSTALL_ROOT}/cache"
MODEL_DIR="/var/lib/super-soaker/models"
STT_DIR="/var/lib/super-soaker/stt"
TTS_DIR="/var/lib/super-soaker/tts"
SERVICE_USER="super-soaker"

SKIP_BUILD=0
DO_REBOOT=1
FORCE_JOBS=0
LLM_MODEL=""
REASONING_BUDGET_ARG=""

log() { echo -e "\n[bootstrap] $*"; }
die() { echo "[bootstrap] ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1 ;;
        --no-reboot)   DO_REBOOT=0 ;;
        --force-jobs)  FORCE_JOBS=1 ;;
        --model)       if [ $# -lt 2 ]; then
                           echo "--model requires a value: fast, balanced, or thorough" >&2
                           exit 2
                       fi
                       LLM_MODEL="$2"; shift ;;
        --model=*)     LLM_MODEL="${1#--model=}" ;;
        --reasoning-budget)
                       if [ $# -lt 2 ]; then
                           echo "--reasoning-budget requires a value: N tokens, off, on, or auto" >&2
                           exit 2
                       fi
                       REASONING_BUDGET_ARG="$2"; shift ;;
        --reasoning-budget=*)
                       REASONING_BUDGET_ARG="${1#--reasoning-budget=}" ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
case "$LLM_MODEL" in
    ""|fast|balanced|thorough) ;;
    *) die "unknown --model '$LLM_MODEL' (fast | balanced | thorough)" ;;
esac
if [ -n "$REASONING_BUDGET_ARG" ]; then
    case "$REASONING_BUDGET_ARG" in
        off|on|auto) ;;
        ''|*[!0-9]*) die "--reasoning-budget must be a non-negative integer or off|on|auto (got '$REASONING_BUDGET_ARG')" ;;
        *) [ "$REASONING_BUDGET_ARG" -le 8192 ] || die "--reasoning-budget too large (max 8192)" ;;
    esac
fi

# 1 compile thread by default; >2 needs --force-jobs (deps/limit-jobs.sh).
JOBS="${JOBS:-1}"
export JOBS
if [ "${FORCE_JOBS}" = "1" ]; then
    export FORCE=1
fi

[ "$(id -u)" -eq 0 ] || die "run as root: sudo $0"

# ---- 0. platform sanity ----------------------------------------------------
log "checking platform"
if [ -f /etc/nv_tegra_release ]; then
    grep -qE "R[0-9]+.*REVISION" /etc/nv_tegra_release || \
        die "/etc/nv_tegra_release present but not parseable"
else
    grep -qiE "jetson|orin" /proc/device-tree/model 2>/dev/null || \
        die "this does not look like a Jetson (no /etc/nv_tegra_release, no Jetson device-tree model)"
fi
L4T="$(grep -oE 'R[0-9]+[^,]*, REVISION: [0-9.]+' /etc/nv_tegra_release 2>/dev/null | head -1)"
echo "  L4T release: ${L4T:-unknown (device-tree: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo '?'))}"
ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || die "expected aarch64, got ${ARCH}"
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "expected Ubuntu, got ${ID:-unknown}"

# ---- 1. apt dependencies ---------------------------------------------------
log "installing apt packages (build toolchain, CUDA 12.6, cuDNN, OpenBLAS, espeak)"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config wget curl ca-certificates \
    python3 python3-venv python3-dev python3-pip pybind11-dev \
    cuda-nvcc-12-6 cuda-cudart-dev-12-6 libcublas-dev-12-6 libcurand-dev-12-6 \
    libcufft-dev-12-6 libcusparse-dev-12-6 cuda-nvrtc-dev-12-6 \
    libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    libopenblas-dev libgomp1 espeak-ng alsa-utils
# ORT needs cmake >= 3.28 (system cmake is 3.22 on Ubuntu 22.04)
if ! cmake --version | head -1 | awk '{exit !($3>="3.28.0")}'; then
    log "installing cmake >= 3.28 via pip (for onnxruntime build)"
    pip3 install --no-cache-dir "cmake==3.31.6"
fi

# ---- 2. service user + directories ----------------------------------------
log "creating ${SERVICE_USER} user and directory layout"
id -u "${SERVICE_USER}" >/dev/null 2>&1 || \
    useradd --system --create-home --home-dir /var/lib/super-soaker \
        --shell /usr/sbin/nologin "${SERVICE_USER}"
# GPU access on L4T r36 needs TWO groups:
#   video  — /dev/nvhost-* and /dev/nvgpu/igpu0/* device nodes
#   render — /dev/dri/renderD128, which libcuda probes during device
#            enumeration; without it CUDA silently reports 0 devices and
#            every later call fails with "operation not supported"
usermod -aG video,render "${SERVICE_USER}" 2>/dev/null || true

mkdir -p "${INSTALL_ROOT}" "${CACHE_DIR}" "${MODEL_DIR}" "${STT_DIR}" "${TTS_DIR}"
mkdir -p /etc/super-soaker
chown -R "${SERVICE_USER}:${SERVICE_USER}" /var/lib/super-soaker

# ---- 3. pinned native builds ----------------------------------------------
if [ "${SKIP_BUILD}" = "1" ]; then
    log "--skip-build: reusing ${CACHE_DIR}"
else
    log "building llama.cpp (pinned b11552 / 23b0202a1), CTranslate2 4.5.0, ONNX Runtime 1.22.0"
    log "  JOBS=${JOBS} — this takes a WHILE at 1-2 threads (1-2 h total); it is"
    log "  RAM-critical on the 8 GB box, so do not raise JOBS past 2 without"
    log "  --force-jobs on a 16 GB device (see deps/limit-jobs.sh)"
    bash "${REPO_DIR}/deps/build-llamacpp.sh"
    bash "${REPO_DIR}/deps/build-ctranslate2.sh"
    bash "${REPO_DIR}/deps/build-onnxruntime.sh"
fi

CT2_WHEEL="$(ls -t "${CACHE_DIR}"/ctranslate2-*.whl | head -1)" || die "no ctranslate2 wheel in ${CACHE_DIR}"
ORT_WHEEL="$(ls -t "${CACHE_DIR}"/onnxruntime_gpu-*.whl | head -1)" || die "no onnxruntime wheel in ${CACHE_DIR}"
[ -x "${INSTALL_ROOT}/llama.cpp/bin/llama-server" ] || die "llama-server missing (build first)"
echo "  ctranslate2 wheel: $(basename "${CT2_WHEEL}")"
echo "  onnxruntime wheel: $(basename "${ORT_WHEEL}")"

# ---- 4. Python environments (pinned) --------------------------------------
log "creating service venvs with pinned Python packages"

log "  STT venv (wyoming-faster-whisper 3.3.1 + CUDA ctranslate2)"
python3 -m venv "${INSTALL_ROOT}/venv-stt"
"${INSTALL_ROOT}/venv-stt/bin/pip" install --no-cache-dir -q --upgrade pip setuptools wheel
"${INSTALL_ROOT}/venv-stt/bin/pip" install --no-cache-dir -q \
    "wyoming-faster-whisper==3.3.1"
# replace the CPU-only ctranslate2 that pip pulled in with our CUDA build
"${INSTALL_ROOT}/venv-stt/bin/pip" install --no-cache-dir -q --force-reinstall --no-deps "${CT2_WHEEL}"
# the locally built ctranslate2 wheel links against /opt/super-soaker/ctranslate2/lib
export LD_LIBRARY_PATH="${INSTALL_ROOT}/ctranslate2/lib:${LD_LIBRARY_PATH:-}"

log "  TTS venv (kokoro-onnx 0.5.0 + CUDA onnxruntime)"
python3 -m venv "${INSTALL_ROOT}/venv-tts"
"${INSTALL_ROOT}/venv-tts/bin/pip" install --no-cache-dir -q --upgrade pip setuptools wheel
"${INSTALL_ROOT}/venv-tts/bin/pip" install --no-cache-dir -q \
    "kokoro-onnx==0.5.0" "espeakng-loader==0.2.4" "phonemizer-fork==3.3.2" \
    "numpy==2.2.6" "soundfile==0.13.1" "wyoming==1.8.0"
# kokoro-onnx drags in the CPU onnxruntime (a *different* distribution name,
# so pip keeps both); uninstall it or the CPU files shadow our CUDA build
"${INSTALL_ROOT}/venv-tts/bin/pip" uninstall -y onnxruntime >/dev/null 2>&1 || true
# force the CUDA wheel over whatever onnxruntime files remain
"${INSTALL_ROOT}/venv-tts/bin/pip" install --no-cache-dir -q --force-reinstall --no-deps "${ORT_WHEEL}"

# verify GPU providers are actually present
"${INSTALL_ROOT}/venv-tts/bin/python3" - <<'EOF'
import onnxruntime as rt
p = rt.get_available_providers()
assert "CUDAExecutionProvider" in p, f"CUDAExecutionProvider missing: {p}"
print("  onnxruntime", rt.__version__, "providers:", p)
EOF
"${INSTALL_ROOT}/venv-stt/bin/python3" - <<'EOF'
import ctranslate2
print("  ctranslate2", ctranslate2.__version__,
      "cuda devices:", ctranslate2.get_cuda_device_count())
EOF

# ---- 5. models (pinned URLs + sha256) -------------------------------------
log "fetching models (skipped if present and hash-verified)"
fetch() {
    # fetch <url> <dest> <sha256>
    local url="$1" dest="$2" sha="$3"
    if [ -f "${dest}" ] && echo "${sha}  ${dest}" | sha256sum -c --quiet 2>/dev/null; then
        echo "  have $(basename "${dest}")"
        return 0
    fi
    echo "  downloading $(basename "${dest}")"
    wget -q --show-progress -c -O "${dest}.part" "${url}"
    echo "${sha}  ${dest}.part" | sha256sum -c --quiet || {
        rm -f "${dest}.part"; die "checksum mismatch for ${dest}"; }
    mv "${dest}.part" "${dest}"
}

# effective model: explicit flag, else installed value, else default
LLM_ENV=/etc/super-soaker/llm.env
EFF_MODEL="$LLM_MODEL"
if [ -z "$EFF_MODEL" ] && [ -f "$LLM_ENV" ]; then
    EFF_MODEL=$(grep -E "^LLM_MODEL=" "$LLM_ENV" | head -1 | cut -d= -f2 || true)
fi
# legacy profile names (pre-rename) still resolve so old env files keep working
case "$EFF_MODEL" in
    lfm2.5-1.2b-chat)        EFF_MODEL=fast ;;
    lfm2.5-2.6b-reasoning)   EFF_MODEL=balanced ;;
    lfm2.5-8b-a1b-reasoning) EFF_MODEL=thorough ;;
esac
EFF_MODEL="${EFF_MODEL:-balanced}"
case "$EFF_MODEL" in
    fast|balanced|thorough) ;;
    *) die "unknown LLM_MODEL '$EFF_MODEL' in ${LLM_ENV} (valid profiles: fast | balanced | thorough)" ;;
esac

case "$EFF_MODEL" in
  balanced)
    log "  LLM: LFM2.5-2.6B QAD-Q4_0 (balanced profile)"
    fetch "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-QAD-Q4_0.gguf" \
          "${MODEL_DIR}/LFM2.5-2.6B-QAD-Q4_0.gguf" \
          "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03"
    fetch "https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF/resolve/main/LFM2.5-2.6B-DSpark-Q8_0.gguf" \
          "${MODEL_DIR}/LFM2.5-2.6B-DSpark-Q8_0.gguf" \
          "85a98fafdaf1328b6876fd1360d7ed69e74c6cefc14dd07fb6306e1940386c87"
    ;;
  thorough)
    log "  LLM: LFM2.5-8B-A1B UD-IQ2_M (thorough profile)"
    # Unsloth UD-IQ2_M is the largest 8B-A1B quant that fits with STT+TTS
    # resident on the 8 GB box without touching swap (see README "Model
    # choice"); the drafter is LiquidAI's official DSpark-Q8_0.
    fetch "https://huggingface.co/unsloth/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-UD-IQ2_M.gguf" \
          "${MODEL_DIR}/LFM2.5-8B-A1B-UD-IQ2_M.gguf" \
          "463670a7140a5d9e2a05f05afce1cd55e81ec764bbfd26e80287727d3e48645f"
    fetch "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF/resolve/main/LFM2.5-8B-A1B-DSpark-Q8_0.gguf" \
          "${MODEL_DIR}/LFM2.5-8B-A1B-DSpark-Q8_0.gguf" \
          "78c6edb3ec97d54fa2e6ec1ba44c394db0a1ae62fcd854d2193389c1295e0a6d"
    ;;
  fast)
    log "  LLM: LFM2.5-1.2B-Instruct Q8_0 (fast profile)"
    # Q8_0 chosen over QAD-Q4_0 for fidelity insurance (notes/07: battery
    # accuracy is identical at QAD-Q4_0/Q8_0/F16 — the cap is the model —
    # but Q8_0 removes quant noise from the equation at a modest speed cost).
    fetch "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q8_0.gguf" \
          "${MODEL_DIR}/LFM2.5-1.2B-Instruct-Q8_0.gguf" \
          "f6b981dcb86917fa463f78a362320bd5e2dc45445df147287eedb85e5a30d26a"
    fetch "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-DSpark-GGUF/resolve/main/LFM2.5-1.2B-Instruct-DSpark-Q8_0.gguf" \
          "${MODEL_DIR}/LFM2.5-1.2B-Instruct-DSpark-Q8_0.gguf" \
          "9f2fd36f4ea26e66b2732cefead2586d90220d829d70e63de8030e3878d49ef5"
    ;;
esac
fetch "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx" \
      "${TTS_DIR}/kokoro-v1.0.fp16.onnx" \
      "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204"
fetch "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
      "${TTS_DIR}/voices-v1.0.bin" \
      "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
# kokoro-onnx expects voices.npy; the release ships the same npy blob as .bin
[ -f "${TTS_DIR}/voices.npy" ] || cp "${TTS_DIR}/voices-v1.0.bin" "${TTS_DIR}/voices.npy"

# whisper tiny (en): pre-warm the huggingface hub cache into STT_DIR so first
# boot works offline (wyoming-faster-whisper resolves "tiny" -> this repo).
log "  pre-warming whisper-tiny model cache"
HF_HOME="${STT_DIR}" "${INSTALL_ROOT}/venv-stt/bin/python3" - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-tiny")
print("  whisper tiny cached")
EOF

chown -R "${SERVICE_USER}:${SERVICE_USER}" /var/lib/super-soaker

# ---- 6. install service code + config -------------------------------------
log "installing service code into ${INSTALL_ROOT}/services"
mkdir -p "${INSTALL_ROOT}/services" "${INSTALL_ROOT}/deps"
cp -a "${REPO_DIR}/services/." "${INSTALL_ROOT}/services/"
cp -a "${REPO_DIR}/deps/." "${INSTALL_ROOT}/deps/"
chmod +x "${INSTALL_ROOT}"/services/*/serve.sh "${INSTALL_ROOT}"/deps/*.sh
# service dirs renamed/removed in the repo must not linger on installed nodes
for d in "${INSTALL_ROOT}/services"/*/; do
    name="$(basename "$d")"
    if [ ! -d "${REPO_DIR}/services/${name}" ]; then
        log "  removing stale service dir ${name}"
        rm -rf "$d"
    fi
done

# env files: written once; edits are preserved on re-run
write_env() {
    local file="$1"; shift
    if [ -f "${file}" ]; then
        echo "  keeping existing ${file}"
    else
        printf '%s\n' "$@" > "${file}"
        echo "  wrote ${file}"
    fi
}
# llm.env reflects the chosen serving profile. It is rewritten when an
# explicit --model or --reasoning-budget differs from what is installed
# (those are the supported ways to switch config on an installed node);
# otherwise existing edits are preserved. EFF_MODEL (resolved before the
# model fetch above) is the effective profile.
# sensible thinking-budget defaults per model
case "$EFF_MODEL" in
    fast) DEFAULT_BUDGET=0 ;;      # no reasoning in this profile
    *)    DEFAULT_BUDGET=128 ;;    # balanced / thorough
esac
if [ -f "$LLM_ENV" ]; then
    cur_model=$(grep -E "^LLM_MODEL=" "$LLM_ENV" | head -1 | cut -d= -f2 || true)
    cur_budget=$(grep -E "^REASONING_BUDGET=" "$LLM_ENV" | head -1 | cut -d= -f2 || true)
    need_rewrite=0
    if [ -n "$LLM_MODEL" ] && ! grep -q "^LLM_MODEL=$LLM_MODEL$" "$LLM_ENV"; then
        need_rewrite=1
    fi
    if [ -n "$REASONING_BUDGET_ARG" ] && [ "$cur_budget" != "$REASONING_BUDGET_ARG" ]; then
        need_rewrite=1
    fi
    # legacy profile name stored in the env file: heal it to the new name
    case "${cur_model:-}" in
        lfm2.5-1.2b-chat|lfm2.5-2.6b-reasoning|lfm2.5-8b-a1b-reasoning) need_rewrite=1 ;;
    esac
    if [ "$need_rewrite" = "1" ]; then
        echo "  updating LLM config (model=$EFF_MODEL): rewriting $LLM_ENV"
        rm -f "$LLM_ENV"
    fi
fi
write_env "$LLM_ENV" \
    "# super-soaker LLM tuning (see notes/03-final-config.md)" \
    "PORT=8080" "N_BATCH=1024" "UBATCH=256" "KV_TYPE=q8_0" \
    "SPEC=auto" "CACHE_REUSE=256" "PRIO=2" \
    "SELF_TEST=on" \
    "LLM_MODEL=$EFF_MODEL" \
    "REASONING_BUDGET=${REASONING_BUDGET_ARG:-$DEFAULT_BUDGET}" \
    "# per-model defaults (CTX, model paths) are applied from LLM_MODEL in" \
    "# serve.sh; REASONING_BUDGET above: N caps thinking at N tokens," \
    "# 0/off disables thinking, on = always think, auto = detect from template." \
    "#CTX=32768"
write_env /etc/super-soaker/stt.env \
    "# super-soaker STT tuning" \
    "PORT=10300" "MODEL=tiny" "LANGUAGE=en" "COMPUTE_TYPE=int8" "BEAM_SIZE=5" \
    "SELF_TEST=on"
write_env /etc/super-soaker/tts.env \
    "# super-soaker TTS tuning" \
    "WYOMING_PORT=10210" "KOKORO_QUANTIZATION=fp16" "KOKORO_VOICE=af_heart" \
    "KOKORO_SPEED=1.0" "KOKORO_GPU_MEM_MB=1024" \
    "KOKORO_MODEL_DIR=/var/lib/super-soaker/tts" "SELF_TEST=on"

# ---- 7. system config: no service pages in swap, MAXN_SUPER power mode -----
log "applying system configuration (swap policy, power mode)"
cat > /etc/sysctl.d/90-super-soaker.conf <<'EOF'
# super-soaker: keep voice-stack latency deterministic.
# Services are also pinned with MemorySwapMax=0; this discourages the kernel
# from pushing anything anonymous (incl. the LLM) to zram under pressure.
vm.swappiness=10
EOF
sysctl --system >/dev/null

# Power mode: MAXN_SUPER raises the frequency ceilings (CPU/GPU/EMC boost to
# their max under load). Nothing is pinned: idle DVFS scales clocks down.
# (jetson_clocks pinning was deliberately removed 2026-10-14 — it cost ~1.15 W
# constant idle power on an always-on box. nvpmodel MAXN_SUPER does NOT lock
# clocks; it only raises the ceilings.) The mode persists in
# /var/lib/nvpmodel and NVIDIA's stock nvpmodel.service re-applies it on
# every boot — no super-soaker unit involved.
MODE_ID="$(awk -F'[= >]' '/POWER_MODEL/ && /NAME=MAXN_SUPER/ {for(i=1;i<=NF;i++) if($i=="ID") print $(i+1)}' /etc/nvpmodel.conf | head -1)"
if [ -z "${MODE_ID}" ]; then
    # fall back to whatever mode is the highest-power one currently set
    MODE_ID="$(nvpmodel -q | awk '/NV Power Mode/{getline; print; exit}')"
    log "MAXN_SUPER not found in nvpmodel.conf; keeping mode ${MODE_ID}"
else
    log "nvpmodel -m ${MODE_ID} (MAXN_SUPER)"
    nvpmodel -m "${MODE_ID}"
fi

cp "${REPO_DIR}/systemd/super-soaker-stt.service" /etc/systemd/system/
cp "${REPO_DIR}/systemd/super-soaker-llm.service" /etc/systemd/system/
cp "${REPO_DIR}/systemd/super-soaker-tts.service" /etc/systemd/system/
# the old clock-lock unit/script are gone: purge them from installed nodes
if [ -f /etc/systemd/system/super-soaker-clocks.service ]; then
    log "removing stale unit super-soaker-clocks.service (clock pinning removed)"
    systemctl disable --now super-soaker-clocks.service >/dev/null 2>&1 || true
    rm -f /etc/systemd/system/super-soaker-clocks.service
fi
rm -f "${INSTALL_ROOT}/deps/lock-clocks.sh"
systemctl daemon-reload
systemctl enable super-soaker-llm.service \
                  super-soaker-stt.service \
                  super-soaker-tts.service

log "starting services"
systemctl restart super-soaker-llm.service
systemctl restart super-soaker-stt.service
systemctl restart super-soaker-tts.service

# ---- 8. wait for healthy + self-tests -------------------------------------
log "waiting for services to pass their built-in self-tests"
fail=0
for svc in llm stt tts; do
    echo -n "  ${svc}: "
    for i in $(seq 1 240); do
        state="$(systemctl is-active "super-soaker-${svc}.service" || true)"
        if [ "${state}" = "active" ] && \
           journalctl -u "super-soaker-${svc}.service" --since "-10 min" -q 2>/dev/null | grep -q "self-test PASSED"; then
            echo "OK"
            break
        fi
        if [ "$i" = "240" ]; then
            echo "FAILED (see: journalctl -u super-soaker-${svc})"
            fail=1
        fi
        sleep 2
    done
done
[ "${fail}" = "0" ] || die "one or more services failed self-test"

# ---- 9. summary -----------------------------------------------------------
log "done — service status"
systemctl --no-pager --type=service status "super-soaker-*" 2>/dev/null | \
    grep -E "^●|Active:|MemorySwapMax" || true
free -h | head -3
cat <<'EOF'

[bootstrap] stack ready:
  STT  wyoming   tcp://<host>:10300   (1-stt-wyoming-whisper-tiny)
  LLM  OpenAI    http://<host>:8080   (2-llm-llama.cpp-super-soaker-managed)
  TTS  wyoming   tcp://<host>:10210   (3-tts-wyoming-kokoro-82m)

  logs:   journalctl -u super-soaker-{stt,llm,tts} -f
  tests:  /opt/super-soaker/services/*/selftest.py
  RAM:    all three services pinned with MemorySwapMax=0 (never swapped)
EOF

if [ "${DO_REBOOT}" = "1" ]; then
    log "rebooting in 5s (services enabled to start on boot)"
    sleep 5
    reboot
else
    log "--no-reboot given; reboot when convenient"
fi