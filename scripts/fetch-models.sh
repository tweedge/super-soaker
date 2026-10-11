#!/bin/bash
# Fetch GGUF weights from Hugging Face into MODELS_DIR (data cache used as
# the Docker build context). Run once on the Jetson before build-runtime.
#
# Usage:
#   ./scripts/fetch-models.sh                          # default pair
#   ./scripts/fetch-models.sh LFM2.5-2.6B-Q5_K_M.gguf  # any official file
#
# Files resolve against LiquidAI/LFM2.5-2.6B-GGUF and
# LiquidAI/LFM2.5-2.6B-DSpark-GGUF. Default: LFM2.5-2.6B-QAD-Q4_0.gguf
# (target) + LFM2.5-2.6B-DSpark-Q8_0.gguf (drafter).
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/.cache/super-soaker/models}"
BASE="https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main"
DSB="https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF/resolve/main"

mkdir -p "$MODELS_DIR"

download() {
    local name="$1" out="$MODELS_DIR/$1"
    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "have $out"
        return 0
    fi
    for url in "$BASE/$name" "$DSB/$name"; do
        echo "downloading $name"
        if curl -fL --retry 5 --retry-delay 3 -C - -o "$out.part" "$url"; then
            mv "$out.part" "$out"
            return 0
        fi
    done
    echo "ERROR: could not fetch $name from either GGUF repo" >&2
    return 1
}

if [ $# -eq 0 ]; then
    set -- LFM2.5-2.6B-QAD-Q4_0.gguf LFM2.5-2.6B-DSpark-Q8_0.gguf
fi
for f in "$@"; do download "$f"; done
echo "done -> $MODELS_DIR"
