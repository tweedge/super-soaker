#!/bin/bash
# Fetch extra GGUF weights from Hugging Face for bench sweep matrices.
# (bootstrap.sh already fetches the default serving pair; this is for the
# additional quant/drafter variants used by bench/matrix.sh.)
#
# Usage:
#   ./scripts/fetch-models.sh LFM2.5-2.6B-Q5_K_M.gguf LFM2.5-2.6B-DSpark-F16.gguf
#
# Files resolve against these repos (first hit wins):
#   LiquidAI/LFM2.5-2.6B-GGUF, LiquidAI/LFM2.5-2.6B-DSpark-GGUF
#   LiquidAI/LFM2.5-8B-A1B-GGUF, LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF
#   unsloth/LFM2.5-8B-A1B-GGUF (Unsloth Dynamic quants, UD-*)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/var/lib/super-soaker/models}"
REPOS=(
    "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main"
    "https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF/resolve/main"
    "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main"
    "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF/resolve/main"
    "https://huggingface.co/unsloth/LFM2.5-8B-A1B-GGUF/resolve/main"
)

mkdir -p "$MODELS_DIR"

download() {
    local name="$1" out="$MODELS_DIR/$1"
    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "have $out"
        return 0
    fi
    for base in "${REPOS[@]}"; do
        echo "downloading $name"
        if curl -fL --retry 5 --retry-delay 3 -C - -o "$out.part" "$base/$name"; then
            mv "$out.part" "$out"
            return 0
        fi
    done
    echo "ERROR: could not fetch $name from any known GGUF repo" >&2
    return 1
}

if [ $# -eq 0 ]; then
    set -- LFM2.5-2.6B-QAD-Q4_0.gguf LFM2.5-2.6B-DSpark-Q8_0.gguf
fi
for f in "$@"; do download "$f"; done
echo "done -> $MODELS_DIR"