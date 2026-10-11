#!/bin/bash
# Post-quants chain (runs inside dev container): quality comparison at fixed
# thinking budget, then draft sweep, then flag sweep. Target quant: QAD-Q4_0.
set -uo pipefail
export TARGET=QAD-Q4_0
export DRAFT=Q8_0

echo "=== quality comparison at reasoning-budget 256 ==="
for q in QAD-Q4_0 Q4_K_M Q5_K_M; do
    m="/models/LFM2.5-2.6B-${q}.gguf"
    bash /app/bench/matrix.sh one "${q}-spec-rb256" \
        -m "$m" -ngl 99 -c 32768 -np 1 -fa on --host 127.0.0.1 --port 8080 \
        --jinja --alias LFM2.5-2.6B --metrics --cache-reuse 256 \
        --reasoning-budget 256 \
        -md /models/LFM2.5-2.6B-DSpark-Q8_0.gguf \
        --spec-type draft-dspark --spec-draft-n-max 10 --spec-draft-n-min 0
done

echo "=== draft sweep ==="
bash /app/bench/matrix.sh draft

echo "=== flags sweep ==="
bash /app/bench/matrix.sh flags

echo "=== chain complete ==="
