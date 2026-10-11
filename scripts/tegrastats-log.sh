#!/bin/bash
# Host-side: log tegrastats + memory while benchmarks run.
# Usage: sudo bash scripts/tegrastats-log.sh [outfile.log]
set -uo pipefail
OUT="${1:-$HOME/.cache/super-soaker/results/tegrastats-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$OUT")"
echo "logging tegrastats to $OUT (ctrl-c to stop)"
exec tegrastats --interval 200 --logfile "$OUT"
