# shellcheck shell=bash
# deps/limit-jobs.sh — shared RAM guard for the deps/build-*.sh source builds.
# Sourced, not executed.
#
# The Jetson Orin Nano Super has 8 GB of RAM shared with its GPU, and one nvcc
# thread compiling ONNX Runtime's flash-attention kernels peaks near 3.5 GB
# RSS. Two threads at once were observed to wedge the box (load 30, RAM and
# swap full, power reset required). So:
#
#   JOBS=1  default — safe everywhere, but expect the build to take a WHILE.
#   JOBS=2  tested maximum on the 8 GB Orin Nano Super.
#   JOBS>2  refused unless you pass --force (or set FORCE=1) — intended for
#           16 GB devices like the Jetson Orin NX, where the operator has
#           the RAM headroom to accept the risk.
#
# Usage from a build script:
#   . "$(dirname "${BASH_SOURCE[0]}")/limit-jobs.sh"

FORCE="${FORCE:-0}"
for _arg in "$@"; do
    if [ "$_arg" = "--force" ]; then FORCE=1; fi
done

JOBS="${JOBS:-1}"

case "$JOBS" in
    ''|*[!0-9]*|0)
        echo "ERROR: JOBS must be a positive integer (got: '${JOBS}')" >&2
        exit 1
        ;;
esac

if [ "$JOBS" -gt 2 ] && [ "$FORCE" != "1" ]; then
    cat >&2 <<'EOF'
ERROR: JOBS > 2 is refused on this device class.

  Each compile thread can consume multiple GB of RAM (ONNX Runtime's nvcc
  flash-attention kernels peak near 3.5 GB RSS each), and the 8 GB Orin
  Nano Super shares that RAM with the GPU and the voice stack. Running
  with more than 2 build threads wedged this machine and required a power
  reset.

  Recommended:
    JOBS=1   default; safe, but the build can take a WHILE (1-2 hours).
    JOBS=2   tested maximum on the 8 GB Orin Nano Super.

  If you are on a 16 GB device (e.g. Jetson Orin NX) and accept the risk:

    JOBS=4 bash build-onnxruntime.sh --force
EOF
    exit 1
fi

if [ "$JOBS" -eq 1 ]; then
    echo "build: JOBS=1 — safe on 8 GB devices; expect the build to take a WHILE."
elif [ "$FORCE" = "1" ]; then
    echo "build: JOBS=${JOBS} with --force — operator accepts the RAM risk."
else
    echo "build: JOBS=${JOBS} — tested maximum for the 8 GB Orin Nano Super."
fi
true