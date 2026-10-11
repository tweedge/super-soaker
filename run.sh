#!/bin/bash
# Host-side convenience: build and run super-soaker on the Jetson.
# Usage:
#   ./run.sh build-dev
#   ./run.sh build-runtime [model.gguf] [draft.gguf]
#   ./run.sh serve          # run the runtime image with self-test
#   ./run.sh bench          # interactive dev container with models mounted
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/.cache/super-soaker/models}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/.cache/super-soaker/results}"
IMAGE_DEV="super-soaker:dev"
IMAGE_RT="super-soaker:latest"
DOCKER="docker"

run_docker() {
    if docker info >/dev/null 2>&1; then
        $DOCKER "$@"
    else
        sudo $DOCKER "$@"
    fi
}

case "${1:-help}" in
build-dev)
    run_docker build --target dev -t "$IMAGE_DEV" .
    ;;
build-runtime)
    MODEL="${2:-LFM2.5-2.6B-QAD-Q4_0.gguf}"
    DRAFT="${3:-LFM2.5-2.6B-DSpark-Q8_0.gguf}"
    mkdir -p models
    cp -f "$MODELS_DIR/$MODEL" "models/$MODEL"
    cp -f "$MODELS_DIR/$DRAFT" "models/$DRAFT"
    run_docker build --target runtime -t "$IMAGE_RT" \
        --build-arg MODEL_FILE="$MODEL" --build-arg DRAFT_FILE="$DRAFT" .
    ;;
serve)
    mkdir -p "$RESULTS_DIR"
    run_docker rm -f super-soaker 2>/dev/null || true
    run_docker run -d --name super-soaker \
        --restart always \
        --runtime nvidia --network host \
        --memory=5g --memory-swap=5g \
        -e SELF_TEST=on \
        "$IMAGE_RT"
    run_docker logs -f super-soaker
    ;;
bench)
    mkdir -p "$RESULTS_DIR"
    run_docker run --rm -it \
        --runtime nvidia --network host \
        -v "$MODELS_DIR":/models:ro \
        -v "$RESULTS_DIR":/results \
        -v "$(pwd)":/app-src \
        "$IMAGE_DEV" bash
    ;;
help|*)
    echo "usage: $0 {build-dev|build-runtime [model] [draft]|serve|bench}"
    ;;
esac
