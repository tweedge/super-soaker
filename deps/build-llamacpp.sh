#!/usr/bin/env bash
# deps/build-llamacpp.sh — build llama.cpp (CUDA, SM 87) at a pinned tag.
#
# Pin: v0.9.0 (cd959bae99) — the release that adds LFM3 support (llama.cpp
# #31252, LFM3-1.7B needs >= b11960) and the LFM3 spec-decode plumbing (#31598).
# It is the exact build validated by the super-soaker self-test/bench for the
# `frontier` profile, and the LFM2.5 profiles pass their self-test on it too.
# Flags mirror the original Dockerfile: FA_ALL_QUANTS + GRAPHS on, GGML_CUDA_NO_VMM=ON
# (Jetson iGPU VMM crash workaround, llama.cpp #29142).
#
# Output: /opt/super-soaker/llama.cpp/{bin,lib,include}
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-v0.9.0}"
LLAMA_COMMIT="cd959bae99ff4bed22bef2604f45241c09bbb8c8"   # pinned: v0.9.0
CUDA_ARCH="${CUDA_ARCH:-87}"
SRC_ROOT="${SRC_ROOT:-/opt/super-soaker/build}"
PREFIX="${PREFIX:-/opt/super-soaker/llama.cpp}"
. "$(dirname "${BASH_SOURCE[0]}")/limit-jobs.sh"   # JOBS guard: 1 default, 2 max, >2 needs --force

# nvcc is not on the default PATH on L4T — point CMake at it explicitly.
if [ -z "${CUDACXX:-}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDACXX=/usr/local/cuda/bin/nvcc
    else
        echo "ERROR: nvcc not found (install cuda-nvcc-12-6)" >&2
        exit 1
    fi
fi

mkdir -p "${SRC_ROOT}" "${PREFIX}"

if [ ! -d "${SRC_ROOT}/llama.cpp/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "${SRC_ROOT}/llama.cpp"
fi
cd "${SRC_ROOT}/llama.cpp"
git fetch origin --tags --force
git checkout --force "${LLAMA_COMMIT}"
git submodule update --init --recursive

# super-soaker local patch: Orin (SM 8.7) MMVQ->MMQ crossover for IQ/simple
# quants (upstream PR #28285 covered K-quants only). Idempotent.
PATCH_FILE="$(dirname "${BASH_SOURCE[0]}")/patches/mmvq-orin-crossover.patch"
if [ -f "$PATCH_FILE" ] && ! grep -q "SUPER_SOAKER_ORIN_PATCH" ggml/src/ggml-cuda/mmvq.cu; then
    git apply "$PATCH_FILE" || { echo "WARN: mmvq patch did not apply" >&2; }
fi

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_NO_VMM=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DBUILD_SHARED_LIBS=OFF
cmake --build build --parallel "${JOBS}" --target llama-server

# Install just the server (cmake --install would try to ship unbuilt side
# targets like batched-bench and fail).
mkdir -p "${PREFIX}/bin"
cp -f build/bin/llama-server "${PREFIX}/bin/"

echo "OK: ${PREFIX}/bin/llama-server"
"${PREFIX}/bin/llama-server" --version | head -2