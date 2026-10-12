#!/usr/bin/env bash
# deps/build-llamacpp.sh — build llama.cpp (CUDA, SM 87) at a pinned tag.
#
# Pin: b11552 (23b0202a1) — the exact build validated by the super-soaker
# self-test/bench (system_fingerprint "b11552-23b0202a1"). Flags mirror the
# original Dockerfile: FA_ALL_QUANTS + GRAPHS on, GGML_CUDA_NO_VMM=ON (Jetson
# iGPU VMM crash workaround, llama.cpp #29142).
#
# Output: /opt/super-soaker/llama.cpp/{bin,lib,include}
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b11552}"
LLAMA_COMMIT="23b0202a189c44a54625aadcb37a946dd1d6278d"   # pinned: b11552
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