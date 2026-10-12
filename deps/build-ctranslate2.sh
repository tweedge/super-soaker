#!/usr/bin/env bash
# deps/build-ctranslate2.sh — build CTranslate2 (CUDA + cuDNN + OpenBLAS) for
# L4T r36 / SM 87 and package a Python wheel for the service venv.
#
# The PyPI ctranslate2 wheel for aarch64 is CPU-only, so this compiles from
# source at a pinned tag (same approach as the old wyoming-whisper-gpu image).
#
# Output: /opt/super-soaker/cache/ctranslate2-<ver>-cp310-*.whl
set -euo pipefail

CT2_VERSION="4.5.0"                                    # pinned
CT2_TAG="v${CT2_VERSION}"
CUDA_ARCH="${CUDA_ARCH:-87}"
SRC_ROOT="${SRC_ROOT:-/opt/super-soaker/build}"
CACHE_DIR="${CACHE_DIR:-/opt/super-soaker/cache}"
. "$(dirname "${BASH_SOURCE[0]}")/limit-jobs.sh"   # JOBS guard: 1 default, 2 max, >2 needs --force

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDNN_STAGE="${CACHE_DIR}/cudnn-stage"
export PATH="${HOME}/.local/bin:${PATH}"

# CT2's FindCUDA needs the unversioned cublas/curand dev symlinks.
if [ ! -e "${CUDA_HOME}/lib64/libcublas.so" ] && [ ! -e /usr/lib/aarch64-linux-gnu/libcublas.so ]; then
    echo "ERROR: libcublas.so dev symlink missing (install libcublas-dev-12-6)" >&2
    exit 1
fi

mkdir -p "${SRC_ROOT}" "${CACHE_DIR}"

mkdir -p "${CUDNN_STAGE}/include" "${CUDNN_STAGE}/lib"
for h in /usr/include/cudnn*.h /usr/include/aarch64-linux-gnu/cudnn*.h; do
    [ -f "$h" ] && ln -sf "$h" "${CUDNN_STAGE}/include/"
done
for l in /usr/lib/aarch64-linux-gnu/libcudnn.so* /usr/lib/aarch64-linux-gnu/libcudnn_*; do
    [ -e "$l" ] && ln -sf "$l" "${CUDNN_STAGE}/lib/"
done

if [ ! -d "${SRC_ROOT}/CTranslate2/.git" ]; then
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git "${SRC_ROOT}/CTranslate2"
fi
cd "${SRC_ROOT}/CTranslate2"
git fetch --tags --force
git checkout "${CT2_TAG}"
git submodule sync
git submodule update --init --recursive

# ---- C++ library -----------------------------------------------------------
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/super-soaker/ctranslate2 \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DWITH_MKL=OFF \
    -DWITH_OPENBLAS=ON \
    -DOPENMP_RUNTIME=COMP \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    -DCUDNN_INCLUDE_DIR="${CUDNN_STAGE}/include" \
    -DCUDNN_LIBRARY="${CUDNN_STAGE}/lib/libcudnn.so" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build build --parallel "${JOBS}"
cmake --install build

# ---- Python wheel linked against the library above --------------------------
python3 -m venv "${CACHE_DIR}/ct2-wheel-venv"
"${CACHE_DIR}/ct2-wheel-venv/bin/pip" install --no-cache-dir -q setuptools wheel pybind11
CT2_CUDA_ARCHS="${CUDA_ARCH}" \
CT2_CUDA_ROOT="${CUDA_HOME}" \
CTRANSLATE2_ROOT=/opt/super-soaker/ctranslate2 \
CMAKE_ARGS="-DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
            -DCUDNN_INCLUDE_DIR=${CUDNN_STAGE}/include \
            -DCUDNN_LIBRARY=${CUDNN_STAGE}/lib/libcudnn.so \
            -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
CFLAGS="-I/opt/super-soaker/ctranslate2/include" \
LDFLAGS="-L/opt/super-soaker/ctranslate2/lib -Wl,-rpath,/opt/super-soaker/ctranslate2/lib" \
"${CACHE_DIR}/ct2-wheel-venv/bin/pip" wheel \
    --no-build-isolation --no-cache-dir \
    -w "${CACHE_DIR}" ./python

rm -rf "${CACHE_DIR}/ct2-wheel-venv"
echo "OK: ${CACHE_DIR}/$(ls -t "${CACHE_DIR}"/ctranslate2-*.whl | head -1 | xargs basename)"