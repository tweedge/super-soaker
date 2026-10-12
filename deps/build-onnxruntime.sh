#!/usr/bin/env bash
# deps/build-onnxruntime.sh — build ONNX Runtime (CUDA) for L4T r36 / Python 3.10.
#
# No aarch64 CUDA wheels exist on PyPI (1.29+ aarch64 links CUDA 13, which the
# L4T 36 driver (CUDA 12.6) cannot load), so the runtime is built from source at
# a pinned tag and packaged as a wheel for the service venv.
#
# Output: /opt/super-soaker/cache/onnxruntime_gpu-<ver>-cp310-*.whl
set -euo pipefail

ORT_VERSION="1.22.0"                                   # pinned: matches kk-onnx validation
ORT_TAG="v${ORT_VERSION}"
SRC_ROOT="${SRC_ROOT:-/opt/super-soaker/build}"
CACHE_DIR="${CACHE_DIR:-/opt/super-soaker/cache}"
. "$(dirname "${BASH_SOURCE[0]}")/limit-jobs.sh"   # JOBS guard: 1 default, 2 max, >2 needs --force

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# ORT's build requires CMake >= 3.28; prefer a pip-installed cmake if present.
export PATH="${HOME}/.local/bin:${PATH}"
CMAKE_MIN="3.28.0"
if ! command -v cmake >/dev/null 2>&1 || \
   [ "$(printf '%s\n' "$CMAKE_MIN" "$(cmake --version | head -1 | awk '{print $3}')" | sort -V | head -1)" != "$CMAKE_MIN" ]; then
    echo "ERROR: cmake >= ${CMAKE_MIN} required (pip install cmake)" >&2
    exit 1
fi
# cuDNN 9 from apt installs headers into /usr/include — stage the layout ORT expects.
CUDNN_STAGE="${CACHE_DIR}/cudnn-stage"

if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "ERROR: nvcc not found (install cuda-nvcc-12-6 / cuda-toolkit-12-6)" >&2
    exit 1
fi

mkdir -p "${SRC_ROOT}" "${CACHE_DIR}"

# ---- stage cuDNN include/lib layout ----------------------------------------
mkdir -p "${CUDNN_STAGE}/include" "${CUDNN_STAGE}/lib"
for h in /usr/include/cudnn*.h /usr/include/aarch64-linux-gnu/cudnn*.h; do
    [ -f "$h" ] && ln -sf "$h" "${CUDNN_STAGE}/include/"
done
for l in /usr/lib/aarch64-linux-gnu/libcudnn.so* /usr/lib/aarch64-linux-gnu/libcudnn_*; do
    [ -e "$l" ] && ln -sf "$l" "${CUDNN_STAGE}/lib/"
done
if [ ! -e "${CUDNN_STAGE}/include/cudnn.h" ] && [ ! -e "${CUDNN_STAGE}/include/cudnn_v9.h" ]; then
    echo "ERROR: cuDNN headers not found (install libcudnn9-dev-cuda-12)" >&2
    exit 1
fi

# ---- source ----------------------------------------------------------------
if [ ! -d "${SRC_ROOT}/onnxruntime/.git" ]; then
    git clone --recursive https://github.com/microsoft/onnxruntime.git "${SRC_ROOT}/onnxruntime"
fi
cd "${SRC_ROOT}/onnxruntime"
git fetch --tags --force
git checkout "${ORT_TAG}"
git submodule sync
git submodule update --init --recursive

# ---- pinned dependency overrides -------------------------------------------
# Eigen is fetched by URL+SHA1 and GitLab regenerated the archive zip (hash
# drift). Pin it the content-addressed way instead: a git clone at the exact
# commit ORT itself pins, fed to FetchContent via FETCHCONTENT_SOURCE_DIR.
EIGEN_COMMIT="1d8b82b0740839c0de7f1242a3585e3390ff5f33"
if [ ! -d "${SRC_ROOT}/eigen3/.git" ]; then
    git clone https://gitlab.com/libeigen/eigen.git "${SRC_ROOT}/eigen3"
fi
cd "${SRC_ROOT}/eigen3"
git fetch origin --force
git checkout --force "${EIGEN_COMMIT}"
EIGEN_SRC="${SRC_ROOT}/eigen3"
cd "${SRC_ROOT}/onnxruntime"

# ---- build-time Python with NumPy 2.x ABI ---------------------------------
# The service venvs use NumPy 2 (kokoro-onnx requires >=2.0.2). If the wheel's
# pybind bindings compile against NumPy 1.x headers they crash at import with
# "_ARRAY_API not found", so the wheel must be built under NumPy 2.
ORT_BUILD_VENV="${SRC_ROOT}/ort-build-venv"
if [ ! -x "${ORT_BUILD_VENV}/bin/python3" ]; then
    python3 -m venv "${ORT_BUILD_VENV}"
fi
"${ORT_BUILD_VENV}/bin/pip" install -q --upgrade "pip" "wheel" "setuptools" "numpy>=2,<3"
echo "build python: $("${ORT_BUILD_VENV}/bin/python3" -c 'import sys, numpy; print(sys.executable, "numpy", numpy.__version__)')"
export VIRTUAL_ENV="${ORT_BUILD_VENV}"
export PATH="${ORT_BUILD_VENV}/bin:${PATH}"

# Ubuntu's python3-numpy drops 1.x headers into /usr/include/python3.10/numpy,
# which shadow the venv's NumPy 2 headers (first match wins), producing a
# NumPy-1-ABI wheel that dies on "_ARRAY_API not found" under NumPy 2. Force
# the venv include to the front with -I (beats -isystem).
NUMPY_INC="$("${ORT_BUILD_VENV}/bin/python3" -c 'import numpy; print(numpy.get_include())')"
echo "numpy include (forced first): ${NUMPY_INC}"

# ---- build CUDA-only (TensorRT EP intentionally omitted) --------------------
./build.sh \
    --config Release \
    --update --build --build_wheel --build_shared_lib \
    --parallel "${JOBS}" \
    --skip_tests \
    --allow_running_as_root \
    --cuda_home "${CUDA_HOME}" \
    --cudnn_home "${CUDNN_STAGE}" \
    --use_cuda \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 \
    --cmake_extra_defines CMAKE_CXX_FLAGS="-I${NUMPY_INC}" \
    --cmake_extra_defines FETCHCONTENT_SOURCE_DIR_EIGEN3="${EIGEN_SRC}" \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF

WHEEL="$(ls -t build/Linux/Release/dist/onnxruntime_gpu-*.whl | head -1)"
cp -f "${WHEEL}" "${CACHE_DIR}/"
echo "OK: ${CACHE_DIR}/$(basename "${WHEEL}")"