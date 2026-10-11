# super-soaker — LFM2.5-2.6B for Jetson Orin Nano Super (SM 8.7 / L4T 36.4 / CUDA 12.6)
#
# Everything is built and served in Docker on the Jetson. Multi-target:
#   dev     — full toolchain (llama-server/bench/quantize/cli + python3) for the
#             benchmark matrix; bind-mount models, do not bake them.
#   runtime — final self-contained image: llama-server + baked GGUFs + self-test.
#
# Build (on the Jetson):
#   docker build --target dev     -t super-soaker:dev     .
#   docker build --target runtime -t super-soaker:latest  .
#
# Run:
#   docker run --runtime nvidia --network host super-soaker:dev   # bench shell
#   docker run --runtime nvidia --network host -p 8080:8080 super-soaker:latest

ARG CUDA_ARCH=87
ARG LLAMA_CPP_SHA=23b0202a189c44a54625aadcb37a946dd1d6278d
ARG BASE_BUILDER=nvcr.io/nvidia/12.6.11-devel:12.6.11-devel-aarch64-ubuntu22.04
ARG BASE_RUNTIME=nvcr.io/nvidia/l4t-cuda:12.6.11-runtime

# =============================================================================
# builder — compile llama.cpp for SM 8.7
# =============================================================================
FROM ${BASE_BUILDER} AS builder

ARG CUDA_ARCH
ARG LLAMA_CPP_SHA

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && git checkout "${LLAMA_CPP_SHA}"

# GGML_CUDA_NO_VMM: avoids the 32GB VMM reservation that crashes Jetson iGPU
# (llama.cpp issue #29142). FA_ALL_QUANTS: flash-attn for every KV quant.
# GRAPHS: CUDA graphs on (default, made explicit). SM 8.7 = Orin.
RUN cmake -S llama.cpp -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_NO_VMM=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --parallel $(nproc) \
    && cmake --install build --prefix /opt/llama \
    && ls -la /opt/llama/bin \
    && find /opt/llama -name '*.so*' -o -name '*.a' | head -20

# CUDA runtime libs the binaries link against (cudart/cublas/cublasLt)
RUN mkdir -p /opt/cuda-libs \
    && cp -a /usr/local/cuda/lib64/libcudart.so* /opt/cuda-libs/ \
    && cp -a /usr/local/cuda/lib64/libcublas.so* /opt/cuda-libs/ \
    && cp -a /usr/local/cuda/lib64/libcublasLt.so* /opt/cuda-libs/ \
    && ls -la /opt/cuda-libs

# =============================================================================
# dev — benchmark image (tools + python, models bind-mounted at /models)
# =============================================================================
FROM ${BASE_RUNTIME} AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/llama /opt/llama
COPY --from=builder /opt/cuda-libs /opt/cuda-libs
ENV PATH="/opt/llama/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/cuda-libs:/opt/llama/lib:/opt/llama/lib64:${LD_LIBRARY_PATH}"

RUN for b in /opt/llama/bin/*; do echo "== $b"; ldd "$b" | grep -E "not found" && exit 1 || true; done

WORKDIR /app
COPY bench/ /app/bench/
COPY entrypoint.sh selftest.py /app/

ENV MODEL_PATH=/models/LFM2.5-2.6B-Q4_K_M.gguf \
    DRAFT_PATH=/models/LFM2.5-2.6B-DSpark-Q8_0.gguf

CMD ["bash"]

# =============================================================================
# runtime — final self-contained serving image
# =============================================================================
FROM dev AS runtime

# Weights baked into the image (override at build time with --build-arg).
ARG MODEL_FILE=LFM2.5-2.6B-Q4_K_M.gguf
ARG DRAFT_FILE=LFM2.5-2.6B-DSpark-Q8_0.gguf

COPY models/${MODEL_FILE} /models/${MODEL_FILE}
COPY models/${DRAFT_FILE} /models/${DRAFT_FILE}

ENV MODEL_PATH=/models/${MODEL_FILE} \
    DRAFT_PATH=/models/${DRAFT_FILE}

EXPOSE 8080
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
