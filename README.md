# super-soaker

Ultra low-latency, OpenAI-compatible LLM service for [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) + [LFM2.5-2.6B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark) speculative decoding on the **NVIDIA Jetson Orin Nano Super**, built and served entirely in Docker.

Designed for a single caller (Home Assistant + [hass_local_openai_llm](https://github.com/skye-harris/hass_local_openai_llm) / Home Assistant Voice) where every millisecond of response latency is felt.

## Prerequisites

Everything builds and runs in Docker on the Jetson itself. On a freshly flashed Jetson Orin Nano Super (8 GB):

1. **JetPack 6.x** (L4T r36.x, CUDA 12.6) — the image targets SM 8.7 and the L4T driver stack; other devices/versions are not supported.
2. **Docker Engine** — `sudo apt-get install docker.io` (or NVIDIA's documented install), and your user in the `docker` group.
3. **NVIDIA Container Toolkit** so `--runtime nvidia` works (JetPack installs this via SDK Manager; otherwise follow [NVIDIA's Jetson install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)). Verify with `docker info | grep -i runtime` showing `nvidia`.
4. **`git` and `curl`** on the host (fetch script and clone).
5. **Disk**: ~25 GB free during the build (builder image, ~15 GB final image, 2 GB model cache).
6. **Network** during build only (GitHub, Hugging Face, `nvcr.io`, Ubuntu apt). The finished container needs no network access.

Recommended host setup for best numbers: `sudo nvpmodel -m 0` (MAXN / MAXN_SUPER power mode) and `sudo jetson_clocks` to lock clocks. The measurements below were taken that way.

## Quick start

```bash
./scripts/fetch-models.sh        # downloads the GGUFs (~1.8 GB) into the model cache
./run.sh build-runtime           # compiles llama.cpp for SM 8.7 + bakes the weights
./run.sh serve                   # self-tests end to end, then serves on :8080
```

Point Home Assistant's *Local OpenAI LLM* integration at `http://<jetson-ip>:8080/v1` (server type **llama.cpp**, model `LFM2.5-2.6B`, API key blank). The container is created with `--restart always`, so the service (and its boot self-test) comes back automatically after power loss.

## How it works

- **llama.cpp** (pinned commit, see `Dockerfile`) compiled in a multi-stage build for CUDA SM 8.7 with flash attention, CUDA graphs, and the Jetson VMM workaround; static binaries on an L4T CUDA runtime base.
- **LFM2.5-2.6B-QAD-Q4_0** (Liquid's quality-preserving distillation of Q4_0) as the target, **LFM2.5-2.6B-DSpark-Q8_0** as the speculative-decoding drafter. Both GGUFs are baked into the image; `MODEL_PATH`/`DRAFT_PATH` env vars can point at others.
- **DSpark speculative decoding** gives 2.2–3.0x decode throughput on this bandwidth-limited platform. Flash attention is required by the drafter and always enabled.
- **Boot self-test** (`selftest.py`): model listing, warmups, streaming speed measurements, forced + auto tool-call round trips, multi-turn cache probe. Results are logged and written to `/tmp/selftest.json`; the service serves regardless (set `SELF_TEST_STRICT=on` to exit on failure).

## Measured performance

Jetson Orin Nano Super 8 GB, MAXN_SUPER + locked clocks, default config (QAD-Q4_0 + DSpark-Q8_0, thinking budget 128, 32k context):

| scenario | TTFT | total | decode |
|---|---|---|---|
| short QA (streaming) | 58 ms | 1.98 s | 71 tok/s |
| **HA Assist warm (voice steady state)** | **251 ms** | **2.21 s** | **82 tok/s** |
| HA Assist multi-turn | 415 ms | 3.07 s | 67 tok/s |
| tool call round trip | — | 2.67 s | 60 tok/s |
| HA Assist cold (13.7k-token prompt) | 15.1 s | — | prefill 1007 tok/s |

- Without DSpark the same model decodes at 24–30 tok/s; the drafter is worth **2.2–3.0x**.
- **Turbo voice mode** (`REASONING_BUDGET=0`): short turn **0.39 s**, warm HA turn **0.85 s**, tool round trip **0.51 s** — at a small quality cost (multi-step reasoning degrades; 8/10 vs 10/10 on the bundled battery).
- Memory: the LLM service holds **~2.6 GB** of the 8 GB SoC RAM (measured via stop/start deltas — most of it is GPU-driver memory that `docker stats` undercounts), leaving ~3 GB for the OS and other services.
- Boot → ready with self-test: ~50 s.

Full methodology and the decision log behind every default live in `notes/` in the working tree (not published to git).

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `PORT` | `8080` | listen port |
| `CTX` | `32768` | context tokens (KV cache is cheap for LFM2.5) |
| `REASONING_BUDGET` | `128` | thinking budget; `0` = turbo voice, `off` = unlimited |
| `SPEC` | `auto` | DSpark speculative decoding (`auto`/`on`/`off`) |
| `KV_TYPE` | `q8_0` | KV cache type (`f16` = +0.25 GB) |
| `N_BATCH` / `UBATCH` | `1024` / `256` | prefill batching (bounds GPU memory) |
| `CACHE_REUSE` | `256` | prompt-cache KV-shift chunk |
| `PRIO` | `2` | process priority (-1..3) |
| `MODEL_PATH` / `DRAFT_PATH` | baked | override GGUF paths |
| `SELF_TEST` | `on` | e2e self-test + speed measurement on boot |
| `API_KEY` | unset | optional bearer key |

## Home Assistant notes

- Works with `hass_local_openai_llm` streaming + tool calling (`--jinja`, LFM2.5 chat template). The integration strips ` think ` tags; the server also extracts them to `reasoning_content`.
- Give Assist at least 10k context; 32k is provisioned. For very long conversations, use the integration's history-trim option.
- For maximum snappiness on voice-only households, set `REASONING_BUDGET=0`.

## Benchmarks

> Ops note: benchmark runs add memory pressure and page-cache churn — avoid running the full matrix while you are actively using other latency-sensitive services on the box.

```bash
./run.sh bench                          # dev container, models mounted at /models
bash /app/bench/matrix.sh quants        # quant x spec-decoding matrix
bash /app/bench/matrix.sh draft         # drafter quant x n-max sweep
bash /app/bench/matrix.sh flags         # cache/fa/prio/reasoning sweeps
python3 /app/bench/bench_openai.py      # real-world OpenAI suite
python3 /app/bench/quality_check.py     # quality battery
```

Benchmark sweeps beyond the default pair need the extra GGUFs, e.g. `./scripts/fetch-models.sh LFM2.5-2.6B-Q4_0.gguf LFM2.5-2.6B-DSpark-F16.gguf`. Results land in `MODELS_DIR`'s sibling `results/` directory (see `run.sh`).

## Hardware scope

This image is built for exactly one configuration: **Jetson Orin Nano Super (SM 8.7) on JetPack 6.x / CUDA 12.6**. It will not run on other architectures without rebuilding (`CUDA_ARCH` build arg exists but is untested elsewhere) and no other platform is supported or optimized.
