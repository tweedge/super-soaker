# super-soaker

[![License](https://img.shields.io/github/license/tweedge/super-soaker)](https://github.com/tweedge/super-soaker)
[![Written By](https://img.shields.io/badge/written%20by-some%20nerd-red.svg)](https://chris.partridge.tech)
[![Follow author on Bluesky](https://img.shields.io/bluesky/followers/tweedge.net)](https://bsky.app/profile/tweedge.net)
[![Follow author on Mastodon](https://img.shields.io/mastodon/follow/108210086817505115?domain=https%3A%2F%2Fcybersecurity.theater)](https://cybersecurity.theater/@tweedge)

A complete voice assistant stack for the **NVIDIA Jetson Orin Nano Super (8 GB)**: Whisper tiny/en for speech-to-text, an [LFM2.5](https://huggingface.co/collections/LiquidAI/lfm25) model with [DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark) speculative decoding for the assistant, and Kokoro-82M for text-to-speech. All three are built locally at pinned versions, run natively under systemd, and stay resident in RAM with none of it swapped. On this box, a page in swap is latency you can hear.

It is built for one caller - Home Assistant's Assist pipeline through [hass_local_openai_llm](https://github.com/skye-harris/hass_local_openai_llm) or a Home Assistant Voice device - where the gap between "user stops talking" and "assistant starts talking" is the whole experience. This README covers installing and running it. The measurements, model shootout, RAM and power numbers, and rejected experiments live in [PERFORMANCE.md](PERFORMANCE.md).

| # | service | function | protocol | port |
|---|---|---|---|---|
| 1 | `1-stt-wyoming-whisper-tiny` | speech → text (Whisper tiny/en, CUDA) | Wyoming | `tcp/10300` |
| 2 | `2-llm-llama.cpp-super-soaker-managed` | text → text (LFM2.5 profile + DSpark draft) | OpenAI HTTP | `tcp/8080` |
| 3 | `3-tts-wyoming-kokoro-82m` | text → speech (Kokoro-82M fp16, CUDA, streaming) | Wyoming | `tcp/10210` |

The directory names encode pipeline order and function; the systemd units are `super-soaker-{stt,llm,tts}.service`.

## What you need

- An NVIDIA Jetson Orin Nano Super with 8 GB of RAM, running JetPack 6.x (L4T r36.x, CUDA 12.6). This targets SM 8.7 and the L4T driver stack; a CUDA-13-linked build will not load on it.
- About 20 GB of free disk during the build.
- Network access for the one-time build and model download. The finished stack does not need it.
- Time. The first build takes one to two hours because it compiles llama.cpp, CTranslate2, and ONNX Runtime from source for the Jetson.

There is no Docker here, and that is deliberate. On a board where the OS, GPU, and models share one 8 GB pool, the daemon and per-container library copies cost hundreds of megabytes before a model loads. Native processes share one toolchain and one copy of every CUDA library. The full argument is in [PERFORMANCE.md](PERFORMANCE.md#why-native-and-not-docker).

## Install

```bash
git clone https://github.com/tweedge/super-soaker.git
cd super-soaker
sudo ./bootstrap.sh
```

That is the whole install. `bootstrap.sh`:

1. installs the build toolchain, CUDA 12.6, cuDNN, and OpenBLAS from apt;
2. builds llama.cpp (pinned `b11552` / `23b0202a1`), CTranslate2 (`4.5.0`, CUDA), and ONNX Runtime (`v1.22.0`, CUDA) into `/opt/super-soaker`;
3. creates two pinned Python virtualenvs (`venv-stt`, `venv-tts`);
4. downloads the model pair with SHA-256 verification into `/var/lib/super-soaker`;
5. installs and enables three systemd services with `Restart=always` and `MemorySwapMax=0`, so service pages can never be swapped;
6. sets the `MAXN_SUPER` power mode. Clocks boost under load and scale down at idle; nothing is pinned;
7. waits for each service's built-in self-test to pass, then reboots.

Re-running it is safe and is the normal way to change profiles. Flags are `--model {fast|balanced|thorough}`, `--reasoning-budget {N|off|on|auto}`, `--skip-build`, `--no-reboot`, and `--force-jobs`. Tuning lives in `/etc/super-soaker/*.env`, which is written once and preserved across re-runs.

### Build parallelism

The build is RAM-bound, not CPU-bound. ONNX Runtime's nvcc flash-attention kernels peak near 3.5 GB RSS per compile thread, and this board shares that RAM with the GPU. High parallelism has wedged this class of machine into a swap spiral that needed a power reset.

`JOBS=1` is the default and safe everywhere, at the cost of a slow build. `JOBS=2` is the tested maximum on the 8 GB Orin Nano Super. Anything higher is refused unless `--force-jobs` is passed, and only makes sense on a 16 GB device such as the Orin NX:

```bash
sudo JOBS=4 ./bootstrap.sh --force-jobs
```

On an 8 GB box, leave it alone and let it grind.

### Optional: remove the desktop

The stack does not need a GUI. If this machine is headless, `clean.sh` removes the GNOME desktop and idle services for several hundred megabytes of extra RAM. It is strictly optional, never called by `bootstrap.sh`, and reversible:

```bash
sudo ./clean.sh            # dry run: lists what would go
sudo ./clean.sh --apply    # actually remove
```

## Choose a model

All three profiles run the same frozen 30-scenario Home Assistant battery, so the comparisons are honest. The short version:

| profile | model | decode (battery mean) | average reply | tool / quality accuracy |
|---|---|---|---|---|
| `fast` | LFM2.5-1.2B-Instruct Q8_0 | 84.4 tok/s | 0.49 s | 85% / 70% |
| `balanced` (default) | LFM2.5-2.6B QAD-Q4_0 | 63.2 tok/s | 1.38 s | 90% / 80% |
| `thorough` | LFM2.5-8B-A1B UD-IQ2_M | 41.6 tok/s | 3.60 s | 95–100% / 90% |

`balanced` is the default because an assistant has to be disciplined about tool calls and facts without feeling slow, and after optimization it is not slow: 1.38 s average replies with 90% tool and 80% quality accuracy.

Pick `fast` if latency is all that matters. It is the quickest on every axis, with tool accuracy close to `balanced`, but it gives up the reasoning pass that lets the larger models check arithmetic and letter-counting.

Pick `thorough` if correctness matters more than reply latency. The 8B-A1B is a reasoning-only MoE: it thinks before answering, so replies average about 3.6 s, but it is the strongest profile on tool use and factual quality.

Only one LLM profile can be resident at a time. Two dense LLMs together measured 7.32 of 7.62 GB with 105 MB free, which is not a safe margin. Run one.

Switching profiles rewrites `LLM_MODEL` in `/etc/super-soaker/llm.env` and fetches the matching SHA-256-pinned GGUF pair:

```bash
sudo ./bootstrap.sh --model fast
sudo ./bootstrap.sh --model balanced
sudo ./bootstrap.sh --model thorough
```

The reasoning budget is set the same way (`--reasoning-budget 256`, or `off`/`on`/`auto`). Defaults are per model: 0 for `fast`, 256 for `balanced` and `thorough`. An explicit flag always wins, and unpassed values are preserved across re-runs. If the budget force-closes thinking, a required wrap-up (`REASONING_BUDGET_MESSAGE`, which by default asks for a brief spoken answer) is injected before the end-of-thinking tag, so a cut lands on a clean hand-off instead of a ragged mid-thought dump. `bootstrap.sh` always writes it into `llm.env` and heals env files that lack it.

All profiles use a 16k context ceiling. HA prompts run about 2.5k tokens and the longest observed tool loop reached about 3.9k, so 16k keeps more than 4× headroom while saving roughly 350 MB versus 32k. Individual overrides such as `CTX` or `MODEL_PATH` still win over the profile defaults; edit `/etc/super-soaker/llm.env` and restart the service if you prefer to tune by hand.

The full profile results, the optimization campaign, the 8B quant hunt, and the RAM cost of each profile are in [PERFORMANCE.md](PERFORMANCE.md).

## Connect Home Assistant

- **Local OpenAI LLM** → `http://<jetson-ip>:8080/v1`, server type `llama.cpp`, API key blank. Use the model alias from `curl http://<jetson-ip>:8080/v1/models`; it matches the installed GGUF, for example `LFM2.5-2.6B-QAD-Q4_0`.
- **Wyoming STT** → `tcp://<jetson-ip>:10300`.
- **Wyoming TTS** → `tcp://<jetson-ip>:10210`. The default voice is `af_heart`; all 53 Kokoro voices are available, with prefixes `a` (US English), `b` (UK English), `j` (Japanese), `z` (Chinese), `e` (Spanish), `f` (French), `h` (Hindi), `i` (Italian), and `p` (Brazilian Portuguese).

## How it works

All three services start at boot, restart on failure, and run as the unprivileged `super-soaker` user in the `video` and `render` groups. That second group matters: without it, CUDA silently reports zero devices and everything downstream fails with unhelpful errors.

**STT** is [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper) 3.3.1 on CUDA via a locally built CTranslate2, because the PyPI aarch64 wheel is CPU-only. It runs Whisper `tiny` with language fixed to `en`, int8 compute, and beam 5. On start it warms CUDA with two transcription round-trips, then runs its self-test.

**LLM** is `llama-server` at the pinned commit, built with `GGML_CUDA_FA_ALL_QUANTS`, CUDA graphs, and `GGML_CUDA_NO_VMM` (the Jetson iGPU VMM workaround). The model is not in the service name: `LLM_MODEL` in `/etc/super-soaker/llm.env` selects the profile, its target GGUF, and its DSpark-Q8_0 drafter. The server runs a single slot because LFM2's short-conv state corrupts under co-batching, and this stack has one caller anyway. It uses q8_0 KV cache, `--cache-reuse 256` for the Home Assistant prefix, and a reasoning budget that keeps thinking bounded.

**TTS** is [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) 0.5.0 on a locally built ONNX Runtime 1.22 with the CUDA execution provider. The Wyoming wrapper is vendored into the service directory, based on [nullable-eth/kokoro-wyoming](https://github.com/nullable-eth/kokoro-wyoming). It streams audio at the first sentence boundary, strips markdown, thinking blocks, tool JSON, and stray angle brackets before phonemization, and pre-sizes its CUDA arena with a full 503-phoneme batch at boot so no request can grow it into an OOM. Synthesis is serialized behind a lock; concurrent requests queue instead of fighting over the arena cap.

## Day-2 operations

```bash
journalctl -u super-soaker-llm -f          # logs
systemctl restart super-soaker-tts         # bounce one service
free -h; swapon --show                     # RAM and swap picture
grep VmSwap /proc/$(pgrep -f serve.sh)/status   # should be zero for every service
```

On disk:

```
/opt/super-soaker/          builds, virtualenvs, service code, pinned wheels (cache/)
/var/lib/super-soaker/      models and caches, owned by super-soaker
/etc/super-soaker/          per-service env files (llm.env, stt.env, tts.env)
/etc/systemd/system/        super-soaker-{llm,stt,tts}.service
```

The env files are the supported tuning surface. They are written once by `bootstrap.sh` and preserved on re-runs, so local edits survive. `LLM_MODEL`, `REASONING_BUDGET`, `CTX`, `MODEL_PATH`, and `DRAFT_PATH` are the ones most people touch; `stt.env` and `tts.env` expose the model, beam size, voice, speed, and arena cap.

Every service unit sets `MemorySwapMax=0`, and `bootstrap.sh` sets `vm.swappiness=10`. Nothing should be swapped by construction. If `VmSwap` is not zero, something is wrong; check `free -h` and whether another process is eating the budget.

## Self-tests

Each service ships a stdlib-only, end-to-end self-test that runs at every start (`SELF_TEST=on`) and can be run by hand:

```bash
sudo -u super-soaker /opt/super-soaker/venv-stt/bin/python3 \
    /opt/super-soaker/services/1-stt-wyoming-whisper-tiny/selftest.py

sudo -u super-soaker python3 \
    /opt/super-soaker/services/2-llm-llama.cpp-super-soaker-managed/selftest.py

sudo -u super-soaker /opt/super-soaker/venv-tts/bin/python3 \
    /opt/super-soaker/services/3-tts-wyoming-kokoro-82m/selftest.py
```

The STT test speaks "hello world" with espeak-ng and asserts the transcript round-trips. The LLM test checks the model list, warmup completions, streaming speed, a tool-calling round trip like the one `hass_local_openai_llm` uses, and prompt-cache behavior. The TTS test covers plain and streaming synthesis plus a worst-case 503-phoneme long-form batch, asserting 24 kHz audio, plausible duration, non-silence RMS, and a first streaming chunk under 3 s (10 s for the long-form gate).

## Troubleshooting

- **The build thrashes or the machine wedges.** You raised `JOBS`. Go back to 1, or 2 at most on this board.
- **The LLM service exits at boot.** Check `journalctl -u super-soaker-llm` for an unknown `LLM_MODEL` or a missing model file. Legacy names in `llm.env` are still accepted, and re-running `bootstrap.sh` heals them.
- **CUDA reports zero devices.** The service user is missing the `video` or `render` group. `usermod -aG video,render super-soaker` and restart the services.
- **A service restart-loops.** Its self-test is failing. Run the matching command above by hand; it prints the actual failure instead of the systemd wrapper.
- **The first reply after a restart is very slow.** That is a cold prompt cache. The static HA Assist prefix prefills at roughly 15 s cold versus 0.3 s warm; it is a one-time cost per context change.
- **The TTS dies mid-answer.** Check whether the arena cap was lowered or a second synthesis was allowed to run concurrently. The default `KOKORO_GPU_MEM_MB=1024` is sized for a full 503-phoneme batch, and the service serializes synthesis for a reason.
- **Home Assistant gets no tool calls.** Confirm the model alias from `/v1/models` and run the LLM self-test's tool-calling probe. The dense models occasionally map "turn on the lights" to `HassLightSet` instead of `HassTurnOn`; a nudge in HA's own tool descriptions is the fix, not a server change.
- **Home Assistant hears nothing.** Confirm its Wyoming STT endpoint points at `tcp://<jetson-ip>:10300`, then run the STT self-test. If transcripts work but replies do not, check the LLM endpoint and the model alias.

## Development

The dependency builds are in `deps/`: `build-llamacpp.sh`, `build-ctranslate2.sh`, `build-onnxruntime.sh`, and `limit-jobs.sh`, the shared RAM guard. Every dependency is a pinned tag or commit, and wheels land in `/opt/super-soaker/cache` for offline rebuilds. The local llama.cpp patch in `deps/patches/mmvq-orin-crossover.patch` extends the Orin MMVQ→MMQ crossover to IQ and simple quants, worth 4–10% on speculative verify batches.

The Home Assistant battery lives in `bench/ha_benchmark.py` and `bench/ha_battery.json`; archived results are in `bench/results/`. To run it against a live service:

```bash
python3 bench/ha_benchmark.py --base-url http://127.0.0.1:8080 \
    --model <alias from /v1/models> --label my-run
```

`bench/bench_openai.py`, `bench/quality_check.py`, and `bench/matrix.sh` are the older interactive sweeps. `scripts/fetch-models.sh` pulls extra GGUFs for quant matrices, and `scripts/quantize-8b-variants.sh` holds the custom 8B recipes.

For the measured history and the reasoning behind the defaults, read [PERFORMANCE.md](PERFORMANCE.md) and [AGENTS.md](AGENTS.md). The latter is the institutional knowledge file: methodology, quirks, and the things that were tried and rejected so nobody has to rediscover them.

## License

MIT. This project is not affiliated with NVIDIA, LiquidAI, or the upstream projects it builds on. Model weights and those upstream projects keep their own licenses.
