# super-soaker

Ultra low-latency voice assistant stack for the **NVIDIA Jetson Orin Nano Super** (8 GB): a [LiquidAI LFM2.5](https://huggingface.co/collections/LiquidAI/lfm25) LLM (1.2B, 2.6B, or [8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) MoE — pick at install) with [DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark) speculative decoding, tiny-en Whisper STT, and Kokoro-82M TTS — all three **built locally at pinned versions, running natively (no Docker), resident in RAM with none of it swapped**.

Designed for a single caller (Home Assistant + [hass_local_openai_llm](https://github.com/skye-harris/hass_local_openai_llm) / Home Assistant Voice) where every millisecond of response latency is felt. The three services run in pipeline order and are named so it shows at a glance:

| # | service | function | protocol | port |
|---|---------|----------|----------|------|
| 1 | `1-stt-wyoming-whisper-tiny` | speech → text (Whisper tiny/en, CUDA) | Wyoming | `tcp/10300` |
| 2 | `2-llm-llama.cpp-super-soaker-managed` | text → text (LFM2.5 profile + DSpark draft) | OpenAI HTTP | `tcp/8080` |
| 3 | `3-tts-wyoming-kokoro-82m` | text → speech (Kokoro-82M fp16, CUDA, streaming) | Wyoming | `tcp/10210` |

## Quick start (new Jetson)

On a freshly flashed Jetson Orin Nano Super with JetPack 6.x (L4T r36.x, CUDA 12.6) and network access:

```bash
git clone https://github.com/tweedge/super-soaker.git
cd super-soaker
sudo ./bootstrap.sh
```

That is the whole install. `bootstrap.sh`:

1. installs build toolchain + CUDA 12.6 + cuDNN + OpenBLAS (apt),
2. builds **llama.cpp** (pinned `b11552` / `23b0202a1`), **CTranslate2** (pinned `4.5.0`, CUDA), and **ONNX Runtime** (pinned `v1.22.0`, CUDA) into `/opt/super-soaker`,
3. creates two pinned Python venvs (`venv-stt`, `venv-tts`),
4. downloads models with SHA-256 verification into `/var/lib/super-soaker`,
5. installs three systemd services (enabled at boot, `Restart=always`, `MemorySwapMax=0` so service pages can never be swapped),
6. locks power to `MAXN_SUPER` + `jetson_clocks`,
7. waits for each service's built-in self-test to pass, then reboots.

Re-running is safe (idempotent). Flags: `--skip-build` (reuse built wheels), `--no-reboot`, `--force-jobs` (see below), `--model {lfm2.5-1.2b-chat|lfm2.5-2.6b-reasoning|lfm2.5-8b-a1b-reasoning}` (default `lfm2.5-1.2b-chat`; see "Model choice"). Service tuning lives in `/etc/super-soaker/*.env` (written once, preserved across re-runs).

### Build threads: 1 by default, 2 maximum

The compile jobs are RAM-bound, not CPU-bound. ONNX Runtime's nvcc flash-attention kernels peak near **3.5 GB RSS per thread**, and the 8 GB Orin Nano Super shares that RAM with the GPU and the voice stack — running the build with high parallelism has wedged this machine (load 30, RAM and swap full, power reset needed). The build scripts enforce:

| `JOBS` | behavior |
|---|---|
| `1` (default) | safe everywhere — **but expect the build to take a WHILE** (1–2 h) |
| `2` | tested maximum on the 8 GB Orin Nano Super |
| `>2` | **refused with an error** unless `--force` / `--force-jobs` is passed |

The `--force` escape hatch (build scripts) and `--force-jobs` (bootstrap.sh) exist for 16 GB devices such as the **Jetson Orin NX**, where the operator has the headroom to accept the risk: `sudo JOBS=4 ./bootstrap.sh --force-jobs`. On an 8 GB box, leave it alone and let it grind.

### Optional: strip the desktop

The stack does not need a GUI. If this machine is headless, `clean.sh` removes the GNOME desktop and idle services for several hundred MB of extra RAM. It is **strictly optional** and never called by `bootstrap.sh` — machines that still need their monitor output are unaffected:

```bash
sudo ./clean.sh            # dry run: lists what would go
sudo ./clean.sh --apply    # actually remove (reversible)
```

## End-to-end latency: when does speech start?

The pipeline is: **user stops speaking → STT finalizes → LLM generates → TTS streams**. Two moments matter perceptually:

- **reply-speech starts** — the first audio sample of the answer is played;
- **reply-speech completes** — the last sample of the answer finishes.

Component costs (measured on the box, services warm, MAXN_SUPER + jetson_clocks):

| stage | cost | note |
|---|---|---|
| STT finalize (3 s utterance) | ~0.3–1.0 s | whisper tiny/en int8, CUDA; scales with utterance length |
| LLM TTFT (warm) | 0.4–0.5 s | prompt-cache hit on the HA Assist prefix |
| LLM decode | 31–34 tok/s | ~17–24 tok/s accepted with DSpark draft |
| first sentence → TTS | ~8–12 tokens | TTS starts at the first sentence boundary |
| TTS time-to-first-chunk | 0.48–0.68 s | Kokoro fp16, CUDA, Wyoming streaming |
| TTS sustained | 5.2–7.1 s audio / s wall | ~5× faster than playback — never the bottleneck |

### Worked example — typical short answer

"One or two sentences" (25 tokens of text, ~6 s of reply audio):

| milestone | time after user stops speaking |
|---|---|
| STT transcript ready | ~0.5 s |
| LLM first token (warm) | ~1.0 s |
| first sentence complete (TTS can start) | ~1.3–1.5 s |
| **reply-speech starts** | **~1.8–2.2 s** |
| **reply-speech completes** | **~8 s** (speech-start + 6 s of playback) |

Speech-completion is dominated by *playback duration of the answer*, not synthesis: TTS generates ~5× faster than real time and streams sentence-by-sentence, so audio production never lags the speaker. Longer answers extend speech-complete proportionally but do not delay speech-start.

### Cold vs warm

| path | latency |
|---|---|
| warm (prompt cache hit, short reply) | speech starts ~2 s; completes ~answer length later |
| cold (13.7k-token prompt, cache miss) | LLM prefill ~15.1 s (1007 tok/s) before first token; speech starts ~17 s |
| service restart (model reload) | +2–4 s once, self-test then ready |

The prompt cache (`--cache-reuse 256`) is what keeps the warm path warm: the static Home Assistant Assist prefix is reused across turns, so only the new utterance is prefilled.

## Model choice: three serving profiles, one battery

Three serving profiles share one battery (`bench/ha_benchmark.py` + `bench/ha_battery.json`: 20 tool scenarios from the HA Assist API surface + 10 factual/brevity scenarios, run identically against each config; raw results in `bench/results/`):

| profile | model | warm TTFT | avg reply | decode | tool acc | quality | notes |
|---|---|---|---|---|---|---|---|
| **lfm2.5-1.2b-chat** (default) | LFM2.5-1.2B-Instruct QAD-Q4_0 | **0.07 s** | **0.45 s** | **55.9 w/s** | 85% | 70% | snappiest; weak at multi-digit arithmetic and letter-counting |
| **lfm2.5-2.6b-reasoning** | LFM2.5-2.6B QAD-Q4_0 | 0.32 s | 2.28 s | 12.4 w/s | 90% | 70% | facts always right; verbose for voice; never over-triggers |
| **lfm2.5-8b-a1b-reasoning** | LFM2.5-8B-A1B UD-IQ2_M | 0.50 s | 2.64 s | 10.5 w/s | **100%** | **90%** | MoE (8.3B total / 1.5B active); best accuracy; optimized spec stack (below) |

Measured on the box, identical battery/prompt; tool scenarios include no-tool negatives (small talk, nonsense, goodbye) that punish models which hallucinate device actions. All three pair their target GGUF with a DSpark-Q8_0 speculative drafter (~29–30% of drafted tokens accepted). The 1.2B and 2.6B share one quirk: "turn on the lights" sometimes maps to HassLightSet instead of HassTurnOn — worth a prompt nudge in HA's own tool descriptions if it shows up in practice. The 8B is the only profile that never gets a tool call wrong on this battery.

**Why lfm2.5-1.2b-chat is the default:** benchmark-measured ~2x lower end-to-end latency than the 2.6B (0.45 s vs 2.28 s average reply) and ~10x lower than the 8B, with tool accuracy close to the 2.6B (85% vs 90%). What it gives up: self-corrected arithmetic and letter-counting (no reasoning pass to check itself). A fourth profile — the 2.6B with reasoning disabled — was benchmarked and rejected: it over-triggers tools on non-commands, keeps the 2.6B memory cost, and is still 2x slower than the 1.2B.

**Pick lfm2.5-2.6b-reasoning if** the assistant must be disciplined about tool calls and facts (the agentic profile), and ~2 s replies are acceptable — the tradeoff this project historically ran with.

**Pick lfm2.5-8b-a1b-reasoning if** tool-call and answer correctness matter more than reply latency. The 8B-A1B is a reasoning-only MoE: it spends its 1.5B active parameters on a chain-of-thought pass before answering, so replies take ~4.7 s on average (vs 2.3 s for the 2.6B) but it is the only profile here that scores 100% on tool accuracy and 90% on factual quality. It also answers some scenarios the dense models simply cannot (reliable letter-counting, multi-digit arithmetic self-checks).

**Only one LLM fits at a time:** measured 7.32/7.62 GB with both dense LLMs resident (105 MB free) — below any safe margin. The 8B-A1B is worse still: even at IQ2-class quants it needs ~4.3 GB of RAM by itself (see below). Run one.

### RAM cost per profile (measured, LLM + DSpark draft + KV + nvgpu pool)

| profile | resident LLM | peak (LLM+TTS concurrent) | fits with STT+TTS? |
|---|---|---|---|
| lfm2.5-1.2b-chat | ~2.6 GB | ~2.6 GB | yes, ~1.8 GB headroom |
| lfm2.5-2.6b-reasoning | ~3.0 GB | ~3.2 GB | yes, ~1.4 GB headroom |
| lfm2.5-8b-a1b-reasoning (UD-IQ2_M) | ~4.3 GB | ~4.8 GB (7.04/7.62 GB board) | yes, but only ~580 MB headroom |

The 8B-A1B leaves little slack: every quant larger than UD-IQ2_M was measured and rejected (see "8B-A1B quant hunt" in AGENTS.md). Q4_0 (4.8 GB of weights) does not fit at all — it starts, then the decode collapses from 33 t/s to 2.4 t/s as the box thrashes. UD-IQ3_XXS completes the battery with the best quality score but pushes 321 MB into zram and peaks at 7.49/7.62 GB — a violation of the stack's no-swap rule and too close to the edge for a concurrent TTS peak. UD-Q2_K_XL fits (3 MB swap) but is dominated by UD-IQ2_M on every axis: slower decode, lower accuracy, more RAM. UD-IQ2_M is the only 8B quant that runs the whole battery with zero swap and real headroom.

### Switching profiles

```bash
sudo ./bootstrap.sh --model lfm2.5-2.6b-reasoning
# or: sudo ./bootstrap.sh --model lfm2.5-8b-a1b-reasoning
# or: sudo ./bootstrap.sh --model lfm2.5-1.2b-chat (default)
```

On an installed node this rewrites `/etc/super-soaker/llm.env` (`LLM_MODEL=...`) and fetches the matching GGUF pair (sha256-pinned): LFM2.5-1.2B-Instruct QAD-Q4_0 + DSpark-Q8_0, LFM2.5-2.6B QAD-Q4_0 + DSpark-Q8_0, or Unsloth LFM2.5-8B-A1B UD-IQ2_M + DSpark-Q8_0. You can also edit `LLM_MODEL=` in `/etc/super-soaker/llm.env` and `systemctl restart super-soaker-llm`. Individual overrides (`CTX`, `REASONING_BUDGET`, `MODEL_PATH`, ...) still win over the profile defaults.

All profiles run with the same reasoning and context settings where their model supports it: `CTX=32768`, `REASONING_BUDGET=128` for the two reasoning profiles (the 1.2B chat profile runs `CTX=8192`, no reasoning). The 8B-A1B is a reasoning-only model — the budget bounds its chain-of-thought so voice replies stay predictable.

### 8B-A1B performance optimization (measured, see notes/04)

The 8B profile ships an optimized speculative-decoding stack: `--spec-type draft-dspark,ngram-mod --spec-draft-n-max 5 --spec-draft-p-min 0.4`, plus a llama.cpp kernel patch (`deps/patches/mmvq-orin-crossover.patch`) that extends the Orin MMVQ→MMQ crossover (upstream PR #28285 covered K-quants only) to IQ/simple quants. Measured effects on the frozen battery (accuracy unchanged: tool 100%, quality 90%):

| metric | before (n_max 9, plain dspark) | after (optimized stack) | gain |
|---|---|---|---|
| latency probe (repeated utterance) | 4.67 s | 2.64 s | 1.77x |
| unique tool scenarios (20) | 4.29 s | 3.88 s | 1.10x |
| free-form varied decode | 35 tok/s | 38 tok/s | 1.10x |
| templated voice traffic | 35 tok/s | 119–175 tok/s | 3.4–5x |

The spread is the honest story: batch-1 GEMVs on this GPU stream at a fixed ~24% of LPDDR5 bandwidth, so non-spec decode is capped near 35–48 tok/s (1.3–1.6x) by bytes-per-token. Speculative decoding escapes that cap by batching verify steps into fat GEMMs — but on a 4-of-32 MoE the expert-union traffic saturates with draft length, so the gain needs high draft acceptance. Novel free-form speech accepts ~2–3 drafted tokens/step (the drafter was distilled against an F16 target — llama.cpp issue #25618); templated HA traffic (fixed tool JSON, repeated phrasing) accepts 5+ and reaches the 119–175 tok/s ceiling. In practice a voice assistant lives between the two columns. **2–3x on repetitive traffic, ~1.1x on novel speech; 4–5x only when content is highly predictable.** Deep dive, rejected experiments (custom quants, TQ2_0, F16 drafter, micro knobs) and the constraint analysis: `notes/04-decode-optimization-research.md`.

## Design tradeoffs: quality vs intelligence vs speed

The guiding rule: **maximum performance inside a hard 8 GB RAM envelope, while tolerating a slightly slower reply when quality would otherwise suffer.** Every component below is the highest-quality option that still fits and keeps the conversation fluid.

### LLM — LFM2.5 family + DSpark-Q8_0 draft (profile-selected)

| decision | chosen | rejected | why |
|---|---|---|---|
| model size | 1.2B / 2.6B / 8B-A1B (profile) | 7B dense, 20B+ | the three profiles are the LFM2.5 sizes that fit alongside STT+TTS; dense 8B-class weights do not fit at usable quants (see quant hunt below) |
| target quant | QAD-Q4_0 (dense), UD-IQ2_M (8B-A1B) | Q4_K_M, Q5_K_M for the 8B | QAD (quantization-aware distill) is LiquidAI's own quality-preserving recipe; for the 8B-A1B MoE only IQ2-class quants fit — Unsloth's UD-IQ2_M was the measured winner of a four-quant shootout |
| draft model | DSpark-Q8_0 (on) | no draft, or Q4 draft | spec decoding raises accepted decode to ~1.5–2.2× with *identical output distribution* (quality-neutral speed); Q8_0 draft is more accurate than Q4, and its RAM cost is affordable (0.35–0.36 GB) |
| KV cache | q8_0 @ 32k ctx | f16 | halves KV RAM with no observed quality hit; keeps the long HA prompt + history in context instead of truncating it (intelligence win) |
| reasoning budget | 128 tokens | 0 / unlimited | bounded scratch thinking: answers keep some reasoning quality, latency stays predictable; unlimited thinking would blow the reply budget |
| runtime | llama.cpp, single slot | vLLM, batching | vLLM's pool + scheduler overhead measured 2.8–3.3 GB earlier; single-caller makes batching pure loss. `FA_ALL_QUANTS` + CUDA graphs recover the speed instead |

Net effect: a slightly slower per-token rate than a bigger model would need to be *fast at all costs* — accepted only where it buys answer quality. The speculative-decoding stack is the opposite trade: free speed that does not touch quality, so it is on by default.

### STT — Whisper tiny/en, int8, beam 5

| decision | chosen | rejected | why |
|---|---|---|---|
| model | tiny/en | small, medium | small/medium add RAM and latency for gains that matter little on clean home-assistant audio; tiny is highly accurate on clear near-field speech |
| language | fixed `en` | auto-detect | skipping language detection saves a model pass and removes misdetection flaps |
| compute type | int8 | float16 | int8 on Orin is ~2× faster with negligible WER change for tiny |
| beam size | 5 | greedy (1) | beam 5 measurably improves name/command recognition for pennies in latency — quality bought, not sold |
| VAD/streaming | wyoming chunked | whole-utterance only | chunks start decoding as audio arrives |

### TTS — Kokoro-82M fp16 (ONNX Runtime, CUDA, streaming)

| decision | chosen | rejected | why |
|---|---|---|---|
| engine | Kokoro-82M | Piper, espeak | Kokoro's prosody/naturalness is a clear tier above Piper at comparable cost |
| precision | fp16 | int8, fp32 | int8 is *slower* on this GPU (quantized ops fall back to CPU) and dulls prosody; fp32 wastes RAM for no audible gain. fp16 measured clean and fast |
| runtime | ONNX Runtime CUDA | PyTorch, CPU | PyTorch path measured +1.5–1.9 GB and OOM'd the box; CPU synthesis is slower than real time (rtf 1.4–1.6) — unusable |
| delivery | Wyoming streaming | buffer-everything | speech starts at the first sentence boundary while the LLM is still generating; buffered TTS would add the full synthesis wait to every reply |
| arena cap | 1024 MB | unbounded | caps GPU/RAM arena so the LLM's pool cannot be squeezed out |

### System — native processes, no Docker

| decision | chosen | rejected | why |
|---|---|---|---|
| packaging | systemd + pinned builds | Docker images | Docker daemon + per-container library copies cost several hundred MB and pushed the box into a swap spiral; native processes share one toolchain and one copy of every library |
| swap policy | `MemorySwapMax=0` | default swapping | with 8 GB, any swapped service page is latency jitter; nothing may swap, and the budget guarantees nothing has to |
| power | MAXN_SUPER + jetson_clocks | ondemand clocks | first-token latency depends on sustained clocks; clock ramping is pure loss here |

## RAM budget (measured on the live box, L4T 36.4.4)

All figures are `MemTotal − MemFree − Cached` deltas at 100 ms sampling, which captures GPU-driver (`nvgpu`) kernel memory that `docker stats`/RSS hide. The GPU shares system RAM on Jetson, so "GPU memory" here is RAM.

| component | resident | peak (LLM+TTS concurrent) |
|---|---|---|
| LLM lfm2.5-1.2b-chat (llama.cpp + nvgpu pool) | ~2.6 GB | flat under load |
| LLM lfm2.5-2.6b-reasoning | ~3.0 GB | flat under load |
| LLM lfm2.5-8b-a1b-reasoning (UD-IQ2_M + DSpark) | ~4.3 GB | ~4.8 GB |
| STT (whisper tiny/en int8) | ~0.7 GB | flat |
| TTS (kokoro fp16, ONNX CUDA) | ~1.1 GB | ~1.5 GB |
| OS (headless) | ~0.8 GB | — |
| **total with default 1.2B profile** | **~5.2 GB / 7.4 GB** | **~5.6 GB / 7.4 GB** |
| **total with 8B-A1B profile** | **~6.9 GB / 7.4 GB** | **~7.0 GB / 7.4 GB** |

~1.8 GB headroom at peak on the default profile, ~0.5 GB on the 8B-A1B profile (measured 7.04/7.62 GB peak with zero swap — see the quant hunt notes); every service unit pins `MemorySwapMax=0` and `vm.swappiness=10`, so the answer to "is anything swapped?" is *no* by construction. Verify with `grep VmSwap /proc/$(pgrep -f serve.sh)/status` or `free -h`.

Earlier docker-based deployments of this stack did **not** fit: the daemon, per-container library duplication, and the LLM's GPU pool together pushed the box into a zram swap spiral (TTS OOM-killed, machine wedged). Running native is the make-or-break difference.

## Power draw (measured, `tegrastats` VDD_IN)

Whole-board input power at `MAXN_SUPER` + `jetson_clocks`, all three services resident (8B-A1B profile, the heaviest config):

| state | VDD_IN | note |
|---|---|---|
| boot idle (services stopped) | ~6.0 W | clocks locked, no work |
| stack resident, no traffic | ~6.9 W | three models in RAM, GPU idle |
| LLM decode + TTS synthesis | ~16.5–17 W | sustained, well under the 25 W MAXN_SUPER envelope |

The light profiles (1.2B/2.6B) idle a few hundred mW lower (smaller weights to keep refreshed, less memory pressure) and peak a bit under load. In practice: an always-on voice assistant costs about **7 W idle**, with brief **17 W bursts** during replies. At €0.30/kWh that is roughly €18/year for the compute; the rest of the board (NVMe, display if attached) adds a little. Speech-completion latency, not power, is the binding constraint on this box.

## Pointing Home Assistant at it

- *Local OpenAI LLM* → `http://<jetson-ip>:8080/v1`, server type **llama.cpp**, model `LFM2.5-2.6B` (or whatever the installed profile's alias is — check `curl http://<jetson-ip>:8080/v1/models`), API key blank.
- *Wyoming STT* → `tcp://<jetson-ip>:10300`.
- *Wyoming TTS* → `tcp://<jetson-ip>:10210`, voice e.g. `af_heart` (53 Kokoro voices; prefix `a`=US, `b`=UK, `j`=ja, `z`=zh, `e`=es, `f`=fr, `h`=hi, `i`=it, `p`=pt-BR).

## Service internals

**1-stt-wyoming-whisper-tiny** — [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper) 3.3.1 on CUDA via a locally built CTranslate2 (the PyPI aarch64 wheel is CPU-only). Model `tiny`, language `en`, compute int8, beam 5. On start it warms CUDA with two transcription round-trips, then runs its self-test (espeak-ng speaks "hello world", the service must transcribe it).

**2-llm-llama.cpp-super-soaker-managed** — llama-server at the pinned commit with `GGML_CUDA_FA_ALL_QUANTS`, CUDA graphs, and `GGML_CUDA_NO_VMM` (Jetson iGPU VMM crash workaround, llama.cpp #29142). The model is not in the service name: `LLM_MODEL` in `/etc/super-soaker/llm.env` selects the profile, which picks the target GGUF and its `DSpark-Q8_0` drafter (`--spec-type draft-dspark`).

| knob | value | why |
|---|---|---|
| `-np 1` | single slot | LFM2 short-conv state corrupts under co-batching (llama.cpp #29002); single caller anyway |
| `-c 32768` | context tokens (8192 on the 1.2B profile) | matches long HA system prompt + history |
| `-ctk/-ctv q8_0` | KV cache quant | ~half the KV RAM of f16, no quality hit observed |
| `-b 1024 -ub 256` | batches | bounds CUDA workspace peak |
| `--cache-reuse 256` | prompt cache | turns the HA Assist prompt into a prefix hit |
| `--reasoning-budget 128` | thinking budget | enough scratch, low latency |
| `GGML_CUDA_NO_VMM=ON` | | avoids 32 GB VMM reservation crash on Jetson (#29142) |
| `GGML_CUDA_GRAPHS=ON` | | decode-speed essential on Orin |

**3-tts-wyoming-kokoro-82m** — [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) 0.5.0 on a locally built ONNX Runtime 1.22 (CUDA EP; no aarch64 CUDA wheel exists on PyPI — 1.29+ links CUDA 13, which the L4T driver cannot load). The Wyoming wrapper is a vendored copy of [nullable-eth/kokoro-wyoming](https://github.com/nullable-eth/kokoro-wyoming), internalized at `services/3-tts-wyoming-kokoro-82m/kokoro_wyoming.py`:

- Wyoming streaming: audio starts at the first sentence boundary
- text sanitation before phonemization (`speech_clean.py`): markdown, thinking/reasoning blocks (including half-open ones from truncated streams), tool-call blocks and JSON payloads, stray angle brackets — espeak reads `<` as "less than", `*` as "asterisk". Unit-tested (25 cases in `test_clean.py`) and gated at every service start
- CUDA arena capped (`KOKORO_GPU_MEM_MB=1024`) — measured comfortable
- **fp16 model** — validated to fit in RAM; `int8` is *slower* on this GPU (quantized ops fall back to CPU), so don't use it
- fails fast if `CUDAExecutionProvider` is missing (no silent CPU fallback)
- warmup synthesis before the port opens

## Self-tests

Each service ships a stdlib-only end-to-end test that runs at every start (`SELF_TEST=on` in the env files) and can be run by hand:

```bash
sudo -u super-soaker /opt/super-soaker/venv-stt/bin/python3 \
    /opt/super-soaker/services/1-stt-wyoming-whisper-tiny/selftest.py
sudo -u super-soaker python3 \
    /opt/super-soaker/services/2-llm-llama.cpp-super-soaker-managed/selftest.py
sudo -u super-soaker /opt/super-soaker/venv-tts/bin/python3 \
    /opt/super-soaker/services/3-tts-wyoming-kokoro-82m/selftest.py
```

- **STT**: speaks "hello world" with espeak-ng, asserts the transcript round-trips.
- **LLM**: model list, warmup completions, streaming speed (TTFT + tok/s), tool-calling round trip (what `hass_local_openai_llm` needs), prompt-cache probe.
- **TTS**: plain + streaming synthesis, asserts 24 kHz audio, plausible duration, non-silence RMS, and streaming first-chunk < 3 s.

## Operations

```bash
journalctl -u super-soaker-llm -f          # logs
systemctl restart super-soaker-tts         # bounce one service
free -h; swapon --show                     # RAM / swap picture
```

Layout on disk:

```
/opt/super-soaker/          builds, venvs, service code, pinned wheels (cache/)
/var/lib/super-soaker/      models, whisper hub cache (owned by super-soaker)
/etc/super-soaker/          per-service env files (llm.env, stt.env, tts.env)
/etc/systemd/system/        super-soaker-{llm,stt,tts,clocks}.service
```

## Benchmarks

The profile-comparison battery lives in `bench/ha_benchmark.py` + `bench/ha_battery.json` (30 scenarios: 20 tool + 10 quality, identical prompts per profile). Archived results in `bench/results/`: `ha_A-1.2b-noreason.json`, `ha_B-2.6b-noreason.json`, `ha_C-2.6b-reason.json`, `ha_D-8b-reason.json`. To re-run against the live service:

```bash
python3 bench/ha_benchmark.py --base-url http://127.0.0.1:8080 \
    --model <alias from /v1/models> --label my-run
```

The interactive bench also lives in `bench/` (the LLM-only sweeps from the earlier docker era are documented in `notes/`):

```bash
python3 bench/bench_openai.py      # real-world OpenAI suite
python3 bench/quality_check.py     # quality battery
bash bench/matrix.sh               # see script for sub-sweeps
```

Extra GGUFs for sweep matrices can be fetched with `scripts/fetch-models.sh` (resolves against the LiquidAI and Unsloth GGUF repos).

## Development notes

- `notes/` — design notes and the measured migration history (why native, why fp16, why one slot, etc.).
- `deps/` — the pinned build scripts (`build-llamacpp.sh`, `build-ctranslate2.sh`, `build-onnxruntime.sh`, `lock-clocks.sh`) plus `limit-jobs.sh`, the shared RAM guard that keeps compile parallelism at 1–2 threads. Every dependency is a pinned tag or commit; wheels land in `/opt/super-soaker/cache` for offline rebuilds.
- Jetson details: SM 8.7 (Orin), L4T r36.x / CUDA 12.6, cuDNN 9 from the L4T repo. `/dev/nvhost*` access comes from the `video` group (the `super-soaker` service user is in it).

## Prerequisites (for reference — bootstrap.sh installs all of this)

1. **JetPack 6.x** (L4T r36.x, CUDA 12.6) — targets SM 8.7 and the L4T driver stack.
2. **Network** for the one-time build/download; the finished stack needs none.
3. **Disk**: ~20 GB free during builds (sources + wheels + models).

## License

MIT — see [LICENSE](LICENSE).
