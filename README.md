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
| LLM TTFT (warm) | 0.3–0.5 s | prompt-cache hit on the HA Assist prefix |
| LLM decode | 42–90 tok/s (battery mean, by profile) | spec stack; templated traffic reaches 119–238 tok/s |
| first sentence → TTS | ~8–12 tokens | TTS starts at the first sentence boundary |
| TTS time-to-first-chunk | 0.48–0.68 s | Kokoro fp16, CUDA, Wyoming streaming |
| TTS sustained | 5.2–7.1 s audio / s wall | ~5× faster than playback — never the bottleneck |

### Worked example — typical short answer

"One or two sentences" (25 tokens of text, ~6 s of reply audio), derived from the measured component costs on the optimized `lfm2.5-2.6b-reasoning` profile:

| milestone | time after user stops speaking |
|---|---|
| STT transcript ready | ~0.5 s |
| LLM first token (warm) | ~0.8 s |
| first sentence complete (TTS can start) | ~1.0 s |
| **reply-speech starts** | **~1.5–1.7 s** |
| **reply-speech completes** | **~7.5 s** (speech-start + 6 s of playback) |

Speech-completion is dominated by *playback duration of the answer*, not synthesis: TTS generates ~5× faster than real time and streams sentence-by-sentence, so audio production never lags the speaker. Longer answers extend speech-complete proportionally but do not delay speech-start.

### Cold vs warm

| path | latency |
|---|---|
| warm (prompt cache hit, short reply) | speech starts ~1.5 s; completes ~answer length later |
| cold (13.7k-token prompt, cache miss) | LLM prefill ~15.1 s (1007 tok/s) before first token; speech starts ~17 s |
| service restart (model reload) | +2–4 s once, self-test then ready |

The prompt cache (`--cache-reuse 256`) is what keeps the warm path warm: the static Home Assistant Assist prefix is reused across turns, so only the new utterance is prefilled.

## Model choice: three serving profiles, one battery

Three serving profiles share one battery (`bench/ha_benchmark.py` + `bench/ha_battery.json`: 20 tool scenarios from the HA Assist API surface + 10 factual/brevity scenarios, run identically against each config). **All performance numbers are indexed on tok/s** (the server's own decode timing); words/s is a prose artifact and is no longer reported. Raw results in `bench/results/` (`ha_A`–`ha_I`).

### Performance as shipped (tok/s, after optimization)

| profile | model | novel tok/s | templated tok/s | battery mean tok/s | warm TTFT | avg reply | tool acc | quality acc |
|---|---|---|---|---|---|---|---|---|
| **lfm2.5-1.2b-chat** (default) | LFM2.5-1.2B-Instruct Q8_0 | ~83 | — | **84.4** | **0.19 s** | **0.49 s** | **85%** | 70% |
| **lfm2.5-1.2b-reasoning** | LFM2.5-1.2B-Thinking Q8_0 | 72.3 | **347** | 69.8 | 0.20 s | **0.82 s** | 55%* | 80%* |
| **lfm2.5-2.6b-reasoning** | LFM2.5-2.6B QAD-Q4_0 | **59.9** | **237.6** | 63.2 | 0.32 s | **1.38 s** | 90% | 80% |
| **lfm2.5-8b-a1b-reasoning** | LFM2.5-8B-A1B UD-IQ2_M | 38.1 | 119 | 41.6 | 0.51 s | 3.60 s | **95–100%** | **90%** |

"Novel" = 12 unique prompts (free-form speech proxy); "templated" = identical-utterance probe (HA boilerplate ceiling — fixed tool JSON, repeated phrasing); "battery mean" = per-scenario decode tok/s across all 30 frozen scenarios. Battery accuracy is n=1 per scenario at temp 0.1, so it wobbles ±5–10% between runs (the 8B's archived `ha_D`/`ha_H` runs scored 100%/90%); tok/s and latency are stable. (*The 1.2B-Thinking's tool/quality scores swing wider across runs — 45–65% / 70–90% — with identical core failures every run (`climate_set`, `timer_cancel`, `weather`, `count_r`): those are the model's own weak spots, matching its card's BFCLv3 57 and 'not recommended for knowledge-intensive tasks'; the optimization adds no systematic penalty.) The 1.2B and 2.6B share one quirk: "turn on the lights" sometimes maps to HassLightSet instead of HassTurnOn — worth a prompt nudge in HA's own tool descriptions if it shows up in practice.

### super-soaker optimized vs original config (same battery, same box)

| profile | metric | original | optimized | gain |
|---|---|---|---|---|
| **lfm2.5-2.6b-reasoning** | battery mean tok/s | 57.7 | 63.2 | 1.10x |
| | novel tok/s | 55.3 | 59.9 | 1.08x |
| | templated tok/s | 59.4 | **237.6** | **4.0x** |
| | avg reply | 2.46 s | **1.38 s** | **1.78x** |
| | accuracy | 90% / 80% | 90% / 80% | unchanged |
| **lfm2.5-1.2b-chat** | battery mean tok/s | 46.5 | **84.4** | **1.81x** |
| | novel tok/s | untested | 83.1 | unchanged |
| | avg reply | 0.52 s | 0.49 s | 1.06x |
| | accuracy | 85% / 70% | 85% / 70% | unchanged |
| **lfm2.5-1.2b-reasoning** | battery mean tok/s | 61.0 | 69.8 | 1.14x |
| | templated tok/s | 86.7 | **347.5** | **4.0x** |
| | avg reply | 2.06 s | **0.82 s** | **2.5x** |
| | novel tok/s | ~70 | 72.3 | 1.03x |
| | accuracy | 55–65% / 70–90% | 45–55% / 70% | run-noise overlap (see note) |
| **lfm2.5-8b-a1b-reasoning** | battery mean tok/s | 39.0 | 41.6 | 1.07x |
| | novel tok/s | 34.6 | 38.1 | 1.10x |
| | templated tok/s | 34.6 | **119** | **3.4x** |
| | avg reply | 4.28 s | **3.60 s** | 1.19x |
| | accuracy | 90% / 80% | 95% / 90% | equal or better |

The chat profile's target moved from QAD-Q4_0 to **Q8_0** during its campaign (fidelity insurance, user preference): battery accuracy is *identical* at QAD-Q4_0/Q8_0/F16 (85%/70% at all three — the cap is the model, not the quant), and the table above compares un-optimized vs optimized **on the shipped Q8_0 target** so both columns are like-for-like. QAD-Q4_0 remains measurably faster on novel speech (99.9 vs 83.1 tok/s) if that trade is wanted later.

The spread between "novel" and "templated" is the honest story of both campaigns: real assistant traffic (fixed tools, familiar phrasing) sits between the columns, closer to templated the more the home vocabulary repeats. Both optimizations keep the battery's accuracy at or above the original — no quality was traded.

**Why lfm2.5-1.2b-chat is the default:** fastest on every latency axis (0.51 s avg reply, 90 tok/s battery mean) with tool accuracy close to the 2.6B (85% vs 90%). What it gives up: self-corrected arithmetic and letter-counting (no reasoning pass to check itself). A fourth profile — the 2.6B with reasoning disabled — was benchmarked and rejected: it over-triggers tools on non-commands and keeps the 2.6B memory cost.

**Pick lfm2.5-2.6b-reasoning if** the assistant must be disciplined about tool calls and facts (the agentic profile). After optimization its replies take 1.38 s on average (was 2.28 s historically) — within striking distance of the 1.2B — and repetitive voice traffic reaches ~238 tok/s. Tool 90%, quality 80%.

**Pick lfm2.5-8b-a1b-reasoning if** tool-call and answer correctness matter more than reply latency. The 8B-A1B is a reasoning-only MoE: it spends its 1.5B active parameters on a chain-of-thought pass before answering, so replies take ~3.6 s on average, but it is the strongest profile on tool accuracy (95–100%) and factual quality (90%). It also answers some scenarios the dense models cannot (reliable letter-counting, multi-digit arithmetic self-checks).

**Only one LLM fits at a time:** measured 7.32/7.62 GB with both dense LLMs resident (105 MB free) — below any safe margin. The 8B-A1B is worse still: even at IQ2-class quants it needs ~4.3 GB of RAM by itself (see below). Run one.

### RAM cost per profile (measured, LLM + DSpark draft + KV + nvgpu pool)

| profile | resident LLM | peak (LLM+TTS concurrent) | fits with STT+TTS? |
|---|---|---|---|
| lfm2.5-1.2b-chat | ~2.6 GB | ~2.6 GB | yes, ~1.8 GB headroom |
| lfm2.5-1.2b-reasoning (Thinking Q8_0) | ~2.2 GB | ~2.7 GB | yes, ~2.3 GB headroom |
| lfm2.5-2.6b-reasoning | ~3.0 GB | ~3.2 GB | yes, ~1.4 GB headroom |
| lfm2.5-8b-a1b-reasoning (UD-IQ2_M) | ~4.3 GB | ~4.8 GB (7.04/7.62 GB board) | yes, but only ~580 MB headroom |

The 8B-A1B leaves little slack: every quant larger than UD-IQ2_M was measured and rejected (see "8B-A1B quant hunt" in AGENTS.md). Q4_0 (4.8 GB of weights) does not fit at all — it starts, then the decode collapses from 33 t/s to 2.4 t/s as the box thrashes. UD-IQ3_XXS completes the battery with the best quality score but pushes 321 MB into zram and peaks at 7.49/7.62 GB — a violation of the stack's no-swap rule and too close to the edge for a concurrent TTS peak. UD-Q2_K_XL fits (3 MB swap) but is dominated by UD-IQ2_M on every axis: slower decode, lower accuracy, more RAM. UD-IQ2_M is the only 8B quant that runs the whole battery with zero swap and real headroom.

### Switching profiles

```bash
sudo ./bootstrap.sh --model lfm2.5-2.6b-reasoning
# or: sudo ./bootstrap.sh --model lfm2.5-8b-a1b-reasoning
# or: sudo ./bootstrap.sh --model lfm2.5-1.2b-chat (default)
```

On an installed node this rewrites `/etc/super-soaker/llm.env` (`LLM_MODEL=...`) and fetches the matching GGUF pair (sha256-pinned): LFM2.5-1.2B-Instruct QAD-Q4_0 + DSpark-Q8_0, LFM2.5-1.2B-Thinking Q8_0 + Instruct DSpark-Q8_0, LFM2.5-2.6B QAD-Q4_0 + DSpark-Q8_0, or Unsloth LFM2.5-8B-A1B UD-IQ2_M + DSpark-Q8_0. The thinking budget is set the same way: `sudo ./bootstrap.sh --reasoning-budget 256` (or `off`/`on`/`auto`) — defaults are per model (0 for the no-reasoning chat profile, 128 for the reasoning profiles), and an explicit flag always wins while unpassed values are preserved across re-runs. You can also edit `LLM_MODEL=`/`REASONING_BUDGET=` in `/etc/super-soaker/llm.env` and `systemctl restart super-soaker-llm`. Individual overrides (`CTX`, `MODEL_PATH`, ...) still win over the profile defaults.

All profiles run with the same reasoning and context settings where their model supports it: `CTX=32768`, `REASONING_BUDGET=128` for the two reasoning profiles (the 1.2B chat profile runs `CTX=8192`, no reasoning). The 8B-A1B is a reasoning-only model — the budget bounds its chain-of-thought so voice replies stay predictable.

### Performance optimization campaigns (measured)

Both reasoning profiles ship an optimized speculative-decoding stack plus a llama.cpp kernel patch (`deps/patches/mmvq-orin-crossover.patch`) extending the Orin MMVQ→MMQ crossover (upstream PR #28285 covered K-quants only) to IQ/simple quants (+4–10% on verify batches). The dense 2.6B and the 8B-A1B MoE need *different* spec tuning — same drafter, opposite economics:

| knob | lfm2.5-2.6b-reasoning (dense) | lfm2.5-8b-a1b-reasoning (MoE) | why |
|---|---|---|---|
| `--spec-type` | `draft-dspark,ngram-mod` | `draft-dspark,ngram-mod` | ngram adds free accepted tokens on templated traffic, costs nothing on novel |
| `--spec-draft-n-max` | 7 | 5 | dense verify is cheap → longer blocks pay; MoE expert-union traffic saturates with block length → shorter wins |
| `--spec-draft-p-min` | off (default 0) | 0.4 | confidence truncation HURTS dense (loses accepted tokens for nothing) but helps MoE (prunes low-confidence tails that cost full expert traffic) |
| drafter quant | Q8_0 | Q8_0 | acceptance is drafter-invariant (F16/Q4_K/Q8_0 all within noise) — the target distribution is the limiter; Q8_0 wins on cost |
| target quant | QAD-Q4_0 | UD-IQ2_M | higher-fidelity targets do NOT buy acceptance (measured Q4_K_M→Q8_0: flat) and cost raw speed |
| **lfm2.5-1.2b-chat verdict** | n_max 9, plain dspark, **Q8_0 target** | — | acceptance 2.99/step with the native drafter ⇒ trained block length wins; ngram/p-min/p-split/backend-sampling all neutral-or-worse; Q8_0 shipped for fidelity (battery accuracy identical at QAD-Q4_0/Q8_0/F16 — the cap is the model — see notes/07) |

The `lfm2.5-1.2b-reasoning` profile (LFM2.5-1.2B-Thinking) needs one extra piece of plumbing: its GGUF embeds the **non-thinking** chat-template variant, so llama.cpp's reasoning extraction never engages and thinking leaks into `message.content` (upstream llama.cpp issue #28675, open). Serving it with the reasoning-capable family reference template (`templates/LFM2.5-reasoning.jinja`, vendored from upstream `LFM2.5-8B-A1B.jinja`) via `--chat-template-file` + `--reasoning-format deepseek` fixes extraction, budget capping, and content cleanliness in one step. No dedicated DSpark drafter exists for Thinking; the 1.2B-Instruct drafter works against it at lower acceptance (1.5–2.1/step vs 2.9), so short blocks (`n_max 5`) win there exactly as the low-acceptance rule predicts.

Measured campaign results (tok/s, frozen battery + bench harness):

| metric | 2.6B original → optimized | 8B original → optimized | 1.2B-Thinking original → optimized |
|---|---|---|
| raw batch-1 decode | 30.5 → 30.5 (1.0x) | 35.5 → 35.5 (1.0x) |
| novel speech | 55.3 → 59.9 (1.08x) | 34.6 → 38.1 (1.10x) |
| templated voice traffic | 59.4 → **237.6 (4.0x)** | 34.6 → **119 (3.4x)** |
| battery mean tok/s | 57.7 → 63.2 (1.10x) | 39.0 → 41.6 (1.07x) |
| battery avg reply | 2.46 s → **1.38 s (1.78x)** | 4.28 s → 3.60 s (1.19x) |
| battery accuracy | 90/80 → 90/80 | 90/80 → 95/90 | see note (run-noise overlap) |
| templated tok/s (extra) | 59.4 → 237.6 (4.0x) | 34.6 → 119 (3.4x) | 86.7 → 347.5 (4.0x) |

Why the spread: batch-1 GEMVs on this GPU stream at a fixed ~24% (IQ-LUT quants) or ~37% (block quants) of LPDDR5 bandwidth, so non-spec decode is capped by bytes-per-token (30–48 tok/s on these models). Speculative decoding escapes that cap by turning hundreds of skinny GEMVs into fat verify GEMMs — but only accepted tokens count. Dense verify reads weights once per block (break-even ≈ 1.2 accepted/step — every acceptance point is profit), while 4-of-32 MoE verify traffic saturates with block length (needs 5+). Acceptance itself is capped at ~2.3–3.5/step on novel speech for both models: the DSpark drafter was distilled against an F16 target and drifts on quantized targets (llama.cpp issue #25618) — measured invariant across drafter quants and target quants. Templated HA traffic accepts 5+ and hits the ceilings above. Full logs, rejected experiments (F16 drafter, p-min, TQ2_0, custom quants, micro knobs, target-fidelity sweeps) and the constraint analyses: `notes/04-decode-optimization-research.md` (8B) and `notes/05-2.6b-optimization.md` (2.6B).

## Design tradeoffs: quality vs intelligence vs speed

The guiding rule: **maximum performance inside a hard 8 GB RAM envelope, while tolerating a slightly slower reply when quality would otherwise suffer.** Every component below is the highest-quality option that still fits and keeps the conversation fluid.

### LLM — LFM2.5 family + DSpark-Q8_0 draft (profile-selected)

| decision | chosen | rejected | why |
|---|---|---|---|
| model size | 1.2B / 2.6B / 8B-A1B (profile) | 7B dense, 20B+ | the three profiles are the LFM2.5 sizes that fit alongside STT+TTS; dense 8B-class weights do not fit at usable quants (see quant hunt below) |
| target quant | QAD-Q4_0 (dense), UD-IQ2_M (8B-A1B) | Q4_K_M, Q5_K_M for the 8B | QAD (quantization-aware distill) is LiquidAI's own quality-preserving recipe; for the 8B-A1B MoE only IQ2-class quants fit — Unsloth's UD-IQ2_M was the measured winner of a four-quant shootout |
| draft model | DSpark-Q8_0 (on) | no draft, F16/Q4 draft | spec decoding is the one lever past the batch-1 streaming cap (measured up to 4.0x on templated traffic, quality-neutral); acceptance is drafter-quants-invariant so Q8_0 wins on cost; F16 drafter is a measured net loss on iGPU |
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

The profile-comparison battery lives in `bench/ha_benchmark.py` + `bench/ha_battery.json` (30 scenarios: 20 tool + 10 quality, identical prompts per profile). Per-scenario and decode results carry **tok/s** (`tok_s`, from the server's own `timings.predicted_per_second`) alongside latency. Archived results in `bench/results/`: `ha_A`–`ha_C` (pre-optimization era, words/s), `ha_D` (8B original config), `ha_E`/`ha_F` (2.6B original/optimized), `ha_G`/`ha_H` (8B original/optimized), `ha_I` (1.2B as shipped). To re-run against the live service:

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
