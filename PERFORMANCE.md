# Performance notes

These are the measurements behind super-soaker's defaults, taken on the target hardware: an NVIDIA Jetson Orin Nano Super (8 GB) running L4T 36.4.4, JetPack 6.x, CUDA 12.6, with the full voice stack resident. The README tells you how to install and run it; this file is for deciding whether to change a knob, pick a different profile, or repeat an experiment.

The short version: the stack fits, nothing swaps, and speculative decoding is the only lever that gets past the GPU's batch-1 memory-bandwidth ceiling without trading away quality. Most of the obvious alternatives were measured and rejected; they are listed at the end so nobody has to rediscover them.

## How these numbers were measured

- **RAM** is `MemTotal − MemFree − Cached` (or `tegrastats` RAM used/total) at 100 ms sampling, not process RSS. On Jetson the iGPU driver allocates kernel memory that RSS cannot see; a `llama-server` showing 700 MB RSS can be sitting on several gigabytes of real footprint.
- **Swap** is `/proc/<pid>/status` `VmSwap`, plus `free -h` and `swapon --show`. Every service is pinned `MemorySwapMax=0`, so the expected answer is zero.
- **Reboot between memory experiments.** Stopping `llama-server` does not reliably return the nvgpu pool and page cache. A stopped stack once read 4.5 GB used; after a reboot, 2.2 GB. If a measurement looks wrong by about 2 GB, suspect stale pool, not the model.
- **Speed** is quoted as decode tok/s from llama-server's own `timings.predicted_per_second`. Words per second was used in the earlier battery and is a prose artifact; the current battery reports tok/s so profiles are comparable.
- **Power** is whole-board `VDD_IN` from `tegrastats`, sampled at 0.5 s, with the stack resident and the clock state reported.
- **Accuracy** is n=1 per scenario at temperature 0.1 against a frozen battery. It wobbles ±5–10% between runs; tok/s and latency are stable. Read accuracy as a signal, not a decimal.

## Where the latency goes

The pipeline is: user stops speaking → STT finalizes → LLM generates → TTS streams. Two moments matter to a person: when reply speech *starts*, and when it *finishes*.

| stage | cost | note |
|---|---|---|
| STT finalize (3 s utterance) | ~0.3–1.0 s | Whisper tiny/en int8 on CUDA; scales with utterance length |
| LLM TTFT (warm) | 0.3–0.5 s | prompt-cache hit on the HA Assist prefix |
| LLM decode | 42–90 tok/s battery mean, by profile | templated traffic reaches 119–238 tok/s |
| first sentence → TTS | ~8–12 tokens | TTS starts at the first sentence boundary |
| TTS time-to-first-chunk | 0.48–0.68 s | Kokoro fp16 on CUDA, Wyoming streaming |
| TTS sustained | 5.2–7.1 s of audio per second of wall time | roughly 5× faster than playback; never the bottleneck |

### Worked example: a typical short answer

"One or two sentences" is about 25 tokens, or roughly 6 s of reply audio. On the optimized `balanced` profile:

| milestone | time after the user stops speaking |
|---|---|
| STT transcript ready | ~0.5 s |
| LLM first token (warm) | ~0.8 s |
| first sentence complete, TTS can start | ~1.0 s |
| **reply speech starts** | **~1.5–1.7 s** |
| **reply speech completes** | **~7.5 s** (speech start + 6 s of playback) |

Speech completion is dominated by the playback duration of the answer, not synthesis. TTS generates about 5× faster than real time and streams sentence by sentence, so longer answers extend completion proportionally without delaying the start.

### Cold versus warm

| path | latency |
|---|---|
| warm (prompt-cache hit, short reply) | speech starts ~1.5 s; finishes one answer-length later |
| cold (13.7k-token prompt, cache miss) | ~15.1 s prefill at 1007 tok/s before the first token; speech starts ~17 s |
| service restart (model reload) | +2–4 s once, then the self-test clears and the service is ready |

The prompt cache (`--cache-reuse 256`) is what keeps the warm path warm. The static Home Assistant Assist prefix is reused across turns, so only the new utterance is prefilled. Anything that churns the prefix - changing the tool list or the system prompt often - pays the cold prefill again.

## The Home Assistant battery

`bench/ha_benchmark.py` and `bench/ha_battery.json` hold a frozen 30-scenario battery: 20 tool scenarios from the HA Assist API surface (including no-tool negatives that punish over-triggering) and 10 factual/brevity scenarios. Every profile gets identical prompts and `max_tokens`. The battery reports per-scenario decode tok/s from the server, end-to-end latency, tool accuracy, quality accuracy, and brevity.

Archived results are in `bench/results/`: `ha_A`–`ha_C` are the early words/s era, `ha_D` the original 8B, `ha_E`/`ha_F` the 2.6B original and optimized, `ha_G`/`ha_H` the 8B original and optimized, `ha_I` the 1.2B as shipped, and `ha_lfm3-*` the LFM3-1.7B quant/spec/reasoning sweeps (2026-11). Old result files are never edited.

`bench/profile_battery.sh` is the harness used for the LFM3 sweeps: it starts `llama-server` with exactly `serve.sh`'s flags, refuses to run if a server already owns the port, runs a memory watchdog (kills the server below 80 MB `MemAvailable`), runs N battery reps, and writes a `mem_<label>.json` with peak `MemTotal−MemFree−Cached` and the server's `VmSwap`. `bench/sweep_frontier.sh` drives the quant / spec / reasoning stages on top of it.

"Novel" in the tables below means 12 unique prompts as a free-form speech proxy. "Templated" is an identical-utterance probe: the HA boilerplate ceiling with fixed tool JSON and repeated phrasing. The battery mean is per-scenario decode across all 30 scenarios.

## Profiles as shipped

| profile | model | novel tok/s | templated tok/s | battery mean | warm TTFT | average reply | tool / quality |
|---|---|---|---|---|---|---|---|
| `fast` | LFM2.5-1.2B-Instruct Q8_0 | ~83 | - | **84.4** | **0.19 s** | **0.49 s** | **85% / 70%** |
| `balanced` (default) | LFM2.5-2.6B QAD-Q4_0 | **59.9** | **237.6** | 63.2 | 0.32 s | **1.38 s** | 90% / 80% |
| `thorough` | LFM2.5-8B-A1B UD-IQ2_M | 38.1 | 119 | 41.6 | 0.51 s | 3.60 s | **95–100% / 90%** |
| `frontier` | LFM3-1.7B QAD-Q4_0 + ngram-mod | ~42 | 75–97 (steady) | 75–97 (steady) | 0.26–0.35 s | 1.6–2.3 s | 85–90% / 80–90% |

The archived 8B runs scored 95–100% tool and 90% quality; the range is battery noise at n=1 per scenario, not a configuration change. tok/s and latency are stable.

`frontier`'s battery mean is not a single number because ngram-mod keeps filling its n-gram table across repeated scenario phrasing: the first rep of a fresh server runs at the no-spec rate (39.6 tok/s) and steady-state reps reach 75.5 / 80.0 / 87.3 / 96.6 tok/s (5-rep run). Novel-prompt decode is ~42 tok/s throughout, unaffected. Accuracy across those five reps was tool 85/90/90/90/85 and quality 90/80/90/90/80 — statistically the same as the no-spec twin (tool 85–95, quality 70–90 across five reps) at ~half the reply time.

The two dense models share one quirk: "turn on the lights" sometimes maps to `HassLightSet` instead of `HassTurnOn`. That is worth a nudge in Home Assistant's own tool descriptions if it shows up in practice; it is not a serving bug.

The `fast` profile moved from QAD-Q4_0 to Q8_0 during its campaign, for fidelity insurance and because that was the user preference. Battery accuracy is identical at QAD-Q4_0, Q8_0, and F16 (85% tool / 70% quality at all three), so the cap is the model, not the quant. QAD-Q4_0 remains measurably faster on novel speech (99.9 vs 83.1 tok/s) if that trade is wanted later. The optimization table below compares un-optimized and optimized on the shipped Q8_0 target so both columns are like-for-like.

## What optimization bought

| profile | metric | original | optimized | gain |
|---|---|---|---|---|
| `balanced` | battery mean tok/s | 57.7 | 63.2 | 1.10× |
| | novel tok/s | 55.3 | 59.9 | 1.08× |
| | templated tok/s | 59.4 | **237.6** | **4.0×** |
| | average reply | 2.46 s | **1.38 s** | **1.78×** |
| | accuracy | 90% / 80% | 90% / 80% | unchanged |
| `fast` | battery mean tok/s | 46.5 | **84.4** | **1.81×** |
| | novel tok/s | untested | 83.1 | - |
| | average reply | 0.52 s | 0.49 s | 1.06× |
| | accuracy | 85% / 70% | 85% / 70% | unchanged |
| `thorough` | battery mean tok/s | 39.0 | 41.6 | 1.07× |
| | novel tok/s | 34.6 | 38.1 | 1.10× |
| | templated tok/s | 34.6 | **119** | **3.4×** |
| | average reply | 4.28 s | **3.60 s** | 1.19× |
| | accuracy | 90% / 80% | 95% / 90% | equal or better |

The spread between novel and templated speech is the honest story of both campaigns. Real assistant traffic - fixed tools, familiar phrasing - sits between the columns, closer to templated the more the home vocabulary repeats. Both optimizations keep battery accuracy at or above the original, so no quality was traded for the speed.

## Why the gains happen

At batch 1, GEMVs on this GPU stream at a fixed fraction of LPDDR5 bandwidth: roughly 24% of peak for IQ-LUT quants and 37% for block quants. Dequantization hides under the memory stalls, so speed is approximately bytes per token divided by that rate. Non-speculative decode is therefore capped by weight size, which is why "just requantize lighter" buys 1.3–1.6× and why a smaller quant is not automatically faster.

Speculative decoding escapes the cap by turning hundreds of skinny GEMVs into fat verify GEMMs. It only wins on *accepted* tokens, and the economics differ by architecture:

- **Dense (2.6B):** verify reads the weights once per block. Break-even is about 1.2 accepted tokens per step; every acceptance above that is profit, so longer draft blocks pay. `n_max` 7 wins, and confidence truncation only loses accepted tokens for nothing.
- **4-of-32 MoE (8B-A1B):** verify traffic saturates as the block lengthens because the union of experts touched approaches the whole model (roughly 23–27 of 32 at k=9). It needs about 5 accepted tokens per step to pay, so shorter blocks and `p-min` 0.4 win: prune the low-confidence tails that cost full expert traffic.

Acceptance is drafter-versus-target-family limited, not quant-noise limited. It is invariant across drafter quants (F16, Q4_K, Q8_0) and target quants (Q4_0 through Q8_0), so always ship Q8_0: same acceptance, less cost. The DSpark drafter was distilled against an F16 target and drifts on quantized targets; novel speech accepts about 2.3–3.5 tokens per step, while templated HA traffic accepts 5+ and reaches the ceilings above.

The stack also carries a local llama.cpp patch, `deps/patches/mmvq-orin-crossover.patch`, which extends the Orin MMVQ→MMQ crossover to IQ and simple quants (upstream PR #28285 covered K-quants only). It is worth 4–10% on verify batches and changes nothing at batch 1.

### Speculative-decoding settings

| knob | `balanced` (dense) | `thorough` (MoE) | why |
|---|---|---|---|
| `--spec-type` | `draft-dspark,ngram-mod` | `draft-dspark,ngram-mod` | ngram adds free accepted tokens on templated traffic and costs nothing on novel speech |
| `--spec-draft-n-max` | 7 | 5 | dense verify is cheap, so longer blocks pay; MoE expert-union traffic saturates with block length, so shorter wins |
| `--spec-draft-p-min` | off (0) | 0.4 | truncation hurts dense by dropping accepted tokens; it helps MoE by pruning tails that cost full expert traffic |
| drafter quant | Q8_0 | Q8_0 | acceptance is drafter-invariant; Q8_0 wins on cost |
| target quant | QAD-Q4_0 | UD-IQ2_M | higher-fidelity targets do not buy acceptance and cost raw speed |
| `fast` verdict | - | - | plain `draft-dspark`, `n_max` 9, Q8_0 target; 2.99 accepted per step means the native drafter's trained block length wins, and ngram/p-min/p-split/backend sampling were all neutral or worse |

## The LFM3-1.7B hunt (the `frontier` profile)

LFM3-1.7B is Liquid's successor to LFM2.5. It is a 1.73B tri-hybrid: 24 layers made of 13 gated short convolutions, 5 sliding-window attention layers (256-token window) and 6 global attention layers, plus a 128K vocabulary and a 256K-token context ceiling. It is reasoning-on by default, and it needs llama.cpp b11960 or newer — older builds do not know the `lfm3` architecture at all. The repo pin moved from b11552 to llama.cpp **v0.9.0** (`cd959bae99`, build 12110) for this profile. The LFM2.5 rows above were measured on the old binary and were not re-benchmarked; on v0.9.0 all three LFM2.5 profiles still assemble the expected DSpark/ngram-mod command lines and pass their self-tests (`fast`, `balanced`, `thorough` verified 2026-11-08), so the pin bump did not break them, but treat their accuracy/speed table as b11552 measurements.

**There is no DSpark (or DFlash) drafter for LFM3.** LiquidAI publishes DSpark drafts for LFM2.5-1.2B/2.6B/8B-A1B/24B-A2B and VL-3B only, and a Hugging Face search for `LFM3-DSpark` returns nothing. The alternative is the drafter-free `ngram-mod` speculative mode, which llama.cpp enables for any model; it is measured below.

Sampling per Liquid's card (temperature 0.1, top_k 50, repetition_penalty 1.1) is what the battery already uses. Every run below is the frozen battery, `CTX=16384`, q8_0 KV, `--cache-reuse 256`, STT+TTS resident, with `bench/profile_battery.sh`.

### Quant choice

All five official GGUF quants were downloaded and SHA-256-verified against the Hugging Face LFS object ids. Each ran 3 battery reps at reasoning budget 160 (chosen below).

| quant | size | tool acc | quality acc | battery tok/s | peak RAM (stack resident) | swap |
|---|---|---|---|---|---|---|
| **QAD-Q4_0** | **1.05 GB** | **93.3%** | **90.0%** | **39.6** | **5.22 GB** | **0** |
| Q4_K_M | 1.10 GB | 91.7% | 90.0% | 36.9 | 5.28 GB | 0 |
| Q5_K_M | 1.26 GB | 88.3% | 63.3% | 34.6 | 5.35 GB | 0 |
| Q6_K | 1.43 GB | 90.0% | 73.3% | 34.0 | 5.52 GB | 0 |
| Q8_0 | 1.85 GB | 90.0% | 76.7% | 28.7 | 5.97 GB | 0 |
| F16 | 3.47 GB | — | — | — | 7.41 GB | **190 MB (watchdog kill)** |

The counter-intuitive result repeats from the 8B-A1B hunt: the smallest official quant wins on accuracy *and* speed. QAD-Q4_0 is Liquid's quant-aware distilled build; Q5_K_M's 63% quality is not a quant-cliff but a reasoning-length interaction (below), and Q8_0 is the slowest because larger weights cost more bandwidth at batch 1 on this GPU. F16 does not fit with the voice stack resident: it peaked at 7.41 GB with 190 MB swapped, the memory watchdog killed the server, and every request after that failed with connection-refused. Selection: **QAD-Q4_0**.

### Reasoning budget is the accuracy lever, not the quant

LFM3 always thinks before answering, and its thinking is longer than LFM2.5's. The frozen battery's `max_tokens` is fixed at 160 for tool scenarios and 200 for quality scenarios. If the reasoning budget is at or above those caps, the budget wrap-up never fires, `max_tokens` truncates mid-thought, and the scenario fails with an empty answer. That is exactly what happened at the first-pass budget of 256: quality scores of 60–80% that looked like a quant problem but moved with the budget instead.

Sweeping the budget on QAD-Q4_0:

| reasoning budget | tool acc | quality acc | note |
|---|---|---|---|
| 64 | 95 / 90 | 80 / 80 | too short: some reasoning-heavy answers lose their check |
| 96 | 90 / 90 | 80 / 80 | |
| 128 | 95 / 95 | 80 / 80 | |
| **160** | **95 / 90** | **100 / 90** | best: wrap-up fires inside the answer budget |
| 192 | 90 / 90 | 80 / 80 | budget at the tool cap: `implicit_climate` truncates |
| 256 | 90–95 | 60–80 | budget above `max_tokens`: answers truncated mid-thought |

The shipped `frontier` default is **160**. Note this is a property of the battery's `max_tokens`, not of Home Assistant: a real HA request that allows a longer answer can use a larger budget. It is the same trade the LFM2.5 profiles make with 256, adjusted for LFM3's longer drafts.

### ngram-mod speculative decoding

With no drafter available, `ngram-mod` is the only speculative mode. It is quality-neutral (verified distributions are unchanged) and it pays off on Home Assistant's templated tool traffic: repeated tool JSON and phrasing become draftable n-grams.

| config | tool acc (5 reps) | quality acc (5 reps) | battery tok/s | average reply | novel tok/s |
|---|---|---|---|---|---|
| QAD-Q4_0, budget 160, no spec | 85/90/95/90/90 | 90/70/90/70/80 | 39.7 flat | 3.46–3.57 s | ~40 |
| **QAD-Q4_0, budget 160, ngram-mod n-max 7** | **85/90/90/90/85** | **90/80/90/90/80** | **39.6 → 75.5 → 80.0 → 87.3 → 96.6** | **1.56–2.32 s** | **~42** |

The first rep after a restart runs at the no-spec rate while the n-gram table fills; steady-state replies are ~2× faster. Accuracy is statistically indistinguishable between the two rows at n=5. `frontier` ships with `ngram-mod` on by default. `SPEC=off` in `llm.env` reverts to the non-speculative path, and `SPEC_TYPE`/`SPEC_N_MAX` are honoured as for the other profiles.

### Frontier power

`tegrastats` `VDD_IN` at 0.5 s during the 5-rep ngram run: min 4.80 W, median 16.49 W, p95 17.32 W, max 17.47 W — the same envelope as the other profiles (whole-board, `MAXN_SUPER`, clocks un-pinned).

### Why `frontier` is not the default

It ties or beats `balanced` on accuracy and its steady-state speed is excellent, but its *cold* reply (first request after a restart, before n-gram tables fill, and with a cold prompt cache) is slower than `balanced`'s, and its reasoning pass makes worst-case replies less predictable. `balanced` remains the default for a voice assistant; `frontier` is the profile to choose when the newest Liquid model's higher ceiling matters more than a predictable floor. This node is left running `frontier` (`LLM_MODEL=frontier` in `llm.env`) because that is the model this work was asked to install and validate; switching back is one `sudo ./bootstrap.sh --model balanced`.

## The 8B-A1B quant hunt

The 8B-A1B is 8.3B total parameters and 1.5B active (MoE: 32 experts, 4 active per token; gated short-conv LFM2 hybrid; reasoning-only). Total parameters drive RAM, active parameters drive speed. That makes it fast for its size, but it still needs its whole quant resident alongside STT and TTS.

Four quants were measured with the real battery, CTK/CTV q8_0, a DSpark-Q8_0 draft, 32k context, and a 128-token reasoning budget (the campaign settings; the shipped `thorough` profile now uses 16k and 256):

| quant | size | peak RAM | swap | tool accuracy | quality | verdict |
|---|---|---|---|---|---|---|
| LiquidAI Q4_0 | 4.84 GB | - | thrash | - | - | **does not fit**: watchdog killed it at 72 MB free; decode collapsed from 33 to 2.4 tok/s |
| Unsloth UD-IQ3_XXS | 3.27 GB | 7.49 / 7.62 GB | 321 MB | 95% | 100% | best quality, but violates no-swap and leaves no room for a TTS peak |
| Unsloth UD-Q2_K_XL | 2.93 GB | 7.27 / 7.62 GB | 3 MB | 90% | 90% | fits, but dominated by UD-IQ2_M on every axis |
| **Unsloth UD-IQ2_M** | **2.75 GB** | **7.04 / 7.62 GB** | **0** | **100%** | **90%** | **chosen**: zero swap, ~580 MB headroom, fastest decode |

The counter-intuitive result: the smaller quant also won on tool accuracy. Battery noise is real at n=1 per scenario, but the takeaway holds - on this box the 8B-A1B has one honest fit, and IQ2-class quants are not automatically broken for MoE models. Only 4 of 32 experts activate per token, so much of the weight is rarely hot.

Prefer official Unsloth or LiquidAI GGUFs. Third-party requantized uploads vary in quality and provenance.

## RAM budget

All figures are `MemTotal − MemFree − Cached` deltas with the stack resident, which captures the nvgpu kernel allocations that `docker stats` and RSS hide. The GPU shares system RAM on Jetson, so "GPU memory" here is RAM.

| component | resident | peak (LLM + TTS concurrent) |
|---|---|---|
| LLM `fast` (llama.cpp + nvgpu pool) | ~2.6 GB | flat under load |
| LLM `balanced` | ~3.0 GB | flat under load |
| LLM `thorough` (UD-IQ2_M + DSpark) | ~4.3 GB | ~4.8 GB |
| LLM `frontier` (LFM3-1.7B QAD-Q4_0, no drafter) | ~1.0 GB | flat under load |
| STT (Whisper tiny/en int8) | ~0.7 GB | flat |
| TTS (Kokoro fp16, ONNX CUDA; arena pre-sized) | ~1.3 GB | ~1.3 GB (high-water reached at boot) |
| OS (headless) | ~0.8 GB | - |
| **total, `balanced`** | **~5.6 / 7.4 GB** | **~6.2 / 7.4 GB** |
| **total, `fast`** | **~5.2 / 7.4 GB** | **~5.6 / 7.4 GB** |
| **total, `thorough`** | **~6.9 / 7.4 GB** | **~7.0 / 7.4 GB** |
| **total, `frontier`** | **~5.3 / 7.4 GB** | **~5.3 / 7.4 GB** |

That is roughly 1.2 GB of headroom at peak on `balanced` (~1.8 GB on `fast`), ~0.5 GB on `thorough`, and ~2.1 GB on `frontier`. The `frontier` figure is measured (not estimated) with STT and TTS resident: 5.13 GB after model load, 5.34 GB peak across five battery reps, server `VmSwap` 0. LFM3-1.7B is simply a small model — its QAD-Q4_0 weights are 1.05 GB — so it is the roomiest profile despite carrying more capable weights than `balanced`.

F16 does **not** fit: with STT+TTS resident it peaked at 7.41 / 7.4 GB with 190 MB of server swap, the watchdog killed it, and every battery request then failed with connection-refused. That is the same failure class as the 8B-A1B Q4_0 attempt. Q8_0 fits (5.97 GB peak, zero swap) but is slower than every smaller quant on Orin, so it was not selected. The measured `thorough` peak was 7.04 / 7.62 GB with zero swap. Every service pins `MemorySwapMax=0` and the system sets `vm.swappiness=10`, so "is anything swapped?" should be *no* by construction.

On 2026-10-17 the TTS began sizing its CUDA arena at service start with a full-size 503-phoneme batch. Before that, the arena high-water (~0.87 GB) grew on the first long answer, and that growth was what pushed the board into a global OOM mid-request. Measured after the change on `balanced`, TTS nvmap is flat at ~1.2 GB across 8- to 99-word requests and full-size batches, with zero swap. Long answers no longer need mid-request headroom, but the steady-state free figure is now ~0.2–0.8 GB depending on page-cache state (88–207 MB free, 0.3–0.9 GB reclaimable cache) rather than the ~1.2 GB implied by the older peak numbers above.

Only one LLM fits at a time. Two dense LLMs resident measured 7.32 / 7.62 GB with 105 MB free, which is below any safe margin. The 8B-A1B is tighter still.

### Why native and not Docker

Docker was tried first and is a dead end on this hardware. The daemon plus per-container library duplication costs several hundred MB of the 8 GB before any model loads; the container boundary prevents the three services from sharing one toolchain and one copy of every CUDA library; and the combined footprint pushed the box into a zram swap spiral where the TTS was OOM-killed and the machine needed a power reset.

On a device where the OS and GPU share one small RAM pool, container isolation buys nothing: it is a single-tenant, single-purpose appliance, and isolation costs the one scarce resource. Native systemd units with `MemorySwapMax=0` are the difference between fitting and not.

## Power

Whole-board input power at `MAXN_SUPER`, clocks un-pinned, sampled at 0.5 s on 2026-10-14 with the `balanced` profile:

| state | VDD_IN | note |
|---|---|---|
| boot idle (services stopped) | ~4.8 W | 4.77–4.85 W measured |
| stack resident, no traffic | ~4.8 W | three models in RAM, GPU idle; resident weights cost almost nothing at idle |
| LLM decode | ~17 W peak | brief bursts; ~16.5–17 W sustained alongside TTS synthesis, well under the 25 W `MAXN_SUPER` envelope |

With the old `jetson_clocks` pinning (EMC held at max, CPU idle states disabled), stack-resident idle measured ~5.95 W. The ~1.15 W the lock burned while doing nothing is why it was removed; decode measured the same either way, and EMC boosts from 2.13 to 3.2 GHz under load on its own. Profile choice barely moves idle power: resident versus services-stopped differed by ~0.04 W in the same session.

In practice an always-on voice assistant costs about 5 W idle with brief 17 W bursts during replies. At €0.30/kWh that is roughly €13/year for the compute; NVMe, a display, or other peripherals add a little. Speech-completion latency, not power, is the binding constraint on this box.

## Design decisions

The guiding rule: maximum performance inside a hard 8 GB envelope, while tolerating a slightly slower reply when quality would otherwise suffer. Every component below is the highest-quality option that still fits and keeps the conversation fluid.

### LLM

| decision | chosen | rejected | why |
|---|---|---|---|
| model size | 1.2B / 2.6B / 8B-A1B (`fast` / `balanced` / `thorough`) | 7B dense, 20B+ | the three profiles are the LFM2.5 sizes that fit alongside STT and TTS; dense 8B-class weights do not fit at usable quants |
| target quant | QAD-Q4_0 (dense), UD-IQ2_M (8B-A1B) | Q4_K_M, Q5_K_M for the 8B | QAD is LiquidAI's quality-preserving quantization-aware distill; only IQ2-class quants fit for the 8B, and UD-IQ2_M won the four-quant shootout |
| draft model | DSpark-Q8_0 | no draft, F16 or Q4 draft | speculative decoding is the one lever past the batch-1 streaming cap, and it is quality-neutral; acceptance is drafter-quant invariant, so Q8_0 wins on cost, and an F16 drafter is a measured net loss on the iGPU |
| KV cache | q8_0 at 16k context | f16, 32k | q8_0 halves KV RAM with no observed quality hit; 16k keeps the long HA prompt and history with 4× headroom while saving ~350 MB |
| reasoning budget | 256 tokens | 0 / unlimited | bounded scratch thinking keeps some reasoning quality while keeping reply latency predictable; unlimited thinking blows the voice budget |
| runtime | llama.cpp, single slot | vLLM, batching | vLLM's pool and scheduler overhead measured 2.8–3.3 GB earlier; single-caller makes batching pure loss |

The net effect is a slightly slower per-token rate than a model chosen to be fast at all costs, accepted only where it buys answer quality. Speculative decoding is the opposite trade: free speed that does not touch quality, so it is on by default.

### STT

| decision | chosen | rejected | why |
|---|---|---|---|
| model | Whisper tiny/en | small, medium | larger models add RAM and latency for gains that matter little on clean near-field audio; tiny is highly accurate on clear speech |
| language | fixed `en` | auto-detect | skipping language detection saves a model pass and removes misdetection flaps |
| compute type | int8 | float16 | int8 on Orin is ~2× faster with negligible WER change for tiny |
| beam size | 5 | greedy (1) | beam 5 measurably improves name and command recognition for pennies in latency |
| streaming | Wyoming chunked | whole-utterance only | chunks start decoding as audio arrives |

### TTS

| decision | chosen | rejected | why |
|---|---|---|---|
| engine | Kokoro-82M | Piper, espeak | Kokoro's prosody and naturalness are a clear tier above Piper at comparable cost |
| precision | fp16 | int8, fp32 | int8 is *slower* on this GPU because quantized ops fall back to CPU, and it dulls prosody; fp32 wastes RAM for no audible gain |
| runtime | ONNX Runtime CUDA | PyTorch, CPU | the PyTorch path measured +1.5–1.9 GB and OOM'd the board; CPU synthesis is slower than real time (rtf 1.4–1.6) |
| delivery | Wyoming streaming | buffer everything | speech starts at the first sentence boundary while the LLM is still generating; buffering adds the full synthesis wait to every reply |
| arena cap | 1024 MB | unbounded | caps the GPU/RAM arena so the LLM's pool cannot be squeezed out |

### System

| decision | chosen | rejected | why |
|---|---|---|---|
| packaging | systemd + pinned native builds | Docker images | the daemon and library duplication cost hundreds of MB and pushed the box into a swap spiral |
| swap policy | `MemorySwapMax=0` | default swapping | with 8 GB, any swapped service page is latency jitter; nothing may swap, and the budget guarantees nothing has to |
| power mode | `MAXN_SUPER` (nvpmodel, clocks un-pinned) | `jetson_clocks` pinning, lower modes | `MAXN_SUPER` raises the ceilings so decode boosts under load; pinning clocks high cost ~1.15 W constant idle power for no measured decode gain |

## Rejected experiments

These were measured on the box and lost. Do not re-run them without a new reason.

- **A larger quant for the 8B-A1B.** Q4_0 does not fit (watchdog kill at 72 MB free, decode collapse), IQ3_XXS swaps and leaves no TTS headroom, and Q2_K_XL is slower, less accurate, and larger than UD-IQ2_M.
- **F16 drafter for higher acceptance.** Acceptance is drafter-quant invariant; on the iGPU, dequant cost outweighs any gain. Q8_0 is the ship.
- **`p-min` truncation on the dense model.** It drops accepted tokens for nothing. It helps the MoE, where low-confidence tails cost full expert traffic.
- **ngram, `p-split`, and backend sampling on `fast`.** Neutral or worse at 2.99 accepted tokens per step; the native drafter's trained block length wins.
- **Target-fidelity sweeps.** Q4_0 through Q8_0 measured flat on acceptance, so a higher-fidelity target costs speed without buying anything.
- **int8 Kokoro.** Slower than fp16 on this GPU and duller. `q8f16` was also rejected: it saves ~70 MB, but kokoro-onnx 0.5.0 sends `speed` as int32 while that export expects float, and activations are unchanged.
- **vLLM.** Pool and scheduler overhead measured 2.8–3.3 GB; single-caller batching is pure loss.
- **Docker.** See [Why native and not Docker](#why-native-and-not-docker).
- **Pinned clocks.** ~1.15 W constant idle burn for no measured decode gain.
- **Micro-knobs that did not help:** `GGML_CUDA_MMQ_PREC=q4`, `GGML_CUDA_PDL=1`, `--poll 1` (busy-wait contends with GPU sync), `--ctx-checkpoints 0`, `--spec-draft-p-split 0.0`, and `--no-spec-draft-backend-sampling`.
- **More than one slot, or more than one LLM.** LFM2 short-conv state corrupts under co-batching (llama.cpp #29002), and two resident LLMs leave 105 MB free.

## Reproducing the numbers

Run the frozen battery against the live service:

```bash
python3 bench/ha_benchmark.py --base-url http://127.0.0.1:8080 \
    --model <alias from /v1/models> --label my-run
```

Archive the result as `bench/results/ha_<LETTER>-<label>.json` and never edit an old result. For a like-for-like comparison, keep context, reasoning budget, draft settings, battery, hardware, and power state identical. The battery is only comparable because it is frozen.

The older interactive sweeps are `bench/bench_openai.py`, `bench/quality_check.py`, and `bench/matrix.sh`. `scripts/fetch-models.sh` pulls extra GGUFs, and `scripts/quantize-8b-variants.sh` holds the custom 8B recipes.

More of the measured history, platform quirks, and the reasoning behind the defaults is in [AGENTS.md](AGENTS.md).
