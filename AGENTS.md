# AGENTS.md — conventions and institutional knowledge for super-soaker

Read this before changing code, docs, or the deployed stack. The README documents the system as shipped; this file documents how to work on it without rediscovering everything the hard way.

## Documentation style: soft wrap, never hard wrap

All prose in markdown files (README.md, AGENTS.md, notes/*.md) uses **soft wrapping**: one paragraph = one line, however long. Do not reflow paragraphs to 80 columns. Tables and code blocks wrap naturally; paragraphs do not. This was a deliberate reflow (see git history) — editors handle long lines fine, and diffs stay readable.

## Repository conventions

- **Three services, three directories**: `services/1-stt-*`, `2-llm-llama.cpp-super-soaker-managed`, `3-tts-*`. The numeric prefix sets pipeline order and startup order. The LLM service directory is deliberately model-agnostic — the model is chosen by `LLM_MODEL=` in `/etc/super-soaker/llm.env`, never by renaming the service. Do not put model names back in service paths.
- **Everything pinned**: llama.cpp by commit (`b11552` / `23b0202a1`), CTranslate2 4.5.0, ONNX Runtime 1.22.0, every Python package exact-version, every model URL paired with its SHA-256 in `bootstrap.sh`. If you bump a pin, bump the hash, and re-run the self-tests end to end.
- **bootstrap.sh is idempotent and re-runnable**: env files in `/etc/super-soaker/` are written once and preserved; model fetches skip when hash-verified; service dirs renamed/removed in the repo are pruned from `/opt/super-soaker/services` on re-run. Re-running on a live node is the normal way to switch profiles (`--model`).
- **Model profiles live in exactly two places**: the fetch case in `bootstrap.sh` (URLs + hashes) and the `case "$LLM_MODEL"` block in `services/2-llm-llama.cpp-super-soaker-managed/serve.sh` (paths + CTX + reasoning budget). A new profile needs both, plus a README row. Keep the three profile names (`fast`, `balanced`, `thorough`) stable — they appear in operators' env files. (They were renamed 2026-10-14 from `lfm2.5-1.2b-chat`/`lfm2.5-2.6b-reasoning`/`lfm2.5-8b-a1b-reasoning`; both scripts still map the legacy names so old env files keep working and bootstrap heals them on re-run.)
- **Service naming for humans**: units are `super-soaker-{stt,llm,tts,clocks}.service`; the directory names encode pipeline order and function. User-facing docs should quote the directory names.

## Testing conventions

- **Self-tests are stdlib-only Python**, run at every service start (`SELF_TEST=on` in the env files) and must pass for the service to stay up. They exercise the real end-to-end path (CUDA warmup, a real inference round trip, tool-calling as the HA integration does it, prompt-cache behavior). If you change serving parameters, run all three by hand (commands in the README) before declaring victory.
- **The model battery** is `bench/ha_benchmark.py` + `bench/ha_battery.json`: 20 tool scenarios (including no-tool negatives that punish over-triggering) + 10 factual/brevity scenarios. Identical prompts and `max_tokens` for every profile — the numbers are only comparable because the battery is frozen. Archive results as `bench/results/ha_<LETTER>-<label>.json` and never edit old result files.
- **When comparing models, control the confounders**: same CTX, same reasoning budget, same draft/speculative settings, same battery, same box, clocks locked. The 8B-A1B profile intentionally matches the 2.6B's `CTX=32768` / `REASONING_BUDGET=128` so the comparison is about the model, not the knobs.
- **Shell edits get `bash -n`** before deploy. There is no linter; `shellcheck`-clean is nice but not enforced.

## Measurement methodology (how the numbers in the README were obtained)

- **Memory**: `tegrastats` `RAM used/total` or `MemTotal − MemFree − Cached` deltas, not RSS — the Jetson iGPU driver (`nvgpu`) allocates kernel memory that per-process RSS cannot see. A `llama-server` at 700 MB RSS can be sitting on 4+ GB of real footprint. `VmSwap` in `/proc/<pid>/status` is the "is anything swapped" oracle; the services are pinned `MemorySwapMax=0` so the answer must be zero.
- **Reboot between memory experiments.** Stopping llama-server does not reliably return its memory immediately: the nvgpu pool and page cache linger and inflate the next measurement (a stopped stack once read 4.5 GB used; after reboot, 2.2 GB). If a measurement looks wrong by ~2 GB, suspect stale pool, not the model.
- **`/tmp` is wiped on reboot** on the target OS. Keep measurement harnesses and results in `/var/tmp/` when the experiment involves reboots.
- **When loading an unknown-size model on an 8 GB box, run a watchdog**: a small loop that kills the server if available RAM drops under ~80 MB or swap grows past baseline. The watchdog saved the box during the Q4_0 attempt (decode collapsing into zram thrash); without it the attempt ends in a power reset.
- **Power**: `tegrastats` `VDD_IN` is whole-board input power. Sample at 0.5–1 s and report min/median/max; idle numbers are only meaningful with the full stack resident and clocks locked.

## The 8B-A1B quant hunt (why the profile uses UD-IQ2_M)

LFM2.5-8B-A1B is 8.3B total / 1.5B active (MoE: 32 experts, 4 active per token, gated short-conv LFM2 hybrid, reasoning-only). Total params drive RAM, active params drive speed — so it runs fast (10.5 w/s decode on this box) but needs its whole quant in RAM alongside STT+TTS. Four quants were measured on-box with the real battery (CTK/CTV q8_0, DSpark-Q8_0 draft, 32k ctx, reasoning 128):

| quant | size | peak RAM | swap | tool acc | quality | verdict |
|---|---|---|---|---|---|---|
| LiquidAI Q4_0 | 4.84 GB | — | thrash | — | — | **does not fit**: watchdog kill at 72 MB free; decode collapsed 33 → 2.4 t/s |
| Unsloth UD-IQ3_XXS | 3.27 GB | 7.49/7.62 GB | 321 MB | 95% | 100% | best quality, but violates no-swap and leaves no room for a TTS peak |
| Unsloth UD-Q2_K_XL | 2.93 GB | 7.27/7.62 GB | 3 MB | 90% | 90% | fits, but dominated by UD-IQ2_M on every axis |
| **Unsloth UD-IQ2_M** | **2.75 GB** | **7.04/7.62 GB** | **0** | **100%** | **90%** | **chosen**: zero swap, ~580 MB headroom, fastest decode |

The counter-intuitive part: the smaller quant won on accuracy too (100% tool). Battery noise is real at n=1 per scenario, but the takeaway stands — on this box the 8B-A1B only has one honest fit, and IQ2-class quants are not automatically broken for MoE models (only 4 of 32 experts activate per token; a lot of the weights are rarely hot). If someone proposes "just use Q4_K_M", the answer is the table above.

Prefer **Unsloth or LiquidAI official** GGUFs. Third-party requantized uploads vary in quality and provenance; if a new quant is needed, prefer `unsloth/LFM2.5-8B-A1B-GGUF` (Unsloth Dynamic quants) or `LiquidAI/*-GGUF` first.

## Performance quirks worth not re-discovering

- **Speculative decoding (DSpark) is quality-neutral speed** — the target distribution is unchanged, so there is never a reason to turn it off on a memory-fit profile. Measured ~29–30% of drafted tokens accepted on the 8B-A1B (2.7–3.1 accepted per verification step). The drafter is only ~0.36 GB at Q8_0 — always take it.
- **Quantized ops can be slower than fp16 on this GPU.** The TTS int8 model was measurably slower than fp16 because quantized kernels fall back to CPU. "Smaller = faster" is false on Orin; measure every quant change.
- **CUDA graphs are load-bearing** (`GGML_CUDA_GRAPHS=ON`): decode speed on the iGPU drops sharply without them. `FA_ALL_QUANTS` keeps flash-attention available for every quant.
- **`GGML_CUDA_NO_VMM=ON` is not optional** on Jetson — the driver's virtual-memory-manager path reserves ~32 GB of address space and crashes (llama.cpp #29142).
- **Single slot (`-np 1`) is required** for LFM2: short-conv state corrupts under co-batching (llama.cpp #29002). Fine anyway — this stack has one caller.
- **Prompt cache (`--cache-reuse 256`) is the warm-path magic**: the static HA Assist prefix is reused across turns; cold prefill of the full prompt is ~15 s vs ~0.3 s warm TTFT. Don't let context changes churn the prefix.
- **KV cache at q8_0** halves RAM vs f16 with no observed quality hit at 32k ctx. f16 KV is a rejected trade here.
- **Bounded reasoning (`--reasoning-budget 128`)** keeps reply latency predictable. The 8B-A1B is reasoning-*only* — it always thinks first — so the budget is what makes it voice-usable at all.
- **Words/s vs tok/s**: the battery reports words/s; llama-server logs tok/s. For English text, roughly 1 w/s ≈ 1.3–1.5 t/s. Compare like with like.

## The 8B-A1B decode-optimization campaign (2026-10, full log in notes/04)

- **Batch-1 GEMVs stream at a fixed efficiency on this GPU**: ~24% of LPDDR5 peak for IQ-LUT quants (IQ2_XXS/IQ3_XXS/IQ2_S), ~37% for block quants (q2_0/q4_0/q2_k). Dequant cost hides under memory stalls at batch 1 (PR #26705 shows the same on DGX Spark); speed ≈ bytes/token ÷ that rate. This is why "just requantize lighter" only buys 1.3–1.6x and why the oQ4-style "smaller is faster" intuition is wrong here.
- **The quality cliff is expert-type + protection, and it is sharp**: IQ2_XXS/IQ3_XXS experts → 100% tool acc; Q2_K experts → 80%; Q2_0/TQ2_0 → destroyed. Attention/embeddings below Q5_K/Q6_K → tool emission breaks ("no tool call" failures) before factual quality does. If you change the quant recipe, gate against `bench/ha_battery.json` — synthetic perplexity will not tell you this.
- **Spec-decode is the only escape from the batch-1 cap — via acceptance.** Verify batches become fat GEMMs and amortize the tied 128K output head. Economics differ sharply by architecture (measured both, 2026-10): **dense** (2.6B) breaks even at ~1.2 accepted/step and earns profit above that — same drafter gives 2x baseline / 4x templated; **4-of-32 MoE** (8B-A1B) verify traffic saturates (union ≈ 23–27/32 experts at k=9) and needs ≥5 accepted/step. So `n_max` and `p-min` must be tuned per model, not copied: dense wants n_max 7 and NO p-min truncation (it loses accepted tokens for nothing); MoE wants n_max 5 and p-min 0.4 (prunes tails that cost full expert traffic).
- **Acceptance is drafter-vs-target-family limited, not quant-noise limited**: invariant across drafter quants (F16/Q4_K/Q8_0 — so always ship Q8_0) and target quants (Q4_0→Q8_0 flat). The drafter was distilled against F16 (llama.cpp #25618); templated traffic sidesteps it via ngram-mod (comma-combine with draft-dspark). Do not re-run target-fidelity sweeps expecting acceptance gains.
- **The DSpark drafter was distilled against an F16 target** (llama.cpp #25618): on quantized targets it drifts on novel content (acceptance 2–3/step vs Liquid's 6.95/10 on F16). F16 drafter is a net loss on iGPU (dequant cost > acceptance gain — Strix-Halo datapoint in PR #27383 agrees). Higher fidelity on the drafter-read layers (2/6/10/14/18) helps templated content only, and costs RAM/quality.
- **A local mmvq.cu patch** (`deps/patches/mmvq-orin-crossover.patch`, applied by `deps/build-llamacpp.sh`) extends the Orin MMVQ→MMQ crossover to IQ/simple quants — upstream PR #28285 covered K-quants only. +4–10% on verify batches, no batch-1 change. Submit upstream if it survives longer.
- **Custom quants live in `scripts/quantize-8b-variants.sh`** (llama-quantize `--tensor-type` per-tensor recipes + unsloth imatrix; watch first-match-wins pattern ordering). All 11 variants were built and gated on the box; none beat UD-IQ2_M on tool accuracy. The BF16 source + imatrix are not fetched by default (17 GB).
- **A model's GGUF can embed the WRONG chat-template variant.** If a reasoning model leaks thinking into `message.content` or ignores `--reasoning-budget`, its embedded chat template is likely the non-reasoning variant — llama.cpp's reasoning extraction keys on the markers in the reasoning-capable template, not on flags (regression class: llama.cpp #28675). Compare the embedded template against `ggml-org/llama.cpp/models/templates/` first — do not assume the GGUF ships the right one. Fix: serve with `--chat-template-file` pointed at the reasoning-capable family reference + `--reasoning-format deepseek` (`TEMPLATE_FILE`/`REASONING_FORMAT` knobs exist in serve.sh). (`--reasoning-format`/`-rea` flags alone do NOT fix it — the template content is what the parser keys on.)
- **A mismatched DSpark drafter/target pair is a HARD CRASH**, not a soft incompatibility: llama-server aborts at load (`GGML_ASSERT(ggml_can_repeat)` in `build_dspark_markov_head`) when the `-md` drafter belongs to a different model than `-m` (e.g. 8B target + 1.2B drafter). Drafters are per-model pairs; always ship the matching DSpark repo file. Same-family different-quant pairs are fine.
- **Quantization is not always the accuracy limit.** For LFM2.5-1.2B-Instruct the frozen battery scores identical tool/quality at QAD-Q4_0, Q8_0, and F16 (85/70 at all three) — the cap is the model. Measure the battery across fidelities before blaming quants for accuracy (and before paying the speed cost of stepping up: 99.9 → 83.1 → 68.1 tok/s novel).
- **Micro-knobs that did NOT help**: `GGML_CUDA_MMQ_PREC=q4`, `GGML_CUDA_PDL=1`, `--poll 1` (busy-wait contends with GPU sync on 6 A78AE cores — measurably worse), `--ctx-checkpoints 0`, `--spec-draft-p-min`, `--spec-draft-p-split 0.0`, `--no-spec-draft-backend-sampling` (slightly worse). Do not re-run these without a reason.
- **Measure acceptance, not just tok/s**: `curl /metrics | grep spec_decode` (drafted/accepted/drafts counters + per-position breakdown) is the fastest diagnostic for "why did spec stop helping".

## Jetson-specific things to keep in mind

- **Unified memory**: the GPU is an iGPU with no dedicated VRAM. "GPU memory" is system RAM, and the driver's allocations live in kernel space where `docker stats`/RSS can't see them. Budget in terms of board RAM, always.
- **8 GB is the whole budget** — OS (~0.8 GB) + STT (~0.7 GB) + TTS (~1.1–1.5 GB) is ~3 GB before the LLM has loaded a byte. Measure candidate models *with the voice stack resident*, not on an idle box.
- **Power modes matter for latency**: `MAXN_SUPER` + `jetson_clocks` locks clocks; without it, first-token latency wobbles as the DVFS governor ramps. The `super-soaker-clocks.service` unit exists to hold this at boot.
- **Whole-board power is modest**: measured ~6.9 W with the stack resident idle, ~16.5–17 W during LLM decode + TTS synthesis. Cooling headroom is rarely the problem; RAM is.
- **`/dev/nvhost*` and `renderD128` access**: service users need the `video` **and** `render` groups or CUDA silently reports zero devices and everything downstream fails with unhelpful errors.
- **L4T ties your CUDA version to your driver**: you cannot load a CUDA-13-linked binary on a CUDA-12.6 L4T driver. This is why ONNX Runtime is built locally at 1.22 (the PyPI aarch64 CUDA wheels don't exist anyway, and 1.29+ links CUDA 13).
- **nvcc is RAM-hungry**: ONNX Runtime's flash-attention kernels peak near 3.5 GB RSS per compile thread. `JOBS=1` (2 max) on 8 GB — the build scripts enforce it; `--force-jobs` is for 16 GB devices. High parallelism has wedged this class of machine into a swap spiral requiring a power reset.
- **zram swap exists on the stock image** (3.8 GB across 6 devices). Under memory pressure the box does not OOM immediately — it thrashes into zram and latency dies first. Thrash is the failure mode to watch for, not the OOM killer.
- **Kernel/systemd version quirks**: `MemorySwapMax=0` works on this L4T systemd; `sysctl --system` may print harmless "Invalid argument" lines for keys the Tegra kernel lacks — ignore.

## Why not Docker (and don't let anyone relitigate it)

Docker was tried first and is a dead end on this hardware. The daemon plus per-container library duplication costs several hundred MB of the 8 GB before any model loads; the container boundary prevents the three services from sharing one toolchain and one copy of every CUDA library; and the combined footprint pushed the box into a zram swap spiral where the TTS got OOM-killed and the machine required a power reset. On a device where the OS and the GPU *share* one small RAM pool, container isolation buys nothing (single tenant, single purpose) and costs the one resource that is scarce. Native systemd units with `MemorySwapMax=0` are the difference between "fits" and "doesn't".

## Process notes from building this

- **Measure first, theorize second.** Several "obvious" conclusions were wrong on the box: int8 quantization slowed TTS down; a smaller LLM quant beat a larger one on accuracy; a service that was "stopped" still held ~2 GB. Every configuration claim in the README is backed by a measurement on the actual hardware.
- **Benchmark the candidates head-to-head before committing** (same battery, same knobs), then pick — and archive the losing measurements with their settings so the decision is reproducible. `bench/results/` and the quant table above exist so nobody has to rerun the shootout.
- **Use a memory watchdog when pushing the RAM envelope**, and reboot for clean baselines between runs.
- **Keep infrastructure tests cheap and honest**: the self-tests run at every service start and must fail when the service genuinely can't serve. A self-test that passes on a broken stack is worse than none.
- **Pin everything at the moment it works**, including model hashes from the exact bytes you validated — HF "main" branches move.
