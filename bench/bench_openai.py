#!/usr/bin/env python3
"""super-soaker real-world OpenAI-compatible benchmark suite (stdlib only).

Measures what Home Assistant / hass_local_openai_llm actually experiences:
TTFT, decode tok/s, prefill tok/s, multi-turn prompt-cache behavior,
tool-call round trips — using streaming SSE exactly like the integration.

Usage:
  python3 bench/bench_openai.py --base-url http://127.0.0.1:8080 \
      --model LFM2.5-2.6B --label q4km-spec --reps 3 --out-dir results

Scenarios:
  short_qa_stream    short prompt, streaming (the snappy-voice core metric)
  short_qa_nostream  same, non-streaming (total latency)
  ha_assist_cold     ~10k-token Home-Assistant-like system prompt, first use
  ha_assist_warm     identical request again (prompt-cache hit = steady state)
  ha_assist_delta    multi-turn voice conversation, growing history
  tool_call          tool schema + tool_call response, HA-style
  long_prompt        4k-token user message (prefill speed probe)
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------
# deterministic synthetic Home-Assistant-like context
# ----------------------------------------------------------------------------

def ha_system_prompt(n_entities=280, seed=42):
    rng = random.Random(seed)
    areas = ["kitchen", "living_room", "bedroom", "office", "garage", "garden",
             "bathroom", "hallway", "basement", "attic", "dining_room", "porch"]
    kinds = [
        ("light", "Light", "dimmable light"),
        ("switch", "Switch", "on/off switch"),
        ("sensor", "Sensor", "temperature sensor"),
        ("binary_sensor", "Binary sensor", "door sensor"),
        ("climate", "Climate", "thermostat"),
        ("cover", "Cover", "blinds"),
        ("media_player", "Media player", "speaker"),
        ("fan", "Fan", "ceiling fan"),
    ]
    lines = [
        "You are a voice assistant for a smart home. Answer briefly and "
        "clearly for text-to-speech. Use the exposed entities below when a "
        "command targets a device. Prefer the most specific matching entity.",
        "",
        "Exposed entities:",
    ]
    for i in range(n_entities):
        kind, cname, desc = kinds[i % len(kinds)]
        area = areas[i % len(areas)]
        name = f"{area.replace('_', ' ')} {desc} {i + 1}"
        lines.append(
            f"- {kind}.{area}_device_{i + 1}: {name} (area: {area.replace('_', ' ')}, "
            f"id: {kind}.{area}_device_{i + 1}, states: on/off, "
            f"attributes: brightness, color_temp, friendly_name: {name})")
    lines += [
        "",
        "Rules:",
        "1. If the user requests a device action, call the appropriate tool.",
        "2. Never invent entities that are not listed above.",
        "3. Keep answers under three sentences unless asked for detail.",
        "4. If a request is ambiguous, ask one short clarifying question.",
    ]
    return "\n".join(lines)


HA_TOOLS = [{
    "type": "function",
    "function": {
        "name": "execute_services",
        "description": "Call a Home Assistant service to control devices.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Service domain, e.g. light"},
                "service": {"type": "string", "description": "Service name, e.g. turn_on"},
                "target": {"type": "object",
                           "properties": {"entity_id": {"type": "string"}},
                           "description": "Target entity"},
                "data": {"type": "object", "description": "Service data"},
            },
            "required": ["domain", "service", "target"],
        },
    },
}]

# ----------------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------------

def stream_chat(url, payload, timeout=300):
    payload = dict(payload, stream=True)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    parts = []
    timings = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            for ch in (obj.get("choices") or []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    parts.append(piece)
            if obj.get("timings"):
                timings = obj["timings"]
    return {"ttft_s": ttft, "total_s": time.perf_counter() - t0,
            "text": "".join(parts), "timings": timings}


def chat_json(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode())
    out = {"ttft_s": None, "total_s": time.perf_counter() - t0,
           "text": "", "timings": obj.get("timings"),
           "tool_calls": None}
    msg = (obj.get("choices") or [{}])[0].get("message", {})
    out["text"] = msg.get("content") or ""
    out["tool_calls"] = msg.get("tool_calls")
    return out


def mem_snapshot():
    snap = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("MemAvailable", "MemTotal", "SwapFree", "SwapTotal"):
                    snap[k] = int(v.split()[0])  # kB
    except OSError:
        pass
    for p in ("/sys/fs/cgroup/memory.current",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(p) as f:
                snap["cgroup_bytes"] = int(f.read().strip())
                break
        except OSError:
            continue
    return snap


# ----------------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------------

def base_payload(model, messages, max_tokens, tools=None):
    p = {"model": model, "messages": messages, "max_tokens": max_tokens,
         "temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1}
    if tools:
        p["tools"] = tools
        p["tool_choice"] = "auto"
    return p


def run_reps(name, n, fn, extra=None):
    runs = []
    for i in range(n):
        r = fn(i)
        r["name"] = f"{name}#{i+1}"
        if extra:
            r.update(extra)
        runs.append(r)
        t = r.get("timings") or {}
        print(f"  {r['name']}: ttft={r['ttft_s'] and round(r['ttft_s'],3)}s "
              f"total={round(r['total_s'],3)}s "
              f"prompt={t.get('prompt_n')}@{t.get('prompt_ms')}ms "
              f"pred={t.get('predicted_n')}@{round(t.get('predicted_per_second') or 0,1)}tps "
              f"draft={t.get('draft_n')}/{t.get('draft_n_accepted')}")
    return runs


def summarize(runs):
    def vals(key, sub=None):
        out = []
        for r in runs:
            if sub:
                t = r.get("timings") or {}
                v = t.get(sub)
            else:
                v = r.get(key)
            if isinstance(v, (int, float)):
                out.append(v)
        return out

    ttfts = [v for v in vals("ttft_s") if v]
    totals = vals("total_s")
    tps = vals(None, "predicted_per_second")
    ptps = vals(None, "prompt_per_second")
    acc = vals(None, "draft_n_accepted")
    dn = vals(None, "draft_n")
    s = {}
    if ttfts:
        s["ttft_ms_med"] = round(statistics.median(ttfts) * 1000)
        s["ttft_ms_min"] = round(min(ttfts) * 1000)
    if totals:
        s["total_ms_med"] = round(statistics.median(totals) * 1000)
    if tps:
        s["decode_tps_med"] = round(statistics.median(tps), 1)
        s["decode_tps_max"] = round(max(tps), 1)
    if ptps:
        s["prefill_tps_med"] = round(statistics.median(ptps), 1)
    if acc and dn:
        s["draft_accept_med"] = round(statistics.median(acc), 2)
        s["draft_n_med"] = round(statistics.median(dn), 2)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="LFM2.5-2.6B")
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--scenario", action="append",
                    help="subset; default all. repeatable")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    chat_url = f"{base}/v1/chat/completions"
    want = set(args.scenario) if args.scenario else None

    sys_prompt = ha_system_prompt()
    approx_prompt_tokens = len(sys_prompt) // 4
    print(f"[bench] label={args.label} url={base} "
          f"HA system prompt ≈{approx_prompt_tokens} tokens")

    report = {"label": args.label, "base_url": base, "model": args.model,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "scenarios": {}, "mem": {}}
    report["mem"]["before"] = mem_snapshot()

    scenarios = []

    def scen(name):
        return want is None or name in want

    if scen("short_qa_stream"):
        print("[bench] short_qa_stream")
        msgs = [{"role": "user", "content": "What is the capital of France?"}]

        def f(i):
            return stream_chat(chat_url, base_payload(
                args.model, msgs, args.max_tokens))
        report["scenarios"]["short_qa_stream"] = {
            "runs": run_reps("short_qa_stream", args.reps, f)}

    if scen("short_qa_nostream"):
        print("[bench] short_qa_nostream")
        msgs = [{"role": "user", "content": "What is the capital of France?"}]

        def f(i):
            return chat_json(chat_url, base_payload(
                args.model, msgs, args.max_tokens))
        report["scenarios"]["short_qa_nostream"] = {
            "runs": run_reps("short_qa_nostream", args.reps, f)}

    if scen("ha_assist_cold") or scen("ha_assist_warm"):
        print("[bench] ha_assist_cold/warm")
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Turn on the kitchen light."}]

        def f(i):
            return stream_chat(chat_url, base_payload(
                args.model, msgs, args.max_tokens, tools=HA_TOOLS))
        cold = run_reps("ha_assist_cold", 1, f)
        warm = run_reps("ha_assist_warm", max(args.reps - 1, 1), f)
        if scen("ha_assist_cold"):
            report["scenarios"]["ha_assist_cold"] = {"runs": cold}
        if scen("ha_assist_warm"):
            report["scenarios"]["ha_assist_warm"] = {"runs": warm}

    if scen("ha_assist_delta"):
        print("[bench] ha_assist_delta (multi-turn voice conversation)")
        history = [{"role": "system", "content": sys_prompt}]
        turns = ["Turn on the kitchen light.",
                 "Set it to 50 percent brightness.",
                 "Also turn on the living room speaker.",
                 "What is the temperature in the bedroom?",
                 "Thanks, good night."]

        def make_fn(hist):
            def f(i):
                return stream_chat(chat_url, base_payload(
                    args.model, hist, args.max_tokens, tools=HA_TOOLS))
            return f
        runs = []
        for ti, turn in enumerate(turns):
            history.append({"role": "user", "content": turn})
            r = make_fn(list(history))(ti)
            r["name"] = f"ha_assist_delta#{ti+1}"
            runs.append(r)
            history.append({"role": "assistant", "content": r["text"][:400] or "ok"})
            t = r.get("timings") or {}
            print(f"  turn {ti+1}: ttft={r['ttft_s'] and round(r['ttft_s'],3)}s "
                  f"prompt={t.get('prompt_n')} pred={t.get('predicted_n')}")
        report["scenarios"]["ha_assist_delta"] = {"runs": runs}

    if scen("tool_call"):
        print("[bench] tool_call")
        msgs = [{"role": "system",
                 "content": "You are a home assistant. Use tools to act."},
                {"role": "user", "content": "Turn on the kitchen light."}]

        def f(i):
            return chat_json(chat_url, base_payload(
                args.model, msgs, args.max_tokens, tools=HA_TOOLS))
        runs = run_reps("tool_call", args.reps, f)
        ok = sum(1 for r in runs if r.get("tool_calls"))
        print(f"  tool_calls produced: {ok}/{len(runs)}")
        report["scenarios"]["tool_call"] = {"runs": runs, "tool_call_ok": ok}

    if scen("long_prompt"):
        print("[bench] long_prompt (prefill probe)")
        filler = ("The home automation system monitors sensors and reports "
                  "readings every minute. ") * 220  # ~4k tokens
        msgs = [{"role": "user",
                 "content": filler + "\n\nSummarize this in one sentence."}]

        def f(i):
            return stream_chat(chat_url, base_payload(
                args.model, msgs, args.max_tokens))
        report["scenarios"]["long_prompt"] = {
            "runs": run_reps("long_prompt", args.reps, f)}

    report["mem"]["after"] = mem_snapshot()

    for name, sc in report["scenarios"].items():
        sc["summary"] = summarize(sc["runs"])

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.out_dir, f"{stamp}-{args.label}.json")
    md_path = os.path.join(args.out_dir, f"{stamp}-{args.label}.md")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    lines = [f"# bench {args.label} ({report['started']})", "",
             "| scenario | TTFT med (ms) | total med (ms) | decode tok/s med | "
             "prefill tok/s med | draft accept |",
             "|---|---|---|---|---|---|"]
    for name, sc in report["scenarios"].items():
        s = sc["summary"]
        lines.append(
            f"| {name} | {s.get('ttft_ms_med', '-')} | {s.get('total_ms_med', '-')} "
            f"| {s.get('decode_tps_med', '-')} | {s.get('prefill_tps_med', '-')} "
            f"| {s.get('draft_accept_med', '-')} |")
    lines += ["", "## memory", "",
              f"- before: {report['mem']['before']}",
              f"- after:  {report['mem']['after']}"]
    md = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md)
    print(md)
    print(f"[bench] wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
