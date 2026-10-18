#!/usr/bin/env python3
"""super-soaker end-to-end self-test (stdlib only).

Runs against a local OpenAI-compatible endpoint after boot:
  1. model list sanity
  2. warmup completions (pays CUDA/graph warmup cost, verifies inference)
  3. streaming speed measurement (TTFT + decode tok/s)
  4. tool-calling round trip (what hass_local_openai_llm needs)
  5. multi-turn prompt-cache probe

Exit code 0 = pass. Writes JSON results to --out (default /tmp/selftest.json).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

MIN_TPS_FAIL = 8.0   # below this, inference is broken (or not on GPU)
MIN_TPS_WARN = 15.0


def http_json(url, payload, timeout=120):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stream_chat(url, payload, timeout=180):
    """POST chat completion with stream=true. Returns (ttft_s, text, timings, chunks)."""
    payload = dict(payload, stream=True)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    text_parts = []
    timings = None
    chunks = 0
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
            chunks += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    text_parts.append(piece)
            if obj.get("timings"):
                timings = obj["timings"]
    total = time.perf_counter() - t0
    return ttft, "".join(text_parts), timings, total, chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="LFM2.5-2.6B")
    ap.add_argument("--out", default="/tmp/selftest.json")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    chat_url = f"{base}/v1/chat/completions"

    report = {"base_url": base, "model": args.model, "runs": [], "pass": True,
              "warnings": []}
    failures = []

    def fail(msg):
        failures.append(msg)
        report["pass"] = False
        print(f"[selftest] FAIL: {msg}")

    # 1. /v1/models
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
            models = json.loads(r.read().decode())
        ids = [m.get("id") for m in models.get("data", [])]
        print(f"[selftest] models: {ids}")
        report["models"] = ids
        if not ids:
            fail("no models listed on /v1/models")
    except Exception as e:
        fail(f"/v1/models unreachable: {e}")
        print(json.dumps(report, indent=2))
        return 1

    def chat_once(messages, name, stream=True, tools=None, max_tokens=128,
                  tool_choice=None):
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        try:
            if stream:
                ttft, text, timings, total, chunks = stream_chat(chat_url, payload)
                run = {"name": name, "ttft_s": round(ttft, 3),
                       "total_s": round(total, 3), "text": text[:200],
                       "timings": timings, "chunks": chunks}
            else:
                t0 = time.perf_counter()
                obj = http_json(chat_url, payload)
                total = time.perf_counter() - t0
                msg = (obj.get("choices") or [{}])[0].get("message", {})
                run = {"name": name, "total_s": round(total, 3),
                       "text": (msg.get("content") or "")[:200],
                       "tool_calls": msg.get("tool_calls"),
                       "timings": obj.get("timings")}
                text = msg.get("content") or ""
            report["runs"].append(run)
            print(f"[selftest] {name}: " +
                  ", ".join(f"{k}={run[k]}" for k in ("ttft_s", "total_s") if k in run))
            return run, text
        except Exception as e:
            fail(f"{name}: {e}")
            return None, ""

    # 2. warmups (first is slow: CUDA init / graph capture)
    chat_once([{"role": "user", "content": "Say hello in one word."}],
              "warmup_1", stream=False, max_tokens=16)
    chat_once([{"role": "user", "content": "Say hello in one word."}],
              "warmup_2", stream=True, max_tokens=16)

    # 3. streaming speed measurement (3 reps)
    tps_vals, ttft_vals = [], []
    for i in range(3):
        run, _ = chat_once(
            [{"role": "user",
              "content": "Write a short paragraph about home automation."}],
            f"speed_{i+1}", stream=True, max_tokens=96)
        if run and run.get("timings"):
            t = run["timings"]
            tps = t.get("predicted_per_second")
            if tps:
                tps_vals.append(tps)
            if run.get("ttft_s") is not None:
                ttft_vals.append(run["ttft_s"])
    if tps_vals:
        best_tps = max(tps_vals)
        report["best_tps"] = round(best_tps, 2)
        report["ttft_s_list"] = ttft_vals
        print(f"[selftest] best decode: {best_tps:.1f} tok/s, "
              f"TTFTs: {[round(x, 2) for x in ttft_vals]}")
        if best_tps < MIN_TPS_FAIL:
            fail(f"decode speed {best_tps:.1f} tok/s below {MIN_TPS_FAIL} "
                 "(broken or not on GPU)")
        elif best_tps < MIN_TPS_WARN:
            report["warnings"].append(
                f"decode speed {best_tps:.1f} tok/s below {MIN_TPS_WARN}")

    # 4. tool-calling round trip (hass_local_openai_llm style)
    tools = [{
        "type": "function",
        "function": {
            "name": "turn_on_light",
            "description": "Turn on a light in the home.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string",
                                  "description": "The light entity id, "
                                                 "e.g. light.kitchen"},
                },
                "required": ["entity_id"],
            },
        },
    }]
    # 4a. forced tool_choice: validates the tool-call plumbing end to end
    run, _ = chat_once(
        [{"role": "system",
          "content": "You are a home assistant. Use the provided tools."},
         {"role": "user", "content": "Turn on the kitchen light."}],
        "tool_call_forced", stream=False, tools=tools, max_tokens=384,
        tool_choice={"type": "function",
                     "function": {"name": "turn_on_light"}})
    if run is not None:
        if not run.get("tool_calls"):
            fail("forced tool_choice produced no tool_calls")
        else:
            print(f"[selftest] tool_calls(forced): "
                  f"{json.dumps(run['tool_calls'])[:300]}")
            report["tool_call_forced_ok"] = True

    # 4b. auto tool_choice: realistic HA behavior (system prompt carries the
    # entity context, like hass_local_openai_llm's exposed-entities prompt)
    run, _ = chat_once(
        [{"role": "system",
          "content": "You are a home assistant for a smart home. Available "
                     "entities: light.kitchen (Kitchen light). When the user "
                     "asks to control a device, immediately call the tool "
                     "with the matching entity_id. Do not ask questions."},
         {"role": "user", "content": "Turn on the kitchen light."}],
        "tool_call_auto", stream=False, tools=tools, max_tokens=384)
    if run is not None:
        tc = run.get("tool_calls")
        if not tc:
            fail("tool call not produced for 'Turn on the kitchen light.'")
        else:
            print(f"[selftest] tool_calls(auto): {json.dumps(tc)[:300]}")
            report["tool_call_ok"] = True

    # 5. multi-turn (prompt cache probe: second turn should TTFT much faster)
    turns = [
        [{"role": "user", "content": "My name is Alex and I live in Berlin."}],
        [{"role": "user", "content": "My name is Alex and I live in Berlin."},
         {"role": "assistant", "content": "Nice to meet you, Alex."},
         {"role": "user", "content": "What is my name and city?"}],
    ]
    for i, msgs in enumerate(turns):
        chat_once(msgs, f"multiturn_{i+1}", stream=True, max_tokens=32)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[selftest] report written to {args.out}")

    if report["warnings"]:
        for w in report["warnings"]:
            print(f"[selftest] WARN: {w}")
    if failures:
        print(f"[selftest] RESULT: FAIL ({len(failures)} failures)")
        return 1
    print("[selftest] RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
