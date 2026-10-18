#!/usr/bin/env python3
"""super-soaker LLM stress test (stdlib only).

Sends plausible Home Assistant Assist traffic to the OpenAI-compatible
endpoint: tool-calling control commands (the hass_local_openai_llm path),
factual questions, and multi-turn follow-ups, half streaming and half
non-streaming, with up to --concurrency requests in flight. The server runs a
single slot (-np 1), so concurrency > 1 exercises request queueing.

Every failure is logged immediately; a summary (counts, p50/p95 latency,
error breakdown) is printed at the end. --duration 0 runs until Ctrl-C.

    python3 stress.py --duration 300 --concurrency 2
"""
import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "turn_on",
            "description": "Turn on a device in the home.",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_off",
            "description": "Turn off a device in the home.",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Read the current temperature of a room.",
            "parameters": {
                "type": "object",
                "properties": {"room": {"type": "string"}},
                "required": ["room"],
            },
        },
    },
]

SYSTEM = (
    "You are a home assistant for a smart home. Available entities: "
    "light.kitchen (Kitchen light), light.living_room (Living room light), "
    "sensor.living_room_temperature (Living room temperature), "
    "lock.front_door (Front door). When the user asks to control a device, "
    "call the matching tool immediately. Keep spoken answers short."
)

PROMPTS = [
    ("Turn on the kitchen light.", True),
    ("What is the temperature in the living room?", False),
    ("Turn off the living room light.", True),
    ("Is the front door locked?", False),
    ("Set a timer for ten minutes.", False),
    ("Turn on the kitchen light and tell me the living room temperature.", True),
    ("What can you do?", False),
    ("Make the living room light a bit brighter.", True),
    ("What is the working families party?", False),
    ("Thank you, that is all.", False),
]


def chat(url, model, messages, stream, timeout):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.1,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    if not stream:
        payload["stream"] = False
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            obj = json.loads(r.read().decode())
        total = time.perf_counter() - t0
        choice = (obj.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        return {"ttft": None, "total": total, "content": content,
                "tool_calls": tool_calls, "timings": obj.get("timings")}

    payload["stream"] = True
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST")
    t0 = time.perf_counter()
    ttft = None
    content = []
    tool_calls = 0
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
            if obj.get("timings"):
                timings = obj["timings"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_calls += len(delta["tool_calls"])
    return {"ttft": ttft, "total": time.perf_counter() - t0,
            "content": "".join(content), "tool_calls": tool_calls,
            "timings": timings}


def worker(idx, args, deadline, stats, lock, stop):
    rng = random.Random(args.seed + idx)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    while not stop.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            return
        prompt, wants_tool = rng.choice(PROMPTS)
        streaming = rng.random() < 0.5
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        t0 = time.monotonic()
        try:
            r = chat(url, args.model, messages, streaming, args.timeout)
            error = None
            if not r["content"].strip() and not r["tool_calls"]:
                error = "empty response (no content, no tool_calls)"
            elif r["total"] > args.timeout:
                error = f"slow response {r['total']:.1f}s"
            with lock:
                stats["done"] += 1
                if error:
                    stats["errors"][error.split(" ")[0]] += 1
                else:
                    stats["latencies"].append(r["total"])
                    if r["ttft"] is not None:
                        stats["ttfts"].append(r["ttft"])
                    stats["tokens"] += 0
                if stats["done"] % 20 == 0:
                    tps = (r.get("timings") or {}).get("predicted_per_second")
                    print(f"[stress-llm] {stats['done']} requests, "
                          f"{stats['errors'].total()} errors, "
                          f"last {r['total']:.2f}s"
                          + (f" {tps:.0f} tok/s" if tps else ""), flush=True)
            if error:
                print(f"[stress-llm] FAIL worker={idx} stream={streaming} "
                      f"prompt={prompt!r} {error}", flush=True)
            if args.pause:
                time.sleep(args.pause)
        except Exception as exc:
            wall = time.monotonic() - t0
            with lock:
                stats["done"] += 1
                stats["errors"][type(exc).__name__] += 1
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode(errors="replace")[:200]
                except Exception:
                    pass
            print(f"[stress-llm] FAIL worker={idx} stream={streaming} "
                  f"after {wall:.2f}s: {type(exc).__name__}: {exc} {detail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="LFM2.5-2.6B")
    ap.add_argument("--duration", type=int, default=300,
                    help="seconds to run; 0 = until Ctrl-C")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    deadline = None if args.duration == 0 else time.monotonic() + args.duration
    stats = {"done": 0, "errors": Counter(), "latencies": [], "ttfts": [], "tokens": 0}
    lock = threading.Lock()
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(i, args, deadline, stats, lock, stop),
                                daemon=True)
               for i in range(args.concurrency)]
    started = time.monotonic()
    for t in threads:
        t.start()
    print(f"[stress-llm] {args.concurrency} workers -> {args.base_url}, "
          f"model={args.model}, duration={args.duration or 'until Ctrl-C'}s", flush=True)
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()
        print("[stress-llm] interrupted", flush=True)
        for t in threads:
            t.join(timeout=60)
    elapsed = time.monotonic() - started

    lat = sorted(stats["latencies"])
    p50 = statistics.median(lat) if lat else 0.0
    p95 = lat[int(len(lat) * 0.95) - 1] if len(lat) >= 20 else (lat[-1] if lat else 0.0)
    ttft = sorted(stats["ttfts"])
    ttft_p50 = statistics.median(ttft) if ttft else 0.0
    print(f"[stress-llm] summary: {stats['done']} requests in {elapsed:.1f}s "
          f"({stats['done'] / elapsed:.2f}/s), {stats['errors'].total()} errors, "
          f"p50={p50:.2f}s p95={p95:.2f}s, stream ttft p50={ttft_p50:.2f}s", flush=True)
    for err, count in stats["errors"].most_common():
        print(f"[stress-llm]   {count} x {err}", flush=True)
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
