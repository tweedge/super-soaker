#!/usr/bin/env python3
"""Quality smoke test for LFM2.5-2.6B quants / reasoning budgets (stdlib only).

Not an academic eval: a small battery of objectively-scorable items plus
transcripts for eyeballing. Used to pick the fastest quant that still passes,
and to measure what a thinking budget costs/buys.

Usage:
  python3 bench/quality_check.py --base-url http://127.0.0.1:8080 \
      --model LFM2.5-2.6B --label q4km-budget256 --out-dir results
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

ITEMS = [
    {
        "id": "math_mul",
        "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}],
        "expect": [r"\b391\b"],
    },
    {
        "id": "math_word",
        "messages": [{"role": "user", "content": "A bat and a ball cost $1.10 in total. "
                     "The bat costs $1.00 more than the ball. How much does the ball cost? "
                     "Reply with just the number in dollars."}],
        "expect": [r"\$?0\.05\b", r"5\s*cents"],
    },
    {
        "id": "math_multistep",
        "messages": [{"role": "user", "content": "If 3 machines make 3 widgets in 3 minutes, "
                     "how many minutes do 9 machines need to make 9 widgets? "
                     "Reply with just the number."}],
        "expect": [r"\b3\b"],
    },
    {
        "id": "logic",
        "messages": [{"role": "user", "content": "Alice is taller than Bob. Bob is taller than Carol. "
                     "Who is the shortest? Reply with one name."}],
        "expect": [r"Carol"],
    },
    {
        "id": "factual",
        "messages": [{"role": "user", "content": "What is the capital of France? One word only."}],
        "expect": [r"Paris"],
    },
    {
        "id": "instruction_exact",
        "messages": [{"role": "user", "content": "Reply with exactly one word: banana"}],
        "expect": [r"^\s*banana\s*$"],
    },
    {
        "id": "json_format",
        "messages": [{"role": "user", "content": 'Return a JSON object with keys "city" (Paris) '
                     'and "country" (France). Output only the JSON.'}],
        "expect": [r'\{[^}]*"city"[^}]*"Paris"', r'\{[^}]*"country"[^}]*"France"'],
    },
    {
        "id": "ha_action",
        "tools": [{
            "type": "function",
            "function": {
                "name": "execute_services",
                "description": "Call a Home Assistant service to control devices.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "service": {"type": "string"},
                        "target": {"type": "object",
                                   "properties": {"entity_id": {"type": "string"}}},
                    },
                    "required": ["domain", "service", "target"],
                },
            },
        }],
        "messages": [{"role": "system", "content": "You are a home assistant. Use tools to act."},
                     {"role": "user", "content": "Turn on the kitchen light."}],
        "expect_tool": {"name": "execute_services",
                        "args_any": [r'"service"\s*:\s*"turn_on"',
                                     r"'service'\s*:\s*'turn_on'"],
                        "args_entity": [r"kitchen"]},
    },
    {
        "id": "ha_answer",
        "messages": [{"role": "system",
                      "content": "You are a voice assistant. Answer in one short sentence."},
                     {"role": "user", "content": "What should I wear if it's raining?"}],
        "expect": [r"rain|jacket|coat|umbrella|waterproof"],
    },
    {
        "id": "reasoning",
        "messages": [{"role": "user", "content": "I have 5 apples. I eat 2 and buy 6 more, "
                     "then give away half of what I have. How many apples do I have now? "
                     "Reply with just the number."}],
        "expect": [r"\b4\b", r"\b4\.5\b"],
    },
]


def chat(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode())
    total = time.perf_counter() - t0
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    return {"total_s": total, "text": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls"), "timings": obj.get("timings")}


def score_item(item, result):
    text = result["text"]
    failures = []
    if "expect" in item:
        if not any(re.search(p, text, re.I | re.M) for p in item["expect"]):
            failures.append(f"no expected pattern {item['expect']} in reply")
    if "expect_tool" in item:
        et = item["expect_tool"]
        tcs = result.get("tool_calls") or []
        if not tcs:
            failures.append("no tool_calls produced")
        else:
            names = [tc.get("function", {}).get("name") for tc in tcs]
            if et["name"] not in names:
                failures.append(f"tool name {names} != {et['name']}")
            raw_args = tcs[0].get("function", {}).get("arguments", "")
            if isinstance(raw_args, str):
                args = raw_args
            else:
                args = json.dumps(raw_args)
            if not any(re.search(p, args) for p in et.get("args_any", [])):
                failures.append(f"tool args missing expected fields: {args[:120]}")
            if not any(re.search(p, args, re.I) for p in et.get("args_entity", [])):
                failures.append(f"tool args missing entity ref: {args[:120]}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="LFM2.5-2.6B")
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    results = []
    n_pass = 0
    for item in ITEMS:
        payload = {"model": args.model, "messages": item["messages"],
                   "max_tokens": args.max_tokens, "temperature": 0.1,
                   "top_k": 50, "repetition_penalty": 1.1}
        if item.get("tools"):
            payload["tools"] = item["tools"]
            payload["tool_choice"] = "auto"
        try:
            r = chat(url, payload)
        except Exception as e:
            r = {"total_s": None, "text": f"<ERROR {e}>", "tool_calls": None,
                 "timings": None}
        failures = score_item(item, r)
        ok = not failures
        n_pass += ok
        t = r.get("timings") or {}
        print(f"[quality] {item['id']}: {'PASS' if ok else 'FAIL'} "
              f"({round(r['total_s'] or 0, 1)}s, pred={t.get('predicted_n')})"
              + (f" — {failures}" if failures else ""))
        results.append({"id": item["id"], "pass": ok, "failures": failures,
                        "text": r["text"][:600],
                        "tool_calls": r.get("tool_calls"),
                        "total_s": r.get("total_s"), "timings": t})

    total = len(ITEMS)
    print(f"[quality] {args.label}: {n_pass}/{total} passed")
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out_dir, f"{stamp}-{args.label}-quality.json")
    with open(path, "w") as f:
        json.dump({"label": args.label, "pass": n_pass, "total": total,
                   "items": results}, f, indent=2)
    print(f"[quality] wrote {path}")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
