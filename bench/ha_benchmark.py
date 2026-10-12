#!/usr/bin/env python3
"""Home Assistant voice-assistant benchmark for super-soaker.

Model-agnostic battery runner: point it at any OpenAI-compatible server
(llama-server) and it measures

  1. latency  — cold TTFT, warm TTFT, end-to-end time, decode tok/s
  2. tool use — a battery of HA utterances against the real HA Assist tool
                surface (mocked schemas), incl. no-tool negatives
  3. quality  — short-answer factual battery with deterministic assertions
                + brevity checks for voice UX

Results: JSON at --out plus a human-readable summary on stdout.

Usage:
  python3 bench/ha_benchmark.py --base-url http://127.0.0.1:8098 \
      --model LFM2.5-1.2B-Instruct --label 1.2b-noreason \
      [--battery bench/ha_battery.json] [--runs 5] [--out FILE]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BATTERY = os.path.join(HERE, "ha_battery.json")


def http_json(url, payload, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stream_chat(base_url, payload, timeout=180):
    """Streaming chat; returns (ttft_s, text, reasoning, total_s, tool_calls, n_chunks)."""
    t0 = time.perf_counter()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({**payload, "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    ttft = None
    text = []
    reasoning = []
    tool_calls = {}   # index -> {"name":..., "args": [fragments]}
    total = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if ttft is None and time.perf_counter() - t0 > 0:
                ttft = time.perf_counter() - t0
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content"):
                    text.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"name": "", "args": []})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"].append(fn["arguments"])
        total = time.perf_counter() - t0
    calls = []
    for idx in sorted(tool_calls):
        slot = tool_calls[idx]
        try:
            args = json.loads("".join(slot["args"]))
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": "".join(slot["args"])}
        calls.append({"name": slot["name"], "args": args})
    return (ttft, "".join(text), "".join(reasoning), total, calls)


def chat_plain(base_url, payload, timeout=180):
    """Non-streaming chat; returns dict with message parts."""
    t0 = time.perf_counter()
    resp = http_json(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    total = time.perf_counter() - t0
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    usage = resp.get("usage", {})
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": [
            {"name": (tc.get("function") or {}).get("name", ""),
             "args": _parse_args((tc.get("function") or {}).get("arguments"))}
            for tc in (msg.get("tool_calls") or [])],
        "total_s": round(total, 3),
        "completion_tokens": usage.get("completion_tokens"),
    }


def _parse_args(args):
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args) if args else {}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": args or ""}


def words(s):
    return len(re.findall(r"\S+", s or ""))


def check_args(expect, got):
    """Returns list of failure strings (empty = pass)."""
    fails = []
    checks = []

    def keys_match(spec):
        f = []
        for k, v in spec.items():
            g = got.get(k)
            if isinstance(v, (int, float)):
                try:
                    if abs(float(g) - float(v)) > 1e-6:
                        f.append(f"{k}={g!r}!={v}")
                except (TypeError, ValueError):
                    f.append(f"{k}={g!r}!={v}")
            elif isinstance(v, str):
                if g is None or v.lower() not in str(g).lower():
                    f.append(f"{k}={g!r}!~{v}")
            else:
                f.append(f"{k} bad type")
        return f

    if "args" in expect:
        checks.append(("exact", expect["args"]))
    if "args_any" in expect:
        checks.append(("any", expect["args_any"]))
    if "args_check" in expect:
        checks.append(("special", expect["args_check"]))

    any_ok = False
    special_fails = []
    for kind, spec in checks:
        if kind == "exact":
            fails.extend(keys_match(spec))
        elif kind == "any":
            # args_any: {"k": v, ...} (flat, one candidate) OR
            #           {"alt1": {"k": v...}, "alt2": {...}} (dict of candidates)
            cands = [spec]
            if isinstance(spec, dict) and spec and all(
                    isinstance(v, dict) for v in spec.values()):
                cands = list(spec.values())
            if not any(not keys_match(c) for c in cands):
                fails.append(f"no arg alternative matched (got {got})")
        elif kind == "special":
            name = spec
            ok = False
            if name == "brightness_40_or_102":
                b = got.get("brightness", got.get("brightness_pct"))
                try:
                    ok = b is not None and (abs(float(b) - 40) < 1 or abs(float(b) - 102) < 1)
                except (TypeError, ValueError):
                    ok = False
            elif name == "duration_10_minutes":
                d = str(got.get("duration", "")).lower()
                ok = ("10" in d and ("minute" in d or "00:10" in d or d == "600")) or d == "600"
            elif name == "volume_0_5":
                v = got.get("volume_level")
                try:
                    ok = v is not None and (abs(float(v) - 0.5) < 0.01 or abs(float(v) - 50) < 1)
                except (TypeError, ValueError):
                    ok = False
            elif name == "expr_has_345_and_12":
                e = str(got.get("expression", ""))
                ok = "345" in e and "12" in e
            elif name == "temp_up_68_80":
                # accept Fahrenheit (68-80) or Celsius (20-27); units are
                # ambiguous in the prompt, a sane raise of either kind passes
                t = got.get("temperature")
                try:
                    ok = t is not None and (68 <= float(t) <= 80
                                            or 20 <= float(t) <= 27)
                except (TypeError, ValueError):
                    ok = False
            if not ok:
                special_fails.append(f"check {name} failed (args={got})")
    fails.extend(special_fails)
    return fails


def run_battery(args):
    battery = json.load(open(args.battery))
    base = args.base_url
    label = args.label
    results = {"label": label, "model": args.model, "base_url": base,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---------------- latency ----------------
    sys_p = battery["latency_prompt_system"]
    tools = battery["tools"]
    lat = {"runs": []}
    for i in range(args.runs):
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": battery["latency_utterance"]},
            ],
            "tools": tools, "tool_choice": "auto",
            "max_tokens": 128, "temperature": 0.1,
        }
        try:
            ttft, text, reasoning, total, calls = stream_chat(base, payload)
            lat["runs"].append({
                "run": i, "cold": i == 0 and args.cold_first,
                "ttft_s": round(ttft, 3) if ttft else None,
                "total_s": round(total, 3),
                "reasoning_chars": len(reasoning),
                "tool_call": (calls[0]["name"] if calls else None),
            })
        except Exception as e:
            lat["runs"].append({"run": i, "error": str(e)[:200]})
    ok = [r for r in lat["runs"] if r.get("ttft_s")]
    if ok:
        warm = [r["ttft_s"] for r in ok if not r.get("cold")]
        cold = [r["ttft_s"] for r in ok if r.get("cold")]
        lat["warm_ttft_s"] = round(sum(warm) / len(warm), 3) if warm else None
        lat["cold_ttft_s"] = round(cold[0], 3) if cold else None
        lat["total_avg_s"] = round(sum(r["total_s"] for r in ok) / len(ok), 3)
    results["latency"] = lat

    # decode speed: longer generation, no tools
    try:
        ttft, text, reasoning, total, _ = stream_chat(base, {
            "model": args.model,
            "messages": [
                {"role": "system", "content": battery["chat_system"]},
                {"role": "user", "content":
                 "Explain in exactly 5 sentences how a smart home hub works."},
            ],
            "max_tokens": 200, "temperature": 0.7,
        })
        w = words(text)
        results["decode"] = {
            "words": w, "total_s": round(total, 3),
            "words_per_s": round(w / total, 1) if total else None,
            "reasoning_chars": len(reasoning),
            "ttft_s": round(ttft, 3) if ttft else None,
        }
    except Exception as e:
        results["decode"] = {"error": str(e)[:200]}

    # ---------------- tool battery ----------------
    tool_results = []
    for sc in battery["tool_scenarios"]:
        expect = sc["expect"]
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": battery["tool_system"]},
                {"role": "user", "content": sc["utterance"]},
            ],
            "tools": tools, "tool_choice": "auto",
            "max_tokens": 160, "temperature": 0.1,
        }
        item = {"id": sc["id"], "utterance": sc["utterance"]}
        try:
            r = chat_plain(base, payload)
            item["reasoning_chars"] = len(r["reasoning"])
            item["content_snip"] = (r["content"] or "")[:80]
            item["completion_tokens"] = r["completion_tokens"]
            item["latency_s"] = r["total_s"]
            if expect.get("tool") is None:
                if r["tool_calls"]:
                    item["pass"] = False
                    item["fail"] = f"called {r['tool_calls'][0]['name']} (expected no tool)"
                else:
                    item["pass"] = True
            else:
                if not r["tool_calls"]:
                    item["pass"] = False
                    item["fail"] = "no tool call"
                else:
                    tc = r["tool_calls"][0]
                    item["tool"] = tc["name"]
                    item["args"] = tc["args"]
                    expected_tools = (expect["tool"] if isinstance(
                        expect["tool"], list) else [expect["tool"]])
                    if tc["name"] not in expected_tools:
                        item["pass"] = False
                        item["fail"] = f"tool {tc['name']} != {expected_tools[0]}"
                    else:
                        af = check_args(expect, tc["args"]) \
                            if tc["name"] == expected_tools[0] else []
                        if af:
                            item["pass"] = False
                            item["fail"] = "; ".join(af)
                        else:
                            item["pass"] = True
        except Exception as e:
            item["pass"] = False
            item["fail"] = f"error: {str(e)[:160]}"
        tool_results.append(item)
    results["tool"] = tool_results
    results["tool_acc"] = round(100 * sum(1 for t in tool_results if t["pass"])
                                / len(tool_results), 1)

    # ---------------- quality battery ----------------
    q_results = []
    for sc in battery["quality_scenarios"]:
        item = {"id": sc["id"], "utterance": sc["utterance"]}
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": battery["chat_system"]},
                {"role": "user", "content": sc["utterance"]},
            ],
            "max_tokens": 200, "temperature": 0.1,
        }
        try:
            r = chat_plain(base, payload)
            text = r["content"] or ""
            low = text.lower()
            item["reasoning_chars"] = len(r["reasoning"])
            item["n_words"] = words(text)
            item["completion_tokens"] = r["completion_tokens"]
            item["latency_s"] = r["total_s"]
            fails = []
            if sc.get("max_words") and item["n_words"] > sc["max_words"]:
                fails.append(f"{item['n_words']} words > {sc['max_words']}")
            for kw in sc.get("require_any", []):
                if kw.lower() not in low:
                    fails.append(f"missing {kw!r}")
            for kw in sc.get("forbid_any", []):
                if kw.lower() in low:
                    fails.append(f"forbidden {kw!r}")
            item["text_snip"] = text[:120]
            item["pass"] = not fails
            if fails:
                item["fail"] = "; ".join(fails)
        except Exception as e:
            item["pass"] = False
            item["fail"] = f"error: {str(e)[:160]}"
        q_results.append(item)
    results["quality"] = q_results
    results["quality_acc"] = round(100 * sum(1 for q in q_results if q["pass"])
                                   / len(q_results), 1)

    results["reasoning_used"] = any(
        (t.get("reasoning_chars") or 0) > 0 for t in tool_results + q_results)

    # ---------------- summary ----------------
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== {label} ({args.model}) ===")
    print(f"  latency   : cold TTFT {lat.get('cold_ttft_s')}s | warm TTFT "
          f"{lat.get('warm_ttft_s')}s | avg total {lat.get('total_avg_s')}s")
    d = results.get("decode", {})
    print(f"  decode    : {d.get('words_per_s')} words/s "
          f"(reasoning {d.get('reasoning_chars', 0)} chars)")
    print(f"  tool acc  : {results['tool_acc']}%  "
          f"({sum(1 for t in tool_results if t['pass'])}/{len(tool_results)})")
    for t in tool_results:
        if not t["pass"]:
            print(f"    FAIL {t['id']}: {t.get('fail', '')[:110]}")
    print(f"  quality   : {results['quality_acc']}%  "
          f"({sum(1 for q in q_results if q['pass'])}/{len(q_results)})")
    for q in q_results:
        if not q["pass"]:
            print(f"    FAIL {q['id']}: {q.get('fail', '')[:110]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--battery", default=DEFAULT_BATTERY)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cold-first", action="store_true",
                    help="first latency run is right after server start (cold)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(HERE, "results", f"ha_{a.label}.json")
    run_battery(a)


if __name__ == "__main__":
    main()