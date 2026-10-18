#!/usr/bin/env python3
"""super-soaker TTS stress test (stdlib only).

Hammers the Wyoming TTS with plausible assistant answers: short replies,
medium paragraphs, a long multi-sentence answer, and the worst-case
503-phoneme run-on batch. Up to --concurrency connections are active at once.
Every failure is logged immediately; a summary (counts, p50/p95 latency,
error breakdown) is printed at the end. --duration 0 runs until Ctrl-C.

    python3 stress.py --duration 300 --concurrency 2
"""
import argparse
import random
import statistics
import sys
import threading
import time
from collections import Counter

from longform import WARMUP_TEXT
from selftest import rms_int16, synthesize

CORPUS = [
    ("short", "The kitchen light is now on."),
    ("short2", "I set a timer for ten minutes."),
    ("short3", "The front door is locked."),
    ("medium", "It is currently twenty one degrees Celsius in the living room, "
               "and the humidity is forty eight percent."),
    ("medium2", "The front door is locked. The porch light is off, and the "
                "garage door is closed."),
    ("medium3", "Your evening routine has three steps. First, the living room "
                "lights will dim to thirty percent. Second, the thermostat will "
                "drop to nineteen degrees. Third, the front door will lock and "
                "the porch light will turn off."),
    ("long", "The Working Families Party is a minor American political party "
             "that was founded to support labor unions, working families, and "
             "progressive economic policies across several states, and the "
             "organization often cross endorses candidates from other parties "
             "while also running its own ballot line in places where election "
             "law allows that kind of fusion politics to happen in practice."),
    ("long2", "Here is what I found. The nearest grocery store is two "
              "kilometres away and closes at nine in the evening. The pharmacy "
              "next door closes at eight, and the hardware store on the corner "
              "opens again tomorrow at seven. Traffic on the main road is light, "
              "so the drive should take about six minutes."),
    ("wmax", WARMUP_TEXT),
]
WEIGHTS = [7, 5, 4, 4, 3, 3, 2, 2, 1]


def worker(idx, args, deadline, stats, lock, stop):
    rng = random.Random(args.seed + idx)
    while not stop.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            return
        name, text = rng.choices(CORPUS, WEIGHTS)[0]
        streaming = rng.random() < 0.8
        t0 = time.monotonic()
        try:
            r = synthesize(args.host, args.port, text, args.voice, streaming)
            wall = time.monotonic() - t0
            dur = len(r["audio"]) / (r["rate"] * r["width"] * r["channels"])
            rms = rms_int16(r["audio"])
            error = None
            if r["rate"] != 24000:
                error = f"unexpected rate {r['rate']}"
            elif dur < 0.4:
                error = f"implausible duration {dur:.2f}s"
            elif rms < 300:
                error = f"silent audio (rms={rms:.0f})"
            elif r["ttfc_ms"] is not None and r["ttfc_ms"] > 15000:
                error = f"ttfc {r['ttfc_ms']:.0f}ms > 15000ms"
            with lock:
                stats["done"] += 1
                stats["audio_s"] += dur
                if error:
                    stats["errors"][error.split(" ")[0]] += 1
                else:
                    stats["latencies"].append(wall)
                if stats["done"] % 25 == 0:
                    print(f"[stress-tts] {stats['done']} requests, "
                          f"{stats['errors'].total()} errors, "
                          f"last {name}/{ 'stream' if streaming else 'plain'} "
                          f"{wall:.2f}s", flush=True)
            if error:
                print(f"[stress-tts] FAIL worker={idx} kind={name} "
                      f"mode={'stream' if streaming else 'plain'} {error}", flush=True)
            if args.pause:
                time.sleep(args.pause)
        except Exception as exc:
            wall = time.monotonic() - t0
            with lock:
                stats["done"] += 1
                stats["errors"][type(exc).__name__] += 1
            print(f"[stress-tts] FAIL worker={idx} kind={name} "
                  f"mode={'stream' if streaming else 'plain'} after {wall:.2f}s: "
                  f"{type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10210)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--duration", type=int, default=300,
                    help="seconds to run; 0 = until Ctrl-C")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to sleep between requests in each worker")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    deadline = None if args.duration == 0 else time.monotonic() + args.duration
    stats = {"done": 0, "errors": Counter(), "latencies": [], "audio_s": 0.0}
    lock = threading.Lock()
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(i, args, deadline, stats, lock, stop),
                                daemon=True)
               for i in range(args.concurrency)]
    started = time.monotonic()
    for t in threads:
        t.start()
    print(f"[stress-tts] {args.concurrency} workers -> tcp://{args.host}:{args.port}, "
          f"duration={args.duration or 'until Ctrl-C'}s", flush=True)
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()
        print("[stress-tts] interrupted", flush=True)
        for t in threads:
            t.join(timeout=30)
    elapsed = time.monotonic() - started

    lat = sorted(stats["latencies"])
    p50 = statistics.median(lat) if lat else 0.0
    p95 = lat[int(len(lat) * 0.95) - 1] if len(lat) >= 20 else (lat[-1] if lat else 0.0)
    print(f"[stress-tts] summary: {stats['done']} requests in {elapsed:.1f}s "
          f"({stats['done'] / elapsed:.2f}/s), {stats['audio_s']:.0f}s audio, "
          f"{stats['errors'].total()} errors, p50={p50:.2f}s p95={p95:.2f}s", flush=True)
    for err, count in stats["errors"].most_common():
        print(f"[stress-tts]   {count} x {err}", flush=True)
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
