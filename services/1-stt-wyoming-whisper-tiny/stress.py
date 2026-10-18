#!/usr/bin/env python3
"""super-soaker STT stress test (stdlib only).

Generates plausible voice-assistant utterances with espeak-ng (16-bit mono
WAVs, real sample rate sent in audio-start), then streams them through the
Wyoming STT protocol with up to --concurrency requests in flight. Every
failure is logged immediately; a summary (counts, p50/p95 latency, error
breakdown) is printed at the end. --duration 0 runs until Ctrl-C.

    python3 stress.py --duration 300 --concurrency 2
"""
import argparse
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import Counter

UTTERANCES = [
    "Turn on the kitchen light.",
    "What is the temperature in the living room?",
    "Set a timer for ten minutes.",
    "Is the front door locked?",
    "Play some jazz music in the living room.",
    "What is the weather going to be like tomorrow?",
    "Remind me to take out the trash at seven o'clock.",
    "Turn off all the lights downstairs.",
    "What is the working families party?",
    "Add milk and eggs to the shopping list.",
]


def send_event(sock, event_type, data=None, payload=b""):
    data_bytes = json.dumps(data).encode() if data is not None else b""
    header = {"type": event_type, "version": "1.0"}
    if data_bytes:
        header["data_length"] = len(data_bytes)
    if payload:
        header["payload_length"] = len(payload)
    sock.sendall(json.dumps(header).encode() + b"\n" + data_bytes + payload)


def recv_event(sock, buf):
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("server closed connection")
        buf.extend(chunk)
    line, rest = buf.split(b"\n", 1)
    buf[:] = rest
    header = json.loads(line)
    data_length = header.get("data_length", 0)
    while len(buf) < data_length:
        buf.extend(sock.recv(65536))
    if data_length:
        header.setdefault("data", {}).update(json.loads(bytes(buf[:data_length])))
        del buf[:data_length]
    payload_length = header.get("payload_length", 0)
    while len(buf) < payload_length:
        buf.extend(sock.recv(65536))
    if payload_length:
        header["payload"] = bytes(buf[:payload_length])
        del buf[:payload_length]
    return header


def make_samples(tmpdir):
    samples = []
    for i, text in enumerate(UTTERANCES):
        path = os.path.join(tmpdir, f"utt{i}.wav")
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", "150", "-w", path, text],
            check=True, capture_output=True,
        )
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            assert w.getsampwidth() == 2 and w.getnchannels() == 1
            pcm = w.readframes(w.getnframes())
        samples.append((text, rate, pcm))
    return samples


def transcribe(host, port, rate, pcm, timeout):
    sock = socket.create_connection((host, port), timeout=30)
    sock.settimeout(timeout)
    buf = bytearray()
    send_event(sock, "transcribe", {"language": "en"})
    fmt = {"rate": rate, "width": 2, "channels": 1}
    send_event(sock, "audio-start", fmt)
    for i in range(0, len(pcm), 3200):
        send_event(sock, "audio-chunk", fmt, pcm[i:i + 3200])
    send_event(sock, "audio-stop")
    text = ""
    done = False
    try:
        while True:
            ev = recv_event(sock, buf)
            typ = ev["type"]
            if typ == "transcript":
                text += ev.get("data", {}).get("text", "")
            elif typ == "transcribe-done":
                done = True
                break
            elif typ == "error":
                raise RuntimeError(f"server error: {ev.get('data')}")
    except (ConnectionError, EOFError):
        done = True
    finally:
        sock.close()
    return text, done


def worker(idx, args, deadline, samples, stats, lock, stop):
    rng = random.Random(args.seed + idx)
    while not stop.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            return
        text, rate, pcm = rng.choice(samples)
        t0 = time.monotonic()
        try:
            transcript, done = transcribe(args.host, args.port, rate, pcm, args.timeout)
            wall = time.monotonic() - t0
            error = None
            if not transcript.strip():
                error = "empty transcript"
            elif not done:
                error = "no transcribe-done"
            with lock:
                stats["done"] += 1
                if error:
                    stats["errors"][error.split(" ")[0]] += 1
                else:
                    stats["latencies"].append(wall)
                if stats["done"] % 20 == 0:
                    print(f"[stress-stt] {stats['done']} requests, "
                          f"{stats['errors'].total()} errors, last {wall:.2f}s",
                          flush=True)
            if error:
                print(f"[stress-stt] FAIL worker={idx} audio={text!r} {error}", flush=True)
            if args.pause:
                time.sleep(args.pause)
        except Exception as exc:
            wall = time.monotonic() - t0
            with lock:
                stats["done"] += 1
                stats["errors"][type(exc).__name__] += 1
            print(f"[stress-stt] FAIL worker={idx} audio={text!r} "
                  f"after {wall:.2f}s: {type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10300)
    ap.add_argument("--duration", type=int, default=300,
                    help="seconds to run; 0 = until Ctrl-C")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        samples = make_samples(td)
        print(f"[stress-stt] generated {len(samples)} utterances "
              f"({sum(len(p) / (r * 2) for _, r, p in samples):.1f}s total)", flush=True)
        deadline = None if args.duration == 0 else time.monotonic() + args.duration
        stats = {"done": 0, "errors": Counter(), "latencies": []}
        lock = threading.Lock()
        stop = threading.Event()
        threads = [threading.Thread(target=worker,
                                    args=(i, args, deadline, samples, stats, lock, stop),
                                    daemon=True)
                   for i in range(args.concurrency)]
        started = time.monotonic()
        for t in threads:
            t.start()
        print(f"[stress-stt] {args.concurrency} workers -> tcp://{args.host}:{args.port}, "
              f"duration={args.duration or 'until Ctrl-C'}s", flush=True)
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(0.5)
        except KeyboardInterrupt:
            stop.set()
            print("[stress-stt] interrupted", flush=True)
            for t in threads:
                t.join(timeout=30)
        elapsed = time.monotonic() - started

        lat = sorted(stats["latencies"])
        p50 = statistics.median(lat) if lat else 0.0
        p95 = lat[int(len(lat) * 0.95) - 1] if len(lat) >= 20 else (lat[-1] if lat else 0.0)
        print(f"[stress-stt] summary: {stats['done']} requests in {elapsed:.1f}s "
              f"({stats['done'] / elapsed:.2f}/s), {stats['errors'].total()} errors, "
              f"p50={p50:.2f}s p95={p95:.2f}s", flush=True)
        for err, count in stats["errors"].most_common():
            print(f"[stress-stt]   {count} x {err}", flush=True)
        return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
