#!/usr/bin/env python3
"""super-soaker TTS end-to-end self-test (stdlib only).

Speaks a phrase via the Wyoming protocol (plain + streaming) and asserts audio
comes back at the expected rate with plausible duration and non-silence.
Exit code 0 = pass.
"""
import argparse
import json
import math
import os
import socket
import struct
import sys
import time

from longform import WARMUP_TEXT


def _send_event(sock: socket.socket, event_type: str, data: dict = None, payload: bytes = b"") -> None:
    data_bytes = json.dumps(data).encode() if data is not None else b""
    header = {"type": event_type, "version": "1.0"}
    if data_bytes:
        header["data_length"] = len(data_bytes)
    if payload:
        header["payload_length"] = len(payload)
    sock.sendall(json.dumps(header).encode() + b"\n" + data_bytes + payload)


def _recv_event(sock: socket.socket, buf: bytearray) -> dict:
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


def synthesize(host: str, port: int, text: str, voice: str, streaming: bool):
    """Returns dict with rate, audio bytes, ttfc_ms, total_ms."""
    sock = socket.create_connection((host, port), timeout=30)
    sock.settimeout(120)
    buf = bytearray()

    _send_event(sock, "describe")
    ev = _recv_event(sock, buf)
    tts = (ev.get("data", {}).get("tts") or [{}])[0]
    if not tts.get("supports_synthesize_streaming"):
        raise RuntimeError("server does not advertise supports_synthesize_streaming")

    t0 = time.perf_counter()
    if streaming:
        _send_event(sock, "synthesize-start", {"voice": {"name": voice}})
        _send_event(sock, "synthesize-chunk", {"text": text})
        _send_event(sock, "synthesize-stop")
    else:
        _send_event(sock, "synthesize", {"text": text, "voice": {"name": voice}})

    rate, width, channels = 24000, 2, 1
    audio = bytearray()
    ttfc = None
    while True:
        ev = _recv_event(sock, buf)
        typ = ev["type"]
        if typ == "audio-start":
            rate = ev.get("data", {}).get("rate", rate)
            width = ev.get("data", {}).get("width", width)
            channels = ev.get("data", {}).get("channels", channels)
        elif typ == "audio-chunk":
            if ttfc is None:
                ttfc = (time.perf_counter() - t0) * 1000
            audio += ev.get("payload", b"")
        elif typ in ("audio-stop", "synthesize-stopped"):
            break
        elif typ == "error":
            raise RuntimeError(f"server error: {ev}")
    total_ms = (time.perf_counter() - t0) * 1000
    sock.close()
    return {"rate": rate, "width": width, "channels": channels,
            "audio": bytes(audio), "ttfc_ms": ttfc, "total_ms": total_ms}


def rms_int16(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10210)
    ap.add_argument("--voice", default="af_heart")
    args = ap.parse_args()

    failures = []

    def fail(msg):
        failures.append(msg)
        print(f"[selftest-tts] FAIL: {msg}")

    # gate 1: text sanitation unit tests (thinking/tool leaks, stray '<')
    import subprocess
    cleaner_test = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "test_clean.py")
    try:
        rc = subprocess.run([sys.executable, cleaner_test]).returncode
        if rc != 0:
            fail(f"text sanitation unit tests failed (rc={rc})")
    except Exception as e:
        fail(f"text sanitation unit tests could not run: {e}")

    # gate 2: Wyoming end-to-end audio
    for streaming in (False, True):
        label = "streaming" if streaming else "plain"
        t = "This is a super soaker self test. One two three four five."
        try:
            r = synthesize(args.host, args.port, t, args.voice, streaming)
        except Exception as e:
            fail(f"{label}: {e}")
            continue

        bytes_per_sec = r["rate"] * r["width"] * r["channels"]
        dur = len(r["audio"]) / bytes_per_sec
        rms = rms_int16(r["audio"])
        print(f"[selftest-tts] {label}: {dur:.2f}s audio, rate={r['rate']}, "
              f"ttfc={r['ttfc_ms']:.0f}ms, total={r['total_ms']:.0f}ms, rms={rms:.0f}")

        if r["rate"] != 24000:
            fail(f"{label}: unexpected sample rate {r['rate']}")
        if dur < 1.0 or dur > 15.0:
            fail(f"{label}: implausible duration {dur:.2f}s")
        if rms < 300:
            fail(f"{label}: audio too quiet (rms={rms:.0f}), likely silence")
        if streaming and r["ttfc_ms"] is not None and r["ttfc_ms"] > 3000:
            fail(f"{label}: time to first chunk {r['ttfc_ms']:.0f}ms > 3000ms")

    # gate 3: worst-case long-form batch (one 503-phoneme sentence). This is the
    # largest synthesis the runtime can produce; it must complete without an
    # arena allocation failure (the 2026-10-17 OOM killed the service here).
    try:
        r = synthesize(args.host, args.port, WARMUP_TEXT, args.voice, True)
        bytes_per_sec = r["rate"] * r["width"] * r["channels"]
        dur = len(r["audio"]) / bytes_per_sec
        rms = rms_int16(r["audio"])
        ttfc = r["ttfc_ms"] or 0.0
        print(f"[selftest-tts] longform: {dur:.2f}s audio, rate={r['rate']}, "
              f"ttfc={ttfc:.0f}ms, total={r['total_ms']:.0f}ms, rms={rms:.0f}")
        if r["rate"] != 24000:
            fail(f"longform: unexpected sample rate {r['rate']}")
        if dur < 5.0:
            fail(f"longform: implausible duration {dur:.2f}s")
        if rms < 300:
            fail(f"longform: audio too quiet (rms={rms:.0f}), likely silence")
        if ttfc > 10000:
            fail(f"longform: time to first chunk {ttfc:.0f}ms > 10000ms")
    except Exception as e:
        fail(f"longform: {e}")

    if failures:
        print(f"[selftest-tts] {len(failures)} failure(s)")
        return 1
    print("[selftest-tts] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())