#!/usr/bin/env python3
"""
Wyoming faster-whisper warmup script.

Sends two real transcription requests (1 second of silence each) through the
Wyoming protocol and logs the wall-clock time for each. Ensures the model is
fully loaded and CUDA kernels are JIT-compiled before real traffic arrives.

Usage: python3 warmup.py [--host HOST] [--port PORT] [--runs N]
"""

import argparse
import json
import socket
import struct
import sys
import time

# ---------------------------------------------------------------------------
# Wyoming wire protocol helpers (no wyoming package dependency)
# ---------------------------------------------------------------------------

def _send_event(sock: socket.socket, event_type: str, data: dict, payload: bytes = b"") -> None:
    data_bytes = json.dumps(data, ensure_ascii=False).encode()
    header = {"type": event_type, "version": "1.0"}
    if data_bytes:
        header["data_length"] = len(data_bytes)
    if payload:
        header["payload_length"] = len(payload)
    line = json.dumps(header, ensure_ascii=False).encode() + b"\n"
    sock.sendall(line + data_bytes + payload)


def _recv_event(sock: socket.socket, buf: bytearray) -> dict:
    """Read one newline-terminated JSON header line, plus optional data/payload."""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed connection during warmup")
        buf.extend(chunk)

    line, buf[:] = buf.split(b"\n", 1)[0], bytearray(buf.split(b"\n", 1)[1])
    header = json.loads(line)

    data_length = header.get("data_length", 0)
    while len(buf) < data_length:
        buf.extend(sock.recv(4096))
    if data_length:
        extra_data = json.loads(bytes(buf[:data_length]))
        header.setdefault("data", {}).update(extra_data)
        buf[:] = buf[data_length:]

    payload_length = header.get("payload_length", 0)
    while len(buf) < payload_length:
        buf.extend(sock.recv(4096))
    if payload_length:
        buf[:] = buf[payload_length:]

    return header


def _make_silence(duration_s: float = 1.0, rate: int = 16000, width: int = 2, channels: int = 1) -> bytes:
    """Generate raw PCM silence (all zeros)."""
    n_samples = int(rate * duration_s)
    return b"\x00" * (n_samples * width * channels)


def run_warmup(host: str, port: int, runs: int) -> None:
    rate, width, channels = 16000, 2, 1
    silence = _make_silence(1.0, rate, width, channels)

    print(f"[warmup] Connecting to Wyoming server at {host}:{port}", flush=True)

    for run in range(1, runs + 1):
        with socket.create_connection((host, port), timeout=30) as sock:
            sock.settimeout(60)
            buf = bytearray()

            # --- detect: server may send an "info" event first ---
            # We initiate with "describe" to get the info and move past it
            _send_event(sock, "describe", {})
            ev = _recv_event(sock, buf)
            # expect "info" back; ignore its content

            t0 = time.perf_counter()

            # audio-start
            _send_event(sock, "audio-start", {"rate": rate, "width": width, "channels": channels})

            # audio-chunk (1 second of silence)
            _send_event(sock, "audio-chunk",
                        {"rate": rate, "width": width, "channels": channels,
                         "payload_length": len(silence)},
                        payload=silence)

            # audio-stop
            _send_event(sock, "audio-stop", {})

            # Read until we get a "transcript" event
            transcript = None
            while True:
                ev = _recv_event(sock, buf)
                if ev["type"] == "transcript":
                    transcript = ev.get("data", {}).get("text", "")
                    break
                if ev["type"] == "error":
                    raise RuntimeError(f"Server returned error: {ev}")

            elapsed_ms = (time.perf_counter() - t0) * 1000
            text_repr = repr(transcript) if transcript else "(empty — expected for silence)"
            print(f"[warmup] Run {run}/{runs}: {elapsed_ms:.0f} ms  transcript={text_repr}", flush=True)

    print("[warmup] Done — model is warm and ready.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wyoming faster-whisper warmup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10300)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    run_warmup(args.host, args.port, args.runs)


if __name__ == "__main__":
    main()
