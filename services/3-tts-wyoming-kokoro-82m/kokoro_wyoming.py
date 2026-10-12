#!/usr/bin/env python3
"""Wyoming server for Kokoro TTS (GPU).

Changes from v1:
- CUDA arena capped (KOKORO_GPU_MEM_MB) so VRAM can't creep into the NVENC budget
- fails fast at startup if the CUDA provider is missing (no silent CPU fallback)
- synthesis runs in an executor: the event loop keeps answering Describe while busy
- Wyoming streaming (synthesize-start/chunk/stop): audio begins at the first
  sentence boundary while the LLM is still generating
- warmup synth before the port opens, so the first real request is warm
- per-request voice validated; samples clipped before int16 (no wraparound clicks)
"""
import asyncio
import logging
import os
import re
import signal
from functools import partial

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

_LOGGER = logging.getLogger(__name__)

WYOMING_PORT = os.environ.get("WYOMING_PORT", "10210")
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))
KOKORO_QUANTIZATION = os.environ.get("KOKORO_QUANTIZATION", "fp32")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")  # default when client names none
KOKORO_GPU_MEM_MB = int(os.environ.get("KOKORO_GPU_MEM_MB", "2048"))

# voice prefix -> BCP-47, per hexgrad/Kokoro-82M voice naming
_LANGS = {"a": "en_US", "b": "en_GB", "j": "ja", "z": "zh", "e": "es", "f": "fr",
          "h": "hi", "i": "it", "p": "pt_BR"}

_SENTENCE_END = re.compile(r"(.*?[.!?;:](?:\s|$))", re.DOTALL)


def _to_int16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class KokoroEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, kokoro, voices: set, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.kokoro = kokoro
        self.voices = voices
        self._stream_text = ""
        self._stream_voice = None
        self._stream_started = False

    def _pick_voice(self, synth_voice) -> str:
        name = synth_voice.name if synth_voice else KOKORO_VOICE
        if name not in self.voices:
            _LOGGER.warning("Unknown voice %r, using %s", name, KOKORO_VOICE)
            name = KOKORO_VOICE
        return name

    async def _synth(self, text: str, voice: str):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self.kokoro.create, text, voice=voice, speed=KOKORO_SPEED)
        )

    async def _speak(self, text: str, voice: str, send_start: bool) -> int:
        """Synthesize one piece and write audio events. Returns sample rate."""
        samples, rate = await self._synth(text, voice)
        if send_start:
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        pcm = _to_int16(samples)
        for i in range(0, len(pcm), 16384):
            await self.write_event(
                AudioChunk(audio=pcm[i:i + 16384], rate=rate, width=2, channels=1).event()
            )
        return rate

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event(self.wyoming_info_event)
                return True

            if Synthesize.is_type(event.type):
                if self._stream_started:
                    return True  # streaming already handled this utterance
                synthesize = Synthesize.from_event(event)
                voice = self._pick_voice(synthesize.voice)
                _LOGGER.info("Synthesize (%s): %r", voice, synthesize.text)
                await self._speak(synthesize.text, voice, send_start=True)
                await self.write_event(AudioStop().event())
                return True

            if SynthesizeStart.is_type(event.type):
                start = SynthesizeStart.from_event(event)
                self._stream_text = ""
                self._stream_voice = self._pick_voice(start.voice)
                self._stream_started = False
                return True

            if SynthesizeChunk.is_type(event.type):
                self._stream_text += SynthesizeChunk.from_event(event).text
                # speak every complete sentence we have so far
                while True:
                    m = _SENTENCE_END.match(self._stream_text)
                    if not m or not m.group(1).strip():
                        break
                    sentence = m.group(1)
                    self._stream_text = self._stream_text[len(sentence):]
                    await self._speak(sentence.strip(), self._stream_voice,
                                      send_start=not self._stream_started)
                    self._stream_started = True
                return True

            if SynthesizeStop.is_type(event.type):
                rest = self._stream_text.strip()
                if rest:
                    await self._speak(rest, self._stream_voice,
                                      send_start=not self._stream_started)
                    self._stream_started = True
                if self._stream_started:
                    await self.write_event(AudioStop().event())
                await self.write_event(SynthesizeStopped().event())
                return True

        except Exception:
            _LOGGER.exception("Synthesis failed")
            # tell the client we're done rather than leaving it hanging
            try:
                await self.write_event(AudioStop().event())
            except Exception:
                pass
            return False
        return True


def load_kokoro() -> Kokoro:
    model_dir = os.environ.get("KOKORO_MODEL_DIR", "/var/lib/super-soaker/tts")
    quant = {"fp32": "kokoro-v1.0.onnx", "fp16": "kokoro-v1.0.fp16.onnx",
             "int8": "kokoro-v1.0.int8.onnx"}
    if KOKORO_QUANTIZATION not in quant:
        raise SystemExit(f"Invalid KOKORO_QUANTIZATION {KOKORO_QUANTIZATION!r}")
    model_path = os.path.join(model_dir, quant[KOKORO_QUANTIZATION])
    if not os.path.exists(model_path):
        raise SystemExit(f"Model file not found: {model_path}")

    if "CUDAExecutionProvider" not in rt.get_available_providers():
        raise SystemExit(
            "CUDAExecutionProvider not available "
            f"(providers: {rt.get_available_providers()}). Refusing to run on CPU."
        )
    providers = [(
        "CUDAExecutionProvider",
        {
            "gpu_mem_limit": str(KOKORO_GPU_MEM_MB * 1024 * 1024),
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
        },
    )]
    sess = rt.InferenceSession(model_path, providers=providers)
    return Kokoro.from_session(sess, os.path.join(model_dir, "voices.npy"))


async def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    if not 0.5 <= KOKORO_SPEED <= 2.0:
        raise SystemExit(f"KOKORO_SPEED must be within 0.5-2.0, got {KOKORO_SPEED}")

    _LOGGER.info("Loading Kokoro (%s, speed %sx, arena cap %d MiB)...",
                 KOKORO_QUANTIZATION, KOKORO_SPEED, KOKORO_GPU_MEM_MB)
    kokoro = load_kokoro()
    voices = sorted(kokoro.get_voices())
    if KOKORO_VOICE not in voices:
        raise SystemExit(f"KOKORO_VOICE {KOKORO_VOICE!r} not in model voices")

    _LOGGER.info("Warming up...")
    kokoro.create("Ready.", voice=KOKORO_VOICE, speed=KOKORO_SPEED)
    _LOGGER.info("Kokoro loaded with %d voices", len(voices))

    wyoming_info = Info(tts=[TtsProgram(
        name="kokoro",
        version="2.0",
        description=f"Kokoro TTS ({KOKORO_QUANTIZATION}, GPU)",
        attribution=Attribution(name="Kokoro",
                                url="https://huggingface.co/hexgrad/Kokoro-82M"),
        installed=True,
        supports_synthesize_streaming=True,
        voices=[TtsVoice(
            name=v,
            description=f"Kokoro voice: {v}",
            attribution=Attribution(name="Kokoro", url=""),
            installed=True,
            languages=[_LANGS.get(v[:1], "en_US")],
            version="1.0",
        ) for v in voices],
    )])

    uri = f"tcp://0.0.0.0:{WYOMING_PORT}"
    _LOGGER.info("Starting server on %s", uri)
    server = AsyncServer.from_uri(uri)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    serve = asyncio.create_task(
        server.run(partial(KokoroEventHandler, wyoming_info, kokoro, set(voices)))
    )
    await asyncio.wait([serve, asyncio.create_task(stop.wait())],
                       return_when=asyncio.FIRST_COMPLETED)
    serve.cancel()


if __name__ == "__main__":
    asyncio.run(main())
