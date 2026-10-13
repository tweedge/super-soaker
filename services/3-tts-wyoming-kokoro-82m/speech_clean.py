#!/usr/bin/env python3
"""Text sanitation for speech synthesis (stdlib only).

Shared by kokoro_wyoming.py and selftest.py. Everything a model may leak
into visible content — markdown, thinking/reasoning blocks, tool-call
blocks, JSON payloads, stray angle brackets from truncated streams — must
be removed before phonemization: espeak reads '*' as "asterisk", '<' as
"less than", and so on, straight into the audio.

Angle-bracket literals in tests should be built with chr(60)/chr(62) so
the test source itself cannot be mistaken for markup by tooling.
"""
import re

_SENTENCE_END = re.compile(r"(.*?[.!?;:](?:\s|$))", re.DOTALL)

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF\U0000FE0F\U0000200D]", flags=re.UNICODE)

# Reasoning / tool-call plumbing that models leak into content: thinking
# tags (paired OR half-open — a truncated stream can end mid-tag), tool-call
# blocks, and JSON payloads. Case-insensitive, across newlines; a stray
# angle bracket must never reach the phonemizer.
_THINK_TAG = r"(?:think(?:ing)?|reasoning|thought|analysis)"
_THINK_BLOCK = re.compile(
    r"<" + _THINK_TAG + r"\b[^>]*>.*?(?:</" + _THINK_TAG + r"\s*>|$)"
    r"|</?" + _THINK_TAG + r"\s*>",
    re.I | re.S)
_TOOL_BLOCK = re.compile(
    r"```\s*(?:json|tool_call|function_call)?\s*\{.*?\}\s*```"   # fenced tool JSON
    r"|```.*?```"                                                # any fenced blob
    r"|<\s*(?:tool_call|function_call|function)\b[^>]*>.*?(?:</\s*(?:tool_call|function_call|function)\s*>|$)"
    r"|</?\s*(?:tool_call|function_call|function)\s*>",
    re.I | re.S)
_TO_MARKER = re.compile(r"\bto=[A-Za-z_][\w.\-]*\s*")           # to=tools  tokens
_JSON_BLOB = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")     # bare/nested JSON
_JSON_SHELL = re.compile(r"\[[\s,.\]]*\]")                      # leftover [ , ] arrays
_ANGLE_TAG = re.compile(r"</?[A-Za-z][^<>]*>/?")                 # well-formed tags
_ANGLE_FRAG = re.compile(r"</?[A-Za-z][\w.:\-]*")               # truncated tags (no '>')
_STRAY_ANGLE = re.compile(r"<\s*/?|/?>")                         # bare <, </, >, />


def clean_for_speech(text: str) -> str:
    """Strip markdown, model plumbing, and symbols the phonemizer would read.

    Sanitize at the TTS boundary so the fix holds no matter which model or
    integration produced the text — including truncated streams that stop
    mid-block.
    """
    if not text or not text.strip():
        return text
    t = text
    # model plumbing first: whole blocks, not char soup
    t = _THINK_BLOCK.sub(" ", t)              # thinking/reasoning (any state)
    t = _TOOL_BLOCK.sub(" ", t)               # tool tags + fenced JSON
    t = _TO_MARKER.sub(" ", t)                # to=tools  tokens
    for _ in range(8):                        # nested JSON, innermost first
        t2 = _JSON_BLOB.sub(" ", t)
        if t2 == t:
            break
        t = t2
    t = _JSON_SHELL.sub(" ", t)               # leftover [ , ] arrays
    t = _ANGLE_TAG.sub(" ", t)                # well-formed tags
    t = _ANGLE_FRAG.sub(" ", t)               # truncated tags (name, no '>')
    t = _STRAY_ANGLE.sub(" ", t)              # bare angle brackets
    # then markdown and phonemizer-hostile symbols
    t = re.sub(r"```.*?```", " code. ", t, flags=re.S)   # fenced blocks
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)        # [text](url) -> text
    t = re.sub(r"https?://\S+", "", t)                    # bare URLs: noise
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)    # **bold**
    t = re.sub(r"__(.+?)__", r"\1", t, flags=re.S)        # __bold__
    t = re.sub(r"\*(.+?)\*", r"\1", t, flags=re.S)        # *italic*
    t = re.sub(r"[*_`#|]{1,3}", "", t)                    # stray markers
    t = re.sub(r"(?m)^\s*[-+\u2022]\s*", "", t)           # bullet markers
    t = re.sub(r"(?m)^\s*\d+[.)]\s+", "", t)              # numbered lists
    t = re.sub(r"(?m)^>{1,}\s?", "", t)                   # blockquotes
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"[{}\[\]]", " ", t)                       # JSON/markup shells
    t = re.sub(r"\s+", " ", t)                            # newlines -> spaces
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)                # space before punct
    return t.strip()
