#!/usr/bin/env python3
"""Unit tests for speech_clean.clean_for_speech (stdlib only).

Angle-bracket literals are built with chr(60)/chr(62) so this source never
contains markup-looking text. Exit 0 = pass. Invoked by selftest.py before
the Wyoming end-to-end checks.
"""
import sys

from speech_clean import clean_for_speech

LT, GT = chr(60), chr(62)   # < >

def tag(name: str, body: str, close: bool = True) -> str:
    out = LT + name + GT + body
    if close:
        out += LT + "/" + name + GT
    return out

CASES = [
    # (input, must_contain, must_not_contain)
    ("Turn on the kitchen lights.", "kitchen lights", None),
    ("**Bold** and *italic* keep words.", "Bold and italic", "*"),
    ("* item one\n* item two", "item one item two", None),
    ("The answer is **42**.", "42", None),
    ("Let me check.\n\nThe kitchen is on.", "The kitchen is on", None),
    # thinking / reasoning leaks (the reported bug class)
    (tag("thinking", "counting steps") + " It is on.", "It is on", "counting"),
    (tag("think", "hmm") + " Sure.", "Sure", "hmm"),
    (tag("reasoning", "why") + " Done.", "Done", "why"),
    (tag("analysis", "deliberating") + " Done.", "Done", "deliberating"),
    # half-open / truncated blocks (stream cut mid-tag): body after an
    # UNCLOSED thinking tag is thinking content — strip through end of text
    (LT + "thinking" + GT + " cut off here", "", "cut off"),
    (tag("thinking", "secret", close=False), "", "secret"),
    # truncated close tag: the name is markup debris, what follows is prose
    (LT + "/" + "thinking" + " done.", "done", "thinking"),
    # tool-call leaks
    ('{"tool_calls": [{"name": "HassTurnOn", "args": {"entity_id": "light.kitchen"}}]} turning them on.',
     "turning them on", "HassTurnOn"),
    ("to=tools\nIt is done.", "It is done", "to="),
    ("to=HassGetWeather Getting the weather.", "Getting the weather", "HassGet"),
    ('Call {"name": "HassGetWeather"} to check.', "Call to check", "name"),
    ("```\n{\"temperature\": 21}\n``` The hall is 21.", "The hall is 21", "temperature"),
    (tag("function_call", '{"a": 1}') + " ok.", "ok", "function"),
    # stray / malformed angle brackets (literal report: reply began with '<')
    (LT + " It is on.", "It is on", None),
    (LT + "/" + " It is on.", "It is on", None),
    (LT + "/" + "function" + GT + "done.", "done", "function"),
    (tag("unk", "hello world"), "hello world", "unk"),
    ("half-open " + LT + "tag at end", "half-open at end", None),
    # JSON in prose
    ('mix text {"a": {"b": 1}} tail', "mix text tail", "{"),
    ("done.", "done.", None),
]

bad = 0
for inp, want, forbidden in CASES:
    out = clean_for_speech(inp)
    problems = []
    if want and want not in out:
        problems.append(f"want {want!r}")
    if forbidden and forbidden in out:
        problems.append(f"forbid {forbidden!r}")
    if LT in out or GT in out:
        problems.append("stray angle bracket")
    if problems:
        bad += 1
        print(f"FAIL {inp[:60]!r}\n   -> {out!r}\n   {'; '.join(problems)}")
if bad:
    print(f"{bad} failing cases")
    sys.exit(1)
print(f"clean_for_speech: {len(CASES)} cases OK")
