"""Worst-case long-form probe text shared by the service and its self-test.

One sentence with no internal [.!?;:] punctuation and 503 phonemes under the
pinned phonemizer, so kokoro-onnx emits exactly one full-size batch (its hard
cap is 510 phonemes). That batch is the largest activation footprint the model
can ask the CUDA arena for. kokoro_wyoming.py synthesizes it once before the
port opens so the arena's high-water mark is reached at service start instead
of on the first long answer, and selftest.py sends it through the Wyoming
streaming path as a long-form gate.
"""

WARMUP_TEXT = (
    "When a home assistant answers a question about politics and public life "
    "it should speak clearly and naturally using complete sentences that "
    "carry the listener from one idea to the next without rushing and it "
    "should keep the volume steady while it explains how a political party "
    "organizes its supporters how it chooses the candidates it endorses how "
    "it raises the money it needs to operate and how it measures whether its "
    "message is reaching the people it hopes to persuade"
)
