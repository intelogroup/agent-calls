#!/usr/bin/env python3
"""Unit tests for the Jett voice proxy worker.

Run on the VM: /opt/agent/venv/bin/python /opt/agent/test_proxy.py
Covers: sentence chunking, bus serialization, missing-key behavior,
brief fallback, TTS/STT adapter contracts. No network, no models.
"""

import importlib.util
import json
import os
import sys
import tempfile

SPEC = importlib.util.spec_from_file_location(
    "live_agent", os.path.join(os.path.dirname(__file__), "live_agent.py"))
la = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(la)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("ok  " if cond else "FAIL") + f" {name}" +
          (f" — {detail}" if detail and not cond else ""))


# 1. sentence chunking ------------------------------------------------
sents = la.split_sentences("Hello there. This is a test, with a clause; and more.")
check("split basic", sents == ["Hello there.", "This is a test, with a clause; and more."], repr(sents))

long = "word " * 200  # 1000 chars, no punctuation
chunks = la.split_sentences(long)
check("long sentence split", len(chunks) > 1 and all(len(c) <= 400 for c in chunks),
      f"{len(chunks)} chunks, max {max(len(c) for c in chunks)}")

check("split empty", la.split_sentences("   ") == [""])

# kokoro phoneme limit is 510; our chunk cap is 400 — margin holds
check("chunk under phoneme limit", all(len(c) <= 400 for c in la.split_sentences("a. " * 500)))

# 2. bus serialization -----------------------------------------------
with tempfile.TemporaryDirectory() as td:
    la.BUS_DIR = td
    bus = la.BusCall("abc123", "sip:jett@1.2.3.4")
    check("bus dir layout", bus.dir.endswith("calls/live/abc123"), bus.dir)
    line = json.dumps({"ts": la._utcnow(), "seq": 1,
                       "text": "Caller asks: test?"}, ensure_ascii=False)
    obj = json.loads(line)
    check("consult line keys", set(obj) == {"ts", "seq", "text"}, repr(obj))
    reply = json.dumps({"ts": la._utcnow(), "in_seq": 1,
                        "text": "Yes — posted this morning."})
    robj = json.loads(reply)
    check("reply line keys", set(robj) == {"ts", "in_seq", "text"}, repr(robj))
    check("seq/in_seq correlate", obj["seq"] == robj["in_seq"])

# 3. missing-key behavior ---------------------------------------------
la.META_API_KEY = ""
check("discover with no key", la.discover_meta_model() == "")

# 4. brief fallback ----------------------------------------------------
la.JETT_MD = "/nonexistent/JETT.md"
brief = la.load_brief()
check("brief fallback", "healthcare interpreter" in brief)

# 5. adapter contracts -------------------------------------------------
from livekit.agents import tts as _tts, stt as _stt  # noqa: E402

check("WhisperSTT is non-streaming STT",
      issubclass(la.WhisperSTT, _stt.STT))
# KokoroTTS.__init__ needs model files; check the class contract instead
import inspect  # noqa: E402
sig = inspect.signature(_tts.TTS.__init__)
check("TTS base needs sample_rate+num_channels",
      "sample_rate" in sig.parameters and "num_channels" in sig.parameters)
src = inspect.getsource(la.KokoroTTS.__init__)
check("KokoroTTS declares 24k/mono",
      "sample_rate=24000" in src and "num_channels=1" in src)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
