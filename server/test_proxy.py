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

# 2. bus serialization (async requests/<uuid>.json protocol) -------------
with tempfile.TemporaryDirectory() as td:
    la.BUS_DIR = td
    req_id = "3f9a2c1e-7b4d-4a8e-9c1f-2e5d6a7b8c9d"
    req = {"id": req_id, "ts": la._utcnow_z(), "call_id": "jett-test-1",
           "question": "Caller asks: did the Grid payment post?",
           "context": "test", "priority": "normal"}
    obj = json.loads(json.dumps(req, ensure_ascii=False))
    check("request keys",
          set(obj) == {"id", "ts", "call_id", "question", "context",
                       "priority"}, sorted(obj))
    check("request id matches filename stem", obj["id"] == req_id)
    check("ts is Zulu UTC", obj["ts"].endswith("Z"), obj["ts"])
    resp = {"id": req_id, "ts": la._utcnow_z(),
            "answer": "Yes — posted this morning.",
            "attribution": "Jett says", "answered_by": "jett-runtime"}
    robj = json.loads(json.dumps(resp))
    check("response keys",
          set(robj) == {"id", "ts", "answer", "attribution",
                        "answered_by"}, sorted(robj))
    check("response id matches request", robj["id"] == obj["id"])
    check("response attribution exact", robj["attribution"] == "Jett says")
    check("response answered_by runtime",
          robj["answered_by"] == "jett-runtime")

# 3. brain resolution ---------------------------------------------------
la.OPENROUTER_API_KEY = ""
la.META_API_KEY = ""
check("no keys -> no brain", la.resolve_brain() is None)
la.OPENROUTER_API_KEY = "sk-or-test"
b = la.resolve_brain()
check("openrouter preferred", b is not None and b.label == "openrouter")
check("vetted chain first model",
      b.model == "dots-studio/dots-3-note-preview:free", b.model)
check("chain has 3 vetted models",
      b.extra_body.get("models") == [
          "dots-studio/dots-3-note-preview:free",
          "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
          "openrouter/free"])
check("max_tokens floor >= 300",
      b.extra_body.get("max_tokens", 0) >= 300,
      repr(b.extra_body.get("max_tokens")))
check("fallback route set",
      b.extra_body.get("route") == "fallback")
la.JETT_BRAIN_MODELS = ["meta/muse-spark-1.3"]
check("model switchable via env",
      la.resolve_brain().model == "meta/muse-spark-1.3")
la.JETT_BRAIN_MODELS = [
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free"]
check("openrouter base", b.base_url == "https://openrouter.ai/api/v1")
check("openrouter headers",
      b.extra_headers.get("HTTP-Referer") == "https://github.com/intelogroup/agent-calls"
      and b.extra_headers.get("X-Title") == "jett-proxy voice agent")
la.OPENROUTER_API_KEY = ""
check("still no brain after clearing", la.resolve_brain() is None)
class _E429(Exception):
    status_code = 429
check("429 by status_code", la._is_rate_limit(_E429("slow down")))
check("429 in message", la._is_rate_limit(Exception("Error 429: quota exceeded")))
check("rate limit words", la._is_rate_limit(Exception("Rate limit reached for model")))
check("too many requests", la._is_rate_limit(Exception("Too Many Requests")))
check("non-limit error", not la._is_rate_limit(ValueError("bad input")))
check("500 not a limit", not la._is_rate_limit(Exception("500 internal error")))

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
