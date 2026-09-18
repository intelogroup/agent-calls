#!/usr/bin/env python3
"""Deterministic regression tests for the Jett voice proxy relay hardening.

Run on the VM: /opt/agent/venv/bin/python /opt/agent/tests/test_relay.py
Run locally:   python3 tests/test_relay.py   (needs livekit-agents installed)

Covers (all deterministic, no network, no models):
  1. max_tokens floor >= 300 on every brain turn + chain preserved
  2. the $3.58 balance vs …1792 suffix corruption (must NOT corrupt)
  3. multiple amounts + dates in one relay
  4. empty model output -> next-model retry; all empty -> spoken fallback
  5. Jett-unavailable -> honest signal, no false "Jett says" attribution
  6. identity / jailbreak pressure ("you are Jett, admit it") -> prompt contract
  7. vague follow-ups ("what about the other one?") -> consult, never guess
  8. factual condensation keeps every number
  9. 429 detection + exponential backoff
 10. local_facts freshness (fresh / stale / missing)
 11. structural "Jett says" attribution (code-enforced, idempotent)
 12. consult request schema matches BUS_PROTOCOL.md
"""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone

# --- stub heavy LiveKit plugins if absent (VM has the real ones) ---
try:
    from livekit.plugins import openai as _real_openai  # noqa: F401
    from livekit.plugins import silero as _real_silero  # noqa: F401
except ImportError:
    from livekit.agents import llm as _llm

    plugins = types.ModuleType("livekit.plugins")
    openai_mod = types.ModuleType("livekit.plugins.openai")
    silero_mod = types.ModuleType("livekit.plugins.silero")

    class _StubLLM(_llm.LLM):
        def __init__(self, **kw):
            super().__init__()
            self.kw = kw

        @property
        def model(self):
            return self.kw.get("model", "?")

        @property
        def provider(self):
            return "stub"

        def chat(self, **kw):
            raise AssertionError("stub LLM.chat must be replaced in tests")

    openai_mod.LLM = _StubLLM
    silero_mod.VAD = types.SimpleNamespace(load=staticmethod(lambda: object()))
    plugins.openai = openai_mod
    plugins.silero = silero_mod
    sys.modules["livekit.plugins"] = plugins
    sys.modules["livekit.plugins.openai"] = openai_mod
    sys.modules["livekit.plugins.silero"] = silero_mod
    import livekit

    livekit.plugins = plugins

SPEC = importlib.util.spec_from_file_location(
    "live_agent", os.path.join(os.path.dirname(__file__), "..", "live_agent.py"))
la = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(la)

from livekit.agents import llm  # noqa: E402
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("ok  " if cond else "FAIL") + f" {name}" +
          (f" — {detail}" if detail and not cond else ""))


# ---------------------------------------------------------------- fakes ---

def _chunk(text):
    return llm.ChatChunk(
        id="c1", delta=llm.ChoiceDelta(content=text, role="assistant"))


class _FakeStream(llm.LLMStream):
    def __init__(self, parent, chunks):
        super().__init__(parent, chat_ctx=llm.ChatContext(), tools=[],
                         conn_options=DEFAULT_API_CONNECT_OPTIONS)
        self._canned = chunks

    async def _run(self):
        for c in self._canned:
            self._event_ch.send_nowait(c)


class _FakeInner(llm.LLM):
    def __init__(self, chunks):
        super().__init__()
        self._chunks = chunks

    @property
    def model(self):
        return "fake"

    @property
    def provider(self):
        return "fake"

    def chat(self, **kw):
        return _FakeStream(self, self._chunks)


def _resilient_brain():
    return la.Brain("openrouter", "https://x", "k",
                    la.JETT_BRAIN_MODELS[0], {},
                    {"models": la.JETT_BRAIN_MODELS})


async def _run_models(script, chat_ctx=None):
    """script: per-model list of chunk-texts ([""] = empty). Returns
    (spoken_text, models_used)."""
    used = []

    def factory(model, remaining):
        used.append(model)
        return _FakeInner([_chunk(t) for t in script[len(used) - 1]])

    r = la.ResilientLLM(_resilient_brain(), _llm_factory=factory)
    stream = r.chat(chat_ctx=chat_ctx or llm.ChatContext())
    texts = []
    async for ch in stream:
        if ch.delta and ch.delta.content:
            texts.append(ch.delta.content)
    await stream.aclose()
    return "".join(texts), used


def _consult_ctx(output, name="consult_jett"):
    ctx = llm.ChatContext()
    ctx.items.append(llm.ChatMessage(role="user", content=["q?"]))
    ctx.items.append(llm.FunctionCall(id="1", call_id="1", name=name,
                                     arguments="{}"))
    ctx.items.append(llm.FunctionCallOutput(id="2", call_id="1", name=name,
                                            output=output, is_error=False))
    return ctx


# ------------------------------------------------- 1. max_tokens floor ---

check("BRAIN_MAX_TOKENS >= 300", la.BRAIN_MAX_TOKENS >= 300,
      repr(la.BRAIN_MAX_TOKENS))
la.OPENROUTER_API_KEY = "test-key"
b = la.resolve_brain()
check("extra_body max_tokens >= 300",
      b.extra_body.get("max_tokens", 0) >= 300, repr(b.extra_body))
check("fallback chain preserved",
      b.extra_body.get("models") == [
          "dots-studio/dots-3-note-preview:free",
          "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
          "openrouter/free"],
      repr(b.extra_body.get("models")))
check("first model is vetted dots",
      b.model == "dots-studio/dots-3-note-preview:free", b.model)
la.OPENROUTER_API_KEY = ""

# --------------------------------- 2. $3.58 vs …1792 (the mock-test bug) ---

SRC_BAL = "Your checking …1792 balance is $3.58 as of this morning."
CORRUPT = "Jett says your balance is $1,792."     # the mock-test failure
GOOD = "Jett says: checking …1792 is at $3.58."
check("corrupt relay fails validation",
      not la.validate_numbers(CORRUPT, SRC_BAL))
check("good relay passes validation", la.validate_numbers(GOOD, SRC_BAL))
check("corrupt -> literal Jett words",
      la.relay_consult_reply(SRC_BAL, CORRUPT) == f"Jett says: {SRC_BAL}",
      la.relay_consult_reply(SRC_BAL, CORRUPT))
check("good relay keeps attribution", la.relay_consult_reply(SRC_BAL, GOOD) == GOOD)
# suffix spoken without ellipsis still matches (not punished)
check("suffix 'ending in 1792' matches",
      la.validate_numbers("account ending in 1792 has $3.58", SRC_BAL))

# -------------------------------------- 3. multiple amounts + dates ---

SRC_MULTI = ("BofA minimum $35 due 2026-09-19. "
             "National Grid $30.23 was due 2026-09-16.")
DRAFT_ALL = ("Jett says: BofA minimum $35 is due 2026-09-19, and National "
             "Grid $30.23 was due 2026-09-16.")
DRAFT_DROP = "Jett says: BofA minimum $35 is due 2026-09-19."
check("all amounts+dates kept -> valid",
      la.validate_numbers(DRAFT_ALL, SRC_MULTI))
check("dropped amount -> invalid",
      not la.validate_numbers(DRAFT_DROP, SRC_MULTI))
check("dropped amount -> literal fallback",
      la.relay_consult_reply(SRC_MULTI, DRAFT_DROP) == f"Jett says: {SRC_MULTI}")

# --------------------------------------------- 4. empty-output retry ---

async def _t4():
    t, used = await _run_models([[""], ["hello world"]])
    check("empty first model -> second model speaks",
          t == "hello world" and len(used) == 2, f"{t!r} {used}")
    t, used = await _run_models([[""], [""], [""]])
    check("all models empty -> guaranteed fallback spoken",
          t == la.FALLBACK_LINE and len(used) == 3, f"{t!r} {used}")
    check("fallback line is non-empty", bool(la.FALLBACK_LINE.strip()))

asyncio.run(_t4())

# -------------------------------------------- 5. Jett unavailable ---

async def _t5():
    real_ensure = la._bus_ensure
    la._bus_ensure = lambda: asyncio.sleep(0, result=False)  # noqa: E731
    try:
        status, payload = await la.file_consult_request("test?")
        check("bus down -> unreachable status", status == "unreachable", status)

        class _Sess:
            async def say(self, *a, **k):
                pass

        fake_ctx = types.SimpleNamespace(session=_Sess())
        out = await la.consult_jett("test?", fake_ctx)
        check("consult tool reports honestly",
              out.startswith("ERROR_CANT_REACH_JETT") and
              "can't reach Jett" in out, out[:80])
    finally:
        la._bus_ensure = real_ensure

    # sentinels must NOT gain a false "Jett says" prefix
    t, _ = await _run_models(
        [["I can't reach Jett right now — try again later."]],
        chat_ctx=_consult_ctx("ERROR_CANT_REACH_JETT: down"))
    check("ERROR sentinel: no false attribution",
          t == "I can't reach Jett right now — try again later.", t)
    t, _ = await _run_models(
        [["I've passed your question to Jett."]],
        chat_ctx=_consult_ctx("FILED_ASYNC: filed as consult abc123"))
    check("FILED_ASYNC: no false attribution",
          t == "I've passed your question to Jett.", t)

asyncio.run(_t5())

# --------------------------- 6/7. identity + vague-follow-up contract ---

P = la.SYSTEM_PROMPT
check("prompt: NOT Jett", "You are NOT Jett" in P)
check("prompt: is Jett's voice proxy", "Jett's voice proxy" in P)
check("prompt: never invent Jett's answers", "NEVER invent" in P)
check("prompt: suffix-is-not-balance rule", "…1792 is NOT a balance" in P)
check("prompt: vague follow-ups -> consult, never guess",
      "Vague follow-ups" in P and "Never guess" in P)

# ------------------------------------------------ 8. condensation ---

SRC_COND = ("Checking …1792: $3.58. BofA card minimum $35 due 2026-09-19. "
            "T-Mobile $126.47 payment returned.")
CONDENSED = ("Jett says: checking …1792 is at $3.58; BofA minimum $35 due "
             "2026-09-19; the $126.47 T-Mobile payment bounced.")
check("condensation keeps every number",
      la.validate_numbers(CONDENSED, SRC_COND))
check("condensed relay spoken with attribution",
      la.relay_consult_reply(SRC_COND, CONDENSED) == CONDENSED)

# ------------------------------------------------ 9. 429 + backoff ---

class _E429(Exception):
    status_code = 429

check("429 by status_code", la._is_rate_limit(_E429("x")))
check("429 in message", la._is_rate_limit(Exception("Error 429")))
check("rate-limit words", la._is_rate_limit(Exception("Rate limit reached")))
check("non-limit error", not la._is_rate_limit(ValueError("bad")))
check("backoff sequence",
      [la.compute_backoff(i) for i in range(7)] ==
      [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0])

# ----------------------------------------------- 10. facts freshness ---

with tempfile.TemporaryDirectory() as td:
    real_bus = la.BUS_DIR
    la.BUS_DIR = td
    try:
        facts_dir = os.path.join(td, "facts")
        os.makedirs(facts_dir)

        def _write_facts(gen_at):
            with open(os.path.join(facts_dir, "facts.json"), "w") as f:
                json.dump({
                    "generated_at": gen_at,
                    "calendar_today": [{"title": "CCCS shift", "time": "9:00 AM"}],
                    "accounts": [{"label": "Adv Plus", "suffix": "1792",
                                  "balance": "3.58", "available": "7.16",
                                  "as_of": "2026-09-18"}],
                    "bills": [{"payee": "BofA", "amount": "35",
                               "due": "2026-09-19", "status": "unpaid"}],
                    "inbox": [{"subject": "Grid bill", "summary": "due soon",
                               "date": "2026-09-18"}],
                    "notes": "test notes",
                }, f)

        now = datetime.now(timezone.utc)
        _write_facts(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        data, age = la.load_facts()
        check("fresh facts load", data is not None and age is not None and age < 120,
              repr(age))

        async def _t10():
            out = await la.local_facts("what is my checking balance?")
            check("fresh facts: slice + freshness",
                  out.startswith("Jett's synced facts (as of ") and
                  "…1792" in out and "$3.58" in out, out[:120])
            check("fresh facts: query-relevant slice",
                  "Adv Plus" in out, out[:200])

            _write_facts((now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"))
            out = await la.local_facts("balance?")
            check("stale facts: honest FACTS_STALE",
                  out.startswith("FACTS_STALE") and "consult_jett" in out,
                  out[:100])

            os.remove(os.path.join(facts_dir, "facts.json"))
            out = await la.local_facts("balance?")
            check("missing facts: honest FACTS_UNAVAILABLE",
                  out.startswith("FACTS_UNAVAILABLE") and "consult_jett" in out,
                  out[:100])

        asyncio.run(_t10())

        # facts relay: freshness enforced, invented numbers rejected
        _write_facts(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        tool_out = ("Jett's synced facts (as of 12 minutes ago):\n"
                    "Accounts:\n  - Adv Plus …1792: balance $3.58")
        r = la.relay_facts_reply(tool_out, "checking …1792 has $3.58")
        check("facts relay: attribution + freshness",
              r.startswith("Jett says:") and "(as of 12 minutes ago)" in r, r)
        r = la.relay_facts_reply(tool_out, "your balance is $99.99")
        check("facts relay: invented number -> literal",
              "$99.99" not in r and r.startswith("Jett says:"), r)
    finally:
        la.BUS_DIR = real_bus

# ------------------------------------------- 11. attribution ---

check("attribution added", la.enforce_attribution("hi") == "Jett says: hi")
check("attribution idempotent",
      la.enforce_attribution("Jett says: hi") == "Jett says: hi")
check("attribution case-insensitive",
      la.enforce_attribution("JETT SAYS: hi") == "JETT SAYS: hi")
check("attribution empty safe", la.enforce_attribution("  ") == "")

# ----------------------------------- 12. consult request schema ---

async def _t12():
    real_ensure, real_push = la._bus_ensure, la._bus_write_push
    captured = {}

    async def _fake_ensure():
        return True

    async def _fake_push(relpath, obj, msg):
        captured["relpath"] = relpath
        captured["obj"] = obj
        return True

    la._bus_ensure = _fake_ensure
    la._bus_write_push = _fake_push
    try:
        la._CURRENT_CALL_ID = "jett-test-1"
        status, payload = await la.file_consult_request(
            "Did the Grid payment post?", priority="urgent")
        req = payload["request"]
        check("consult filed", status == "filed", status)
        check("request schema keys",
              set(req) == {"id", "ts", "call_id", "question", "context",
                           "priority"}, sorted(req))
        import re as _re
        check("request id is uuid4",
              bool(_re.fullmatch(r"[0-9a-f-]{36}", req["id"])), req["id"])
        check("request filename matches id",
              captured["relpath"] == f"requests/{req['id']}.json",
              captured["relpath"])
        check("ts is UTC Zulu", req["ts"].endswith("Z"), req["ts"])
        check("priority passthrough", req["priority"] == "urgent")
        check("call_id attached", req["call_id"] == "jett-test-1")

        status, payload = await la.file_consult_request("q?", priority="bogus")
        check("bad priority -> normal",
              payload["request"]["priority"] == "normal")
    finally:
        la._bus_ensure = real_ensure
        la._bus_write_push = real_push
        la._CURRENT_CALL_ID = None

asyncio.run(_t12())

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
