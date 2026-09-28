#!/usr/bin/env python3
"""Jett's voice proxy — LiveKit agent worker.

One SIP line (sip:jett@129.159.189.244) -> LiveKit room (prefix jett-) ->
this worker. Real-time voice pipeline:

    faster-whisper tiny.en (local STT, via StreamAdapter + silero VAD)
      -> ResilientLLM (OpenRouter free-model chain, empty-output retry)
      -> kokoro-onnx (local warm-voice TTS)

The worker answers from its JETT.md brief + Jett's synced facts snapshot
when confident. For anything needing the real Jett's live memory, tools, or
judgment it calls the consult_jett() function tool: the question is written
to requests/<uuid>.json in the private agent-call-bus repo (async protocol,
see server/BUS_PROTOCOL.md), pushed immediately, and Jett's side answers via
responses/<uuid>.json. If Jett answers within CONSULT_TIMEOUT_SEC the reply
is relayed live ("Jett says …", numbers validated); otherwise the request
stays filed and Jett's answer is delivered afterward (voice callback /
WhatsApp) — the caller is told honestly which happened.

Relay guarantees (enforced in CODE, not just the prompt):
  - every brain turn requests max_tokens >= 300 (floor, not a suggestion);
  - empty model output retries with the next model in JETT_BRAIN_MODELS;
    if all are empty, a guaranteed fallback line is spoken — dead air never;
  - anything sourced from consult_jett or local_facts is spoken prefixed
    with "Jett says", with numbers/dates validated against the source;
  - rate limits back off exponentially between models; every turn logs
    model used, latency, and whether empty-retry/fallback fired.

Without OPENROUTER_API_KEY (or the META_API_KEY fallback) the worker
still registers, but any inbound call gets a graceful "brain not
connected" message instead of silence.

Usage: python live_agent.py start
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobExecutorType,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    llm,
    stt,
    tts,
    vad,
)
from livekit.plugins import openai as openai_plugin
from livekit.plugins import silero
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [proxy] %(message)s",
)
log = logging.getLogger("jett-proxy")

# ---------------------------------------------------------------- config ---

AGENT_DIR = os.environ.get("AGENT_DIR", "/opt/agent")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
META_API_KEY = os.environ.get("META_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Brain model fallback chain (OpenRouter route:"fallback" tries them in order).
# Vetted free models first — the random "openrouter/free" router served tiny /
# code models that blanked or burned the token budget in reasoning. Random
# router stays as last resort. Override with JETT_BRAIN_MODELS (comma-sep);
# the legacy singular JETT_BRAIN_MODEL still works as a one-model chain.
_DEFAULT_BRAIN_MODELS = (
    "dots-studio/dots-3-note-preview:free,"
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,"
    "openrouter/free"
)
if os.environ.get("JETT_BRAIN_MODELS"):
    JETT_BRAIN_MODELS = [m.strip() for m in os.environ["JETT_BRAIN_MODELS"].split(",") if m.strip()]
elif os.environ.get("JETT_BRAIN_MODEL"):
    JETT_BRAIN_MODELS = [os.environ["JETT_BRAIN_MODEL"].strip()]
else:
    JETT_BRAIN_MODELS = [m.strip() for m in _DEFAULT_BRAIN_MODELS.split(",") if m.strip()]
# max_tokens floor: free models blanked on tiny budgets in testing. This is a
# floor, not a suggestion — resolve_brain() enforces >= 300 on every turn.
BRAIN_MAX_TOKENS = max(300, int(os.environ.get("BRAIN_MAX_TOKENS", "300")))
# Spoken when every model returns empty output. Dead air is never acceptable.
FALLBACK_LINE = (
    "Sorry — my brain came back empty just now. "
    "Could you say that once more?"
)
# OpenRouter's recommended attribution headers (their docs ask for these).
OPENROUTER_REFERER = os.environ.get(
    "OPENROUTER_REFERER", "https://github.com/intelogroup/agent-calls")
OPENROUTER_TITLE = os.environ.get("OPENROUTER_TITLE", "jett-proxy voice agent")
# Optional fallback: Meta's direct Model API (unverified endpoint).
META_MODEL = os.environ.get("META_MODEL", "muse-spark-1.3")
META_BASE_URL = "https://api.ai.meta.com/v1"
JETT_MD = os.path.join(AGENT_DIR, "JETT.md")
GREETING = os.environ.get(
    "AGENT_GREETING", "Hey, it's Jett's line. What do you want to talk about?"
)
BUS_DIR = os.environ.get("BUS_DIR", os.path.join(AGENT_DIR, "bus"))
BUS_REPO_SSH = os.environ.get(
    "BUS_REPO_SSH", "git@github.com:intelogroup/agent-call-bus.git"
)
BUS_KEY = os.path.expanduser("~/.ssh/bus_key")
CONSULT_TIMEOUT_SEC = int(os.environ.get("CONSULT_TIMEOUT_SEC", "180"))
MAX_CALL_SEC = int(os.environ.get("AGENT_MAX_CALL_SEC", "600"))
# Synced facts snapshot (published by Track B into the bus repo).
FACTS_STALE_SEC = int(os.environ.get("FACTS_STALE_SEC", "3600"))
KOKORO_MODEL = os.path.join(AGENT_DIR, "models", "kokoro-v1.0.int8.onnx")
KOKORO_VOICES = os.path.join(AGENT_DIR, "models", "voices-v1.0.bin")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
AGENT_NAME = "jett-proxy"

# one call at a time; a second caller gets a polite busy message
_CALL_LOCK = threading.Lock()
_ACTIVE_CALL: str | None = None
# call id of the call currently being served (for consult context)
_CURRENT_CALL_ID: str | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utcnow_z() -> str:
    # bus protocol wants trailing Z, e.g. 2026-09-18T16:40:00Z
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------- structured event logging ---
# Bottleneck instrumentation for the voice line. One JSON object per line
# on stderr (journald picks it up via the service unit). NEVER log secrets,
# keys, tokens, or message content — only ids, counts, lengths, latencies,
# and outcomes. Hot-path overhead is a single json.dumps per event; events
# fire at stage boundaries (boot, call start/end, per brain attempt,
# consult lifecycle), never per audio frame.
_SECRET_FIELD_RE = re.compile(
    r"(key|token|secret|password|credential|cookie|auth)", re.I)
_CONTENT_FIELD_RE = re.compile(
    r"^(question|answer|text|transcript|content|summary)$", re.I)


def _redact_value(v):
    if isinstance(v, str) and len(v) > 96:
        return v[:12] + "\u2026[truncated]"
    return v


def log_event(event: str, **fields) -> None:
    """Emit one structured JSON log line to stderr. Never raises."""
    try:
        clean: dict = {}
        for k, v in fields.items():
            if _SECRET_FIELD_RE.search(k):
                clean[k] = "[redacted]"
            elif _CONTENT_FIELD_RE.match(k):
                clean[k + "_len"] = len(v) if v is not None else 0
            else:
                clean[k] = _redact_value(v)
        rec = {"ts": _utcnow(), "event": event}
        rec.update(clean)
        print(json.dumps(rec, default=str), file=sys.stderr, flush=True)
    except Exception:
        pass  # logging must never break a call


# ------------------------------------------- relay hardening (pure) ---

ATTRIBUTION_PREFIX = "Jett says"

_NUM_TOKEN_RE = re.compile(r"""
    (?:\.\.\.|…)?          # account-suffix ellipsis, e.g. …1792
    \$?                    # optional currency sign
    \d[\d,]*               # digits, keeping thousand separators
    (?:\.\d+)?             # optional decimals
    |
    \b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b   # numeric dates 09/19, 2026-09-19
""", re.VERBOSE)


def _norm_num(tok: str) -> str:
    """Normalize a number token for comparison. Keeps thousand separators
    (so 1,792 != 1792) and decimals, drops $, whitespace, …/.... Trailing
    commas are sentence punctuation, not part of the number."""
    t = tok.strip().lower().replace("…", "").replace("$", "").replace(" ", "")
    if t.startswith("..."):
        t = t[3:]
    return t.rstrip(",")


def extract_num_tokens(text: str) -> set[str]:
    """All number/date tokens in text, normalized. Used to catch relay
    corruption like the …1792 suffix -> $1,792 balance bug."""
    out = set()
    for m in _NUM_TOKEN_RE.finditer(text or ""):
        n = _norm_num(m.group(0))
        if n:
            out.add(n)
    return out


def validate_numbers(spoken: str, source: str,
                     require_all_source: bool = True) -> bool:
    """Check number/date fidelity between a spoken relay and its source.

    require_all_source=True (consult replies): every number in the SOURCE
        must survive into the spoken text — nothing dropped, nothing merged.
    require_all_source=False (facts slices): every number in the SPOKEN text
        must exist in the source — nothing invented or corrupted.
    """
    src = extract_num_tokens(source)
    got = extract_num_tokens(spoken)
    if require_all_source:
        return src <= got
    return got <= src


def enforce_attribution(spoken: str) -> str:
    """Code-enforced 'Jett says' prefix for anything sourced from Jett."""
    s = (spoken or "").strip()
    if not s:
        return s
    if s.lower().startswith("jett says"):
        return s
    return f"{ATTRIBUTION_PREFIX}: {s}"


def relay_consult_reply(answer: str, draft: str) -> str:
    """Build the spoken relay of a consult_jett answer.

    answer: Jett's exact words (responses/<uuid>.json). draft: the model's
    relay text. Numbers are validated; on ANY mismatch we read Jett
    literally rather than risk corruption. Attribution is enforced.
    """
    answer = (answer or "").strip()
    draft = (draft or "").strip()
    if not answer:
        return (f"{ATTRIBUTION_PREFIX}: I didn't get a clear answer from "
                "Jett — I'll ask him again.")
    if not draft or not validate_numbers(draft, answer,
                                         require_all_source=True):
        if draft and draft != answer:
            log.warning("consult relay numbers mismatch; reading Jett literally")
        return enforce_attribution(answer)
    return enforce_attribution(draft)


def relay_facts_reply(tool_output: str, draft: str) -> str:
    """Build the spoken relay of a local_facts tool result.

    The draft may be a slice, so we check the spoken numbers all exist in
    the source (no invented/corrupted numbers) and that the freshness stamp
    survives into speech.
    """
    tool_output = (tool_output or "").strip()
    draft = (draft or "").strip()
    if not draft:
        return enforce_attribution(tool_output)
    if not validate_numbers(draft, tool_output, require_all_source=False):
        log.warning("facts relay numbers mismatch; reading facts literally")
        return enforce_attribution(tool_output)
    out = enforce_attribution(draft)
    m = re.search(r"\(as of ([^)]+)\)", tool_output)
    if m and "as of" not in out.lower():
        out = out.rstrip().rstrip(".") + f" (as of {m.group(1)})."
    return out


def compute_backoff(attempt: int, base: float = 1.0,
                    cap: float = 30.0) -> float:
    """Deterministic exponential backoff (seconds) for rate-limit retries."""
    return min(cap, base * (2 ** max(0, attempt)))


def freshness_str(age_sec: float) -> str:
    if age_sec < 60:
        return "just now"
    mins = int(age_sec // 60)
    if mins < 60:
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    hrs = int(mins // 60)
    return f"{hrs} hour{'s' if hrs != 1 else ''} ago"


def _facts_path() -> str:
    return os.path.join(BUS_DIR, "facts", "facts.json")


def load_facts() -> tuple[dict | None, float | None]:
    """Load the synced facts snapshot. Returns (data, age_seconds);
    (None, None) when missing/unreadable."""
    try:
        with open(_facts_path(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        log.debug("facts snapshot unreadable: %s", e)
        return None, None
    gen = data.get("generated_at")
    age = None
    if gen:
        try:
            gen_dt = datetime.fromisoformat(str(gen).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - gen_dt).total_seconds()
        except ValueError:
            age = None
    return data, age

# ------------------------------------------------------- JETT.md brief ---

DEFAULT_BRIEF = """\
You are Jett's voice proxy. Jett is Jim — a certified healthcare interpreter
(CoreCHI), studying for the USMLE (Step 1) to apply for internal medicine
residency through ERAS. He builds software side projects (ResidencyPhoto,
AIVerse, InfoChir). He's an engineer; talk peer-to-peer, terse, no fluff.
Timezone: America/New_York. Money is tight; checking floor $100.
"""

def load_brief() -> str:
    try:
        with open(JETT_MD, encoding="utf-8") as f:
            return f.read()
    except OSError:
        log.warning("JETT.md not found at %s — using default brief", JETT_MD)
        return DEFAULT_BRIEF


SYSTEM_PROMPT = """\
You are Jett's voice proxy — his AI phone line. You are NOT Jett himself.
You are his voice interface, briefed on his notes below.

Contract (follow it exactly):
- Answer from the brief when you are confident. Keep answers SHORT: this is
  a phone call, one or two sentences when possible. No bullet lists, no
  essays, no markdown.
- For FACTUAL questions (schedule, money, balances, bills, inbox, anything
  with numbers or dates): FIRST call local_facts — it reads Jett's synced
  snapshot. Relay it starting with "Jett says" and include its freshness
  ("as of 12 minutes ago"). If the snapshot is missing or stale, say so
  honestly and use consult_jett instead of guessing.
- If the caller asks anything else that needs the REAL Jett's live memory,
  current information, or real judgment, call consult_jett with a clear,
  self-contained question. It tells the caller you're checking, then waits
  for Jett: his answer may arrive live — relay it as "Jett says" with his
  numbers EXACTLY as stated — or be delivered to him afterward, in which
  case tell the caller honestly you've passed it to Jett. NEVER invent
  Jett's answers. NEVER guess at things only Jett would know (his schedule,
  money, messages, accounts).
- Vague follow-ups ("what about the other one?") → ask what they mean or
  consult_jett. Never guess which "other one" they mean.
- Relay Jett's facts and numbers EXACTLY as he stated them. Never reinterpret,
  round, or merge them: an account suffix like …1792 is NOT a balance, a date
  is a date. When condensing for voice, keep every number and key fact; when
  in doubt, quote him nearly word-for-word.
- Be honest about what comes from you versus from Jett. If you don't know,
  say so and offer to check with Jett.
- The caller is Jim. Warm, direct, peer-to-peer. Skip performative
  assistant-speak ("Great question!"). Just talk.

Jett's brief:
{brief}
"""


def _is_rate_limit(err: Exception) -> bool:
    """True if this looks like an HTTP 429 / rate-limit error, any shape."""
    if getattr(err, "status_code", None) == 429:
        return True
    low = str(err).lower()
    return "429" in low or "rate limit" in low or "too many requests" in low


# ------------------------------------------------------------------ brain ---

class Brain(NamedTuple):
    """Resolved LLM backend for one call: OpenRouter, or Meta direct."""
    label: str          # "openrouter" | "meta-direct"
    base_url: str
    api_key: str
    model: str
    extra_headers: dict
    extra_body: dict


def resolve_brain() -> Brain | None:
    """Pick the LLM backend: OpenRouter first, Meta direct as fallback.

    Every OpenRouter turn carries max_tokens >= BRAIN_MAX_TOKENS (>= 300)
    in extra_body — the floor is enforced here, not hoped for.
    """
    if OPENROUTER_API_KEY:
        return Brain(
            label="openrouter",
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=JETT_BRAIN_MODELS[0],
            extra_headers={
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_TITLE,
            },
            extra_body={
                "models": JETT_BRAIN_MODELS,
                "route": "fallback",
                "max_tokens": BRAIN_MAX_TOKENS,
            },
        )
    if META_API_KEY:
        return Brain(
            label="meta-direct",
            base_url=META_BASE_URL,
            api_key=META_API_KEY,
            model=_discover_meta_model(),
            extra_headers={},
            extra_body={},
        )
    return None


def _discover_meta_model() -> str:
    """Pick the best muse-spark model from Meta's /v1/models; fall back."""
    try:
        import httpx

        r = httpx.get(
            META_BASE_URL + "/models",
            headers={"Authorization": f"Bearer {META_API_KEY}"},
            timeout=10,
        )
        r.raise_for_status()
        ids = [m.get("id", "") for m in r.json().get("data", [])]
        if META_MODEL in ids:
            return META_MODEL
        sparks = sorted([i for i in ids if "muse-spark" in i], reverse=True)
        if sparks:
            log.info("using Meta model %s (preferred %s not listed)",
                     sparks[0], META_MODEL)
            return sparks[0]
        log.warning("/v1/models listed no muse-spark model; using %s",
                    META_MODEL)
    except Exception as e:
        log.warning("model discovery failed (%s); using %s", e, META_MODEL)
    return META_MODEL


# ------------------------------------- resilient LLM (retry wrapper) ---

async def _drain(stream: llm.LLMStream) -> tuple[list, str, bool]:
    """Drain a stream fully. Returns (chunks, text, has_tool_calls)."""
    chunks: list = []
    texts: list[str] = []
    has_tools = False
    async for chunk in stream:
        chunks.append(chunk)
        delta = chunk.delta
        if delta is None:
            continue
        if delta.content:
            texts.append(delta.content)
        if delta.tool_calls:
            has_tools = True
    try:
        await stream.aclose()
    except Exception:
        pass
    return chunks, "".join(texts), has_tools


def _last_tool_output(chat_ctx: llm.ChatContext,
                      names: tuple[str, ...]) -> llm.FunctionCallOutput | None:
    """The most recent FunctionCallOutput for one of `names`, provided no
    user message came after it (i.e. this turn is the relay turn)."""
    last = None
    for item in getattr(chat_ctx, "items", []) or []:
        if getattr(item, "type", None) == "function_call_output" \
                and getattr(item, "name", None) in names:
            last = item
        elif getattr(item, "type", None) == "message" \
                and getattr(item, "role", None) == "user":
            last = None
    return last


def _maybe_relay(chat_ctx: llm.ChatContext, text: str) -> str:
    """Post-process a finished turn: if it relays a consult_jett /
    local_facts tool result, enforce attribution + number fidelity in code."""
    out = _last_tool_output(chat_ctx, ("consult_jett", "local_facts"))
    if out is None:
        return text
    name, output = out.name, (out.output or "")
    if name == "consult_jett":
        if output.startswith(("ERROR_CANT_REACH_JETT", "FILED_ASYNC")):
            return text  # honest proxy message, not Jett's words
        return relay_consult_reply(output, text)
    if name == "local_facts":
        if output.startswith(("FACTS_UNAVAILABLE", "FACTS_STALE")):
            return text  # honest proxy message, not Jett's words
        return relay_facts_reply(output, text)
    return text


class _ReplayStream(llm.LLMStream):
    """Replays pre-drained (and possibly rewritten) chunks to the session."""

    def __init__(self, resilient: "ResilientLLM", *, chat_ctx, tools,
                 conn_options, chunks: list) -> None:
        super().__init__(resilient, chat_ctx=chat_ctx, tools=tools,
                         conn_options=conn_options)
        self._chunks = chunks

    async def _run(self) -> None:
        for chunk in self._chunks:
            self._event_ch.send_nowait(chunk)


class ResilientLLM(llm.LLM):
    """llm.LLM with per-model empty-output retry and relay post-processing.

    chat() returns a stream whose _run() drains each model in
    JETT_BRAIN_MODELS in order: on empty content (or error) it advances to
    the next model, backing off exponentially on rate limits. When the turn
    relays a consult_jett/local_facts tool result, the finished text is
    rewritten through the relay guards (attribution + number validation).
    If every model comes back empty, a guaranteed fallback line is spoken —
    dead air is never acceptable.

    metrics_collected events from the inner LLMs are re-emitted so usage
    telemetry keeps working. Per-turn: model used, latency, empty-retry and
    fallback flags are logged.
    """

    def __init__(self, brain: Brain, _llm_factory=None) -> None:
        super().__init__()
        self._brain = brain
        self._factory = _llm_factory or self._default_factory

    @property
    def model(self) -> str:
        return self._brain.model

    @property
    def provider(self) -> str:
        return "openrouter-resilient"

    def _models(self) -> list[str]:
        if self._brain.label == "openrouter":
            return list(JETT_BRAIN_MODELS)
        return [self._brain.model]

    def _default_factory(self, model: str, remaining: list[str]):
        inner = openai_plugin.LLM(
            model=model,
            api_key=self._brain.api_key,
            base_url=self._brain.base_url,
            extra_headers=self._brain.extra_headers
            if self._brain.extra_headers else NOT_GIVEN,
            extra_body={
                **(self._brain.extra_body or {}),
                "models": remaining,
                "route": "fallback",
                "max_tokens": BRAIN_MAX_TOKENS,
            },
        )
        # usage telemetry keeps working through the wrapper
        inner.on("metrics_collected",
                 lambda m: self.emit("metrics_collected", m))
        # inner "error" events are NOT forwarded: the wrapper absorbs
        # failed attempts (retry / backoff / fallback) by design.
        return inner

    def chat(self, *, chat_ctx: llm.ChatContext,
             tools: list | None = None,
             conn_options=DEFAULT_API_CONNECT_OPTIONS,
             parallel_tool_calls=NOT_GIVEN,
             tool_choice=NOT_GIVEN,
             extra_kwargs=NOT_GIVEN,
             **kwargs) -> llm.LLMStream:
        return _ResilientStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
            extra_kw=kwargs,
        )

    async def _attempt(self, ctx: "_ResilientStream") -> list:
        models = self._models()
        empty_retries = 0
        for i, model in enumerate(models):
            t0 = time.monotonic()
            try:
                inner = self._factory(model, models[i:])
                stream = inner.chat(
                    chat_ctx=ctx._chat_ctx,
                    tools=ctx._tools or None,
                    conn_options=ctx._conn_options,
                    parallel_tool_calls=ctx._parallel_tool_calls,
                    tool_choice=ctx._tool_choice,
                    extra_kwargs=ctx._extra_kwargs,
                    **ctx._extra_kw,
                )
                chunks, text, has_tools = await _drain(stream)
            except Exception as e:
                dt = time.monotonic() - t0
                if _is_rate_limit(e):
                    delay = compute_backoff(i)
                    log_event("brain_attempt", model=model,
                              latency_s=round(dt, 2), outcome="rate_limited",
                              backoff_s=round(delay, 1),
                              fallback_to=models[i + 1] if i + 1 < len(models) else None)
                    log.warning("brain chat: model=%s rate-limited "
                                "(%.1fs); backing off %.1fs then next model",
                                model, dt, delay)
                    await asyncio.sleep(delay)
                else:
                    log_event("brain_attempt", model=model,
                              latency_s=round(dt, 2), outcome="error",
                              error=str(e)[:120],
                              fallback_to=models[i + 1] if i + 1 < len(models) else None)
                    log.warning("brain chat: model=%s errored (%.1fs): %.150s",
                                model, dt, e)
                empty_retries += 1
                continue
            dt = time.monotonic() - t0
            if text.strip() or has_tools:
                log_event("brain_attempt", model=model,
                          latency_s=round(dt, 2), outcome="ok",
                          empty_retries=empty_retries)
                log.info("brain chat: model=%s latency=%.1fs empty_retry=%s",
                         model, dt, empty_retries > 0)
                return self._postprocess(ctx, chunks, text, has_tools)
            empty_retries += 1
            log_event("brain_attempt", model=model,
                      latency_s=round(dt, 2), outcome="empty",
                      fallback_to=models[i + 1] if i + 1 < len(models) else None)
            log.warning("brain chat: model=%s empty content (%.1fs); "
                        "trying next model", model, dt)
        # every model empty/failed — guaranteed spoken fallback, never silence
        log_event("brain_exhausted", models=len(models),
                  fallback_spoken=True)
        log.warning("brain chat: ALL %d models empty/failed; speaking fallback",
                    len(models))
        return [llm.ChatChunk(
            id=f"fallback-{uuid.uuid4().hex[:8]}",
            delta=llm.ChoiceDelta(content=FALLBACK_LINE, role="assistant"),
        )]

    def _postprocess(self, ctx, chunks: list, text: str,
                     has_tools: bool) -> list:
        if has_tools or not text.strip():
            return chunks
        fixed = _maybe_relay(ctx._chat_ctx, text)
        if fixed == text:
            return chunks
        log.info("brain chat: relay post-processed "
                 "(attribution/numbers enforced)")
        return [llm.ChatChunk(
            id=f"relay-{uuid.uuid4().hex[:8]}",
            delta=llm.ChoiceDelta(content=fixed, role="assistant"),
        )]


class _ResilientStream(llm.LLMStream):
    """The stream AgentSession consumes; _run() does drain→retry→replay."""

    def __init__(self, resilient: ResilientLLM, *, chat_ctx,
                 tools: list,
                 conn_options,
                 parallel_tool_calls,
                 tool_choice,
                 extra_kwargs,
                 extra_kw: dict) -> None:
        super().__init__(resilient, chat_ctx=chat_ctx, tools=tools,
                         conn_options=conn_options)
        self._resilient = resilient
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs
        self._extra_kw = extra_kw

    async def _run(self) -> None:
        chunks = await self._resilient._attempt(self)
        for chunk in chunks:
            self._event_ch.send_nowait(chunk)

# ----------------------------------------------------------------- STT ---

class WhisperSTT(stt.STT):
    """faster-whisper tiny.en wrapped as a LiveKit STT (non-streaming)."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=False))
        from faster_whisper import WhisperModel

        log.info("loading faster-whisper tiny.en …")
        self._model = WhisperModel("tiny.en", device="cpu",
                                   compute_type="int8")
        log.info("whisper ready")

    @property
    def model(self) -> str:
        return "tiny.en"

    @property
    def provider(self) -> str:
        return "faster-whisper"

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN,
                              conn_options) -> stt.SpeechEvent:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._transcribe, buffer)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text=text)],
        )

    def _transcribe(self, buffer) -> str:
        # buffer is a single merged rtc.AudioFrame (from StreamAdapter)
        raw = bytes(buffer.data)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if buffer.num_channels > 1:
            audio = audio.reshape(-1, buffer.num_channels).mean(axis=1)
        sr = buffer.sample_rate
        dur_s = round(len(audio) / max(1, sr), 2)
        if audio.size == 0 or float(np.abs(audio).max(initial=0.0)) < 1e-4:
            log_event("stt_empty_audio", duration_s=dur_s)
            return ""
        try:
            if sr != 16000:
                # cheap linear resample to 16 kHz
                dur = len(audio) / sr
                n = int(dur * 16000)
                audio = np.interp(
                    np.linspace(0, len(audio), n, endpoint=False),
                    np.arange(len(audio)), audio).astype(np.float32)
            segments, _ = self._model.transcribe(audio, language="en",
                                                 beam_size=1, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            log_event("stt_error", error=str(e)[:150], duration_s=dur_s)
            return ""  # treat as silence rather than killing the turn


# ----------------------------------------------------------------- TTS ---

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, limit: int = 400) -> list[str]:
    """Split text into TTS-friendly chunks (kokoro phoneme limit is 510)."""
    chunks: list[str] = []
    for part in _SENT_SPLIT.split(text.strip()):
        part = part.strip()
        if not part:
            continue
        while len(part) > limit:
            cut = max(part.rfind(c, 0, limit) for c in (",", ";", ":", "—", "-"))
            if cut <= 0:
                cut = limit
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            chunks.append(part)
    return chunks or [text.strip()]


class KokoroTTS(tts.TTS):
    """kokoro-onnx warm voice as a LiveKit TTS (chunked synthesis)."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        from kokoro_onnx import Kokoro

        if not (os.path.exists(KOKORO_MODEL) and os.path.exists(KOKORO_VOICES)):
            raise FileNotFoundError(
                f"kokoro model files missing: {KOKORO_MODEL}, {KOKORO_VOICES}")
        log.info("loading kokoro-onnx (%s) …", KOKORO_VOICE)
        self._kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        self._voice = KOKORO_VOICE
        log.info("kokoro ready")

    @property
    def model(self) -> str:
        return "kokoro-v1.0-int8"

    @property
    def provider(self) -> str:
        return "kokoro-onnx"

    def _synth(self, text: str) -> tuple[np.ndarray, int]:
        return self._kokoro.create(text, voice=self._voice, speed=1.0,
                                   lang="en-us")

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS
                   ) -> tts.ChunkedStream:
        return _KokoroChunkedStream(tts=self, input_text=text,
                                    conn_options=conn_options)


class _KokoroChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        loop = asyncio.get_running_loop()
        output_emitter.initialize(
            request_id=uuid.uuid4().hex[:12],
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )
        for sent in split_sentences(self._input_text):
            try:
                samples, _sr = await loop.run_in_executor(
                    None, self._tts._synth, sent)
            except Exception as e:
                log_event("tts_error", sentence_len=len(sent),
                          error=str(e)[:150])
                continue
            if samples is None or len(samples) == 0:
                log_event("tts_empty_synth", sentence_len=len(sent))
                continue
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(
                np.int16).tobytes()
            # push in ~20 ms frames to keep the pipeline flowing
            frame_bytes = 24000 * 2 // 50
            for i in range(0, len(pcm), frame_bytes):
                output_emitter.push(pcm[i:i + frame_bytes])
        output_emitter.flush()

# ------------------------------------------------- agent-call-bus ---

def _git_env() -> dict:
    env = dict(os.environ)
    env["GIT_SSH_COMMAND"] = (
        f"ssh -i {BUS_KEY} -o StrictHostKeyChecking=no "
        "-o ConnectTimeout=10 -o BatchMode=yes")
    return env


async def _git(args: list[str], cwd: str = BUS_DIR) -> tuple[int, str]:
    loop = asyncio.get_running_loop()

    def _run() -> tuple[int, str]:
        p = subprocess.run(["git", "-C", cwd, *args], env=_git_env(),
                           capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()[-2000:]

    return await loop.run_in_executor(None, _run)


async def _bus_ensure() -> bool:
    """Clone the bus repo if missing. Returns True if usable.

    Graceful when the VM deploy key hasn't been added to
    intelogroup/agent-call-bus yet — consults then report unreachable
    instead of crashing the call.
    """
    if os.path.isdir(os.path.join(BUS_DIR, ".git")):
        log_event("bus_status", available=True, action="cached")
        return True
    try:
        os.makedirs(BUS_DIR, exist_ok=True)
        rc, out = await _git(["clone", BUS_REPO_SSH, BUS_DIR], cwd="/tmp")
        if rc != 0:
            log_event("bus_status", available=False, reason="clone_failed",
                      detail=out[-150:])
            log.warning("bus clone failed: %s", out[-300:])
            return False
        await _git(["config", "user.email", "jett-proxy@129.159.189.244"])
        await _git(["config", "user.name", "jett-proxy"])
        log_event("bus_status", available=True, action="cloned")
        log.info("bus repo cloned")
        return True
    except Exception as e:
        log_event("bus_status", available=False, reason="exception",
                  error=str(e)[:120])
        log.warning("bus ensure failed: %s", e)
        return False


async def _bus_write_push(relpath: str, obj: dict, msg: str) -> bool:
    """Write one JSON file into the bus repo and push immediately
    (protocol: commit + push, batch window <= 3 s)."""
    try:
        full = os.path.join(BUS_DIR, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        if (await _git(["add", relpath]))[0] != 0:
            return False
        if (await _git(["commit", "-m", msg, "--allow-empty"]))[0] != 0:
            return False
        rc, out = await _git(["push", "origin", "HEAD"])
        if rc != 0:
            # someone (Jett's side) pushed first — rebase once and retry
            log.info("bus push raced; rebasing once")
            await _git(["pull", "--rebase", "--autostash"])
            rc, out = await _git(["push", "origin", "HEAD"])
        if rc != 0:
            log_event("bus_push", relpath=relpath, ok=False,
                      error=out[-150:])
            log.warning("bus push failed: %s", out[-300:])
        else:
            log_event("bus_push", relpath=relpath, ok=True)
        return rc == 0
    except OSError as e:
        log_event("bus_push", relpath=relpath, ok=False,
                  error=str(e)[:120])
        log.warning("bus write failed: %s", e)
        return False


async def _bus_poll_response(req_id: str,
                             timeout_sec: int) -> str | None:
    """Poll responses/<req_id>.json until Jett answers or timeout.

    Returns Jett's answer text, or None on timeout. responses/ and
    deliveries/ are written by Jett's side only — we never write them.
    """
    path = os.path.join(BUS_DIR, "responses", f"{req_id}.json")
    t0 = time.monotonic()
    deadline = t0 + max(0, timeout_sec)
    while time.monotonic() < deadline:
        await _git(["pull", "--ff-only", "-q"])
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, ValueError):
            obj = None
        if obj and obj.get("id") == req_id:
            answer = (obj.get("answer") or "").strip()
            if answer:
                if obj.get("answered_by") != "jett-runtime":
                    log.warning("consult %s answered_by=%r (expected "
                                "'jett-runtime')", req_id[:8],
                                obj.get("answered_by"))
                log_event("consult_answered", id=req_id[:8],
                          latency_s=round(time.monotonic() - t0, 1),
                          answer_len=len(answer))
                log.info("consult %s answered live (%d chars)",
                         req_id[:8], len(answer))
                return answer
        await asyncio.sleep(5)
    log_event("consult_timeout", id=req_id[:8], waited_s=timeout_sec)
    return None


async def file_consult_request(question: str,
                               priority: str = "normal") -> tuple[str, dict]:
    """File a consult to Jett's async judgment queue.

    Returns (status, payload):
      "unreachable" — bus repo unusable; payload {"error": ...}
      "filed"       — request pushed; payload {"request": req_obj}
    The caller (consult_jett tool) then polls for a live answer; on timeout
    the filed request is picked up by Jett's 5-minute watcher and the answer
    is delivered afterward (voice callback / WhatsApp).
    """
    if not await _bus_ensure():
        log_event("consult_unreachable", reason="bus_ensure_failed")
        return ("unreachable",
                {"error": "bus repo unreachable (deploy key not added?)"})
    req_id = str(uuid.uuid4())
    req = {
        "id": req_id,
        "ts": _utcnow_z(),
        "call_id": _CURRENT_CALL_ID or "unknown",
        "question": question.strip(),
        "context": ("Live voice call with Jim on Jett's line "
                    "(sip:jett@129.159.189.244). Asked mid-call; the brief "
                    "and facts snapshot couldn't answer it."),
        "priority": priority if priority in ("normal", "urgent") else "normal",
    }
    log_event("consult_request", id=req_id[:8],
              question_len=len(question),
              priority=req["priority"],
              call_id=req["call_id"])
    ok = await _bus_write_push(f"requests/{req_id}.json", req,
                               f"consult {req_id[:8]}")
    if not ok:
        log_event("consult_unreachable", reason="bus_push_failed",
                  id=req_id[:8])
        return ("unreachable", {"error": "bus push failed"})
    log_event("consult_filed", id=req_id[:8], priority=req["priority"])
    log.info("consult %s filed (priority=%s)", req_id[:8], req["priority"])
    return ("filed", {"request": req})


# ------------------------------------------------------------ tools ---

@function_tool
async def consult_jett(question: str, context: RunContext,
                       priority: str = "normal") -> str:
    """Ask the real Jett something only he would know — his live memory,
    current information, messages, accounts, or real judgment. The question
    is filed to Jett's async judgment queue and pushed immediately. If Jett
    answers within a few minutes his reply is spoken live; otherwise his
    answer is delivered to the caller afterward (voice callback / WhatsApp)
    — tell the caller honestly which happened. NEVER invent Jett's answer.

    priority: "normal", or "urgent" for time-sensitive matters (money
    emergencies, same-day deadlines).
    """
    try:
        await context.session.say(
            "Let me check with Jett on that — one moment…",
            allow_interruptions=True,
        )
    except Exception as e:
        log.debug("holding line failed: %s", e)

    status, payload = await file_consult_request(question, priority)
    if status == "unreachable":
        return ("ERROR_CANT_REACH_JETT: couldn't file the question to Jett "
                f"({payload.get('error')}). Tell the caller honestly: "
                "I can't reach Jett right now — offer to try again later.")

    req = payload["request"]
    answer = await _bus_poll_response(req["id"], CONSULT_TIMEOUT_SEC)
    if answer is not None:
        return answer  # relayed with "Jett says" + number checks by wrapper
    return (f"FILED_ASYNC: filed as consult {req['id'][:8]}. Jett answers "
            "asynchronously — his answer will be delivered to the caller "
            "afterward (voice callback / WhatsApp). Tell the caller "
            "honestly: I've passed your question to Jett, he'll get back "
            "to you shortly. Do NOT invent an answer.")


def _facts_slice(data: dict, query: str) -> str:
    """Compact, query-relevant slice of the facts snapshot."""
    q = (query or "").lower()
    lines: list[str] = []

    def _want(*keys: str) -> bool:
        return not q or any(k in q for k in keys)

    cal = data.get("calendar_today") or []
    if cal and _want("schedul", "calendar", "today", "appointment", "meeting",
                     "booking", "plan"):
        lines.append("Today:")
        for e in cal[:8]:
            lines.append(f"  - {e.get('title', '?')} at {e.get('time', '?')}")
    accts = data.get("accounts") or []
    if accts and _want("account", "balance", "bank", "checking", "money",
                       "dollar"):
        lines.append("Accounts:")
        for a in accts[:6]:
            lines.append(
                f"  - {a.get('label', '?')} …{a.get('suffix', '?')}: "
                f"balance ${a.get('balance', '?')} "
                f"(available ${a.get('available', '?')})")
    bills = data.get("bills") or []
    if bills and _want("bill", "due", "payment", "pay", "owe", "money",
                       "dollar"):
        lines.append("Bills:")
        for b in bills[:8]:
            lines.append(
                f"  - {b.get('payee', '?')}: ${b.get('amount', '?')} "
                f"due {b.get('due', '?')} [{b.get('status', '?')}]")
    inbox = data.get("inbox") or []
    if inbox and _want("email", "inbox", "mail", "message", "subject"):
        lines.append("Inbox:")
        for m in inbox[:6]:
            lines.append(f"  - {m.get('subject', '?')} "
                         f"({m.get('date', '?')}): {m.get('summary', '')}"[:120])
    notes = (data.get("notes") or "").strip()
    if notes and (not lines or _want("note")):
        lines.append(f"Notes: {notes[:400]}")
    if not lines:
        # nothing matched the query — give the compact whole
        return _facts_slice(data, "")
    return "\n".join(lines)


@function_tool
async def local_facts(query: str) -> str:
    """Look up Jett's synced facts snapshot: today's calendar, account
    balances, upcoming bills, recent inbox items. The snapshot refreshes
    every few minutes. ALWAYS prefer this over guessing for factual
    questions. If the snapshot is missing or stale it says so honestly —
    then use consult_jett instead of guessing."""
    data, age = load_facts()
    if data is None:
        log_event("facts_unavailable")
        return ("FACTS_UNAVAILABLE: no synced facts snapshot found. Tell the "
                "caller honestly you don't have Jett's latest facts — offer "
                "to check with Jett via consult_jett instead of guessing.")
    if age is None or age > FACTS_STALE_SEC:
        log_event("facts_stale", age_s=round(age) if age else None)
        stale = freshness_str(age) if age else "unknown age"
        return (f"FACTS_STALE: snapshot is {stale} (stale). Tell the caller "
                "honestly the facts may be out of date — offer consult_jett "
                "instead of guessing.")
    log_event("facts_used", age_s=round(age) if age else None,
              query_len=len(query or ""))
    fresh = freshness_str(age)
    return f"Jett's synced facts (as of {fresh}):\n{_facts_slice(data, query)}"


# ------------------------------------------- entrypoint & call flow ---

def _sip_caller(ctx: JobContext) -> str:
    for p in ctx.room.remote_participants.values():
        return p.identity or "unknown"
    return "unknown"


async def _speak_raw(room: rtc.Room, wav_pcm: bytes, sample_rate: int = 24000):
    """Publish PCM audio directly (no-key fallback; no STT/LLM needed)."""
    source = rtc.AudioSource(sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    await room.local_participant.publish_track(track)
    frame_samples = sample_rate // 50  # 20 ms
    frame_bytes = frame_samples * 2
    try:
        for i in range(0, len(wav_pcm), frame_bytes):
            chunk = wav_pcm[i:i + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk += b"\x00" * (frame_bytes - len(chunk))
            frame = rtc.AudioFrame(
                data=chunk, num_channels=1,
                samples_per_channel=frame_samples,
                sample_rate=sample_rate)
            await source.capture_frame(frame)
        await asyncio.sleep(0.5)
    finally:
        await room.local_participant.unpublish_track(track.sid)


async def _run_no_brain(ctx: JobContext):
    """No brain key: connect, play a spoken notice, hang up gracefully."""
    await ctx.connect()
    log_event("no_brain_notice")
    log.warning("no brain key (OPENROUTER_API_KEY/META_API_KEY) — playing no-key notice")
    tts_engine = KokoroTTS()
    samples, _ = await asyncio.get_running_loop().run_in_executor(
        None, tts_engine._synth,
        "Hey — this is Jett's voice line, but his brain isn't connected "
        "yet. He hasn't given me a key to think with. Try again later.")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    await _speak_raw(ctx.room, pcm)
    await ctx.room.disconnect()


async def _run_call(ctx: JobContext, brain: Brain, stats: dict):
    global _CURRENT_CALL_ID

    log.info("call brain: %s models=%s max_tokens=%s", brain.label,
             brain.extra_body.get("models", brain.model),
             brain.extra_body.get("max_tokens", "?"))
    stt_engine = WhisperSTT()
    stt_stream = stt.StreamAdapter(
        stt=stt_engine, vad=silero.VAD.load())
    tts_engine = KokoroTTS()

    agent_llm = ResilientLLM(brain)

    brief = load_brief()
    agent = Agent(
        instructions=SYSTEM_PROMPT.format(brief=brief),
        tools=[consult_jett, local_facts],
    )
    session = AgentSession(
        stt=stt_stream,
        vad=silero.VAD.load(),
        llm=agent_llm,
        tts=tts_engine,
        max_tool_steps=10,
    )

    # --- brain telemetry: per-turn model/latency via the wrapper's logs,
    # plus usage metrics forwarded through it. Inner "error" events are
    # absorbed by the wrapper's retry/backoff/fallback by design.
    def _on_llm_metrics(metrics) -> None:
        stats["turns"] += 1
        log.info("brain usage: model=%s duration=%.2fs ttft=%.2fs "
                 "tokens prompt=%d completion=%d",
                 brain.model, metrics.duration, metrics.ttft,
                 metrics.prompt_tokens, metrics.completion_tokens)

    agent_llm.on("metrics_collected", _on_llm_metrics)

    await ctx.connect()
    await session.start(agent, room=ctx.room)
    log.info("session started in room %s", ctx.room.name)

    ended = asyncio.Event()

    def _on_participant_left(p: rtc.RemoteParticipant):
        if not ctx.room.remote_participants:
            log.info("caller hung up")
            ended.set()

    ctx.room.on("participant_disconnected", _on_participant_left)

    async def _watchdog():
        await asyncio.sleep(MAX_CALL_SEC)
        log.info("max call duration reached; wrapping up")
        try:
            session.say("I have to run — talk soon.")
        except Exception:
            pass
        await asyncio.sleep(8)
        ended.set()

    wd_task = asyncio.create_task(_watchdog())

    try:
        await session.generate_reply(
            instructions="Greet the caller briefly as Jett's voice proxy: "
                         f"say hello in one short sentence like: {GREETING}")
        await ended.wait()
    finally:
        _CURRENT_CALL_ID = None
        wd_task.cancel()
        try:
            await session.aclose()
        except Exception:
            pass


async def entrypoint(ctx: JobContext):
    global _ACTIVE_CALL, _CURRENT_CALL_ID
    call_id = ctx.room.name.replace("jett-", "", 1) or uuid.uuid4().hex[:8]

    with _CALL_LOCK:
        if _ACTIVE_CALL is not None:
            log_event("call_rejected_busy", room=ctx.room.name)
            log.info("busy; rejecting second call %s", ctx.room.name)
            await ctx.connect()
            engine = KokoroTTS()
            samples, _ = await asyncio.get_running_loop().run_in_executor(
                None, engine._synth,
                "Jett's line is on another call right now. Try again in a bit.")
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(
                np.int16).tobytes()
            await _speak_raw(ctx.room, pcm)
            await ctx.room.disconnect()
            return
        _ACTIVE_CALL = call_id
    _CURRENT_CALL_ID = call_id

    stats = {"t0": time.monotonic(), "turns": 0, "errors": []}
    try:
        await ctx.connect()  # ensure room handle before reading participants
        # the SIP leg may still be joining; wait briefly for the caller
        caller = "unknown"
        for _ in range(16):
            caller = _sip_caller(ctx)
            if caller != "unknown":
                break
            await asyncio.sleep(0.5)
        log_event("call_start", call_id=call_id, room=ctx.room.name,
                  caller=caller)
        log.info("inbound call %s from %s", call_id, caller)
        brain = resolve_brain()
        if brain is not None:
            log_event("brain_resolved", label=brain.label,
                      model=brain.model, chain_len=len(JETT_BRAIN_MODELS))
            await _run_call(ctx, brain, stats)
        else:
            log_event("brain_missing")
            await _run_no_brain(ctx)
    except Exception as e:
        stats["errors"].append(type(e).__name__)
        log_event("call_error", call_id=call_id,
                  error=f"{type(e).__name__}: {str(e)[:150]}")
        log.exception("call %s failed: %s", call_id, e)
    finally:
        with _CALL_LOCK:
            _ACTIVE_CALL = None
        _CURRENT_CALL_ID = None
        log_event("call_end", call_id=call_id,
                  duration_s=round(time.monotonic() - stats["t0"], 1),
                  turns=stats["turns"], errors=stats["errors"])
        log.info("call %s done", call_id)


if __name__ == "__main__":
    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise SystemExit("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are required")
    log_event("boot_config", agent=AGENT_NAME, livekit_url=LIVEKIT_URL,
              brain_models=len(JETT_BRAIN_MODELS),
              brain_max_tokens=BRAIN_MAX_TOKENS,
              consult_timeout_s=CONSULT_TIMEOUT_SEC,
              max_call_s=MAX_CALL_SEC, facts_stale_s=FACTS_STALE_SEC,
              bus_dir=BUS_DIR,
              bus_cloned=os.path.isdir(os.path.join(BUS_DIR, ".git")))
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=AGENT_NAME,
        job_executor_type=JobExecutorType.THREAD,
        ws_url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    ))
