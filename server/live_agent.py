#!/usr/bin/env python3
"""Jett's voice proxy — LiveKit agent worker.

One SIP line (sip:jett@129.159.189.244) -> LiveKit room (prefix jett-) ->
this worker. Real-time voice pipeline:

    faster-whisper tiny.en (local STT, via StreamAdapter + silero VAD)
      -> Muse Spark (LLM, OpenAI-compatible API)
      -> kokoro-onnx (local warm-voice TTS)

The worker answers from its JETT.md brief when confident. For anything
needing the real Jett's live memory, tools, or judgment it calls the
consult_jett() function tool: the question is appended to
calls/live/<call-id>/in.jsonl in the private agent-call-bus repo, a
background task polls out.jsonl for Jett's reply, and the reply is spoken
back to the caller when it lands.

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
KOKORO_MODEL = os.path.join(AGENT_DIR, "models", "kokoro-v1.0.int8.onnx")
KOKORO_VOICES = os.path.join(AGENT_DIR, "models", "voices-v1.0.bin")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
AGENT_NAME = "jett-proxy"

# one call at a time; a second caller gets a polite busy message
_CALL_LOCK = threading.Lock()
_ACTIVE_CALL: str | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
- If the caller asks anything that needs the REAL Jett's live memory,
  current information, tools, or real judgment, call consult_jett with a
  clear, self-contained question. NEVER invent Jett's answers. NEVER guess
  at things only Jett would know (his schedule, money, messages, accounts).
- After calling consult_jett, tell the caller plainly you're checking with
  Jett ("Let me check with Jett on that, one sec…") and keep them company
  naturally. Jett's reply will be handed to you; relay it faithfully —
  you may say "Jett says…" then his words, unchanged.
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
    """Pick the LLM backend: OpenRouter first, Meta direct as fallback."""
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
            extra_body={"models": JETT_BRAIN_MODELS, "route": "fallback"},
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
            samples, _sr = await loop.run_in_executor(
                None, self._tts._synth, sent)
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
    """Clone the bus repo if missing. Returns True if usable."""
    if os.path.isdir(os.path.join(BUS_DIR, ".git")):
        return True
    try:
        os.makedirs(BUS_DIR, exist_ok=True)
        rc, out = await _git(["clone", BUS_REPO_SSH, BUS_DIR], cwd="/tmp")
        if rc != 0:
            log.warning("bus clone failed: %s", out[-300:])
            return False
        await _git(["config", "user.email", "jett-proxy@129.159.189.244"])
        await _git(["config", "user.name", "jett-proxy"])
        log.info("bus repo cloned")
        return True
    except Exception as e:
        log.warning("bus ensure failed: %s", e)
        return False


async def _bus_push_with_retry(commit_msg: str) -> bool:
    for attempt in range(3):
        rc, _ = await _git(["pull", "--rebase", "--autostash"])
        if rc != 0:
            await asyncio.sleep(2)
            continue
        rc, _ = await _git(["add", "-A"])
        rc, _ = await _git(["commit", "-m", commit_msg, "--allow-empty"])
        rc, out = await _git(["push", "origin", "HEAD"])
        if rc == 0:
            return True
        log.warning("bus push attempt %d failed: %s", attempt + 1,
                    out[-300:])
        await asyncio.sleep(2)
    return False


class BusCall:
    """Per-call bus state: meta.json, in/out.jsonl, consult matching."""

    def __init__(self, call_id: str, caller: str):
        self.call_id = call_id
        self.dir = os.path.join(BUS_DIR, "calls", "live", call_id)
        self.in_path = os.path.join(self.dir, "in.jsonl")
        self.out_path = os.path.join(self.dir, "out.jsonl")
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.hb_path = os.path.join(self.dir, "hb.json")
        self.caller = caller
        self.seq = 0
        self.spoken_out = 0          # highest out.jsonl line spoken/consumed
        self.pending: dict[int, asyncio.Event] = {}
        self.replies: dict[int, str] = {}
        self.delivered: set[int] = set()
        self.timed_out: set[int] = set()
        self.session: AgentSession | None = None
        self.active = True

    async def open(self) -> bool:
        if not await _bus_ensure():
            return False
        os.makedirs(self.dir, exist_ok=True)
        for p in (self.in_path, self.out_path):
            if not os.path.exists(p):
                open(p, "a").close()
        meta = {
            "call_id": self.call_id,
            "mode": "jett-proxy",
            "started_at": _utcnow(),
            "ended_at": None,
            "status": "ringing",
            "caller": self.caller,
            "end_reason": None,
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
        ok = await _bus_push_with_retry(f"call {self.call_id} started")
        log.info("bus call dir ready: %s (push %s)", self.dir,
                 "ok" if ok else "FAILED")
        return True

    async def set_status(self, status: str, end_reason: str | None = None):
        try:
            with open(self.meta_path) as f:
                meta = json.load(f)
        except OSError:
            return
        meta["status"] = status
        if status == "ended":
            meta["ended_at"] = _utcnow()
            meta["end_reason"] = end_reason
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
        await _bus_push_with_retry(f"call {self.call_id} {status}")

    async def consult(self, question: str) -> int | None:
        """Append a consult question; returns seq or None on failure."""
        self.seq += 1
        line = json.dumps({"ts": _utcnow(), "seq": self.seq,
                           "text": question}, ensure_ascii=False)
        try:
            with open(self.in_path, "a") as f:
                f.write(line + "\n")
        except OSError as e:
            log.warning("in.jsonl append failed: %s", e)
            return None
        self.pending[self.seq] = asyncio.Event()
        ok = await _bus_push_with_retry(
            f"consult #{self.seq} on {self.call_id}")
        if not ok:
            log.warning("consult push failed; Jett may never see it")
        return self.seq

    async def poll_once(self):
        """Pull and match any new replies in out.jsonl."""
        rc, _ = await _git(["pull", "--rebase", "--autostash"])
        if rc != 0:
            return
        try:
            with open(self.out_path) as f:
                lines = f.read().splitlines()
        except OSError:
            return
        for line in lines[self.spoken_out:]:
            self.spoken_out += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            in_seq = obj.get("in_seq", 0)
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            if in_seq in self.pending and in_seq not in self.replies:
                self.replies[in_seq] = text
                self.pending[in_seq].set()
                log.info("consult #%d answered (%d chars)", in_seq,
                         len(text))

    async def bus_loop(self):
        """Background task: poll replies, heartbeat, speak answers."""
        hb_every = 12  # ~60 s at 5 s poll interval
        n = 0
        while self.active:
            await asyncio.sleep(5)
            if not self.active:
                break
            n += 1
            try:
                await self.poll_once()
                # deliver fresh replies proactively
                for seq, text in list(self.replies.items()):
                    self.replies.pop(seq, None)
                    self.pending.pop(seq, None)
                    if seq in self.timed_out or seq in self.delivered:
                        continue
                    # deliver fresh replies proactively
                    if self.session is not None:
                        log.info("delivering Jett's reply to consult #%d",
                                 seq)
                        self.session.say(
                            f"Jett says: {text}",
                            allow_interruptions=True,
                        )
                    self.delivered.add(seq)
                if n % hb_every == 0:
                    hb = {"ts": _utcnow(),
                          "last_in_seq": self.seq,
                          "last_out_seq": self.spoken_out}
                    with open(self.hb_path, "w") as f:
                        json.dump(hb, f)
                    await _bus_push_with_retry(
                        f"hb {self.call_id}")
            except Exception as e:
                log.warning("bus_loop error: %s", e)

    async def wait_reply(self, seq: int) -> str | None:
        """Wait up to CONSULT_TIMEOUT_SEC for Jett's reply."""
        ev = self.pending.get(seq)
        if ev is None:
            return None
        try:
            await asyncio.wait_for(ev.wait(), timeout=CONSULT_TIMEOUT_SEC)
            return self.replies.get(seq)
        except asyncio.TimeoutError:
            self.timed_out.add(seq)
            self.pending.pop(seq, None)
            return None


# the call currently being served (set in entrypoint); tools close over it
_CURRENT_BUS: BusCall | None = None


@function_tool
async def consult_jett(question: str) -> str:
    """Ask the real Jett something only he would know — his live memory,
    current info, messages, accounts, or real judgment. Use when the caller's
    question goes beyond your brief. Jett's reply arrives in a minute or two
    and is spoken to the caller automatically; keep the caller company
    meanwhile. NEVER invent Jett's answer."""
    bus = _CURRENT_BUS
    if bus is None or not bus.active:
        return "ERROR: no active call bus — tell the caller you can't reach Jett right now."
    seq = await bus.consult(question)
    if seq is None:
        return ("ERROR: failed to send the question to Jett — tell the "
                "caller honestly that you couldn't reach him.")
    # background waiter: speak the reply (or a timeout note) when it lands
    async def _waiter():
        reply = await bus.wait_reply(seq)
        if seq in bus.delivered or seq in bus.replies:
            return  # delivered (or about to be) by bus_loop
        if bus.session is not None and bus.active:
            bus.session.say(
                "I haven't heard back from Jett yet — he's probably tied up. "
                "I'll make sure he gets your question.",
                allow_interruptions=True,
            )
    asyncio.create_task(_waiter())
    return (f"Question sent to Jett as consult #{seq}. His reply will arrive "
            f"in a minute or two and will be spoken automatically. Tell the "
            f"caller you're checking with Jett and keep them company.")


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


async def _run_no_brain(ctx: JobContext, bus: BusCall):
    """No brain key: connect, play a spoken notice, hang up gracefully."""
    await ctx.connect()
    log.warning("no brain key (OPENROUTER_API_KEY/META_API_KEY) — playing no-key notice")
    tts_engine = KokoroTTS()
    samples, _ = await asyncio.get_running_loop().run_in_executor(
        None, tts_engine._synth,
        "Hey — this is Jett's voice line, but his brain isn't connected "
        "yet. He hasn't given me a key to think with. Try again later.")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    await _speak_raw(ctx.room, pcm)
    await bus.set_status("ended", end_reason="no_brain_key")
    await ctx.room.disconnect()


async def _run_call(ctx: JobContext, bus: BusCall, brain: Brain):
    global _CURRENT_BUS
    _CURRENT_BUS = bus

    log.info("call brain: %s models=%s", brain.label, brain.extra_body.get("models", brain.model))
    stt_engine = WhisperSTT()
    stt_stream = stt.StreamAdapter(
        stt=stt_engine, vad=silero.VAD.load())
    tts_engine = KokoroTTS()

    agent_llm = openai_plugin.LLM(
        model=brain.model,
        api_key=brain.api_key,
        base_url=brain.base_url,
        extra_headers=brain.extra_headers if brain.extra_headers else NOT_GIVEN,
        extra_body=brain.extra_body if brain.extra_body else NOT_GIVEN,
    )

    brief = load_brief()
    agent = Agent(
        instructions=SYSTEM_PROMPT.format(brief=brief),
        tools=[consult_jett],
    )
    session = AgentSession(
        stt=stt_stream,
        vad=silero.VAD.load(),
        llm=agent_llm,
        tts=tts_engine,
        max_tool_steps=10,
    )
    bus.session = session

    # --- brain resilience: per-turn latency log + graceful rate-limit ---
    # Free OpenRouter endpoints can be slower/flakier than paid ones.
    # - "metrics_collected" gives per-turn duration/ttft for visibility.
    # - "error" fires per failed attempt (the stream's built-in retry loop
    #   already backs off); on 429s we tell the caller once (debounced)
    #   instead of leaving silence, and the call never crashes here.
    _last_limit_notice = 0.0  # monotonic ts of last spoken rate-limit notice

    async def _say_brain_note(text: str) -> None:
        try:
            await session.say(text)
        except Exception as e:
            log.warning("brain notice speech failed: %s", e)

    def _on_llm_metrics(metrics) -> None:
        log.info("brain turn: model=%s duration=%.2fs ttft=%.2fs "
                 "tokens prompt=%d completion=%d",
                 brain.model, metrics.duration, metrics.ttft,
                 metrics.prompt_tokens, metrics.completion_tokens)

    def _on_llm_error(llm_error) -> None:
        nonlocal _last_limit_notice
        err = llm_error.error
        is_limit = _is_rate_limit(err)
        log.warning("brain error (recoverable=%s rate_limit=%s): %.200s",
                    llm_error.recoverable, is_limit, err)
        loop = asyncio.get_running_loop()
        if is_limit:
            now = time.monotonic()
            if now - _last_limit_notice > 30:
                _last_limit_notice = now
                loop.create_task(_say_brain_note(
                    "I'm hitting my rate limits \u2014 give me a few seconds "
                    "and I'll be right with you."))
        if not llm_error.recoverable:
            note = ("I'm having trouble reaching my brain right now \u2014 "
                    "could you say that once more?"
                    if is_limit else
                    "Sorry, I lost my train of thought \u2014 "
                    "could you repeat that?")
            loop.create_task(_say_brain_note(note))

    agent_llm.on("metrics_collected", _on_llm_metrics)
    agent_llm.on("error", _on_llm_error)

    await ctx.connect()
    await bus.set_status("active")
    await session.start(agent, room=ctx.room)
    log.info("session started in room %s", ctx.room.name)

    # background tasks
    bus_task = asyncio.create_task(bus.bus_loop())
    ended = asyncio.Event()

    def _on_participant_left(p: rtc.RemoteParticipant):
        if not ctx.room.remote_participants:
            log.info("caller hung up")
            ended.set()

    ctx.room.on("participant_disconnected", _on_participant_left)

    async def _watchdog():
        await asyncio.sleep(MAX_CALL_SEC)
        if bus.active:
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
        bus.active = False
        _CURRENT_BUS = None
        for t in (bus_task, wd_task):
            t.cancel()
        try:
            await bus.set_status("ended", end_reason="call_finished")
        except Exception as e:
            log.warning("final meta update failed: %s", e)
        try:
            await session.aclose()
        except Exception:
            pass


async def entrypoint(ctx: JobContext):
    global _ACTIVE_CALL
    call_id = ctx.room.name.replace("jett-", "", 1) or uuid.uuid4().hex[:8]

    with _CALL_LOCK:
        if _ACTIVE_CALL is not None:
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

    try:
        await ctx.connect()  # ensure room handle before reading participants
        # the SIP leg may still be joining; wait briefly for the caller
        caller = "unknown"
        for _ in range(16):
            caller = _sip_caller(ctx)
            if caller != "unknown":
                break
            await asyncio.sleep(0.5)
        log.info("inbound call %s from %s", call_id, caller)
        bus = BusCall(call_id, caller)
        bus_ok = await bus.open()
        brain = resolve_brain()
        if brain is not None:
            await _run_call(ctx, bus, brain)
        else:
            await _run_no_brain(ctx, bus)
        if not bus_ok:
            log.warning("bus unavailable; call proceeded without consult")
    except Exception as e:
        log.exception("call %s failed: %s", call_id, e)
    finally:
        with _CALL_LOCK:
            _ACTIVE_CALL = None
        log.info("call %s done", call_id)


if __name__ == "__main__":
    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise SystemExit("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are required")
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=AGENT_NAME,
        job_executor_type=JobExecutorType.THREAD,
        ws_url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    ))
