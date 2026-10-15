#!/usr/bin/env python3
"""Regression test: SPECULATIVE_FEED with the new engines (2026-10-15).

Jim's scope note: his latency question was always about the LOCAL
(fallback) pipeline — the Deepgram bridge is untouched by this.
This test exercises the local self-tier turn end to end with the
two new engines in their slots, mirroring _run_call's wiring
(user_input_transcribed -> spec_drafter.observe on interims;
session LLM = SpeculativeLLM; session TTS = the slot engine):

  Phonon-2 interims (fake LiveSession, vendor shapes as in
  test_streaming_feed) -> SpeculativeDrafter drafts on the interim
  -> final transcript arrives -> SpeculativeLLM replays the draft
  WITHOUT touching the inner chain (the turn COMMITS — the old
  streaming-Whisper defect was a ready draft whose turn never
  committed, leaving silence) -> the replayed text goes through
  PocketTTS.synthesize -> _PocketChunkedStream pushes non-silent
  PCM frames (the turn SPEAKS, through the new TTS).

Vendor edges are faked (fermion session, pocket_tts model, inner
LLM); everything between them is the production code.
Run: <venv>/bin/python test_spec_feed_pocket.py
"""
import asyncio
import sys
import types

import numpy as np

sys.path.insert(0, ".")
import live_agent as la  # noqa: E402
from livekit import rtc  # noqa: E402
from livekit.agents import llm, stt  # noqa: E402

la.SPEC_SETTLE_S = 0.02
la.SPEC_DRAFT_WAIT_S = 1.0

FAILED = []
DRAFT = "You have a quiet morning, nothing until later."
FINAL = "what is on my calendar today"


def check(name, cond, extra=""):
    if not cond:
        FAILED.append(name)
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))


# ------------------------------------------------- fake Phonon session
class FakeSpeech:
    def __init__(self):
        self.final_text = FINAL

    def text_at(self, secs):
        if secs < 0.4:
            return ""
        if secs < 0.9:
            return "what is on"
        return self.final_text

    def transcribe_array(self, audio):
        return self.text_at(len(audio) / 16000.0), 0.0, len(audio) / 16000.0


class FakeLiveSession:
    def __init__(self, speech, *, on_partial, on_final):
        self._speech = speech
        self._on_partial = on_partial
        self._on_final = on_final
        self._samples = 0
        self._text = ""

    def feed(self, block):
        self._samples += len(block)
        text = self._speech.text_at(self._samples / 16000.0)
        if text and text != self._text:
            self._text = text
            self._on_partial(text)

    def finish(self):
        if self._text:
            self._on_final(self._text)
        return self._speech.final_text


def frame(seconds, sr=16000):
    n = int(seconds * sr)
    return rtc.AudioFrame(data=b"\x10\x00" * n, sample_rate=sr,
                          num_channels=1, samples_per_channel=n)


# ------------------------------------------------------- fake inner LLM
class _FakeStream(llm.LLMStream):
    def __init__(self, owner, *, chat_ctx, tools, conn_options, text):
        super().__init__(owner, chat_ctx=chat_ctx, tools=tools,
                         conn_options=conn_options)
        self._text = text

    async def _run(self):
        self._event_ch.send_nowait(llm.ChatChunk(
            id="fake-1", delta=llm.ChoiceDelta(content=self._text,
                                               role="assistant")))


class FakeInnerLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self.calls = 0

    @property
    def model(self):
        return "fake-model"

    @property
    def provider(self):
        return "fake"

    def chat(self, *, chat_ctx, tools=None, conn_options=None,
             parallel_tool_calls=None, tool_choice=None,
             extra_kwargs=None, **kw):
        self.calls += 1
        return _FakeStream(self, chat_ctx=chat_ctx, tools=tools or [],
                           conn_options=conn_options,
                           text="the real chain answer")


def ctx_with_user(text):
    return llm.ChatContext(items=[llm.ChatMessage(role="user",
                                                  content=[text])])


async def drain(stream):
    parts = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            parts.append(chunk.delta.content)
    return "".join(parts)


# --------------------------------------------------------- fake Pocket
class FakeTensor:
    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakePocketModel:
    sample_rate = 24000

    @classmethod
    def load_model(cls):
        return cls()

    def get_state_for_audio_prompt(self, voice):
        return {"voice": voice}

    def generate_audio_stream(self, state, text):
        for i in range(6):
            n = 1920
            yield FakeTensor(
                (0.2 * np.sin(np.arange(n) * 0.1 + i)).astype(np.float32))


class FakeEmitter:
    def __init__(self):
        self.kw = None
        self.pushed = []

    def initialize(self, **kw):
        self.kw = kw

    def push(self, data):
        self.pushed.append(bytes(data))

    def flush(self):
        pass


# ----------------------------------------------------------------- main
async def main():
    # --- stage 1: Phonon interims feed the drafter (production wiring)
    gen_bases = []

    async def gen(basis):
        gen_bases.append(basis)
        await asyncio.sleep(0.05)
        return DRAFT

    drafter = la.SpeculativeDrafter(gen)
    engine = la.PhononSTT(
        FakeSpeech(),
        session_factory=lambda sp, *, on_partial, on_final:
            FakeLiveSession(sp, on_partial=on_partial,
                            on_final=on_final))
    stream = engine.stream()
    events = []

    async def collect():
        async for ev in stream:
            events.append(ev)
            # mirror _on_user_transcribed: interims -> drafter.observe
            if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                text = ev.alternatives[0].text.strip()
                if text:
                    drafter.observe(text)
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                break

    collector = asyncio.ensure_future(collect())
    stream.push_frame(frame(0.5))
    await asyncio.sleep(0.25)
    stream.push_frame(frame(0.6))
    await asyncio.sleep(0.3)
    stream.flush()
    try:
        await asyncio.wait_for(collector, 5)
    except asyncio.TimeoutError:
        pass
    await stream.aclose()

    ev_types = [e.type for e in events]
    texts = [e.alternatives[0].text for e in events if e.alternatives]
    check("phonon interims emitted",
          stt.SpeechEventType.INTERIM_TRANSCRIPT in ev_types,
          f"types={ev_types}")
    check("phonon final text", texts and texts[-1] == FINAL,
          f"texts={texts}")
    check("drafter drafted from interims", len(gen_bases) >= 1,
          f"bases={gen_bases}")

    # --- stage 2: the turn commits via the draft (inner untouched)
    inner = FakeInnerLLM()
    spec = la.SpeculativeLLM(inner, drafter)
    out = await drain(spec.chat(chat_ctx=ctx_with_user(
        "What is on my calendar today?"), tools=[]))
    check("turn committed the draft", out == DRAFT, f"out={out!r}")
    check("inner chain untouched on hit", inner.calls == 0,
          f"calls={inner.calls}")

    # --- stage 3: the committed turn SPEAKS through Pocket TTS
    mod = types.ModuleType("pocket_tts")
    mod.TTSModel = FakePocketModel
    sys.modules["pocket_tts"] = mod
    la._PocketEngine.reset()
    tts_engine = la.PocketTTS()
    em = FakeEmitter()
    await tts_engine.synthesize(out)._run(em)
    total = sum(len(p) for p in em.pushed)
    check("pocket pushed frames for the draft", len(em.pushed) > 10,
          f"frames={len(em.pushed)}")
    check("pocket audio bytes flowed", total > 0, f"bytes={total}")
    pcm = np.frombuffer(b"".join(em.pushed), dtype=np.int16)
    rms = float(np.sqrt(np.mean((pcm / 32767.0) ** 2))) if pcm.size else 0.0
    check("spoken draft is non-silent", rms > 0.05, f"rms={rms:.3f}")

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("ALL SPEC-FEED + POCKET REGRESSION TESTS PASSED")


asyncio.run(main())
