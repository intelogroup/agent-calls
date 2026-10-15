#!/usr/bin/env python3
"""Tests for the SPECULATIVE_FEED spike (2026-10-15): streaming Whisper
interims + speculative draft/commit layer in live_agent.py.

Covers:
  - texts_compatible / draft_is_speakable policy functions
  - SpeculativeDrafter: settle-debounce, draft on interim, accept on
    matching final, wait-for-in-flight, reject on mismatch/unspeakable
  - StreamingWhisperSTT: interim events from incremental transcription,
    final on flush, sharing the WhisperSTT model object
  - SpeculativeLLM: draft hit replays without touching the inner chain;
    miss delegates untouched
Run: ~/venv187/bin/python test_streaming_feed.py
"""
import asyncio
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, ".")
import live_agent as la
from livekit import rtc
from livekit.agents import llm, stt

la.SPEC_SETTLE_S = 0.02
la.SPEC_STREAM_INTERVAL_S = 0.05
la.SPEC_DRAFT_WAIT_S = 1.0

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ------------------------------------------------------------ policy fns
def t_policy():
    check("compat exact", la.texts_compatible(
        "what is on my calendar today", "What is on my calendar today?"))
    check("compat prefix (final grew)", la.texts_compatible(
        "what is on my calendar", "what is on my calendar today"))
    check("compat prefix (final shrank)", la.texts_compatible(
        "remind me to call the dentist tomorrow", "remind me to call the dentist"))
    check("compat mismatch", not la.texts_compatible(
        "what is on my calendar today", "call mom and tell her about lunch"))
    check("speakable plain answer", la.draft_is_speakable(
        "You have a quiet morning, nothing until the afternoon."))
    check("unspeakable digits", not la.draft_is_speakable(
        "Your appointment is at 3 in the afternoon."))
    check("unspeakable deferral", not la.draft_is_speakable(
        "Let me check that for you right away."))
    check("unspeakable too short", not la.draft_is_speakable("Sure."))


# --------------------------------------------------------------- drafter
async def t_drafter_accept():
    calls = []

    async def gen(basis):
        calls.append(basis)
        await asyncio.sleep(0.05)
        return "You have a quiet morning, nothing until later."

    d = la.SpeculativeDrafter(gen)
    d.observe("what is on my calendar")
    d.observe("what is on my calendar today")
    await asyncio.sleep(0.3)
    check("drafter drafted latest basis once", calls == ["what is on my calendar today"],
          f"calls={calls}")
    out = await d.take("What is on my calendar today?")
    check("drafter accept on matching final", out == "You have a quiet morning, nothing until later.")


async def t_drafter_waits_for_inflight():
    async def gen(basis):
        await asyncio.sleep(0.25)
        return "Nothing urgent, just the usual morning routine."

    d = la.SpeculativeDrafter(gen)
    d.observe("anything urgent this morning")
    await asyncio.sleep(0.06)  # settle passed, generation in flight
    t0 = time.monotonic()
    out = await d.take("anything urgent this morning")
    waited = time.monotonic() - t0
    check("drafter waits for in-flight draft", out is not None and waited >= 0.15,
          f"waited={waited:.2f}")


async def t_drafter_rejects():
    async def gen(basis):
        return "Your train leaves at 5 from the station."  # digits: unspeakable

    d = la.SpeculativeDrafter(gen)
    d.observe("when does my train leave")
    await asyncio.sleep(0.2)
    out = await d.take("when does my train leave")
    check("drafter rejects unspeakable draft", out is None)

    async def gen2(basis):
        return "A perfectly fine conversational answer here."

    d2 = la.SpeculativeDrafter(gen2)
    d2.observe("tell me a joke about cats")
    await asyncio.sleep(0.2)
    out2 = await d2.take("what is the capital of france")
    check("drafter rejects mismatched final", out2 is None)


async def t_drafter_short_interims_ignored():
    calls = []

    async def gen(basis):
        calls.append(basis)
        return "Some answer that is long enough to speak."

    d = la.SpeculativeDrafter(gen)
    d.observe("hi there")
    await asyncio.sleep(0.15)
    check("short interims never draft", calls == [])


# ------------------------------------------------------------ streaming STT
class FakeModel:
    """Scripted faster-whisper stand-in: text grows with audio length."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, language="en", beam_size=1, vad_filter=True):
        self.calls += 1
        secs = len(audio) / 16000.0
        if secs < 0.4:
            text = ""
        elif secs < 0.9:
            text = "what is on"
        else:
            text = "what is on my calendar today"
        return [SimpleNamespace(text=text)], None


def frame(seconds, sr=16000):
    n = int(seconds * sr)
    return rtc.AudioFrame(data=b"\x10\x00" * n, sample_rate=sr,
                          num_channels=1, samples_per_channel=n)


async def t_streaming_stt():
    model = FakeModel()
    engine = la.StreamingWhisperSTT(model)
    check("streaming capabilities", engine.capabilities.streaming
          and engine.capabilities.interim_results)
    stream = engine.stream()
    events = []

    async def collect():
        async for ev in stream:
            events.append(ev)
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
    types = [e.type for e in events]
    texts = [e.alternatives[0].text for e in events if e.alternatives]
    check("stream emitted interim(s)", stt.SpeechEventType.INTERIM_TRANSCRIPT in types,
          f"types={types}")
    check("stream final text", texts and texts[-1] == "what is on my calendar today",
          f"texts={texts}")
    check("stream final is last", types and types[-1] == stt.SpeechEventType.FINAL_TRANSCRIPT)
    await stream.aclose()


# ---------------------------------------------------------- speculative LLM
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
    return llm.ChatContext(items=[llm.ChatMessage(role="user", content=[text])])


async def drain(stream):
    parts = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            parts.append(chunk.delta.content)
    return "".join(parts)


async def t_spec_llm():
    inner = FakeInnerLLM()

    async def gen(basis):
        return "You have a quiet morning, nothing until later."

    drafter = la.SpeculativeDrafter(gen)
    spec = la.SpeculativeLLM(inner, drafter)
    out = await drain(spec.chat(chat_ctx=ctx_with_user("hello there friend"), tools=[]))
    check("spec miss delegates to inner", out == "the real chain answer" and inner.calls == 1)

    drafter.observe("what is on my calendar today")
    await asyncio.sleep(0.25)
    out2 = await drain(spec.chat(
        chat_ctx=ctx_with_user("What is on my calendar today?"), tools=[]))
    check("spec hit replays draft", out2 == "You have a quiet morning, nothing until later.",
          f"out={out2!r}")
    check("spec hit skipped inner", inner.calls == 1)


async def t_draft_generator():
    class FakeBrain:
        def __init__(self):
            self.seen_tools = None
            self.seen_ctx = None

        def chat(self, *, chat_ctx, tools=None, **kw):
            self.seen_tools = tools
            self.seen_ctx = chat_ctx

            class S:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    raise StopAsyncIteration

            # one chunk then stop: emulate with an async generator stream
            async def agen():
                yield SimpleNamespace(delta=SimpleNamespace(content="Draft answer text."))

            return agen()

    brain = FakeBrain()
    gen = la._make_draft_generator(brain, "SYS", lambda: [("user", "earlier question"),
                                                          ("assistant", "earlier answer")])
    text = await gen("the partial question")
    check("generator drains brain text", text == "Draft answer text.")
    check("generator passes no tools", brain.seen_tools == [])
    roles = [i.role for i in brain.seen_ctx.items]
    check("generator ctx shape", roles == ["system", "user", "assistant", "user"],
          f"roles={roles}")


async def main():
    t_policy()
    await t_drafter_accept()
    await t_drafter_waits_for_inflight()
    await t_drafter_rejects()
    await t_drafter_short_interims_ignored()
    await t_streaming_stt()
    await t_spec_llm()
    await t_draft_generator()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("ALL STREAMING-FEED TESTS PASSED")


asyncio.run(main())
