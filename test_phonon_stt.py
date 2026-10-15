#!/usr/bin/env python3
"""Tests for PhononSTT (2026-10-15): Phonon-2 via fermion-research
as the local pipeline's only STT — Whisper and the interim
Parakeet/Photon build are both gone.

The fermion client is faked — no model download, no fermion import.
The fakes mirror the vendor shapes (phonon2-bench harnesses):
  speech = fermion.load_speech("phonon-2", quiet=True)
  text, decode_s, dur_s = speech.transcribe_array(float32_array)
  sess = LiveSession(speech, on_partial=cb, on_final=cb)
  sess.feed(block)   # 16 kHz float32; decodes synchronously inside
  sess.finish()      # -> full transcript; on_final per cut segment

Covers:
  - capabilities / identity; Whisper + Parakeet classes are gone
  - stream: interims are cumulative (finals + in-flight partial),
    final on flush, final on close, second segment after flush
  - 8 kHz frames resampled to 16 kHz before reaching the session
  - recognize path (_recognize_impl) returns the batch final
  - silence short-circuits the engine
  - engine error mid-segment -> final falls back to last cumulative
  - _PhononEngine singleton: loads once via a stubbed fermion module
Run: ~/venv187/bin/python test_phonon_stt.py
"""
import asyncio
import sys
import types

sys.path.insert(0, ".")
import live_agent as la
from livekit import rtc
from livekit.agents import stt

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ------------------------------------------------------------ fake fermion
class FakeSpeech:
    """Scripted speech stand-in: text grows with seconds of audio."""

    def __init__(self):
        self.final_text = "what is on my calendar today"
        self.batch_calls = 0

    def text_at(self, secs):
        if secs < 0.4:
            return ""
        if secs < 0.9:
            return "what is on"
        return self.final_text

    def transcribe_array(self, audio):
        self.batch_calls += 1
        return self.text_at(len(audio) / 16000.0), 0.0, len(audio) / 16000.0


class FakeLiveSession:
    """LiveSession stand-in with the vendor's callback semantics:
    on_partial = current in-flight hypothesis (segment-local),
    on_final = one cut segment; finish() cuts and joins."""

    def __init__(self, speech, *, on_partial, on_final,
                 fail_after_s=None):
        self._speech = speech
        self._on_partial = on_partial
        self._on_final = on_final
        self._fail_after_s = fail_after_s
        self.samples = 0
        self._text = ""

    def feed(self, block):
        self.samples += len(block)
        secs = self.samples / 16000.0
        if self._fail_after_s and secs >= self._fail_after_s:
            raise RuntimeError("engine boom")
        text = self._speech.text_at(secs)
        if text and text != self._text:
            self._text = text
            self._on_partial(text)

    def finish(self):
        if self._text:
            self._on_final(self._text)
        return self._speech.final_text


def fake_factory(fail_after_s=None):
    def factory(speech, *, on_partial, on_final):
        return FakeLiveSession(speech, on_partial=on_partial,
                               on_final=on_final,
                               fail_after_s=fail_after_s)
    return factory


def frame(seconds, sr=16000, amp=0x10):
    n = int(seconds * sr)
    return rtc.AudioFrame(data=bytes([amp, 0x00]) * n, sample_rate=sr,
                          num_channels=1, samples_per_channel=n)


# ------------------------------------------------------------------ tests
async def collect_until_final(stream, timeout=5):
    events = []

    async def collect():
        async for ev in stream:
            events.append(ev)
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                return

    try:
        await asyncio.wait_for(asyncio.ensure_future(collect()), timeout)
    except asyncio.TimeoutError:
        pass
    return events


def t_removal_is_real():
    check("whisper classes gone",
          not hasattr(la, "WhisperSTT")
          and not hasattr(la, "StreamingWhisperSTT"))
    check("parakeet classes gone",
          not hasattr(la, "StreamingParakeetSTT")
          and not hasattr(la, "_PhotonEngine"))
    check("no STREAMING_STT flag", not hasattr(la, "STREAMING_STT"))
    check("phonon is the STT", hasattr(la, "PhononSTT"))


async def t_capabilities():
    engine = la.PhononSTT(FakeSpeech(), session_factory=fake_factory())
    check("capabilities streaming+interim",
          engine.capabilities.streaming
          and engine.capabilities.interim_results)
    check("identity", engine.model == "phonon-2"
          and engine.provider == "fermion")


async def t_stream_flush_and_second_segment():
    speech = FakeSpeech()
    engine = la.PhononSTT(speech, session_factory=fake_factory())
    stream = engine.stream()
    stream.push_frame(frame(0.5))
    await asyncio.sleep(0.25)
    stream.push_frame(frame(0.6))
    stream.flush()
    events = await collect_until_final(stream)
    types = [e.type for e in events]
    texts = [e.alternatives[0].text for e in events if e.alternatives]
    check("stream emitted interim(s)",
          stt.SpeechEventType.INTERIM_TRANSCRIPT in types,
          f"types={types}")
    check("stream final text",
          texts and texts[-1] == "what is on my calendar today",
          f"texts={texts}")
    check("stream final is last",
          types and types[-1] == stt.SpeechEventType.FINAL_TRANSCRIPT)

    # A flush ends the segment, not the stream: speak again.
    stream.push_frame(frame(1.1))
    stream.flush()
    events2 = await collect_until_final(stream)
    texts2 = [e.alternatives[0].text for e in events2 if e.alternatives]
    check("second segment transcribes",
          texts2 and texts2[-1] == "what is on my calendar today",
          f"texts2={texts2}")
    await stream.aclose()


async def t_stream_final_on_close():
    stream = la.PhononSTT(FakeSpeech(),
                          session_factory=fake_factory()).stream()
    events = []

    async def collect():
        async for ev in stream:
            events.append(ev)

    collector = asyncio.ensure_future(collect())
    stream.push_frame(frame(1.1))
    await asyncio.sleep(0.25)
    await stream.aclose()
    try:
        await asyncio.wait_for(collector, 5)
    except asyncio.TimeoutError:
        pass
    types = [e.type for e in events]
    texts = [e.alternatives[0].text for e in events if e.alternatives]
    check("close emits final",
          types and types[-1] == stt.SpeechEventType.FINAL_TRANSCRIPT,
          f"types={types}")
    check("close final text",
          texts and texts[-1] == "what is on my calendar today",
          f"texts={texts}")


async def t_stream_resamples_8k():
    speech = FakeSpeech()
    holder = {}

    def factory(sp, *, on_partial, on_final):
        holder["sess"] = FakeLiveSession(
            sp, on_partial=on_partial, on_final=on_final)
        return holder["sess"]

    stream = la.PhononSTT(speech, session_factory=factory).stream()
    stream.push_frame(frame(1.0, sr=8000))
    stream.flush()
    await collect_until_final(stream)
    # 1.0 s at 8 kHz must arrive at the session as ~16000 samples.
    got = holder["sess"].samples
    check("8k frames resampled to ~16k samples",
          15000 <= got <= 17000, f"samples={got}")
    await stream.aclose()


async def t_recognize_impl():
    speech = FakeSpeech()
    engine = la.PhononSTT(speech, session_factory=fake_factory())
    ev = await engine._recognize_impl(frame(1.1), conn_options=None)
    check("recognize returns final event",
          ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
    check("recognize text",
          ev.alternatives[0].text == "what is on my calendar today",
          f"text={ev.alternatives[0].text!r}")
    check("recognize used batch decode", speech.batch_calls == 1)


async def t_recognize_silence_skips_engine():
    speech = FakeSpeech()
    engine = la.PhononSTT(speech, session_factory=fake_factory())
    ev = await engine._recognize_impl(frame(1.0, amp=0x00),
                                      conn_options=None)
    check("silence -> empty final, engine untouched",
          ev.alternatives[0].text == "" and speech.batch_calls == 0,
          f"calls={speech.batch_calls}")


async def t_engine_error_falls_back_to_cumulative():
    speech = FakeSpeech()
    stream = la.PhononSTT(
        speech, session_factory=fake_factory(fail_after_s=0.9)).stream()
    stream.push_frame(frame(0.6))   # partial "what is on" lands
    await asyncio.sleep(0.25)
    stream.push_frame(frame(0.6))   # crosses 0.9 s: engine raises
    stream.flush()
    events = await collect_until_final(stream)
    texts = [e.alternatives[0].text for e in events if e.alternatives]
    check("error segment still finalizes with last cumulative",
          texts and texts[-1] == "what is on", f"texts={texts}")
    await stream.aclose()


def t_engine_singleton_loads_once():
    la._PhononEngine.reset()
    loads = []
    fake_speech = FakeSpeech()

    fermion_mod = types.ModuleType("fermion")
    speech_pkg = types.ModuleType("fermion._speech")
    live_mod = types.ModuleType("fermion._speech.live")

    def fake_load_speech(name, quiet=False):
        loads.append(name)
        return fake_speech

    fermion_mod.load_speech = fake_load_speech
    live_mod.warm_up = lambda speech: 0.0
    live_mod.LiveSession = FakeLiveSession
    saved = {k: sys.modules.get(k) for k in
             ("fermion", "fermion._speech", "fermion._speech.live")}
    sys.modules["fermion"] = fermion_mod
    sys.modules["fermion._speech"] = speech_pkg
    sys.modules["fermion._speech.live"] = live_mod
    try:
        s1 = la._PhononEngine.get()
        s2 = la._PhononEngine.get()
    finally:
        for k, v in saved.items():
            if v is None:
                del sys.modules[k]
            else:
                sys.modules[k] = v
        la._PhononEngine.reset()
    check("singleton identity", s1 is s2 and s1 is fake_speech)
    check("singleton loaded once", loads == ["phonon-2"],
          f"loads={loads}")


async def main():
    t_removal_is_real()
    t_engine_singleton_loads_once()
    await t_capabilities()
    await t_stream_flush_and_second_segment()
    await t_stream_final_on_close()
    await t_stream_resamples_8k()
    await t_recognize_impl()
    await t_recognize_silence_skips_engine()
    await t_engine_error_falls_back_to_cumulative()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("ALL PHONON STT TESTS PASSED")


asyncio.run(main())
