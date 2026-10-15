#!/usr/bin/env python3
"""Tests for StreamingParakeetSTT (STREAMING_STT, 2026-10-15):
Parakeet Redux via Moondream Photon as the local pipeline's STT.

The Photon client is faked — no model download, no moondream
import. The fake mirrors the proven bench API shape (stream_bench.py):
  updates = await speech.atranscribe(audio=<async PCM gen>,
                                     sample_rate=16000, stream=True)
  async for upd in updates: upd["text"]  (progressive snapshots)
  final = await updates.aresult()       -> {"text": ...}

Covers:
  - capabilities / identity
  - stream: interims from snapshots, final on flush, final on close,
    a second segment after flush (per-utterance Photon sessions)
  - 8 kHz frames resampled to 16 kHz before reaching Photon
  - recognize path (_recognize_impl) returns the Photon final
  - silence short-circuits the engine
  - engine error mid-segment -> final falls back to last snapshot
  - _PhotonEngine singleton: loads once via a stubbed moondream
Run: ~/venv187/bin/python test_parakeet_stt.py
"""
import asyncio
import sys
import time
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


# ------------------------------------------------------------ fake Photon
class FakeUpdates:
    """Async-iterable snapshot stream + aresult(), as Photon returns."""

    def __init__(self, speech, audio_gen):
        self._speech = speech
        self._gen = audio_gen

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        # Consume in 0.25 s slices regardless of producer chunking:
        # a real engine snapshots as audio arrives, so a failure at
        # 0.9 s still leaves the 0.5 s snapshot on the record.
        total = 0
        async for chunk in self._gen:
            for off in range(0, len(chunk), 4000):
                total += len(chunk[off:off + 4000])
                self._speech.samples_seen = total
                secs = total / 16000.0
                if self._speech.fail_after_s \
                        and secs >= self._speech.fail_after_s:
                    raise RuntimeError("photon boom")
                text = self._speech.text_at(secs)
                if text:
                    yield {"text": text}
        self._speech.consumed = True

    async def aresult(self):
        return {"text": self._speech.final_text}


class FakeSpeech:
    """Scripted Photon stand-in: text grows with seconds of audio."""

    def __init__(self, fail_after_s=None):
        self.samples_seen = 0
        self.consumed = False
        self.calls = 0
        self.sample_rates = []
        self.fail_after_s = fail_after_s
        self.final_text = "what is on my calendar today"

    def text_at(self, secs):
        if secs < 0.4:
            return ""
        if secs < 0.9:
            return "what is on"
        return self.final_text

    async def atranscribe(self, *, audio, sample_rate, stream):
        assert stream is True
        self.calls += 1
        self.sample_rates.append(sample_rate)
        return FakeUpdates(self, audio)


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


async def t_capabilities():
    engine = la.StreamingParakeetSTT(FakeSpeech())
    check("capabilities streaming+interim",
          engine.capabilities.streaming
          and engine.capabilities.interim_results)
    check("identity", engine.model == "parakeet-redux"
          and engine.provider == "moondream-photon")


async def t_stream_flush_and_second_segment():
    speech = FakeSpeech()
    engine = la.StreamingParakeetSTT(speech)
    stream = engine.stream()
    stream.push_frame(frame(0.5))
    await asyncio.sleep(0.2)
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
    check("photon got 16k stream", speech.sample_rates == [16000],
          f"rates={speech.sample_rates}")

    # A flush ends the segment, not the stream: speak again.
    stream.push_frame(frame(1.1))
    stream.flush()
    events2 = await collect_until_final(stream)
    texts2 = [e.alternatives[0].text for e in events2 if e.alternatives]
    check("second segment transcribes",
          texts2 and texts2[-1] == "what is on my calendar today",
          f"texts2={texts2}")
    check("second photon session opened", speech.calls == 2,
          f"calls={speech.calls}")
    await stream.aclose()


async def t_stream_final_on_close():
    speech = FakeSpeech()
    stream = la.StreamingParakeetSTT(speech).stream()
    events = []

    async def collect():
        async for ev in stream:
            events.append(ev)

    collector = asyncio.ensure_future(collect())
    stream.push_frame(frame(1.1))
    await asyncio.sleep(0.2)
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
    stream = la.StreamingParakeetSTT(speech).stream()
    stream.push_frame(frame(1.0, sr=8000))
    stream.flush()
    await collect_until_final(stream)
    # 1.0 s at 8 kHz must arrive at Photon as ~16000 samples at 16 kHz.
    check("8k frames resampled to ~16k samples",
          15000 <= speech.samples_seen <= 17000,
          f"samples={speech.samples_seen}")
    await stream.aclose()


async def t_recognize_impl():
    speech = FakeSpeech()
    engine = la.StreamingParakeetSTT(speech)
    merged = frame(1.1)  # StreamAdapter-style merged buffer
    ev = await engine._recognize_impl(merged, conn_options=None)
    check("recognize returns final event",
          ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
    check("recognize text",
          ev.alternatives[0].text == "what is on my calendar today",
          f"text={ev.alternatives[0].text!r}")


async def t_recognize_silence_skips_engine():
    speech = FakeSpeech()
    engine = la.StreamingParakeetSTT(speech)
    silent = frame(1.0, amp=0x00)
    ev = await engine._recognize_impl(silent, conn_options=None)
    check("silence -> empty final, engine untouched",
          ev.alternatives[0].text == "" and speech.calls == 0,
          f"calls={speech.calls}")


async def t_engine_error_falls_back_to_last_snapshot():
    speech = FakeSpeech(fail_after_s=0.9)
    stream = la.StreamingParakeetSTT(speech).stream()
    stream.push_frame(frame(1.2))
    stream.flush()
    events = await collect_until_final(stream)
    texts = [e.alternatives[0].text for e in events if e.alternatives]
    # Snapshot "what is on" (0.4-0.9 s) landed before the failure; the
    # final must carry it rather than dying with the engine.
    check("error segment still finalizes",
          texts and texts[-1] == "what is on", f"texts={texts}")
    await stream.aclose()


def t_engine_singleton_loads_once():
    la._PhotonEngine.reset()
    loads = []

    class FakeCM:
        def __enter__(self):
            return FakeSpeech()

        def __exit__(self, *a):
            return False

    fake_md = types.ModuleType("moondream")

    def fake_photon(model, device=None):
        loads.append((model, device))
        return FakeCM()

    fake_md.photon = fake_photon
    saved = sys.modules.get("moondream")
    sys.modules["moondream"] = fake_md
    try:
        s1 = la._PhotonEngine.get()
        s2 = la._PhotonEngine.get()
    finally:
        if saved is None:
            del sys.modules["moondream"]
        else:
            sys.modules["moondream"] = saved
        la._PhotonEngine.reset()
    check("singleton identity", s1 is s2)
    check("singleton loaded once", len(loads) == 1
          and loads[0][1] == "cpu", f"loads={loads}")


def t_flag_default_off():
    # STREAMING_STT is read from env at import; in the test env it is
    # unset, so production default stays the Whisper path.
    check("STREAMING_STT default off in test env",
          la.STREAMING_STT is False)


async def main():
    t_flag_default_off()
    t_engine_singleton_loads_once()
    await t_capabilities()
    await t_stream_flush_and_second_segment()
    await t_stream_final_on_close()
    await t_stream_resamples_8k()
    await t_recognize_impl()
    await t_recognize_silence_skips_engine()
    await t_engine_error_falls_back_to_last_snapshot()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("ALL PARAKEET STT TESTS PASSED")


asyncio.run(main())
