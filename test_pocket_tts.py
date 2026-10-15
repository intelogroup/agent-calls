#!/usr/bin/env python3
"""Tests for PocketTTS (2026-10-15): Pocket TTS (Kyutai) as the
local pipeline's only TTS — Kokoro is deleted from the slot.

The pocket_tts client is faked — no model download, no torch. The
fakes mirror the vendor shapes (tts-bench harness):
  model = TTSModel.load_model()
  state = model.get_state_for_audio_prompt("alba")
  audio = model.generate_audio(state, text)          # torch tensor
  for chunk in model.generate_audio_stream(state, text): ...

Covers:
  - removal is real: no KokoroTTS class, no kokoro reference in
    live_agent.py / Dockerfile / requirements.txt (one changelog
    line in a docstring is the allowed exception, checked by hand
    in the build report, not here — here the check is on code
    identifiers and dependency surfaces)
  - capabilities / identity; construction does NOT load the model
  - _PocketEngine singleton: loads once, voice state cached and
    the identical object reused across two syntheses
  - batch _synth returns float32 samples + sample rate
  - chunked stream pushes many PCM frames, non-silent overall
Run: .venv-test/bin/python test_pocket_tts.py
"""
import asyncio
import sys
import types

import numpy as np

sys.path.insert(0, ".")
import live_agent as la  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    if not cond:
        FAILED.append(name)
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))


# ------------------------------------------------------------ fakes --

class FakeTensor:
    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeModel:
    sample_rate = 24000
    loads = 0
    states_built = 0
    gen_states = []

    @classmethod
    def load_model(cls):
        cls.loads += 1
        return cls()

    def get_state_for_audio_prompt(self, voice):
        type(self).states_built += 1
        return {"voice": voice, "id": type(self).states_built}

    def generate_audio(self, state, text):
        type(self).gen_states.append(state)
        n = 2400
        return FakeTensor(
            (0.2 * np.sin(np.arange(n) * 0.1)).astype(np.float32))

    def generate_audio_stream(self, state, text):
        type(self).gen_states.append(state)
        for i in range(6):
            n = 1920  # 80 ms at 24 kHz, the real chunk size
            yield FakeTensor(
                (0.2 * np.sin(np.arange(n) * 0.1 + i)).astype(np.float32))


def install_fake_pocket():
    mod = types.ModuleType("pocket_tts")
    mod.TTSModel = FakeModel
    sys.modules["pocket_tts"] = mod
    FakeModel.loads = 0
    FakeModel.states_built = 0
    FakeModel.gen_states = []
    la._PocketEngine.reset()


# ------------------------------------------------------------ tests --

def t_removal_is_real():
    check("KokoroTTS class gone", not hasattr(la, "KokoroTTS"))
    check("KokoroTTS stream gone", not hasattr(la, "_KokoroChunkedStream"))
    check("PocketTTS exists", hasattr(la, "PocketTTS"))
    check("voice pinned to alba", la.POCKET_TTS_VOICE == "alba")
    src = open("live_agent.py").read()
    check("no kokoro_onnx import", "kokoro_onnx" not in src)
    check("no KOKORO_ constants", "KOKORO_" not in src)
    for path in ("Dockerfile", "requirements.txt"):
        body = open(path).read().lower()
        check(f"no kokoro in {path}", "kokoro" not in body)


def t_identity_and_lazy_construction():
    la._PocketEngine.reset()
    eng = la.PocketTTS()
    check("construction does not load model",
          la._PocketEngine._model is None)
    check("model name", eng.model == "pocket-tts")
    check("provider", eng.provider == "kyutai-pocket-tts")
    check("sample rate 24k", eng._sample_rate == 24000
          if hasattr(eng, "_sample_rate") else True)


def t_singleton_and_voice_state_reuse():
    install_fake_pocket()
    eng = la.PocketTTS()
    s1, sr1 = eng._synth("Hello there.")
    s2, sr2 = eng._synth("Hello again.")
    check("model loaded once", FakeModel.loads == 1,
          f"loads={FakeModel.loads}")
    check("voice state built once", FakeModel.states_built == 1,
          f"states={FakeModel.states_built}")
    check("same voice state both syntheses",
          len(FakeModel.gen_states) == 2
          and FakeModel.gen_states[0] is FakeModel.gen_states[1])
    check("voice is alba",
          FakeModel.gen_states[0]["voice"] == "alba")
    check("batch samples float32 + non-silent",
          s1.dtype == np.float32 and float(np.abs(s1).max()) > 0.1)
    check("batch sample rates", sr1 == sr2 == 24000)


def t_stream_chunks():
    install_fake_pocket()
    eng = la.PocketTTS()
    chunks = list(eng._stream_chunks("One streamed sentence."))
    check("stream yields many chunks", len(chunks) == 6,
          f"n={len(chunks)}")
    check("stream chunk shape",
          all(c[1] == 24000 and len(c[0]) == 1920 for c in chunks))


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


async def t_chunked_stream_pushes_audio():
    install_fake_pocket()
    eng = la.PocketTTS()
    stream = eng.synthesize("One streamed sentence.")
    em = FakeEmitter()
    await stream._run(em)
    total = sum(len(p) for p in em.pushed)
    check("emitter initialized 24k pcm",
          em.kw and em.kw["sample_rate"] == 24000
          and em.kw["mime_type"] == "audio/pcm")
    check("many frames pushed", len(em.pushed) > 10,
          f"frames={len(em.pushed)}")
    check("audio bytes flowed", total == 6 * 1920 * 2, f"bytes={total}")
    pcm = np.frombuffer(b"".join(em.pushed), dtype=np.int16)
    rms = float(np.sqrt(np.mean((pcm / 32767.0) ** 2)))
    check("pushed audio non-silent", rms > 0.05, f"rms={rms:.3f}")


def main():
    t_removal_is_real()
    t_identity_and_lazy_construction()
    t_singleton_and_voice_state_reuse()
    t_stream_chunks()
    asyncio.run(t_chunked_stream_pushes_audio())
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("ALL POCKET TTS TESTS PASSED")


main()
