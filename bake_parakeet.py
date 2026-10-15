#!/usr/bin/env python3
"""Build-time bake for Parakeet Redux (STREAMING_STT, 2026-10-15).

Forces the Photon weights into the image's HF cache so the worker
never downloads at boot, and proves the full load + transcribe path
works inside the image before it ships. Run by the Dockerfile only;
not part of the runtime.
"""
import struct
import wave

path = "/tmp/silence.wav"
w = wave.open(path, "wb")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(16000)
w.writeframes(struct.pack("<16000h", *([0] * 16000)))
w.close()

import moondream as md

with md.photon("moondream/parakeet-redux", device="cpu") as speech:
    out = speech.transcribe(audio=path)
    print("parakeet bake OK:", out)
