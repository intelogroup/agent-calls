#!/usr/bin/env python3
"""Synthesize the spoken message with Kokoro TTS -> message.wav (24 kHz)."""
import sys

from kokoro import KPipeline
import soundfile as sf
import torch

text = sys.argv[1].strip()
if not text:
    sys.exit("empty message text")

pipeline = KPipeline(lang_code="a")  # American English
chunks = []
for _, _, audio in pipeline(text, voice="af_heart"):
    chunks.append(audio)

speech = torch.cat(chunks, dim=0).numpy()
sf.write("message.wav", speech, 24000)
print(f"wrote message.wav ({len(speech) / 24000:.1f}s)")
