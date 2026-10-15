#!/usr/bin/env python3
"""Build-time bake for Pocket TTS (the worker's only TTS, 2026-10-15).

Forces the Pocket TTS weights and the "alba" voice prompt into the
image's HF cache so the worker never downloads at boot, and proves
the full load + streaming-synthesis path works inside the image.
Run by the Dockerfile only; not part of the runtime.

License: the Pocket TTS weights and the "alba" voice are CC-BY-4.0
(attribution in README.md). Other built-in voices (jean, cosette)
are CC-BY-NC — this bake builds the alba state ONLY.
"""
import numpy as np

from pocket_tts import TTSModel

VOICE = "alba"  # CC-BY-4.0. Never bake jean/cosette (CC-BY-NC).

model = TTSModel.load_model()
state = model.get_state_for_audio_prompt(VOICE)

chunks = []
for chunk in model.generate_audio_stream(
        state, "Hi, this is Jett, Jim's assistant."):
    chunks.append(chunk.detach().cpu().numpy().astype(np.float32))
assert len(chunks) > 1, f"expected a chunk stream, got {len(chunks)}"
audio = np.concatenate([c.squeeze() for c in chunks])
rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
print(f"pocket bake: voice={VOICE} chunks={len(chunks)} "
      f"samples={audio.size} sr={model.sample_rate} rms={rms:.4f}")
assert audio.size > 0 and rms > 0.01, "pocket bake: silent output"
print("pocket bake OK")
