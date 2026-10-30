#!/usr/bin/env python3
"""Synthesize the spoken message -> message.wav (24 kHz).

Engine chain (Jim's ruling, 2026-10-30):
  1. Kokoro ONNX (int8 model, voice af_heart) -- the primary, approved voice.
  2. Pocket TTS (voice alba) -- second option, only if Kokoro ONNX fails.
  3. Both fail -> error to stderr, exit 1. The call fails loudly; no
     substitute robot voice is ever rendered.

Kokoro model paths come from env KOKORO_ONNX_MODEL / KOKORO_ONNX_VOICES,
defaulting to ./kokoro-onnx/kokoro-v1.0.int8.onnx and
./kokoro-onnx/voices-v1.0.bin (the workflow downloads them there first).
"""
import os
import re
import sys

import numpy as np

OUT = "message.wav"
KOKORO_VOICE = "af_heart"
POCKET_VOICE = "alba"


def synth_kokoro_onnx(text):
    from kokoro_onnx import Kokoro

    model = os.environ.get(
        "KOKORO_ONNX_MODEL", "./kokoro-onnx/kokoro-v1.0.int8.onnx")
    voices = os.environ.get(
        "KOKORO_ONNX_VOICES", "./kokoro-onnx/voices-v1.0.bin")
    kokoro = Kokoro(model, voices)
    # sentence-split like Jim's approved say.py, 0.12 s gap between parts
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks, sr = [], None
    for p in parts:
        if not p.strip():
            continue
        samples, sr = kokoro.create(p, voice=KOKORO_VOICE, speed=1.0,
                                    lang="en-us")
        chunks.append(np.asarray(samples, dtype=np.float32))
        chunks.append(np.zeros(int(0.12 * sr), dtype=np.float32))
    if not chunks:
        raise RuntimeError("kokoro-onnx produced no audio")
    return np.concatenate(chunks), sr


def synth_pocket(text):
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt(POCKET_VOICE)
    audio = model.generate_audio(state, text)
    samples = audio.detach().cpu().numpy().astype(np.float32).squeeze()
    return samples, int(model.sample_rate)


def main():
    text = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not text:
        print("empty message text", file=sys.stderr)
        return 1

    audio = sr = engine = None
    errors = []
    for name, fn in (("kokoro-onnx", synth_kokoro_onnx),
                     ("pocket", synth_pocket)):
        try:
            audio, sr = fn(text)
            engine = name
            break
        except Exception as e:  # any engine failure -> try the next one
            errors.append(f"{name}: {type(e).__name__}: {e}")
            print(f"[tts] {name} failed: {e}", file=sys.stderr)
    if engine is None:
        print("ERROR: all TTS engines failed:\n" + "\n".join(errors),
              file=sys.stderr)
        return 1

    import soundfile as sf
    tmp = OUT.replace(".wav", ".tmp.wav")  # keep .wav suffix: soundfile needs it
    try:
        sf.write(tmp, audio, sr)
        os.replace(tmp, OUT)
    except Exception:
        for f in (tmp, OUT):
            if os.path.exists(f):
                os.unlink(f)
        raise

    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    print(f"TTS engine: {engine}")
    print(f"wrote message.wav ({len(audio) / sr:.2f}s, sr={sr}, rms={rms:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
