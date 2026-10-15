#!/usr/bin/env python3
"""Warm real-model run for PocketTTS through live_agent's own code
(2026-10-15). Uses the REAL pocket-tts model (cached weights) —
this is the build's proof, not a unit test:

  - _PocketEngine loads once; two consecutive _stream_chunks runs
    reuse the identical model + voice-state objects
  - stream yields many chunks; first chunk (median of 3 reps of the
    greeting) lands in < 1 s warm
  - concatenated output is non-silent (RMS)
Run: .venv-test/bin/python test_pocket_warm.py
"""
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import live_agent as la  # noqa: E402

GREETING = ("Hi, this is Jett, Jim's assistant. Jim can't come to "
            "the phone right now, but I can take a message.")

FAILED = []


def check(name, cond, extra=""):
    if not cond:
        FAILED.append(name)
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))


def main():
    t0 = time.monotonic()
    model1, state1 = la._PocketEngine.get()
    load_s = time.monotonic() - t0
    print(f"engine load: {load_s:.2f}s")
    model2, state2 = la._PocketEngine.get()
    check("engine singleton identity", model1 is model2)
    check("voice state reused across calls", state1 is state2)

    eng = la.PocketTTS()
    firsts, counts, rms_all = [], [], []
    for rep in range(3):
        t0 = time.monotonic()
        first = None
        parts = []
        n = 0
        for samples, sr in eng._stream_chunks(GREETING):
            if first is None:
                first = time.monotonic() - t0
            parts.append(samples)
            n += 1
        total = time.monotonic() - t0
        audio = np.concatenate(parts)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        firsts.append(first)
        counts.append(n)
        rms_all.append(rms)
        print(f"rep {rep}: chunks={n} first={first:.3f}s "
              f"total={total:.2f}s audio={len(audio) / sr:.2f}s "
              f"rms={rms:.4f}")
    med_first = statistics.median(firsts)
    check("many chunks per stream", min(counts) > 10,
          f"counts={counts}")
    check("first chunk < 1s warm (median)", med_first < 1.0,
          f"median={med_first:.3f}s firsts={firsts}")
    check("output non-silent", min(rms_all) > 0.01,
          f"rms={rms_all}")

    samples, sr = eng._synth(GREETING)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    check("batch _synth non-silent", rms > 0.01, f"rms={rms:.4f} sr={sr}")

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("POCKET WARM RUN PASSED")


main()
