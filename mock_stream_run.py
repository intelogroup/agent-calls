#!/usr/bin/env python3
"""MOCK STREAM RUN (2026-10-15): end-to-end timing proof of the
SPECULATIVE_FEED layer with fakes for the brain and the caller.

Simulates a caller speaking for ~1.2s (interims arrive progressively),
a brain that needs 0.8s to answer, and compares:
  A (control, today's pipeline): generation starts at the final
     transcript -> reply ready at final + 0.8s.
  B (speculative): interims feed the drafter while the caller talks ->
     reply ready at ~final + 0s.
Run: ~/venv187/bin/python mock_stream_run.py
"""
import asyncio
import sys
import time

sys.path.insert(0, ".")
import live_agent as la

la.SPEC_SETTLE_S = 0.05

BRAIN_LATENCY_S = 0.8
ANSWER = "You have a quiet morning, nothing until later."
INTERIMS = [
    (0.0, "what is on"),
    (0.4, "what is on my calendar"),
    (0.8, "what is on my calendar today"),
]
FINAL_AT = 1.2
FINAL = "what is on my calendar today"


async def fake_brain(basis):
    await asyncio.sleep(BRAIN_LATENCY_S)
    return ANSWER


async def control():
    t0 = time.monotonic()
    await asyncio.sleep(FINAL_AT)          # caller finishes speaking
    text = await fake_brain(FINAL)         # generation starts now
    return time.monotonic() - t0, text


async def speculative():
    t0 = time.monotonic()
    drafter = la.SpeculativeDrafter(fake_brain)
    for at, text in INTERIMS:
        delay = at - (time.monotonic() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        drafter.observe(text)
    delay = FINAL_AT - (time.monotonic() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    ready_at = time.monotonic() - t0
    text = await drafter.take(FINAL)
    return ready_at, time.monotonic() - t0, text


async def main():
    a_total, a_text = await control()
    b_ready, b_total, b_text = await speculative()
    print(f"A control:     reply ready {a_total:.2f}s after call start "
          f"({a_total - FINAL_AT:.2f}s after the caller stopped)")
    print(f"B speculative: draft ready by {b_ready:.2f}s; take() returned "
          f"at {b_total:.2f}s ({b_total - FINAL_AT:.2f}s after the caller stopped)")
    assert a_text == b_text == ANSWER
    saved = a_total - b_total
    print(f"perceived latency saved: {saved:.2f}s")
    assert saved > 0.5, "speculation did not save meaningful latency"
    print("MOCK STREAM RUN PASSED")


asyncio.run(main())
