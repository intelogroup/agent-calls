#!/usr/bin/env python3
"""Tests for ring-me-first (RING_FIRST, 2026-10-15) in live_agent.py.

Covers the pure gates (quiet hours, decision matrix), the SIP ring
request builder, the ringback tone shape, and the ring race state
machine against fake SIP legs:
  - Jim answers inside the window -> "taken", no cleanup
  - nobody answers -> "no_answer" AT the window, leg cleaned up
  - Jim declines/busy -> "declined" IMMEDIATELY, not at the window
  - caller hangs up mid-ring -> "caller_gone", leg cleaned up
Run: ~/venv187/bin/python test_ring_first.py
"""
import asyncio
import json
import sys
import time
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, ".")
import live_agent as la  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ------------------------------------------------------------- gates
def t_quiet_hours():
    tz = la.JIM_TZ
    def at(h, m=0):
        return datetime(2026, 10, 15, h, m, tzinfo=tz)
    check("quiet 22:00", la.in_quiet_hours(at(22, 0)))
    check("quiet 23:30", la.in_quiet_hours(at(23, 30)))
    check("quiet 00:30", la.in_quiet_hours(at(0, 30)))
    check("quiet 07:59", la.in_quiet_hours(at(7, 59)))
    check("not quiet 08:00", not la.in_quiet_hours(at(8, 0)))
    check("not quiet 21:59", not la.in_quiet_hours(at(21, 59)))
    check("not quiet 12:00", not la.in_quiet_hours(at(12, 0)))


def t_decision():
    pstn = {"channel": "pstn", "direction": "inbound", "tier": ""}
    d = la.ring_first_decision
    check("decision off", d(pstn, enabled=False, quiet=False,
                            trunk_id="ST_x") == "off")
    check("decision ring", d(pstn, enabled=True, quiet=False,
                             trunk_id="ST_x") == "ring")
    check("decision self skips", d(
        {**pstn, "tier": "self"}, enabled=True, quiet=False,
        trunk_id="ST_x") == "skip")
    check("decision family rings", d(
        {**pstn, "tier": "family"}, enabled=True, quiet=False,
        trunk_id="ST_x") == "ring")
    check("decision sip skips", d(
        {**pstn, "channel": "sip"}, enabled=True, quiet=False,
        trunk_id="ST_x") == "skip")
    check("decision outbound skips", d(
        {**pstn, "direction": "outbound"}, enabled=True, quiet=False,
        trunk_id="ST_x") == "skip")
    check("decision quiet hours", d(pstn, enabled=True, quiet=True,
                                    trunk_id="ST_x") == "quiet_hours")
    check("decision self beats quiet", d(
        {**pstn, "tier": "self"}, enabled=True, quiet=True,
        trunk_id="ST_x") == "skip")
    check("decision no trunk", d(pstn, enabled=True, quiet=False,
                                 trunk_id="") == "no_trunk")


def t_request_builder():
    req = la.build_ring_request(
        "jett-abc", "jim-ring-abc", trunk_id="ST_test",
        call_to="+18574261739", from_number="+16176556805")
    check("req trunk", req.sip_trunk_id == "ST_test")
    check("req call_to is Jim's cell", req.sip_call_to == "+18574261739")
    check("req room", req.room_name == "jett-abc")
    check("req identity", req.participant_identity == "jim-ring-abc")
    check("req waits for answer", req.wait_until_answered is True)
    check("req caller id", req.sip_number == "+16176556805")
    bare = la.build_ring_request(
        "jett-abc", "jim-ring-abc", trunk_id="ST_test",
        call_to="+18574261739")
    check("req caller id omittable", bare.sip_number == "")


def t_defaults():
    check("RING_FIRST defaults off in test env", la.RING_FIRST is False)
    check("window defaults to 10s", la.RING_FIRST_WINDOW_S == 10.0)


def t_ringback_pcm():
    tone = la._RingbackTone(room=None)
    pcm = tone._cycle_pcm()
    check("ringback cycle is 6s", len(pcm) == 24000 * 6 * 2,
          f"len={len(pcm)}")
    import numpy as np
    arr = np.frombuffer(pcm, dtype=np.int16)
    check("ringback first 2s audible", int(np.abs(arr[:24000]).max()) > 1000)
    check("ringback last 4s silent", int(np.abs(arr[48000:]).max()) == 0)


# -------------------------------------------------------------- race
class FakeLegs:
    """Fake SIP create/cleanup pair recording what the race did."""

    def __init__(self, behavior, delay=0.15):
        self.behavior = behavior
        self.delay = delay
        self.cleanup_calls = 0
        self.create_cancelled = False

    async def create(self):
        try:
            if self.behavior == "answer":
                await asyncio.sleep(self.delay)
                return SimpleNamespace(sip_call_id="sip-1")
            if self.behavior == "decline":
                await asyncio.sleep(self.delay)
                raise RuntimeError("sip error 486: busy here")
            await asyncio.sleep(60)  # rings forever
            return SimpleNamespace(sip_call_id="sip-late")
        except asyncio.CancelledError:
            self.create_cancelled = True
            raise

    async def cleanup(self, create_task):
        self.cleanup_calls += 1
        if not create_task.done():
            create_task.cancel()
            try:
                await create_task
            except (asyncio.CancelledError, Exception):
                pass


async def t_race_taken():
    legs = FakeLegs("answer", delay=0.15)
    gone = asyncio.Event()
    outcome, elapsed, _ = await la._ring_race(
        create=legs.create, cleanup=legs.cleanup, window_s=5.0,
        caller_gone=gone)
    check("race taken", outcome == "taken", outcome)
    check("race taken fast", elapsed < 1.5, f"{elapsed}s")
    check("race taken leaves the leg alone", legs.cleanup_calls == 0)


async def t_race_no_answer():
    legs = FakeLegs("hang")
    gone = asyncio.Event()
    outcome, elapsed, _ = await la._ring_race(
        create=legs.create, cleanup=legs.cleanup, window_s=0.4,
        caller_gone=gone)
    check("race no_answer", outcome == "no_answer", outcome)
    check("race no_answer at the window", 0.35 <= elapsed <= 1.5,
          f"{elapsed}s")
    check("race no_answer cleans up", legs.cleanup_calls == 1)
    check("race no_answer stops the ring", legs.create_cancelled)


async def t_race_declined():
    legs = FakeLegs("decline", delay=0.1)
    gone = asyncio.Event()
    outcome, elapsed, detail = await la._ring_race(
        create=legs.create, cleanup=legs.cleanup, window_s=5.0,
        caller_gone=gone)
    check("race declined", outcome == "declined", outcome)
    check("race declined immediately, not at window", elapsed < 1.5,
          f"{elapsed}s")
    check("race declined detail carries the SIP error", "486" in detail,
          detail)
    check("race declined cleans up", legs.cleanup_calls == 1)


async def t_race_caller_gone():
    legs = FakeLegs("hang")
    gone = asyncio.Event()
    async def hangup():
        await asyncio.sleep(0.15)
        gone.set()
    asyncio.ensure_future(hangup())
    outcome, elapsed, _ = await la._ring_race(
        create=legs.create, cleanup=legs.cleanup, window_s=5.0,
        caller_gone=gone)
    check("race caller_gone", outcome == "caller_gone", outcome)
    check("race caller_gone fast", elapsed < 1.5, f"{elapsed}s")
    check("race caller_gone stops Jim's ring", legs.cleanup_calls == 1
          and legs.create_cancelled)


async def t_receipt_shape():
    sent = {}
    async def fake_append(relpath, lines, msg):
        sent["relpath"] = relpath
        sent["lines"] = lines
        return True
    orig = la._bus_append_lines
    la._bus_append_lines = fake_append
    try:
        await la._ring_receipt("call-42", "ring_first_taken",
                               elapsed_s=3.2)
    finally:
        la._bus_append_lines = orig
    check("receipt path", sent.get("relpath")
          == "calls/live/call-42/relay.jsonl", str(sent.get("relpath")))
    line = json.loads(sent["lines"][0])
    check("receipt fields", line["event"] == "ring_first_taken"
          and line["call_id"] == "call-42"
          and line["via"] == "ring_first"
          and line["elapsed_s"] == 3.2 and "ts" in line, str(line))


async def main():
    t0 = time.monotonic()
    t_quiet_hours()
    t_decision()
    t_request_builder()
    t_defaults()
    t_ringback_pcm()
    await t_race_taken()
    await t_race_no_answer()
    await t_race_declined()
    await t_race_caller_gone()
    await t_receipt_shape()
    print(f"\n({time.monotonic() - t0:.1f}s)")
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("ALL RING-FIRST TESTS PASSED")


asyncio.run(main())
