#!/usr/bin/env python3
"""Tests for the record-gap close (2026-11-02, Fix C).

The finding, proven from the Mac worker's own log: while the
19:53Z outbound test call held the single-call line for ten
minutes (it ended only at the duration cap), Jim called his line
from Linphone four times — 19:55:42Z, 19:56:03Z, 19:56:12Z,
19:58:11Z. Every attempt reached the agent and was rejected in
the entrypoint busy path, and every one left NO bus record: on
the local worker that path wrote only a stderr log line. The
busy-announcement recording Jim's test call heard was the same
path answering the test call's own inbound leg (19:53:51Z) —
also record-less.

Hard invariants under test:
  - the busy path writes a call_rejected_busy bus stub (via
    _ring_receipt, via="entrypoint", naming the holding call)
    BEFORE the BRIDGE_ONLY split, so the local worker gets it
    too — the simulated stub lands in relay.jsonl with its
    fields intact;
  - the no-brain path writes a brain_missing stub;
  - a bus receipt that FAILS to reach the bus is loud: a
    bus_receipt_failed log event + a durable fallback copy in
    the spool dir (bus-fallback.jsonl) — never the old silent
    `except: pass`;
  - _receipt_fallback itself never raises, even with an
    unwritable spool dir (a receipt path must not crash a call).

Run: <venv>/python test_record_gap.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="gap-bus-")
_SPOOL = tempfile.mkdtemp(prefix="gap-spool-")
os.environ["BUS_DIR"] = _TMP
os.environ["TX_SPOOL_DIR"] = _SPOOL
os.environ.setdefault("GIT_AUTHOR_NAME", "gap-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "gap@test")
os.environ.setdefault("GIT_COMMITTER_NAME", "gap-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "gap@test")
subprocess.run(["git", "init", "-q", _TMP], check=True)

sys.path.insert(0, ".")
import live_agent as la  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


def t_wiring():
    src = open("live_agent.py", encoding="utf-8").read()
    busy = src[src.index("if _ACTIVE_CALL is not None:"):]
    busy = busy[:busy.index("engine = PocketTTS()")]
    check("wiring [busy stub written in entrypoint busy path]",
          '"call_rejected_busy", via="entrypoint"' in busy)
    check("wiring [busy stub names the holding call]",
          "busy_with=_ACTIVE_CALL" in busy)
    check("wiring [busy stub precedes the BRIDGE_ONLY split]",
          busy.index('"call_rejected_busy"')
          < busy.index("if BRIDGE_ONLY:"))
    check("wiring [no-brain stub written]",
          '"brain_missing",' in src
          and src.index('"brain_missing",')
          > src.index("async def _run_no_brain"))
    receipt = src[src.index("async def _ring_receipt"):]
    receipt = receipt[:receipt.index("async def _ring_first")]
    check("wiring [receipt failure is loud + fallback]",
          "bus_receipt_failed" in receipt
          and "_receipt_fallback(line)" in receipt)


async def t_stubs_land():
    # The exact call the entrypoint busy path now makes.
    await la._ring_receipt(
        "_intelogroup_GAPTEST1", "call_rejected_busy",
        via="entrypoint", busy_with="out-10be302c9a",
        room="jett-_intelogroup_GAPTEST1")
    path = os.path.join(_TMP, "calls", "live", "_intelogroup_GAPTEST1",
                        "relay.jsonl")
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
    stub = [ln for ln in lines if ln.get("event") == "call_rejected_busy"]
    check("stub [busy rejection lands in relay.jsonl]",
          len(stub) == 1, f"lines={len(lines)}")
    if stub:
        check("stub [fields intact: via + busy_with + room]",
              stub[0].get("via") == "entrypoint"
              and stub[0].get("busy_with") == "out-10be302c9a"
              and stub[0].get("room") == "jett-_intelogroup_GAPTEST1",
              json.dumps(stub[0])[:160])


async def t_failure_is_loud():
    events = []
    orig_log_event = la.log_event
    orig_append = la._bus_append_lines

    async def _fail(*a, **k):
        return False

    la.log_event = lambda name, **kw: events.append(name)
    la._bus_append_lines = _fail
    try:
        await la._ring_receipt("gap-fail-1", "call_rejected_busy",
                               via="entrypoint", busy_with="out-x")
    finally:
        la.log_event = orig_log_event
        la._bus_append_lines = orig_append
    fb = os.path.join(_SPOOL, "bus-fallback.jsonl")
    got = []
    if os.path.exists(fb):
        with open(fb, encoding="utf-8") as f:
            got = [json.loads(ln) for ln in f if ln.strip()]
    mine = [g for g in got if g.get("call_id") == "gap-fail-1"]
    check("loud [bus_receipt_failed event emitted]",
          "bus_receipt_failed" in events, str(events))
    check("loud [fallback copy holds the lost receipt]",
          len(mine) == 1
          and mine[0].get("event") == "call_rejected_busy"
          and mine[0].get("busy_with") == "out-x",
          json.dumps(mine)[:140])


def t_fallback_never_raises():
    saved = la.TX_SPOOL_DIR
    la.TX_SPOOL_DIR = "/proc/definitely-not-writable-xyz"
    try:
        la._receipt_fallback({"ts": "t", "event": "x", "call_id": "y"})
        ok = True
    except Exception:
        ok = False
    finally:
        la.TX_SPOOL_DIR = saved
    check("loud [fallback write never raises]", ok)


def main():
    t_wiring()
    asyncio.run(t_stubs_land())
    asyncio.run(t_failure_is_loud())
    t_fallback_never_raises()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES: {FAILED}")
        return 1
    print("\nALL RECORD GAP TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
