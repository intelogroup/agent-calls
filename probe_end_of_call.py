#!/usr/bin/env python3
"""PROBE: end-of-call dry run (Jim, 2026-11-02).

Plays scripted fake conversations through the Deepgram bridge's
session loop and prints, per scenario, whether the call would
actually end. Fakes only — no network, no LiveKit server, no SIP,
no Telnyx, no spend. The human never hangs up in the farewell and
silence rows: ending the call is the agent's job there.

    ~/workspace/bridge-outbound-build/.venv-test/bin/python \
        probe_end_of_call.py

Point EOC_MODULE_DIR at a scratch fixed copy to watch the table
flip to PASS.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from end_of_call_harness import (  # noqa: E402
    FAREWELL_SCENARIOS, farewell_script, function_responses,
    run_fake_bridge_session, settings_applied, silence_script,
    user_says)


async def main():
    rows = []
    for label, line in FAREWELL_SCENARIOS:
        r = await run_fake_bridge_session(
            farewell_script(line), patience=4.0)
        resps = [x for x in function_responses(r["sent"])
                 if x.get("name") == "end_call"]
        refused = any("not available" in (x.get("content") or "")
                      for x in resps)
        ended = r["outcome"] == "ended" and not refused
        rows.append((label, ended,
                     "end_call refused by bridge" if refused
                     else "session never ended (caller still "
                     "connected, cap is the only exit)"
                     if r["outcome"] != "ended" else "session ended"))

    r = await run_fake_bridge_session(silence_script(), patience=4.0)
    rows.append(("prolonged silence after a completed exchange",
                 r["outcome"] == "ended",
                 "session ended" if r["outcome"] == "ended"
                 else "no inactivity rule — open until the "
                 "max-duration cap"))

    r = await run_fake_bridge_session(
        [(0.05, settings_applied()), (0.10, user_says("Hello?"))],
        patience=4.0, caller_gone_after=1.0)
    rows.append(("CONTROL: human actually hangs up",
                 r["outcome"] == "ended",
                 "session ended (only working end path today)"
                 if r["outcome"] == "ended" else "session open"))

    print(f"\n{'SCENARIO':<52} RESULT  DETAIL")
    print("-" * 100)
    fails = 0
    for label, ok, detail in rows:
        print(f"{label:<52} {'PASS' if ok else 'FAIL':<7} {detail}")
        fails += 0 if ok else 1
    print("-" * 100)
    print(f"{len(rows) - fails}/{len(rows)} scenarios end the call")
    sys.exit(0 if fails == 0 else 1)


asyncio.run(main())
