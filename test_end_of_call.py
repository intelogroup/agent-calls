#!/usr/bin/env python3
"""End-of-call tests (Jim, 2026-11-02): the agent cannot hang up.

Symptom: the callee says "bye" (or the conversation is obviously
over) and the call stays open. These tests pin the end-of-call
contract for BOTH routes, bridge first (Jim's 2026-10-28 ruling:
all routes use the Deepgram bridge; the local pipeline is fallback
and tools-only):

  BRIDGE — the Deepgram-hosted agent can only end a call through a
  declared `end_call` function (Deepgram Voice Agent API: functions
  live in Settings agent.think.functions). Contract:
    B1/B2  end_call is declared, with and without consult_jett.
    B3     the prompt names the end_call mechanism in every mode.
    B4     a scripted farewell + end_call attempt ENDS the session
           (no refusal, no waiting for the human to hang up).
    B5     prolonged silence after a completed exchange ends the
           session by rule, not by the 10-minute cap.
    T1     when the bridge session does end, live_agent tears the
           call down (room disconnect / participant removal).
  CONTROLS (must pass today — they prove the harness works and
  mark the paths that DO end calls):
    C1     the human hanging up ends the bridge session.
    C2     consult_jett still files mid-call.
    L3     leave_voicemail schedules a hangup (local pipeline).
    L4     the local MAX_CALL_SEC watchdog exists as backstop.
  LOCAL PIPELINE (secondary coverage):
    L1     an end-call tool exists for the agent's brain.
    L2     the outbound brief's "end the call" instruction is
           backed by a real mechanism.

Everything is faked: no network, no LiveKit server, no SIP, no
Telnyx, no spend. Run:
    ~/workspace/bridge-outbound-build/.venv-test/bin/python \
        test_end_of_call.py
Module under test can be pointed at a scratch copy with
EOC_MODULE_DIR (see end_of_call_harness.py).
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import types

# BUS_DIR must be set BEFORE importing live_agent (same harness as
# test_bridge_outbound.py).
_TMP = tempfile.mkdtemp(prefix="eoc-bus-")
os.environ["BUS_DIR"] = _TMP
os.environ.setdefault("GIT_AUTHOR_NAME", "eoc-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "eoc@test")
os.environ.setdefault("GIT_COMMITTER_NAME", "eoc-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "eoc@test")
# The silence scenario waits with patience=4.0s, so the bridge's
# silence watchdog window must be test-scale: the production
# default is 45s (deepgram_bridge reads EOC_SILENCE_WINDOW_S at
# session start). Without this the scenario can never fire in
# time and fails deterministically in a clean environment.
os.environ.setdefault("EOC_SILENCE_WINDOW_S", "2")
subprocess.run(["git", "init", "-q", _TMP], check=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("EOC_MODULE_DIR"):
    # last insert wins: the module under test beats this dir
    sys.path.insert(0, os.environ["EOC_MODULE_DIR"])

import deepgram_bridge as dgb  # noqa: E402
import live_agent as la  # noqa: E402
from end_of_call_harness import (  # noqa: E402
    FAREWELL_SCENARIOS, farewell_script, function_responses,
    run_fake_bridge_session, silence_script, agent_says,
    function_call, settings_applied, user_says)

PASS = []
FAILED = []


def check(name, cond, extra=""):
    PASS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ------------------------------------------------------------- bridge --

def t_settings_declare_end_call():
    for with_consult, label in ((True, "with consult"),
                                (False, "no consults")):
        s = dgb.build_settings("prompt", "", "nova-3", "gpt-4o-mini",
                               with_consult=with_consult)
        fns = (s.get("agent", {}).get("think", {})
               .get("functions") or [])
        names = [f.get("name") for f in fns]
        check(f"B settings declare end_call ({label})",
              "end_call" in names, f"functions={names}")


def t_prompt_names_mechanism():
    for mode in ("self", "untrusted", "notools"):
        for outbound in (False, True):
            p = dgb.build_prompt("Brief.", mode=mode,
                                 outbound=outbound)
            check(f"B prompt names end_call [{mode} "
                  f"outbound={outbound}]", "end_call" in p)


async def t_farewell_scenarios():
    for label, line in FAREWELL_SCENARIOS:
        r = await run_fake_bridge_session(
            farewell_script(line), patience=4.0)
        resps = [x for x in function_responses(r["sent"])
                 if x.get("name") == "end_call"]
        refused = any("not available" in (x.get("content") or "")
                      for x in resps)
        check(f"B farewell ends session: {label}",
              r["outcome"] == "ended" and not refused,
              f"outcome={r['outcome']} refused={refused}")


async def t_silence_ends_by_rule():
    r = await run_fake_bridge_session(silence_script(), patience=4.0)
    check("B silence after completed exchange ends session",
          r["outcome"] == "ended", f"outcome={r['outcome']}")


def t_post_bridge_teardown():
    src = open(os.path.join(
        os.environ.get("EOC_MODULE_DIR",
                       os.path.dirname(os.path.abspath(__file__))),
        "live_agent.py")).read()
    i = src.find("run_bridge(")
    window = src[i:i + 6000] if i >= 0 else ""
    verbs = [v for v in ("room.disconnect(", "delete_room",
                         "remove_participant") if v in window]
    check("T live_agent tears down call after bridge session",
          bool(verbs), f"teardown verbs after run_bridge: {verbs}")


# ----------------------------------------------------------- controls --

async def t_control_caller_hangup():
    r = await run_fake_bridge_session(
        [(0.05, settings_applied()),
         (0.10, user_says("Hello?"))],
        patience=4.0, caller_gone_after=1.0)
    check("C human hangup ends bridge session (control)",
          r["outcome"] == "ended", f"outcome={r['outcome']}")


async def t_control_consult_still_files():
    script = [
        (0.05, settings_applied()),
        (0.10, user_says("Can you ask Jett about my bill?")),
        (0.10, function_call(
            "consult_jett", fid="fn-c",
            arguments={"question": "What is my bill status?",
                       "priority": "normal"})),
    ]
    r = await run_fake_bridge_session(
        script, patience=4.0, caller_gone_after=2.0)
    resps = [x for x in function_responses(r["sent"])
             if x.get("name") == "consult_jett"]
    filed = any("Filed with Jett" in (x.get("content") or "")
                for x in resps)
    check("C consult_jett still files mid-call (control)",
          filed and "dg_consult_filed" in r["events"],
          f"responses={len(resps)}")


# -------------------------------------------------------------- local --

def _tool_fn(tool):
    return (getattr(tool, "_func", None)
            or getattr(tool, "fn", None))


def t_local_end_tool_exists():
    names = [n for n in ("end_call", "hang_up", "hangup_call",
                         "end_conversation")
             if hasattr(la, n)]
    check("L local pipeline exposes an end-call tool",
          bool(names), f"found={names}")


def t_local_brief_mechanism():
    brief = la.outbound_bridge_brief(
        "Base brief.", {"number": "+19787156903"},
        {"brief": "Say hello.", "callee_name": "Beck"})
    instructs = "end the call" in brief
    mechanism = any(hasattr(la, n) for n in
                    ("end_call", "hang_up", "hangup_call"))
    import deepgram_bridge as _d  # brief also feeds the bridge
    settings = _d.build_settings("p", "", "nova-3", "gpt-4o-mini",
                                 with_consult=True)
    fns = [f.get("name") for f in
           settings["agent"]["think"].get("functions") or []]
    check("L 'end the call' instruction is backed by a mechanism",
          instructs and (mechanism or "end_call" in fns),
          f"instructs={instructs} local_tool={mechanism} "
          f"bridge_functions={fns}")


async def t_local_voicemail_hangup_control():
    fn = _tool_fn(la.leave_voicemail)
    check("L voicemail tool fn reachable (control)", fn is not None)
    if fn is None:
        return
    la._ACTIVE_CALL_INFO = {"direction": "outbound",
                            "channel": "pstn"}
    hung = {}
    orig = la._delayed_hangup

    async def fake_hangup(delay):
        hung["delay"] = delay
    la._delayed_hangup = fake_hangup
    try:
        out = await fn("Jett here for Jim — call back please.",
                       types.SimpleNamespace(session=None))
    finally:
        la._delayed_hangup = orig
        la._ACTIVE_CALL_INFO = None
    await asyncio.sleep(0.1)
    check("L leave_voicemail schedules a hangup (control)",
          out.startswith("SAY_THIS_THEN_STOP")
          and 4.0 < hung.get("delay", 0) < 12.0, str(hung))


def t_local_watchdog_control():
    check("L MAX_CALL_SEC watchdog backstop exists (control)",
          isinstance(la.MAX_CALL_SEC, int) and la.MAX_CALL_SEC > 0,
          f"MAX_CALL_SEC={la.MAX_CALL_SEC}")


async def main():
    t_settings_declare_end_call()
    t_prompt_names_mechanism()
    await t_farewell_scenarios()
    await t_silence_ends_by_rule()
    t_post_bridge_teardown()
    await t_control_caller_hangup()
    await t_control_consult_still_files()
    t_local_end_tool_exists()
    t_local_brief_mechanism()
    await t_local_voicemail_hangup_control()
    t_local_watchdog_control()
    print(f"\n{sum(PASS)}/{len(PASS)} passed; "
          f"{len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + "; ".join(FAILED))
    sys.exit(0 if not FAILED else 1)


asyncio.run(main())
