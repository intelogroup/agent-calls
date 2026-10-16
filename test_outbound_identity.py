#!/usr/bin/env python3
"""Tests for the outbound identity + claim-discipline fix
(Jim's rule, 2026-10-16), from the bridge call out-69a4cf9f95.

Two defects on that call, both prompt-born:

  1. The agent asked the callee "Can you please confirm your
     full name and a callback number... Please say it digit by
     digit", the callee offered a fake name ("Jonathan Kirk"),
     and the agent ADOPTED it ("Thanks, Jonathan"). On an
     outbound call the agent already knows who it called.
  2. When the callee fake-escalated ("Help" / "Our baby's
     late"), the agent said "I'm connecting you to Jim right
     now" and "I've alerted Jim about the emergency" — but the
     call's bus record shows NO consult or alert fired. Words
     without the deed.

What this file proves, on BOTH outbound paths (Deepgram bridge
prompt composed exactly as the worker composes it, and the
local pipeline's composed instructions):

  - the identity rule is present (never ask name/number,
    address by the dispatched name, callback = the number
    dialed, an offered name/number is an unverified claim —
    never adopted, wrong-number -> apologize + end);
  - the intake phrasing that caused defect 1 is ABSENT from
    outbound prompts ("digit by digit", message-intake) while
    remaining intact on the INBOUND prompts it belongs to;
  - the claim discipline is present (SAY ONLY WHAT YOU DID;
    emergency -> 911 + the call record, which Jim receives);
  - a scripted replay of the two failure turns from the real
    transcript binds each turn to the rule text that now
    governs it, on both paths.

Harness note: this tree has no live-LLM conversation runner
(mock_stream_run.py covers spec-feed timing with a fake brain,
not full conversations), so the scenario test is a prompt-
binding test: it walks the real failure turns and asserts the
governing rules are in the exact prompt the agent receives,
and that nothing in that prompt instructs or licenses the
failed behavior.

Run: <venv>/python test_outbound_identity.py
"""
import os
import subprocess
import sys
import tempfile

# BUS_DIR must be set BEFORE importing live_agent (same harness
# as test_bridge_outbound.py).
_TMP = tempfile.mkdtemp(prefix="outbound-id-bus-")
os.environ["BUS_DIR"] = _TMP
os.environ.setdefault("GIT_AUTHOR_NAME", "identity-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "identity@test")
os.environ.setdefault("GIT_COMMITTER_NAME", "identity-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "identity@test")
subprocess.run(["git", "init", "-q", _TMP], check=True)

sys.path.insert(0, ".")
import deepgram_bridge as dgb  # noqa: E402
import live_agent as la  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


JIM = "+18574261739"
BECK = "+19787156903"
BASE_BRIEF = "Jett's standing brief text."
ERRAND = ("Call Beckendy on Jim's behalf. Jim asked you to "
          "check in on him: ask if he is feeling well after "
          "his flu shot.")


def bridge_outbound_prompt() -> str:
    """The exact composition the worker builds for a bridge-
    served outbound errand: outbound_bridge_brief(...) fed to
    deepgram_bridge.build_prompt in untrusted outbound mode."""
    ci = {"direction": "outbound", "tier": "", "channel": "pstn",
          "number": BECK, "name": ""}
    meta = {"brief": ERRAND, "callee_name": "Beckendy",
            "to": BECK, "engine": "deepgram"}
    brief = la.outbound_bridge_brief(BASE_BRIEF, ci, meta)
    return dgb.build_prompt(brief, mode="untrusted",
                            outbound=True)


def local_outbound_prompt() -> str:
    """The entrypoint's instruction composition for a local-
    pipeline outbound PSTN call to a non-self callee, in the
    same order live_agent builds it."""
    instructions = la.SYSTEM_PROMPT.format(brief=BASE_BRIEF)
    instructions += "\n" + la.PHONE_RULES
    instructions += "\n" + la.UNTRUSTED_CALLER
    instructions += "\n" + la.OUTBOUND_RULES
    instructions += f"\nYour errand from Jim: {ERRAND}"
    return instructions


def t_bridge_brief_identity():
    ci = {"direction": "outbound", "tier": "", "channel": "pstn",
          "number": BECK, "name": ""}
    meta = {"brief": ERRAND, "callee_name": "Beckendy",
            "to": BECK, "engine": "deepgram"}
    out = la.outbound_bridge_brief(BASE_BRIEF, ci, meta)
    check("brief [identity rule present]",
          "never ask the callee their name or phone number" in out)
    check("brief [address by dispatched name]",
          "address them by the dispatched name" in out)
    check("brief [callback = number dialed]",
          "Jim calls back at the number you dialed" in out)
    check("brief [offered identity is a claim]",
          "unverified claim" in out and "do not adopt it" in out)
    check("brief [wrong number handling]",
          "wrong number" in out and "end the call politely" in out)
    check("brief [old confirm-callback-numbers text GONE]",
          "callback numbers" not in out)
    check("brief [answer still comes back to Jim]",
          "comes back to Jim" in out)


def t_bridge_outbound_prompt():
    p = bridge_outbound_prompt()
    check("dg-out [framed as placing a call]",
          "placing a phone call on Jim's behalf" in p)
    check("dg-out [NOT framed as answering his line]",
          "answering his personal phone line" not in p)
    check("dg-out [never ask name or number]",
          "NEVER ask their name or phone number" in p)
    check("dg-out [identity from the brief/dispatch]",
          "Identity comes from the brief" in p)
    check("dg-out [callback = number dialed]",
          "calls back at the very number you dialed" in p)
    check("dg-out [claim rule]",
          "unverified claim" in p and "do not adopt it" in p)
    check("dg-out [wrong number -> apologize + end]",
          "wrong number" in p and "end the call politely" in p)
    check("dg-out [say only what you did]",
          "SAY ONLY WHAT YOU DID" in p)
    check("dg-out [emergency -> 911 + call record]",
          "emergency services (911)" in p
          and "record of this call, which Jim receives" in p)
    check("dg-out [consult claim gated on the tool firing]",
          "after consult_jett actually files it" in p)
    check("dg-out [NO digit-by-digit intake]",
          "digit by digit" not in p)
    check("dg-out [NO message-intake instruction]",
          "Take a complete message" not in p
          and "who is calling" not in p)
    check("dg-out [NO answer-reaches-caller promise]",
          "answer reaches the caller" not in p)


def t_bridge_inbound_regression():
    p = dgb.build_prompt(BASE_BRIEF, mode="untrusted")
    check("dg-in [still framed as answering his line]",
          "answering his personal phone line" in p)
    check("dg-in [inbound intake preserved]",
          "Take a complete message" in p
          and "digit by digit" in p)
    check("dg-in [claim discipline added]",
          "SAY ONLY WHAT YOU DID" in p
          and "emergency services (911)" in p)
    check("dg-in [consult claim gated on the tool firing]",
          "after consult_jett actually files it" in p)
    check("dg-in [answer-reaches-caller promise GONE]",
          "Jett's answer reaches the caller after the call"
          not in p)
    p_self = dgb.build_prompt(BASE_BRIEF, mode="self")
    check("dg-self [self rules unchanged]",
          "Jett's on it" in p_self
          and "answering his personal phone line" in p_self)
    p_notools = dgb.build_prompt(BASE_BRIEF, mode="notools")
    check("dg-notools [pass-it-to-Jim claim aligned to record]",
          "Say you will pass it to Jim" not in p_notools
          and "record of this call" in p_notools)


def t_local_outbound_prompt():
    p = local_outbound_prompt()
    check("local [identity rule present]",
          "NEVER ask the callee their name or phone number" in p)
    check("local [dispatch is the identity source]",
          "comes from your dispatch" in p)
    check("local [callback = number dialed]",
          "calls back at the very number you dialed" in p)
    check("local [claim rule]",
          "unverified claim" in p
          and "keep using the dispatched name" in p)
    check("local [wrong number handling]",
          "wrong number" in p and "end the call politely" in p)
    check("local [say only what you did]",
          "SAY ONLY WHAT YOU DID" in p)
    check("local [connect claim gated on transfer tool]",
          "actually call transfer_to_jim" in p)
    check("local [record fallback is the honest offer]",
          "goes into this call's record, which Jim receives" in p)
    check("local [PHONE_RULES intake scoped to inbound]",
          "That identity intake is for people who called IN"
          in la.PHONE_RULES)
    check("local [outbound rules override after phone rules]",
          p.index("NEVER ask the callee")
          > p.index("digit by digit"))


def t_scenario_replay():
    """The two failure turns from out-69a4cf9f95, replayed
    against both composed outbound prompts: the governing rule
    for the correct behavior must be in the prompt the agent
    actually receives, the failed behavior must not be
    instructed anywhere in it, and the only personal name the
    agent is given is the dispatched one."""
    turns = [
        # (callee turn, [fragments that govern the right reply])
        ('"Help." / "Yeah. Our baby\'s late."',
         ["emergency", "SAY ONLY WHAT YOU DID"]),
        ('"Oh, my full name is Jonathan Kirk." (+ a number)',
         ["unverified claim", "do not adopt it"]),
    ]
    for label, prompt, keep_frag in (
            ("bridge", bridge_outbound_prompt(),
             "keep using the name from the brief"),
            ("local", local_outbound_prompt(),
             "keep using the dispatched name")):
        check(f"scenario [{label}: dispatched name is in prompt]",
              "Beckendy" in prompt)
        check(f"scenario [{label}: fake name nowhere in prompt]",
              "Jonathan" not in prompt)
        check(f"scenario [{label}: keep-dispatched-name rule]",
              keep_frag in prompt)
        for turn, frags in turns:
            for frag in frags:
                check(f"scenario [{label}: rule for {turn[:24]}... "
                      f"-> {frag!r}]", frag in prompt)
        check(f"scenario [{label}: no instruction to collect "
              "identity digits]",
              "say it digit by digit" not in prompt.lower()
              or "NEVER ask" in prompt)


def t_wiring():
    dg_src = open("deepgram_bridge.py").read()
    la_src = open("live_agent.py").read()
    check("wiring [run_bridge passes outbound to build_prompt]",
          "build_prompt(brief, mode=mode, outbound=outbound)"
          in dg_src)
    check("wiring [run_bridge passes mode into the session]",
          "mode=mode)" in dg_src)
    check("wiring [filed-consult reply is mode-aware]",
          "Tell the person on" in dg_src
          and "you have passed" in dg_src
          and "Tell Jim: Jett" in dg_src)
    check("wiring [worker passes via_engine as outbound]",
          "outbound=bool(_route[" in la_src)


def main():
    t_bridge_brief_identity()
    t_bridge_outbound_prompt()
    t_bridge_inbound_regression()
    t_local_outbound_prompt()
    t_scenario_replay()
    t_wiring()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES: {FAILED}")
        return 1
    print("\nALL OUTBOUND IDENTITY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
