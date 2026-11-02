#!/usr/bin/env python3
"""Tests for the terminal machine-announcement guard (2026-11-02).

The defect (live, 19:53Z): an outbound test call to Jim's own
number was answered by the line's recorded busy announcement —
"Jett's line is on another call right now. Try again in a bit." —
and the agent treated it as a human turn, replying to the
recording 17 s later. The guard: on an OUTBOUND call, the far
end's FIRST turn is matched against terminal carrier/line
announcements; a match ends the call promptly with the outcome
on the record (far_end: machine_announcement + the text).

Hard invariants under test:
  - the matcher fires on terminal announcements (busy line with
    its try-again tail, not-in-service, disconnected / changed
    number, cannot-complete, all-circuits-busy), with labels;
  - it NEVER fires on human first turns, on IVR menus (menus are
    navigable — press_digits exists), or on voicemail greetings
    (the existing voicemail path + Jim's never-leave-a-message
    rule govern those);
  - a receptionist's "is on another call" WITHOUT the carrier
    try-again tail does not fire (she may finish the sentence
    with a message offer, which the voicemail exclusion also
    protects);
  - the local pipeline wiring: the check sits in the STT-final
    handler, gated on direction == outbound, one-shot, writes
    the machine_announcement relay receipt and ends the call;
  - the bridge behavior (fake transport): an outbound session
    whose first user turn is the busy announcement ENDS on its
    own with the receipt on relay.jsonl; a human first turn, a
    menu, a voicemail greeting, or any inbound session does not.

Run: <venv>/python test_announcement_guard.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="announce-bus-")
os.environ["BUS_DIR"] = _TMP
os.environ.setdefault("GIT_AUTHOR_NAME", "announce-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "announce@test")
os.environ.setdefault("GIT_COMMITTER_NAME", "announce-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "announce@test")
subprocess.run(["git", "init", "-q", _TMP], check=True)

sys.path.insert(0, ".")
from announcement_guard import machine_announcement_match as match  # noqa: E402
import deepgram_bridge as dgb  # noqa: E402
import end_of_call_harness as harness  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


# ------------------------------------------------------------ matcher
def t_matcher_fires():
    cases = [
        ("Jeff's line is on another call right now. Try again in a bit.",
         "busy_line"),
        ("The line is on another call. Please try your call again.",
         "busy_line"),
        ("We're sorry, the number you have dialed is not in service.",
         "not_in_service"),
        ("The number you are trying to reach is no longer in service.",
         "not_in_service"),
        ("The number you have called has been disconnected.",
         "disconnected"),
        ("The number you have dialed has been changed.",
         "disconnected"),
        ("Your call cannot be completed as dialed. Please check "
         "the number and try your call again.", "cannot_complete"),
        ("All circuits are busy now. Please try your call again later.",
         "circuits_busy"),
        ("We're sorry, your call did not go through.",
         "cannot_complete"),
    ]
    for text, label in cases:
        got = match(text)
        check(f"match fires [{label}] {text[:44]!r}",
              got == label, f"got={got}")


def t_matcher_never_fires():
    cases = [
        "Hello?",
        "Hi, this is Dana from Brightpath Dental, calling to confirm.",
        "Yeah, speaking. Who's this?",
        "Hey Jim? No wait, who am I talking to?",
        "Sorry, I was on another call earlier — what's up?",
        "He is on another call right now, try his cell instead.",
        "He is on another call — would you like to leave a message?",
        # IVR menus: navigable, not terminal.
        "Thank you for calling Brightpath Dental. For appointments, "
        "press 1. For billing, press 2.",
        "Please listen carefully, as our menu options have changed. "
        "Press 1 for sales.",
        # Voicemail greetings: the existing path governs.
        "Hi, you've reached Mike. I can't take your call right now, "
        "so please leave a message after the beep.",
        "Hello, this is Sarah's voicemail. Leave your message after "
        "the tone and I'll get back to you.",
        "The person you are trying to reach is unavailable. Please "
        "leave a message.",
        "",
        "   ",
    ]
    for text in cases:
        got = match(text)
        check(f"match silent {text[:44]!r}", got is None,
              f"got={got}")


# ------------------------------------------------- local wiring (source)
def t_local_wiring():
    src = open("live_agent.py", encoding="utf-8").read()
    check("local [matcher imported in _run_call]",
          "from announcement_guard import machine_announcement_match"
          in src)
    check("local [guard state armed per call]",
          'announce_guard = {"spent": False}' in src)
    region = src[src.index("def _on_user_transcribed"):]
    region = region[:region.index("def _on_transcription_timeout")]
    check("local [check sits in the STT-final handler]",
          "machine_announcement_match(text)" in region)
    check("local [gated on outbound]",
          'call_info["direction"] == "outbound"' in region)
    check("local [one-shot: spent set before evaluation]",
          'announce_guard["spent"] = True' in region)
    check("local [receipt carries far_end + text]",
          '"machine_announcement",' in region
          and 'far_end="machine_announcement"' in region
          and "text=text[:200]" in region)
    check("local [ends the call: ended event + delayed hangup]",
          'end_holder.get("event")' in region
          and "_delayed_hangup(1.0)" in region)
    check("local [ended event handed to the guard]",
          'end_holder["event"] = ended' in src)


# ------------------------------------------------ bridge behavior (fake)
async def _run_bridge_case(script, *, outbound):
    room = harness.FakeRoom()
    ws = harness.FakeWS(script)
    caller_gone = asyncio.Event()
    appended = []

    async def append_lines(rel, lines, label):
        appended.append((rel, list(lines)))
        return True

    events = []

    def log_event(name, **fields):
        events.append(name)

    orig = (dgb.rtc.AudioStream, dgb.rtc.AudioSource,
            dgb.rtc.LocalAudioTrack)
    dgb.rtc.AudioStream = harness._FakeAudioStream
    dgb.rtc.AudioSource = harness._FakeAudioSource
    dgb.rtc.LocalAudioTrack = harness._FakeLocalAudioTrack
    try:
        task = asyncio.ensure_future(dgb._run_session(
            ws, room, object(),
            call_id="announce-probe",
            prompt=dgb.build_prompt("Probe brief.", mode="untrusted",
                                    outbound=outbound),
            greeting="",
            listen_model="nova-3", think_model="gpt-4o-mini",
            append_lines=append_lines, log_event=log_event,
            caller_gone=caller_gone, stats={},
            file_consult=None, consult_enabled=False,
            mode="untrusted", outbound=outbound))
        try:
            await asyncio.wait_for(task, timeout=1.5)
            outcome = "ended"
        except asyncio.TimeoutError:
            task.cancel()
            outcome = "still_open"
    finally:
        dgb.rtc.AudioStream, dgb.rtc.AudioSource, \
            dgb.rtc.LocalAudioTrack = orig
    receipts = []
    for rel, lines in appended:
        if rel.endswith("relay.jsonl"):
            receipts.extend(json.loads(ln) for ln in lines)
    return outcome, receipts, events


def t_bridge_behavior():
    async def go():
        busy = [harness.settings_applied(),
                harness.user_says(
                    "Jett's line is on another call right now. "
                    "Try again in a bit.")]
        outcome, receipts, events = await _run_bridge_case(
            [(0, m) for m in busy], outbound=True)
        ann = [r for r in receipts
               if r.get("event") == "machine_announcement"]
        check("bridge [busy announcement ends outbound session]",
              outcome == "ended", outcome)
        check("bridge [receipt on relay.jsonl]",
              len(ann) == 1 and ann[0].get("far_end")
              == "machine_announcement"
              and ann[0].get("pattern") == "busy_line"
              and "on another call" in ann[0].get("text", ""),
              json.dumps(ann)[:160])
        check("bridge [dg_machine_announcement logged]",
              "dg_machine_announcement" in events, str(events))

        outcome2, receipts2, _ = await _run_bridge_case(
            [(0, m) for m in busy], outbound=False)
        check("bridge [same text inbound does NOT end session]",
              outcome2 == "still_open"
              and not [r for r in receipts2
                       if r.get("event") == "machine_announcement"],
              outcome2)

        human = [harness.settings_applied(),
                 harness.user_says("Hi, this is Dana, who's this?")]
        outcome3, receipts3, _ = await _run_bridge_case(
            [(0, m) for m in human], outbound=True)
        check("bridge [human first turn keeps session open]",
              outcome3 == "still_open"
              and not [r for r in receipts3
                       if r.get("event") == "machine_announcement"],
              outcome3)

        menu = [harness.settings_applied(),
                harness.user_says(
                    "For appointments press 1, for billing press 2.")]
        outcome4, _, _ = await _run_bridge_case(
            [(0, m) for m in menu], outbound=True)
        check("bridge [IVR menu keeps session open]",
              outcome4 == "still_open", outcome4)

        vm = [harness.settings_applied(),
              harness.user_says(
                  "You've reached Mike — leave a message after "
                  "the beep.")]
        outcome5, _, _ = await _run_bridge_case(
            [(0, m) for m in vm], outbound=True)
        check("bridge [voicemail greeting keeps existing path]",
              outcome5 == "still_open", outcome5)

        second = [harness.settings_applied(),
                  harness.user_says("Hello, Dana speaking."),
                  harness.user_says(
                      "The number you have dialed is not in service.")]
        outcome6, receipts6, _ = await _run_bridge_case(
            [(0, m) for m in second], outbound=True)
        check("bridge [announcement as SECOND turn does not fire]",
              outcome6 == "still_open"
              and not [r for r in receipts6
                       if r.get("event") == "machine_announcement"],
              outcome6)
    asyncio.run(go())


def main():
    t_matcher_fires()
    t_matcher_never_fires()
    t_local_wiring()
    t_bridge_behavior()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES: {FAILED}")
        return 1
    print("\nALL ANNOUNCEMENT GUARD TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
