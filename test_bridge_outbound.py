#!/usr/bin/env python3
"""Tests for the Deepgram bridge OUTBOUND route (Jim, 2026-10-15).

The build: the dialer may select the bridge for an outbound
third-party errand via dispatch metadata engine="deepgram"; the
worker then serves the call on the bridge in guarded "untrusted"
mode with a generalized outbound brief. Hard invariants under
test:

  - the gate matrix (bridge_route): engine on/off x direction x
    tier — self keeps today's modes, inbound is untouched, an
    outbound errand reaches the bridge ONLY via the flag;
  - the flag never touches classification: classify_caller output
    is identical with/without engine metadata (the worker reads the
    flag from job metadata only, never from SIP attributes), and a
    diverted/spoofed call can still not reach self tier or the
    self bridge mode — even if job metadata claims engine deepgram;
  - the third-party brief renders the errand + callee, frames the
    callee as NOT Jim, and contains none of the self-mode
    "to Jim" text; the self append string survives byte-for-byte
    in the source;
  - the dialer: plan metadata carries engine (+callee_name), and
    a brief that plainly needs bridge-missing tools is
    AUTO-ROUTED to local when dispatched --engine deepgram
    (plan metadata engine=local, AUTO-ROUTED line logged), and
    runs as dispatched for --engine local.
  - the dialer dispatches by engine (2026-11-02 split):
    deepgram -> jett-bridge (the Oracle bridge-only worker),
    local -> jett-proxy (the Mac worker); an AGENT_NAME env
    override still wins for both engines; an AUTO-ROUTED plan
    targets jett-proxy, never jett-bridge.
  - the dialer's pre-flight busy check (2026-11-02): dialing
    our OWN number while the line already holds a call (a
    jett- room with a non-agent participant, per the room
    service) refuses before placing anything — LineBusyError,
    REFUSED line, exit 3; a free line, an agent-only room, a
    foreign room, or any other destination proceeds.
  - the outbound prompts (bridge build_prompt AND the local
    OUTBOUND_RULES) carry Jim's 2026-10-16 relay wording (agree
    to relay — the transcript reaches Jim; no can't-relay
    posture) and brief-adherence rule (deliver Jim's message
    verbatim, never add sentiments he did not state).

Run: <venv>/python test_bridge_outbound.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

# BUS_DIR must be set BEFORE importing live_agent (same harness
# as test_forwarding_observability.py).
_TMP = tempfile.mkdtemp(prefix="bridge-out-bus-")
os.environ["BUS_DIR"] = _TMP
os.environ.setdefault("GIT_AUTHOR_NAME", "bridge-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "bridge@test")
os.environ.setdefault("GIT_COMMITTER_NAME", "bridge-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "bridge@test")
subprocess.run(["git", "init", "-q", _TMP], check=True)

sys.path.insert(0, ".")
import live_agent as la  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dial", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "dial.py"))
dial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dial)

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


JIM = "+18574261739"
BECK = "+19787156903"
AGENT = "+16176556805"
BRIEF = "Jett's standing brief text."


def info(direction, tier, **kw):
    d = {"direction": direction, "tier": tier, "channel": "pstn",
         "number": BECK, "name": ""}
    d.update(kw)
    return d


def t_gate_matrix():
    dg = {"engine": "deepgram"}
    cases = [
        # (call_info, job_meta, serve, mode, via_engine, label)
        (info("inbound", "self"), None, True, "self", False,
         "inbound self"),
        (info("inbound", "self"), dg, True, "self", False,
         "inbound self + flag: flag inert"),
        (info("outbound", "self"), None, True, "self", False,
         "outbound to Jim, no flag"),
        (info("outbound", "self"), dg, True, "self", False,
         "outbound to Jim + flag: still self mode"),
        (info("inbound", ""), None, True, "untrusted", False,
         "inbound stranger"),
        (info("inbound", ""), dg, True, "untrusted", False,
         "inbound stranger + flag: unchanged (untrusted)"),
        (info("inbound", "family"), None, True, "untrusted", False,
         "inbound known non-self"),
        (info("outbound", ""), None, False, "untrusted", False,
         "outbound errand, no flag: local (today's behavior)"),
        (info("outbound", ""), {"engine": "local"}, False,
         "untrusted", False, "outbound errand, engine=local"),
        (info("outbound", ""), {"engine": "bogus"}, False,
         "untrusted", False, "outbound errand, garbage engine"),
        (info("outbound", ""), dg, True, "untrusted", True,
         "outbound errand + deepgram flag: BRIDGE (new route)"),
        (info("outbound", "family"), dg, True, "untrusted", True,
         "outbound known non-self + flag: bridge, untrusted"),
        (info("outbound", ""), {"engine": " DeepGram "}, True,
         "untrusted", True, "engine value normalized"),
        (info("inbound", "", diverted=True,
              forwarded_from_cell=True), dg,
         True, "untrusted", False,
         "diverted inbound + flag: still untrusted, not via_engine"),
        (info("outbound", "", diverted=True), dg,
         True, "untrusted", True,
         "engine flag can never undivert/spoof a tier"),
    ]
    for ci, meta, serve, mode, via, label in cases:
        r = la.bridge_route(ci, meta)
        check(f"gate [{label}]",
              r["serve"] == serve and r["mode"] == mode
              and r["via_engine"] == via, str(r))
    # No metadata at all / empty metadata: inbound default holds.
    r = la.bridge_route(info("inbound", ""), {})
    check("gate [empty metadata inbound]", r["serve"] and
          r["mode"] == "untrusted" and not r["via_engine"], str(r))
    # Mode "self" must be reachable ONLY via tier == "self".
    for tier in ("", "family", "known", "unknown"):
        for direction in ("inbound", "outbound"):
            r = la.bridge_route(info(direction, tier), dg)
            check(f"gate [no self mode for tier={tier!r} {direction}]",
                  r["mode"] != "self", str(r))


def t_classification_untouched():
    """The engine flag lives in job metadata; classification reads
    SIP attributes only. Feed classify_caller the diverted/spoofed
    shapes and prove the tier never becomes self."""
    brief = ("Some brief.\nKnown callers:\n"
             f"- Beckendy — {BECK} (family)\n")
    # Diverted: From rewritten to Jim, Diversion from Jim's cell,
    # PAI = the real (stranger) caller.
    attrs = {"sip.from_number": JIM, "sip.diversion": JIM,
             "sip.p_asserted_identity": "+16175550199"}
    ci = la.classify_caller(attrs, f"callee-{AGENT}", brief)
    check("classify [diverted spoof stays non-self]",
          ci.get("tier") != "self", str(ci.get("tier")))
    r = la.bridge_route({**ci, "direction": "inbound"},
                        {"engine": "deepgram"})
    check("classify [diverted spoof + flag: bridge mode untrusted]",
          r["mode"] == "untrusted" and not r["via_engine"], str(r))
    # Outbound classification of the callee number is unaffected by
    # any metadata (metadata is not an input to classify_caller).
    ci2 = la.classify_caller({"sip.from_number": BECK},
                             f"callee-{BECK}", brief)
    check("classify [callee attrs carry no tier upgrade]",
          ci2.get("tier") != "self", str(ci2.get("tier")))


def t_brief_rendering():
    meta = {"brief": "Ask if he is feeling well after his flu "
                     "shot.",
            "callee_name": "Beckendy", "to": BECK,
            "engine": "deepgram"}
    ci = info("outbound", "", number=BECK)
    out = la.outbound_bridge_brief(BRIEF, ci, meta)
    check("brief [keeps base brief]", out.startswith(BRIEF))
    check("brief [names the callee]", "Beckendy" in out)
    check("brief [carries the number]", BECK in out)
    check("brief [carries the errand]",
          "feeling well after his flu" in out)
    check("brief [callee is NOT Jim]", "NOT Jim" in out)
    check("brief [introduce as Jett, calling for Jim]",
          "Introduce yourself as Jett" in out
          and "calling" in out and "for Jim" in out)
    check("brief [bring the answer back]",
          "comes back to Jim" in out)
    check("brief [NO self-mode 'to Jim' text]",
          "you placed this call to Jim" not in out)
    # No name in metadata: falls back to the number, still safe.
    out2 = la.outbound_bridge_brief(
        BRIEF, ci, {"brief": "Say hello.", "to": BECK})
    check("brief [nameless callee uses number]",
          BECK in out2 and "NOT Jim" in out2
          and "you placed this call to Jim" not in out2)
    # The self-mode append string is byte-for-byte what it was.
    src = open("live_agent.py").read()
    check("brief [self append verbatim in source]",
          '"\\n\\nTHIS CALL: you placed this "\n'
          in src.replace("    ", "    ") or
          "THIS CALL: you placed this " in src
          and "call to Jim. Your errand from " in src)
    # Greeting: named and unnamed shapes, never the answering one.
    g1 = la.outbound_bridge_greeting({"callee_name": "Beckendy"})
    check("greeting [named]", "Beckendy" in g1
          and "Jett" in g1 and "on his behalf" in g1, g1)
    g2 = la.outbound_bridge_greeting({})
    check("greeting [unnamed]", "Jett" in g2
          and "can't come to the phone" not in g2, g2)


def t_dialer():
    plan = dial.build_plan(BECK, "Ask how he is.",
                           engine="deepgram", callee_name="Beckendy")
    md = plan["metadata"]
    check("dial [metadata carries engine]",
          md.get("engine") == "deepgram", json.dumps(md))
    check("dial [metadata carries callee_name]",
          md.get("callee_name") == "Beckendy")
    check("dial [metadata keeps direction/to/brief]",
          md.get("direction") == "outbound" and md.get("to") == BECK
          and md.get("brief") == "Ask how he is.")
    plan2 = dial.build_plan(BECK, "Ask how he is.")
    check("dial [default engine is local]",
          plan2["metadata"].get("engine") == "local"
          and "callee_name" not in plan2["metadata"])
    # Agent-name selection by engine (2026-11-02 split), with
    # the AGENT_NAME env override preserved. Scrub/set the env
    # explicitly so the checks are deterministic in any shell.
    _saved_name = os.environ.pop("AGENT_NAME", None)
    try:
        check("dial [agent name: deepgram -> jett-bridge]",
              dial.build_plan(BECK, "Ask how he is.",
                              engine="deepgram")["agent_name"]
              == "jett-bridge")
        check("dial [agent name: local -> jett-proxy]",
              dial.build_plan(BECK, "Ask how he is.",
                              engine="local")["agent_name"]
              == "jett-proxy")
        os.environ["AGENT_NAME"] = "custom-name"
        check("dial [agent name: env override wins]",
              dial.build_plan(BECK, "Ask how he is.",
                              engine="deepgram")["agent_name"]
              == "custom-name")
    finally:
        if _saved_name is None:
            os.environ.pop("AGENT_NAME", None)
        else:
            os.environ["AGENT_NAME"] = _saved_name
    # Tool-needs detection: plainly-tool briefs trip it, plain
    # conversation briefs (incl. the proof brief) do not.
    needs_cases = [
        ("If he doesn't answer, leave a voicemail.", True),
        ("Leave a message on his voicemail please.", True),
        ("Press 1 to confirm, then enter the extension 42.", True),
        ("Call the office and transfer me to Jim.", True),
        ("Ask if he is feeling well after his flu shot.", False),
        ("Check in and bring his answer back to Jim.", False),
        ("Take a message back to Jim if he mentions anything.",
         False),
    ]
    for brief, trips in needs_cases:
        got = dial.brief_tool_needs(brief)
        check(f"dial [tool-needs {brief[:38]!r}...]",
              bool(got) == trips, str(got))
    # End-to-end auto-route: --engine deepgram + tool brief runs
    # as local (route rule enforced, run not failed).
    rc = subprocess.run(
        [sys.executable, "dial.py", "--to", BECK,
         "--brief", "Leave a voicemail if he doesn't answer.",
         "--engine", "deepgram", "--dry-run"],
        capture_output=True, text=True)
    plan_auto = json.loads(rc.stdout) if rc.returncode == 0 else {}
    check("dial [deepgram + voicemail brief AUTO-ROUTED to local]",
          rc.returncode == 0 and "AUTO-ROUTED" in rc.stderr
          and plan_auto.get("metadata", {}).get("engine") == "local",
          f"rc={rc.returncode} err={rc.stderr[:90]}")
    check("dial [auto-routed plan targets jett-proxy]",
          plan_auto.get("agent_name") == "jett-proxy",
          str(plan_auto.get("agent_name")))
    rc2 = subprocess.run(
        [sys.executable, "dial.py", "--to", BECK,
         "--brief", "Leave a voicemail if he doesn't answer.",
         "--engine", "local", "--dry-run"],
        capture_output=True, text=True)
    check("dial [local + voicemail brief allowed]",
          rc2.returncode == 0
          and json.loads(rc2.stdout)["metadata"]["engine"]
          == "local")
    rc3 = subprocess.run(
        [sys.executable, "dial.py", "--to", BECK,
         "--brief", "Ask if he is feeling well after his flu shot.",
         "--engine", "deepgram", "--name", "Beckendy", "--dry-run"],
        capture_output=True, text=True)
    plan3 = json.loads(rc3.stdout) if rc3.returncode == 0 else {}
    check("dial [deepgram + clean brief plans fine]",
          rc3.returncode == 0
          and plan3.get("metadata", {}).get("engine") == "deepgram"
          and plan3.get("metadata", {}).get("callee_name")
          == "Beckendy", f"rc={rc3.returncode}")
    check("dial [deepgram plan targets jett-bridge]",
          plan3.get("agent_name") == "jett-bridge",
          str(plan3.get("agent_name")))


def t_prompt_relay_verbatim():
    """Jim's relay-wording + brief-adherence rules (2026-10-16)
    on BOTH outbound prompt paths: the bridge's build_prompt in
    untrusted+outbound mode and the local pipeline's
    OUTBOUND_RULES. Relay agreement is grounded in the real
    mechanism (transcript + summary are delivered to Jim), the
    can't-relay posture is absent, and Jim's message must be
    delivered in the brief's exact words with nothing added."""
    import deepgram_bridge as dgb
    p = dgb.build_prompt(BRIEF, mode="untrusted", outbound=True)
    check("prompt [relay agreement present]",
          "I will relay the message to him" in p)
    check("prompt [relay grounded in transcript delivery]",
          "transcript and summary are delivered to Jim" in p)
    check("prompt [no can't-relay posture]",
          "can't relay" not in p and "cannot relay" not in p)
    check("prompt [say-only-what-you-did still present]",
          "SAY ONLY WHAT YOU DID" in p)
    check("prompt [verbatim rule present]",
          "in exactly the words the brief states" in p
          and "Never add sentiments" in p)
    check("prompt [local OUTBOUND_RULES carry the pair]",
          "I will relay the message to him" in la.OUTBOUND_RULES
          and "transcript and summary are delivered to Jim"
          in la.OUTBOUND_RULES
          and "in exactly the words the brief states"
          in la.OUTBOUND_RULES
          and "can't relay" not in la.OUTBOUND_RULES
          and "cannot relay" not in la.OUTBOUND_RULES)


def t_dialer_busy():
    """Pre-flight busy check (2026-11-02): dialing our OWN number
    while the line holds a call refuses (exit 3, REFUSED line, no
    SIP participant); a free line or any other number proceeds.
    The LiveKit API is stubbed — the signal under test is the
    room-service occupancy shape, not a live server."""
    import asyncio
    import contextlib
    import io
    import types as _types
    from livekit import api as lk_api
    from livekit.protocol import models as lk_models

    AGENT_KIND = dial._agent_kind()
    SIP_KIND = int(lk_models.ParticipantInfo.Kind.Value("SIP"))
    OWN = "+16176556805"
    OTHER = "+19787156903"

    def part(identity, kind):
        return _types.SimpleNamespace(identity=identity, kind=kind)

    def room(name, parts):
        return (name, parts)

    class RoomSvc:
        def __init__(self, rooms):
            self._rooms = rooms
            self.calls = []

        async def list_rooms(self, req):
            return _types.SimpleNamespace(
                rooms=[_types.SimpleNamespace(name=n)
                       for n, _ in self._rooms])

        async def list_participants(self, req):
            parts = dict(self._rooms)[req.room]
            return _types.SimpleNamespace(participants=parts)

        async def create_room(self, req):
            self.calls.append("create_room")

    class DispatchSvc:
        def __init__(self, calls):
            self._calls = calls

        async def create_dispatch(self, req):
            self._calls.append("create_dispatch")

    class SipSvc:
        def __init__(self, calls):
            self._calls = calls

        async def create_sip_participant(self, req):
            self._calls.append("create_sip_participant")
            return _types.SimpleNamespace(sip_call_id="SCL_stub")

    def stub_client(rooms):
        room_svc = RoomSvc(rooms)
        return _types.SimpleNamespace(
            room=room_svc,
            agent_dispatch=DispatchSvc(room_svc.calls),
            sip=SipSvc(room_svc.calls),
            _room_svc=room_svc,
            aclose=lambda: asyncio.sleep(0))

    busy_rooms = [room("jett-out-abc123",
                       [part("agent-worker", AGENT_KIND),
                        part("sip_+16176556805", SIP_KIND)])]
    free_rooms = [room("jett-out-old999",
                       [part("agent-worker", AGENT_KIND)])]

    ev = asyncio.run(dial.line_busy_evidence(stub_client(busy_rooms)))
    check("dial busy [evidence names room + leg]",
          ev is not None and "jett-out-abc123" in ev
          and "sip_+16176556805" in ev, str(ev))
    check("dial busy [agent-only room is free]",
          asyncio.run(dial.line_busy_evidence(
              stub_client(free_rooms))) is None)
    check("dial busy [no rooms is free]",
          asyncio.run(dial.line_busy_evidence(
              stub_client([]))) is None)
    check("dial busy [foreign room ignored]",
          asyncio.run(dial.line_busy_evidence(stub_client(
              [room("other-room",
                    [part("sip_+16176556805", SIP_KIND)])]))) is None)

    saved_env = {k: os.environ.get(k) for k in
                 ("LIVEKIT_URL", "LIVEKIT_API_KEY",
                  "LIVEKIT_API_SECRET", "JETT_PHONE_E164")}
    saved_api = lk_api.LiveKitAPI
    try:
        os.environ.update({
            "LIVEKIT_URL": "ws://localhost:7880",
            "LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "s",
            "JETT_PHONE_E164": OWN})

        client = stub_client(busy_rooms)
        lk_api.LiveKitAPI = lambda **kw: client
        plan = dial.build_plan(OWN, "Line test.")
        try:
            asyncio.run(dial.place_call(plan))
            raised = ""
        except dial.LineBusyError as e:
            raised = e.evidence
        check("dial busy [own number + busy -> LineBusyError]",
              "jett-out-abc123" in raised, raised)
        check("dial busy [nothing was placed]",
              client._room_svc.calls == [],
              str(client._room_svc.calls))

        client2 = stub_client(free_rooms)
        lk_api.LiveKitAPI = lambda **kw: client2
        out = asyncio.run(dial.place_call(dial.build_plan(
            OWN, "Line test.")))
        check("dial busy [own number + free -> placed]",
              out == "SCL_stub"
              and client2._room_svc.calls == [
                  "create_room", "create_dispatch",
                  "create_sip_participant"],
              f"out={out} calls={client2._room_svc.calls}")

        client3 = stub_client(busy_rooms)
        lk_api.LiveKitAPI = lambda **kw: client3
        out3 = asyncio.run(dial.place_call(dial.build_plan(
            OTHER, "Errand.")))
        check("dial busy [other number ignores occupancy]",
              out3 == "SCL_stub"
              and "create_sip_participant" in client3._room_svc.calls,
              f"out={out3} calls={client3._room_svc.calls}")
    finally:
        lk_api.LiveKitAPI = saved_api
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # main() maps LineBusyError to the REFUSED line + exit 3.
    saved_argv = sys.argv
    saved_pc = dial.place_call

    async def _busy(plan):
        raise dial.LineBusyError("room jett-x holds participant y")

    try:
        dial.place_call = _busy
        sys.argv = ["dial.py", "--to", OWN, "--brief", "Line test."]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = dial.main()
        check("dial busy [main -> REFUSED + exit 3]",
              rc == dial.BUSY_REFUSAL_EXIT == 3
              and "REFUSED: line busy" in err.getvalue()
              and "jett-x" in err.getvalue(),
              f"rc={rc} err={err.getvalue()[:90]}")
    finally:
        dial.place_call = saved_pc
        sys.argv = saved_argv


def main():
    t_gate_matrix()
    t_classification_untouched()
    t_brief_rendering()
    t_dialer()
    t_dialer_busy()
    t_prompt_relay_verbatim()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES: {FAILED}")
        return 1
    print("\nALL BRIDGE OUTBOUND TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
