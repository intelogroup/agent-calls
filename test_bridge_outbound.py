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
    a brief that plainly needs bridge-missing tools is refused
    for --engine deepgram (exit 2) but fine for --engine local.

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
    # End-to-end refusal: --engine deepgram + tool brief exits 2.
    rc = subprocess.run(
        [sys.executable, "dial.py", "--to", BECK,
         "--brief", "Leave a voicemail if he doesn't answer.",
         "--engine", "deepgram", "--dry-run"],
        capture_output=True, text=True)
    check("dial [deepgram + voicemail brief REFUSED]",
          rc.returncode == 2 and "REFUSED" in rc.stderr,
          f"rc={rc.returncode} err={rc.stderr[:90]}")
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


def main():
    t_gate_matrix()
    t_classification_untouched()
    t_brief_rendering()
    t_dialer()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES: {FAILED}")
        return 1
    print("\nALL BRIDGE OUTBOUND TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
