#!/usr/bin/env python3
"""Tests for the forwarding observability pass (2026-10-15).

Jim's requirement: "verify that this capability is working dry
run and deployed first" + "have error loggers and handlers so we
debug." The failure mode being closed: key forwarding-path
decisions were stderr-only log_event calls, and Fly logs are not
readable from the sandbox — a failed forwarded call would leave
no debug record on the agent-call bus.

Coverage:
  - pure helpers: diversion_header_present,
    caller_recovered_from_pai, classify_safely (incl. the
    raising-classifier fallback: untrusted default, error
    returned, no exception escaping)
  - a SIMULATED forwarded call (no PSTN, no LiveKit room) that
    runs the same receipt sequence _run_call now runs and must
    PRODUCE the events on a local bus dir: call_classified (with
    pai_recovered), call_forwarded, ring_first_skipped
    (forwarded_immediate_answer), classification_error for the
    broken-classifier case.

Run: <venv>/python test_forwarding_observability.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile

# BUS_DIR must be set BEFORE importing live_agent: _git's default
# cwd binds the module-level BUS_DIR at definition time.
_TMP = tempfile.mkdtemp(prefix="fwd-obs-bus-")
os.environ["BUS_DIR"] = _TMP
os.environ.setdefault("GIT_AUTHOR_NAME", "obs-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "obs@test")
os.environ.setdefault("GIT_COMMITTER_NAME", "obs-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "obs@test")
subprocess.run(["git", "init", "-q", _TMP], check=True)

sys.path.insert(0, ".")
import live_agent as la  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


BRIEF = """Some brief text.
Known callers:
- Benoucheca Pierre — +1 774-555-0134 (family)
"""
JIM = "+18574261739"
AGENT = "+16176556805"
STRANGER = "+16175550199"
DIVERSION_JIM = ("<sip:+18574261739@sip.telnyx.com>;reason=no-answer;"
                 "privacy=off;counter=1")


# ------------------------------------------------------------ pure
def t_pure():
    check("header present: diversion",
          la.diversion_header_present({"sip.diversion": DIVERSION_JIM}))
    check("header present: history-info spelling",
          la.diversion_header_present({"sip.h.history-info": "<sip:x>"}))
    check("header absent: empty attrs",
          not la.diversion_header_present({}))
    check("header absent: blank value",
          not la.diversion_header_present({"sip.diversion": "  "}))

    attrs_a = {"sip.phoneNumber": STRANGER,
               "sip.trunkPhoneNumber": AGENT,
               "sip.diversion": DIVERSION_JIM}
    info_a, err_a = la.classify_safely(attrs_a, STRANGER, BRIEF)
    check("safely: case (a) classifies, no error",
          err_a == "" and info_a["number"] == STRANGER)
    check("pai_recovered: (a) From intact -> False",
          la.caller_recovered_from_pai(attrs_a, info_a) is False)

    attrs_b = {"sip.phoneNumber": JIM, "sip.trunkPhoneNumber": AGENT,
               "sip.diversion": DIVERSION_JIM,
               "sip.p_asserted_identity": f"<sip:{STRANGER}@x>"}
    info_b, _ = la.classify_safely(attrs_b, JIM, BRIEF)
    check("pai_recovered: (b) From rewritten -> True",
          la.caller_recovered_from_pai(attrs_b, info_b) is True,
          str(info_b))

    real = la.classify_caller

    def boom(attrs, identity, brief):
        raise RuntimeError("kaboom in classification")

    la.classify_caller = boom
    try:
        info_x, err_x = la.classify_safely(attrs_a, STRANGER, BRIEF,
                                           call_id="sim-x")
    finally:
        la.classify_caller = real
    check("safely: exception contained", "kaboom" in err_x, err_x)
    check("safely: fallback is untrusted default",
          info_x["tier"] == "" and info_x["forwarded_from_cell"] is False
          and info_x["channel"] == "sip", str(info_x))


# ------------------------------------------------- simulated call
async def simulate(attrs, identity, call_id):
    """The receipt sequence _run_call now runs for one call,
    replicated against a local bus dir. Returns the relay.jsonl
    lines the sequence produced."""
    info, err = la.classify_safely(attrs, identity, BRIEF,
                                   call_id=call_id)
    if err:
        await la._ring_receipt(call_id, "classification_error",
                               via="classification",
                               stage="classify_caller", error=err)
    await la._ring_receipt(
        call_id, "call_classified", via="classification",
        caller_number=info.get("number", ""),
        channel=info["channel"], direction=info["direction"],
        tier=info["tier"] or "unknown", known=bool(info["name"]),
        diverted=bool(info.get("diverted")),
        diversion_source=info.get("diversion_source", ""),
        diversion_reason=info.get("diversion_reason", ""),
        forwarded_from_cell=bool(info.get("forwarded_from_cell")),
        pai_recovered=la.caller_recovered_from_pai(attrs, info),
        diversion_header_present=la.diversion_header_present(attrs))
    if la.diversion_header_present(attrs) and not info.get("diverted"):
        await la._ring_receipt(call_id, "diversion_unparsed",
                               via="classification",
                               caller_number=info.get("number", ""),
                               tier=info["tier"] or "unknown")
    if info.get("diverted"):
        await la._ring_receipt(
            call_id, "call_forwarded", via="forwarding",
            forwarded_from_cell=bool(info.get("forwarded_from_cell")),
            diverted_from=info.get("diverted_from", ""),
            diversion_reason=info.get("diversion_reason", ""),
            diversion_source=info.get("diversion_source", ""),
            caller_number=info.get("number", ""),
            tier=info["tier"] or "unknown")
    dec = la.ring_first_decision(info, enabled=True, quiet=False,
                                 trunk_id="ST_test")
    if dec in ("off", "skip"):
        reason = ("forwarded_immediate_answer" if info.get("diverted")
                  else "self_tier" if info.get("tier") == "self"
                  else dec)
        await la._ring_receipt(call_id, "ring_first_skipped",
                               reason=reason, decision=dec)
    path = os.path.join(_TMP, "calls", "live", call_id,
                        "relay.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


async def t_sim():
    evs = await simulate(
        {"sip.phoneNumber": STRANGER, "sip.trunkPhoneNumber": AGENT,
         "sip.diversion": DIVERSION_JIM}, STRANGER, "sim-fwd-a")
    names = [e["event"] for e in evs]
    check("sim (a): bus events produced",
          names == ["call_classified", "call_forwarded",
                    "ring_first_skipped"], str(names))
    cc = evs[0]
    check("sim (a): classified by stranger on bus",
          cc["caller_number"] == STRANGER
          and cc["tier"] == "unknown"
          and cc["forwarded_from_cell"] is True
          and cc["diversion_source"] == "diversion"
          and cc["pai_recovered"] is False, json.dumps(cc))
    check("sim (a): ring skipped as forwarded",
          evs[2]["reason"] == "forwarded_immediate_answer")

    evs_b = await simulate(
        {"sip.phoneNumber": JIM, "sip.trunkPhoneNumber": AGENT,
         "sip.diversion": DIVERSION_JIM,
         "sip.p_asserted_identity": f"<sip:{STRANGER}@x>"},
        JIM, "sim-fwd-b")
    cc_b = [e for e in evs_b if e["event"] == "call_classified"][0]
    check("sim (b): PAI recovery on bus",
          cc_b["pai_recovered"] is True
          and cc_b["caller_number"] == STRANGER
          and cc_b["tier"] != "self", json.dumps(cc_b))

    evs_d = await simulate(
        {"sip.phoneNumber": STRANGER, "sip.trunkPhoneNumber": AGENT,
         "sip.diversion": ";;;garbage{{{<>;reason="},
        STRANGER, "sim-fwd-d")
    names_d = [e["event"] for e in evs_d]
    check("sim (d): unparsed header is LOUD on bus",
          "diversion_unparsed" in names_d, str(names_d))

    real = la.classify_caller
    la.classify_caller = lambda a, i, b: (_ for _ in ()).throw(
        RuntimeError("kaboom"))
    try:
        evs_x = await simulate(
            {"sip.phoneNumber": STRANGER}, STRANGER, "sim-fwd-x")
    finally:
        la.classify_caller = real
    names_x = [e["event"] for e in evs_x]
    check("sim (x): classification_error on bus, call survives",
          names_x[0] == "classification_error"
          and "call_classified" in names_x, str(names_x))


def main():
    t_pure()
    asyncio.run(t_sim())
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES: {FAILED}")
        return 1
    print("\nALL FORWARDING OBSERVABILITY TESTS PASSED")
    print(f"simulated bus dir: {_TMP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
