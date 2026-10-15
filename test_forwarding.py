#!/usr/bin/env python3
"""Tests for forwarding-aware answering (Phase 2, 2026-10-15).

Jim will enable conditional (no-answer) forwarding on his cell
(+1 857-426-1739) to the agent line (+1 617-655-6805). Calls he
doesn't answer arrive as DIVERTED calls. Coverage:

  - Diversion / History-Info header parsing (RFC 5806 / RFC 7044
    shapes, as mapped to participant attributes by the trunk)
  - classification by the REAL caller; a diverted call is NEVER
    tier "self" (injection-style: spoofed Jim number + diversion
    must not reach orders mode)
  - real-caller recovery via P-Asserted-Identity when From was
    rewritten to the redirector
  - forwarded greeting wording; ring-first never fires on diverted
    calls
  - loop safety: Jim's cell forwarding our own ring-first leg back
    matches the active ring wait, and the race ends as "loop" the
    moment the abort fires — the waiting call agent-answers and
    the forwarded leg is ended by the busy path, not answered

Run: ~/venv187/bin/python test_forwarding.py
"""
import asyncio
import sys
import threading
import time
from types import SimpleNamespace

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
- Joe Driving School — 781-929-3434 (work)
"""
JIM = "+18574261739"
AGENT_NUM = "+16176556805"
FAMILY = "+17745550134"
STRANGER = "+16175550199"

DIVERSION_JIM = ("<sip:+18574261739@sip.telnyx.com>;reason=no-answer;"
                 "privacy=off;counter=1")
HISTORY_JIM = (
    '<sip:+18574261739@sip.telnyx.com>;index=1, '
    '<sip:+16176556805@sip.telnyx.com'
    '?Reason=SIP%3Bcause%3D408%3Btext%3D%22No%20Reply%22>;index=1.1')


# ------------------------------------------------------------- parse
def t_parse():
    d = la.parse_diversion({"sip.diversion": DIVERSION_JIM})
    check("diversion parsed", d["diverted"] and d["source"]
          == "diversion", str(d))
    check("diversion from is Jim", d["from_digits"] == "8574261739")
    check("diversion reason", d["reason"] == "no-answer")
    d2 = la.parse_diversion({"sip.h.diversion":
                             "<sip:+16175550100@x>;reason=busy"})
    check("diversion sip.h spelling + busy", d2["diverted"]
          and d2["from_digits"] == "6175550100"
          and d2["reason"] == "busy", str(d2))
    multi = ("<sip:+16175550100@x>;reason=busy, "
             "<sip:+18574261739@x>;reason=no-answer")
    d3 = la.parse_diversion({"sip.diversion": multi})
    check("diversion first entry wins (most recent redirector)",
          d3["from_digits"] == "6175550100"
          and d3["reason"] == "busy", str(d3))
    h = la.parse_diversion({"sip.h.history-info": HISTORY_JIM})
    check("history-info parsed", h["diverted"] and h["source"]
          == "history-info", str(h))
    check("history-info from = originally-called (Jim)",
          h["from_digits"] == "8574261739")
    check("history-info cause 408 -> no-answer",
          h["reason"] == "no-answer", str(h))
    none = la.parse_diversion({"sip.phoneNumber": STRANGER})
    check("no headers -> not diverted", not none["diverted"])
    junk = la.parse_diversion({"sip.diversion": "garbage, no uri"})
    check("malformed diversion -> not diverted, no crash",
          not junk["diverted"])
    single = la.parse_diversion(
        {"sip.history_info": "<sip:+16176556805@x>;index=1"})
    check("single history entry (no retarget) -> not diverted",
          not single["diverted"])


def t_pai():
    check("pai sip uri", la.pai_number(
        {"sip.p_asserted_identity": "<sip:+17745550134@x>;tag=1"})
        == "+17745550134")
    check("pai tel uri", la.pai_number(
        {"sip.h.p-asserted-identity": "<tel:+17745550134>"})
        == "+17745550134")
    check("pai absent", la.pai_number({}) == "")


# ---------------------------------------------------------- classify
def t_classify_forwarded():
    fam = la.classify_caller(
        {"sip.phoneNumber": FAMILY, "sip.trunkPhoneNumber": AGENT_NUM,
         "sip.diversion": DIVERSION_JIM},
        "sip_+17745550134", BRIEF)
    check("fwd family: diverted", fam["diverted"]
          and fam["forwarded_from_cell"], str(fam))
    check("fwd family: real caller kept", fam["number"] == FAMILY)
    check("fwd family: tier family", fam["tier"] == "family"
          and fam["name"].startswith("Benoucheca"))
    check("fwd family: reason kept", fam["diversion_reason"]
          == "no-answer")
    st = la.classify_caller(
        {"sip.phoneNumber": STRANGER,
         "sip.trunkPhoneNumber": AGENT_NUM,
         "sip.diversion": DIVERSION_JIM},
        "sip_+16175550199", BRIEF)
    check("fwd stranger: tier empty (message-taking)",
          st["tier"] == "" and st["name"] == "")
    check("fwd stranger: still pstn + diverted",
          st["channel"] == "pstn" and st["diverted"])


def t_classify_spoof_never_self():
    # The security-critical cases: any header arrangement that
    # smells like Jim must NOT produce orders mode once a diversion
    # is present.
    a = la.classify_caller(
        {"sip.phoneNumber": JIM, "sip.trunkPhoneNumber": AGENT_NUM,
         "sip.diversion": DIVERSION_JIM},
        "sip_+18574261739", BRIEF)
    check("spoof: Jim number + diversion from Jim -> NOT self",
          a["tier"] != "self", str(a))
    check("spoof: rewritten From drops the number entirely",
          a["number"] == "" and a["tier"] == "", str(a))
    b = la.classify_caller(
        {"sip.phoneNumber": JIM, "sip.trunkPhoneNumber": AGENT_NUM,
         "sip.diversion": "<sip:+16175550100@x>;reason=unconditional"},
        "sip_+18574261739", BRIEF)
    check("spoof: Jim number + diversion from a stranger -> NOT self",
          b["tier"] != "self", str(b))
    c = la.classify_caller(
        {"sip.phoneNumber": JIM, "sip.trunkPhoneNumber": AGENT_NUM,
         "sip.h.history-info": HISTORY_JIM},
        "sip_+18574261739", BRIEF)
    check("spoof: Jim number + history-info -> NOT self",
          c["tier"] != "self", str(c))


def t_classify_pai_recovery():
    # From rewritten to the redirector (Jim), PAI carries the real
    # caller: classify by PAI.
    r = la.classify_caller(
        {"sip.phoneNumber": JIM, "sip.trunkPhoneNumber": AGENT_NUM,
         "sip.diversion": DIVERSION_JIM,
         "sip.p_asserted_identity": f"<sip:{FAMILY}@sip.telnyx.com>"},
        "sip_+18574261739", BRIEF)
    check("pai recovery: caller is the family number",
          r["number"] == FAMILY and r["tier"] == "family", str(r))
    r2 = la.classify_caller(
        {"sip.phoneNumber": JIM, "sip.trunkPhoneNumber": AGENT_NUM,
         "sip.diversion": DIVERSION_JIM,
         "sip.p_asserted_identity": f"<sip:{STRANGER}@x>"},
        "sip_+18574261739", BRIEF)
    check("pai recovery: stranger stays stranger",
          r2["number"] == STRANGER and r2["tier"] == "", str(r2))


def t_classify_direct_unchanged():
    jim = la.classify_caller(
        {"sip.phoneNumber": JIM, "sip.trunkPhoneNumber": AGENT_NUM},
        "sip_+18574261739", BRIEF)
    check("direct Jim call still self", jim["tier"] == "self"
          and not jim["diverted"], str(jim))
    fam = la.classify_caller(
        {"sip.phoneNumber": FAMILY, "sip.trunkPhoneNumber": AGENT_NUM},
        "sip_+17745550134", BRIEF)
    check("direct family call unchanged", fam["tier"] == "family"
          and not fam["diverted"])
    sip = la.classify_caller({}, "intelogroup", BRIEF)
    check("direct SIP self unchanged", sip["tier"] == "self"
          and sip["channel"] == "sip" and not sip["diverted"])


# ------------------------------------------------------- flow policy
def t_flow_policy():
    fwd = {"channel": "pstn", "direction": "inbound", "tier": "family",
           "diverted": True, "forwarded_from_cell": True}
    check("forwarded call skips ring-first",
          la.ring_first_decision(fwd, enabled=True, quiet=False,
                                 trunk_id="ST_x") == "skip")
    g = la.greeting_for(fwd | {"name": "Benoucheca Pierre"})
    check("forwarded greeting says he couldn't pick up",
          "couldn't pick up" in g and "Benoucheca" in g, g)
    g2 = la.greeting_for(fwd | {"name": ""})
    check("forwarded greeting (unnamed) too",
          "couldn't pick up" in g2, g2)
    direct = {"channel": "pstn", "direction": "inbound",
              "tier": "family", "diverted": False,
              "forwarded_from_cell": False, "name": "Benoucheca Pierre"}
    check("direct greeting unchanged",
          "couldn't pick up" not in la.greeting_for(direct))


def t_loop_match():
    wait = {"call_id": "c1", "caller_digits": "6175550199",
            "abort": threading.Event()}
    probe = la.diversion_probe(
        {"sip.phoneNumber": STRANGER, "sip.diversion": DIVERSION_JIM})
    check("loop probe: diverted from Jim, caller kept",
          probe["diverted"] and probe["from_digits"] == "8574261739"
          and probe["caller_digits"] == "6175550199", str(probe))
    check("loop match: own leg coming back",
          la.loop_leg_match(probe, wait, "8574261739"))
    other = la.diversion_probe(
        {"sip.phoneNumber": FAMILY, "sip.diversion": DIVERSION_JIM})
    check("loop no-match: different original caller",
          not la.loop_leg_match(other, wait, "8574261739"))
    notdiv = la.diversion_probe({"sip.phoneNumber": STRANGER})
    check("loop no-match: not diverted",
          not la.loop_leg_match(notdiv, wait, "8574261739"))
    check("loop no-match: no active wait",
          not la.loop_leg_match(probe, None, "8574261739"))
    otherdiv = la.diversion_probe(
        {"sip.phoneNumber": STRANGER,
         "sip.diversion": "<sip:+16175550100@x>;reason=no-answer"})
    check("loop no-match: diverted from someone else",
          not la.loop_leg_match(otherdiv, wait, "8574261739"))


# ------------------------------------------------------------ race
class FakeLegs:
    def __init__(self):
        self.cleanup_calls = 0
        self.create_cancelled = False

    async def create(self):
        try:
            await asyncio.sleep(60)  # Jim never picks up on his own
            return SimpleNamespace(sip_call_id="late")
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


async def t_race_loop_abort():
    legs = FakeLegs()
    gone = asyncio.Event()
    abort = threading.Event()

    async def fire_abort():
        await asyncio.sleep(0.2)
        abort.set()  # the forwarded leg arrived (other job)
    asyncio.ensure_future(fire_abort())
    outcome, elapsed, detail = await la._ring_race(
        create=legs.create, cleanup=legs.cleanup, window_s=30.0,
        caller_gone=gone, loop_abort=abort)
    check("race loop: outcome loop", outcome == "loop", outcome)
    check("race loop: ends at the abort, not the 30s window",
          elapsed < 2.0, f"{elapsed}s")
    check("race loop: leg cleaned up (Jim's phone stops)",
          legs.cleanup_calls == 1 and legs.create_cancelled)
    check("race loop: detail notes the forwarded leg",
          "forwarded" in detail, detail)


async def main():
    t0 = time.monotonic()
    t_parse()
    t_pai()
    t_classify_forwarded()
    t_classify_spoof_never_self()
    t_classify_pai_recovery()
    t_classify_direct_unchanged()
    t_flow_policy()
    t_loop_match()
    await t_race_loop_abort()
    print(f"\n({time.monotonic() - t0:.1f}s)")
    if FAILED:
        print(f"{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("ALL FORWARDING TESTS PASSED")


asyncio.run(main())
