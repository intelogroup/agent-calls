"""Unit tests for the PSTN telephony layer in live_agent.py (2026-10-14).

Covers the pure helpers and the testable cores behind the new tools:
known-caller parsing from the brief, caller classification from SIP
participant attributes, greeting selection, DTMF cleaning/codes, the
transfer request builder against a fake API client, and the
voicemail tool's direction guard.

Run: cd ~/workspace/jett-voice-image && <env> .venv-drytest/bin/python \
        ~/tmp-pip/test_telephony.py
"""
import asyncio
import sys
import types

sys.path.insert(0, "/home/hatch/tmp-pip/merge187/run")
import live_agent as la  # noqa: E402

PASS = []


def check(name, cond, extra=""):
    PASS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))


BRIEF = """Some brief text.
Known callers:
- Benoucheca Pierre — +1 774-555-0134 (family)
- Joe Driving School — 781-929-3434 (work)
- Dr. Lee office: (617) 555-0182
Random line with a date 2026-10-14 and a zip 02149.
"""


def t_parse():
    k = la.parse_known_callers(BRIEF)
    check("P1 family parsed", k.get("7745550134", {}).get("tier")
          == "family", str(k.get("7745550134")))
    check("P2 name parsed", "Benoucheca" in
          k.get("7745550134", {}).get("name", ""))
    check("P3 work parsed", k.get("7819293434", {}).get("tier")
          == "work")
    check("P4 untagged -> known", k.get("6175550182", {}).get("tier")
          == "known")
    check("P5 Jim's own line is self",
          k.get("8574261739", {}).get("tier") == "self")
    check("P6 date/zip not parsed as callers",
          "20261014" not in k and len(k) == 5, str(sorted(k)))


def t_classify():
    known = la.classify_caller(
        {"sip.phoneNumber": "+17745550134",
         "sip.trunkPhoneNumber": "+16178623921"},
        "sip_+17745550134", BRIEF)
    check("C1 pstn channel", known["channel"] == "pstn")
    check("C2 known name+tier", known["name"].startswith("Benoucheca")
          and known["tier"] == "family")
    check("C3 called number kept",
          known["called_number"] == "+16178623921")
    unk = la.classify_caller({"sip.phoneNumber": "+16175559999"},
                             "x", BRIEF)
    check("C4 unknown pstn", unk["channel"] == "pstn"
          and unk["name"] == "" and unk["tier"] == "")
    sip = la.classify_caller({}, "intelogroup", BRIEF)
    check("C5 sip channel for linphone", sip["channel"] == "sip")
    check("C6 Jim's SIP identity is self (2026-10-15 A/B regression)",
          sip["tier"] == "self" and sip["name"] == "Jim",
          str(sip))
    via_attr = la.classify_caller(
        {"sip.phoneNumber": "intelogroup"}, "sip_intelogroup", BRIEF)
    check("C7 SIP username in phoneNumber attr is self, not pstn",
          via_attr["channel"] == "sip" and via_attr["tier"] == "self"
          and via_attr["number"] == "", str(via_attr))
    stranger = la.classify_caller(
        {"sip.phoneNumber": "someoneelse"}, "sip_someoneelse", BRIEF)
    check("C8 stranger SIP username stays unknown",
          stranger["channel"] == "sip" and stranger["tier"] == ""
          and stranger["name"] == "", str(stranger))


def t_greeting():
    g = la.greeting_for({"name": "Benoucheca Pierre",
                         "tier": "family"})
    check("G1 known greeting names the caller + Jim + consent",
          "Benoucheca" in g and "Jim's assistant" in g
          and "transcript" in g)
    g2 = la.greeting_for({"name": "", "tier": ""})
    check("G2 unknown greeting generic + consent",
          g2.startswith("Hi, this is Jett") and "transcript" in g2)


def t_dtmf():
    check("D1 clean keeps keypad chars",
          la._clean_digits("1-800 #*a") == "1800#*")
    check("D2 codes map", la._DTMF_CODES["#"] == 11
          and la._DTMF_CODES["*"] == 10
          and la._DTMF_CODES["9"] == 9)


class FakeSipService:
    def __init__(self):
        self.reqs = []

    async def transfer_sip_participant(self, req):
        self.reqs.append(req)
        return object()


async def t_transfer():
    fake = types.SimpleNamespace(sip=FakeSipService())
    ok, detail = await la._do_transfer(
        "jett-call1", "sip_+17745550134",
        "sip:intelogroup@sip.linphone.org", fake)
    check("T1 transfer ok", ok and detail == "started")
    req = fake.sip.reqs[0]
    check("T2 request fields", req.room_name == "jett-call1"
          and req.participant_identity == "sip_+17745550134"
          and req.transfer_to
          == "sip:intelogroup@sip.linphone.org"
          and req.play_dialtone is False)


async def t_voicemail_guard():
    fn = getattr(la.leave_voicemail, "_func", None) or getattr(la.leave_voicemail, "fn", None)
    check("V0 tool fn reachable", fn is not None)
    if fn is None:
        return
    la._ACTIVE_CALL_INFO = {"direction": "inbound", "channel": "pstn"}
    ctx = types.SimpleNamespace(session=None)
    out = await fn("Hello, this is Jett.", ctx)
    check("V1 inbound -> not applicable",
          "VOICEMAIL_NOT_APPLICABLE" in out)
    la._ACTIVE_CALL_INFO = {"direction": "outbound", "channel": "pstn"}
    hung = {}

    async def fake_hangup(delay):
        hung["delay"] = delay
    orig = la._delayed_hangup
    la._delayed_hangup = fake_hangup
    try:
        out = await fn("Jett here for Jim — call back please.", ctx)
    finally:
        la._delayed_hangup = orig
        la._ACTIVE_CALL_INFO = None
    check("V2 outbound -> say-then-stop instruction",
          out.startswith("SAY_THIS_THEN_STOP")
          and "call back" in out)
    await asyncio.sleep(0.1)
    check("V3 hangup scheduled with sane delay",
          4.0 < hung.get("delay", 0) < 12.0, str(hung))


async def t_transfer_guard():
    fn = getattr(la.transfer_to_jim, "_func", None) or getattr(la.transfer_to_jim, "fn", None)
    check("W0 tool fn reachable", fn is not None)
    if fn is None:
        return
    la._ACTIVE_CALL_INFO = {"direction": "inbound", "channel": "sip"}
    la._ACTIVE_ROOM = types.SimpleNamespace(
        name="r", remote_participants={})
    ctx = types.SimpleNamespace(session=None)
    out = await fn("urgent", ctx)
    check("W1 sip channel -> unavailable",
          "TRANSFER_UNAVAILABLE" in out)
    la._ACTIVE_CALL_INFO = None
    la._ACTIVE_ROOM = None


async def main():
    t_parse()
    t_classify()
    t_greeting()
    t_dtmf()
    await t_transfer()
    await t_voicemail_guard()
    await t_transfer_guard()
    print(f"\n{sum(PASS)}/{len(PASS)} passed")
    sys.exit(0 if all(PASS) else 1)


asyncio.run(main())
