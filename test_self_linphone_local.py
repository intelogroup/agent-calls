#!/usr/bin/env python3
"""Tests for the self-Linphone local-pipeline toggle (Jim, 2026-10-16).

The build: a bus config file, config/self-linphone-local.enabled,
read per call exactly like the Deepgram bridge kill-switch beside
it. While the file exists, Jim's OWN Linphone calls (inbound, SIP
channel, tier "self") skip the Deepgram bridge and are served by
the LOCAL pipeline (Phonon-2 STT + Pocket TTS + speculative feed),
so his test calls exercise the latest local stack. Deleting the
file is the instant rollback — no redeploy.

Hard invariants under test:
  - the gate matrix (self_linphone_local_route): ONLY
    (inbound, sip, self, toggle on) diverts to local; PSTN self
    stays on the bridge even with the toggle on; non-self SIP
    stays on the bridge; outbound is never affected;
  - bridge_route() itself is unchanged by the toggle's existence
    (the diversion lives at the gate, beside it): route outputs
    for the matrix cases are exactly today's;
  - the file probe (deepgram_bridge.self_linphone_local_on) is a
    plain existence check, independent of the bridge kill-switch
    file in both directions;
  - the gate wiring in the entrypoint: the diverted case emits
    the dg_bridge_skipped bus receipt with reason
    "self_linphone_local", and the bridge block is conditioned on
    NOT diverted, so a diverted call falls through to the local
    pipeline (whose self-tier speculative feed then serves it).

Run: <venv>/python test_self_linphone_local.py
"""
import os
import subprocess
import sys
import tempfile

# BUS_DIR must be set BEFORE importing live_agent (same harness
# as test_bridge_outbound.py).
_TMP = tempfile.mkdtemp(prefix="self-ll-bus-")
os.environ["BUS_DIR"] = _TMP
os.environ.setdefault("GIT_AUTHOR_NAME", "self-ll-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "self-ll@test")
os.environ.setdefault("GIT_COMMITTER_NAME", "self-ll-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "self-ll@test")
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


def info(direction, channel, tier, **kw):
    d = {"direction": direction, "channel": channel, "tier": tier,
         "number": "+18574261739", "name": "Jim"}
    d.update(kw)
    return d


def t_gate_matrix():
    r = la.self_linphone_local_route
    # The one diverting case.
    check("sip self inbound + toggle on -> local",
          r(info("inbound", "sip", "self"), True) is True)
    # Toggle off: today's behavior exactly.
    check("sip self inbound + toggle off -> bridge",
          r(info("inbound", "sip", "self"), False) is False)
    check("sip self inbound + toggle None -> bridge",
          r(info("inbound", "sip", None), False) is False)
    # PSTN self is NOT affected even with the toggle on.
    check("pstn self inbound + toggle on -> bridge (guard)",
          r(info("inbound", "pstn", "self"), True) is False)
    # Non-self SIP is NOT affected even with the toggle on.
    check("sip unknown inbound + toggle on -> bridge (guard)",
          r(info("inbound", "sip", "unknown"), True) is False)
    check("sip empty-tier inbound + toggle on -> bridge (guard)",
          r(info("inbound", "sip", ""), True) is False)
    # Outbound is never affected, in either channel or tier.
    check("sip self outbound + toggle on -> unchanged",
          r(info("outbound", "sip", "self"), True) is False)
    check("pstn self outbound + toggle on -> unchanged",
          r(info("outbound", "pstn", "self"), True) is False)
    check("pstn unknown outbound + toggle on -> unchanged",
          r(info("outbound", "pstn", "unknown"), True) is False)
    # Missing fields fail safe toward the bridge (no diversion).
    check("empty call_info + toggle on -> bridge (fail safe)",
          r({}, True) is False)


def t_file_probe():
    bus = tempfile.mkdtemp(prefix="self-ll-file-")
    cfg = os.path.join(bus, "config")
    check("toggle absent -> off", dgb.self_linphone_local_on(bus)
          is False)
    os.makedirs(cfg)
    check("config dir only -> still off",
          dgb.self_linphone_local_on(bus) is False)
    with open(os.path.join(cfg, "self-linphone-local.enabled"),
              "w") as f:
        f.write("")
    check("toggle present -> on", dgb.self_linphone_local_on(bus)
          is True)
    # Independent of the bridge kill-switch, both directions.
    check("bridge flag still off (independence)",
          dgb.bridge_flag_on(bus) is False)
    with open(os.path.join(cfg, "deepgram-bridge.enabled"),
              "w") as f:
        f.write("")
    check("bridge flag on does not change toggle read",
          dgb.self_linphone_local_on(bus) is True)
    os.remove(os.path.join(cfg, "self-linphone-local.enabled"))
    check("toggle deleted -> off again (rollback)",
          dgb.self_linphone_local_on(bus) is False)
    check("bridge flag unaffected by toggle delete",
          dgb.bridge_flag_on(bus) is True)
    check("relpath is config/self-linphone-local.enabled",
          dgb.SELF_LINPHONE_LOCAL_RELPATH.replace(os.sep, "/")
          == "config/self-linphone-local.enabled")


def t_bridge_route_unchanged():
    # The diversion lives at the gate, NOT inside bridge_route:
    # route outputs for the affected cases are exactly today's.
    r = la.bridge_route
    out = r(info("inbound", "sip", "self"), None)
    check("bridge_route sip self: serve/self (unchanged)",
          out == {"serve": True, "mode": "self",
                  "via_engine": False})
    out = r(info("inbound", "pstn", "self"), None)
    check("bridge_route pstn self: serve/self (unchanged)",
          out == {"serve": True, "mode": "self",
                  "via_engine": False})
    out = r(info("inbound", "sip", "unknown"), None)
    check("bridge_route sip unknown: serve/untrusted (unchanged)",
          out == {"serve": True, "mode": "untrusted",
                  "via_engine": False})
    out = r(info("outbound", "pstn", "unknown"), None)
    check("bridge_route outbound errand: local (unchanged)",
          out["serve"] is False)
    out = r(info("outbound", "pstn", "unknown"),
            {"engine": "deepgram"})
    check("bridge_route outbound errand + flag: bridge (unchanged)",
          out["serve"] is True and out["via_engine"] is True)


def t_gate_wiring():
    src = open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "live_agent.py")).read()
    check("gate reads toggle via deepgram_bridge probe",
          "self_linphone_local_on(BUS_DIR)" in src)
    check("gate computes diversion from predicate",
          "_self_ll = self_linphone_local_route(" in src)
    check("diverted case emits dg_bridge_skipped receipt",
          'reason="self_linphone_local"' in src
          and '"dg_bridge_skipped"' in src)
    check("bridge block conditioned on not diverted",
          'if DEEPGRAM_BRIDGE and _route["serve"] and not _self_ll:'
          in src)
    check("divert branch does not return early (falls to local)",
          'reason="self_linphone_local")' in src
          and "dg_bridge_start" in src)


def main():
    t_gate_matrix()
    t_file_probe()
    t_bridge_route_unchanged()
    t_gate_wiring()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES: {FAILED}")
        return 1
    print("\nALL SELF-LINPHONE-LOCAL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
