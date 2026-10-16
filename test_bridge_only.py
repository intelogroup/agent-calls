#!/usr/bin/env python3
"""Tests for BRIDGE_ONLY receptionist mode (Jim, 2026-10-16 —
"deepgram is primary and livekit secondary").

The build: an env flag (BRIDGE_ONLY=1) turning this worker into
the small always-on PSTN receptionist. It registers as
"jett-bridge", never loads Phonon-2 or Pocket (no warm-up
threads), serves direct calls on the Deepgram bridge exactly as
the big worker does, and REFUSES any call that would fall through
to the local pipeline — cleanly, with a bus receipt — because the
local stack (the secondary) is asleep by design.

Hard invariants under test:
  - default mode (flag unset) is untouched: BRIDGE_ONLY False,
    agent name "jett-proxy", bridge_only_guard ALWAYS None;
  - subprocess with BRIDGE_ONLY=1: BRIDGE_ONLY True, agent name
    "jett-bridge", and the module import pulls in NEITHER torch,
    fermion, nor pocket_tts (the engines import lazily; the mode
    exists to keep it that way);
  - the guard matrix (bridge_only_guard, flag on): self-linphone
    diversion -> "self_linphone_local"; a local-only route (an
    outbound errand without the deepgram flag) ->
    "route_local_only"; a dg_bridge_skipped reason passes through
    ("flag_off" / "no_api_key"); anything else ->
    "bridge_unavailable";
  - the mode through the REAL gate inputs: bridge_route outputs
    feeding the guard — a PSTN self call and a PSTN untrusted call
    both route to the bridge (serve=True), so the guard stays
    silent for exactly the calls the receptionist exists to take;
    an outbound errand with engine="deepgram" also serves; only a
    plain outbound errand (local route) trips the guard;
  - wiring, by source inspection (same style as the other gate
    suites): the BRIDGE_ONLY no-fallback branch emits BOTH
    dg_bridge_fallback_local (shared with the big worker) and
    bridge_only_no_fallback and returns; the choke-point guard
    sits before the local pipeline's SYSTEM_PROMPT construction;
    the busy rejection and the no-brain notice have BRIDGE_ONLY
    branches that receipt + disconnect instead of touching
    PocketTTS; the __main__ warm-up threads are conditioned on
    NOT BRIDGE_ONLY; the fly.bridge.toml shape (app, env flags,
    1x/1024mb machine, no auto-stop).

Run: <venv>/python test_bridge_only.py
"""
import os
import subprocess
import sys
import tempfile

# BUS_DIR must be set BEFORE importing live_agent (same harness
# as test_bridge_outbound.py).
_TMP = tempfile.mkdtemp(prefix="bridge-only-bus-")
os.environ["BUS_DIR"] = _TMP
os.environ.setdefault("GIT_AUTHOR_NAME", "bridge-only-test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "bo@test")
os.environ.setdefault("GIT_COMMITTER_NAME", "bridge-only-test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "bo@test")
subprocess.run(["git", "init", "-q", _TMP], check=True)
os.environ.pop("BRIDGE_ONLY", None)

sys.path.insert(0, ".")
import live_agent as la  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


def info(direction, channel, tier, **kw):
    d = {"direction": direction, "channel": channel, "tier": tier,
         "name": "", "number": ""}
    d.update(kw)
    return d


SRC = open("live_agent.py").read()

# ---------------------------------------------------------- #
# Default mode is untouched.
# ---------------------------------------------------------- #
check("default: BRIDGE_ONLY is False", la.BRIDGE_ONLY is False)
check("default: AGENT_NAME is jett-proxy",
      la.AGENT_NAME == "jett-proxy")

# The guard is inert in big-worker mode for EVERY routing state.
for route in ({"serve": True}, {"serve": False}):
    for self_ll in (False, True):
        for skip in ("", "flag_off", "no_api_key"):
            check(f"default guard None route={route} self_ll={self_ll} "
                  f"skip={skip!r}",
                  la.bridge_only_guard(route, self_ll, skip) is None)

# bridge_route itself is unchanged by the mode's existence:
# the receptionist's bread-and-butter calls still route to the
# bridge, and the local-only route is still local-only.
check("route: PSTN self -> bridge/self",
      la.bridge_route(info("inbound", "pstn", "self"), None)
      == {"serve": True, "mode": "self", "via_engine": False})
check("route: PSTN untrusted -> bridge/untrusted",
      la.bridge_route(info("inbound", "pstn", ""), None)
      == {"serve": True, "mode": "untrusted", "via_engine": False})
check("route: outbound errand, no flag -> local",
      la.bridge_route(info("outbound", "pstn", ""), {"brief": "x"})
      == {"serve": False, "mode": "untrusted", "via_engine": False})
check("route: outbound errand, engine=deepgram -> bridge",
      la.bridge_route(info("outbound", "pstn", ""),
                      {"brief": "x", "engine": "deepgram"})
      == {"serve": True, "mode": "untrusted", "via_engine": True})

# ---------------------------------------------------------- #
# Guard matrix (flag explicitly on).
# ---------------------------------------------------------- #
G = lambda r, s, k="": la.bridge_only_guard(r, s, k, bridge_only=True)  # noqa: E731
check("guard: self-linphone diversion refused as such",
      G({"serve": True}, True) == "self_linphone_local")
check("guard: self_ll wins over route state",
      G({"serve": False}, True) == "self_linphone_local")
check("guard: local-only route refused",
      G({"serve": False}, False) == "route_local_only")
check("guard: flag_off passes through",
      G({"serve": True}, False, "flag_off") == "flag_off")
check("guard: no_api_key passes through",
      G({"serve": True}, False, "no_api_key") == "no_api_key")
check("guard: route beats skip reason (local route decided first)",
      G({"serve": False}, False, "flag_off") == "route_local_only")
check("guard: bare fall-through refused",
      G({"serve": True}, False) == "bridge_unavailable")
check("guard: flag explicitly off -> None even in guard states",
      la.bridge_only_guard({"serve": False}, True, "flag_off",
                           bridge_only=False) is None)

# The mode end-to-end through real gate inputs: guard applied to
# bridge_route outputs exactly as _run_call computes them.
def guard_for(ci, meta=None, self_ll=False, skip=""):
    return la.bridge_only_guard(la.bridge_route(ci, meta), self_ll,
                                skip, bridge_only=True)

# NOTE: at the choke point the guard is only REACHED when the
# bridge did not serve — a serving bridge returns earlier. For a
# serve=True route with the gate open, "bridge_unavailable" is
# therefore the correct backstop refusal ("route said serve, yet
# control still fell through"); the guard is not the routing
# decision itself.
check("e2e: PSTN self fall-through -> bridge_unavailable backstop",
      guard_for(info("inbound", "pstn", "self")) == "bridge_unavailable")
check("e2e: PSTN untrusted fall-through -> bridge_unavailable",
      guard_for(info("inbound", "pstn", "")) == "bridge_unavailable")
check("e2e: plain outbound errand -> route_local_only",
      guard_for(info("outbound", "pstn", ""), {"brief": "x"})
      == "route_local_only")
check("e2e: engine=deepgram errand with key missing -> no_api_key",
      guard_for(info("outbound", "pstn", ""),
                {"brief": "x", "engine": "deepgram"}, skip="no_api_key")
      == "no_api_key")
check("e2e: Linphone self with toggle on -> self_linphone_local",
      guard_for(info("inbound", "sip", "self"), self_ll=True)
      == "self_linphone_local")

# ---------------------------------------------------------- #
# Wiring, by source inspection.
# ---------------------------------------------------------- #
check("wiring: BRIDGE_ONLY read from env",
      'BRIDGE_ONLY = os.environ.get("BRIDGE_ONLY") == "1"' in SRC)
check("wiring: agent name flips under the flag",
      'if BRIDGE_ONLY:\n    AGENT_NAME = "jett-bridge"' in SRC)
check("wiring: fallback branch emits bridge_only_no_fallback",
      '"bridge_only_no_fallback", via="bridge"' in SRC
      or '"bridge_only_no_fallback"' in SRC)
_fb = SRC.index('if BRIDGE_ONLY:', SRC.index("dg_bridge_fallback_local"))
check("wiring: no-fallback branch returns (no fall-through)",
      "return" in SRC[_fb:_fb + 1400])
check("wiring: guard call present",
      "_guard_reason = bridge_only_guard(_route, _self_ll," in SRC)
_guard_at = SRC.index("_guard_reason = bridge_only_guard")
_prompt_at = SRC.index("instructions = SYSTEM_PROMPT.format(brief=brief)")
check("wiring: guard sits BEFORE the local pipeline construction",
      _guard_at < _prompt_at)
check("wiring: refusal receipt emitted at the guard",
      'bridge_only_local_refused' in SRC)
check("wiring: busy rejection has a bridge-only branch",
      'reason="busy_rejection"' in SRC)
check("wiring: no-brain notice has a bridge-only branch",
      'reason="no_brain_notice"' in SRC)
_main = SRC[SRC.index('if __name__ == "__main__":'):]
check("wiring: warm-ups conditioned on NOT BRIDGE_ONLY",
      "if BRIDGE_ONLY:" in _main
      and 'threading.Thread(target=_warm_phonon' in _main
      and _main.index("if BRIDGE_ONLY:") < _main.index(
          "threading.Thread(target=_warm_phonon"))
check("wiring: boot_config carries bridge_only",
      "bridge_only=BRIDGE_ONLY" in _main)
check("wiring: boot_config engines are honest in bridge mode",
      '"deepgram-bridge" if BRIDGE_ONLY' in _main)

# ---------------------------------------------------------- #
# fly.bridge.toml shape.
# ---------------------------------------------------------- #
TOML = open("fly.bridge.toml").read()
check("toml: app is jett-bridge", 'app = "jett-bridge"' in TOML)
check("toml: BRIDGE_ONLY on", 'BRIDGE_ONLY = "1"' in TOML)
check("toml: DEEPGRAM_BRIDGE on", 'DEEPGRAM_BRIDGE = "1"' in TOML)
check("toml: RING_FIRST on", 'RING_FIRST = "1"' in TOML)
check("toml: speculative feed off (local path never runs)",
      'SPECULATIVE_FEED = "0"' in TOML)
check("toml: small machine", 'size = "shared-cpu-1x"' in TOML
      and 'memory = "1024mb"' in TOML)
check("toml: never auto-stops (it is the answering path)",
      "auto_stop_machines = false" in TOML)

# ---------------------------------------------------------- #
# Subprocess: the mode as the worker actually boots it.
# ---------------------------------------------------------- #
probe = r'''
import os, resource, sys
sys.path.insert(0, ".")
import live_agent as la
mods = [m for m in ("torch", "fermion", "pocket_tts")
        if m in sys.modules or any(s.startswith(m + ".")
                                   for s in sys.modules)]
print("AGENT_NAME", la.AGENT_NAME)
print("BRIDGE_ONLY", la.BRIDGE_ONLY)
print("HEAVY_MODULES", ",".join(mods) if mods else "none")
print("MAXRSS_MB", resource.getrusage(
    resource.RUSAGE_SELF).ru_maxrss // 1024)
'''
env = dict(os.environ, BRIDGE_ONLY="1", BUS_DIR=_TMP)
out = subprocess.run([sys.executable, "-c", probe], env=env,
                     capture_output=True, text=True, cwd=".")
check("boot: subprocess ran", out.returncode == 0,
      out.stderr.strip()[-200:] if out.returncode else "")
vals = dict(line.split(None, 1) for line in out.stdout.splitlines()
            if " " in line)
check("boot: agent name is jett-bridge",
      vals.get("AGENT_NAME") == "jett-bridge")
check("boot: BRIDGE_ONLY True", vals.get("BRIDGE_ONLY") == "True")
check("boot: no heavy module imported at boot",
      vals.get("HEAVY_MODULES") == "none")
check("boot: import RSS under 400 MB (measured ~150 MB)",
      int(vals.get("MAXRSS_MB", "99999")) < 400,
      f"maxrss={vals.get('MAXRSS_MB')} MB")

if FAILED:
    print(f"\n{len(FAILED)} FAILURES: {FAILED}")
    sys.exit(1)
print("\nALL PASSED")
