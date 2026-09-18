#!/bin/bash
# Line-health check for Jett's voice proxy. Runs ON the VM (as root).
#
# Checks every bottleneck in the inbound path:
#   1. services active: redis-server, livekit-server, livekit-sip, jett-proxy
#   2. SIP listeners: UDP 5060 and TCP 5060 bound (ss)
#   3. SIP trunk 'jett' + dispatch rule 'jett' exist (LiveKit API,
#      same patterns as setup-sip.py)
#   4. brain resolves (imports live_agent.resolve_brain, like verify-brain.sh)
#
# Prints PASS/FAIL per check. Exits 0 only if all pass; on any FAIL exits
# nonzero — each FAIL line is a one-line reason.
# Never prints secrets: only service states, port states, trunk/rule NAMES,
# and the brain label/model.
set -uo pipefail

AGENT_DIR=/opt/agent
FAIL=0

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAIL=1; }

echo "== services =="
for s in redis-server livekit-server livekit-sip jett-proxy; do
  if [ "$(systemctl is-active "$s" 2>/dev/null)" = "active" ]; then
    pass "service $s active"
  else
    fail "service $s not active (state: $(systemctl is-active "$s" 2>/dev/null || echo unknown))"
  fi
done

echo "== SIP listeners =="
if ss -uln 2>/dev/null | grep -q ":5060 "; then
  pass "UDP 5060 listening"
else
  fail "UDP 5060 not listening — inbound INVITEs (Linphone default) go nowhere"
fi
if ss -tln 2>/dev/null | grep -q ":5060 "; then
  pass "TCP 5060 listening"
else
  fail "TCP 5060 not listening"
fi

echo "== SIP trunk + dispatch rule =="
if [ -f "$AGENT_DIR/livekit.env" ] && [ -x "$AGENT_DIR/venv/bin/python" ]; then
  SIP_CHECK_OUT=$("$AGENT_DIR/venv/bin/python" - <<'EOF' 2>&1
import asyncio, sys
sys.path.insert(0, "/opt/agent/venv/lib/python3.12/site-packages")
from livekit.api import LiveKitAPI
from livekit.protocol import sip as S

def load_env(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out

async def main():
    env = load_env("/opt/agent/livekit.env")
    api = LiveKitAPI(url="ws://localhost:7880",
                     api_key=env["LIVEKIT_API_KEY"],
                     api_secret=env["LIVEKIT_API_SECRET"])
    try:
        trunks = await api.sip.list_inbound_trunk(
            S.ListSIPInboundTrunkRequest())
        rules = await api.sip.list_dispatch_rule(
            S.ListSIPDispatchRuleRequest())
    finally:
        await api.aclose()
    # names only — never ids, keys, or numbers
    tnames = sorted(t.name for t in trunks.items)
    rnames = sorted(r.name for r in rules.items)
    print("trunks: " + (",".join(tnames) if tnames else "(none)"))
    print("rules: " + (",".join(rnames) if rnames else "(none)"))
    return 0 if ("jett" in tnames and "jett" in rnames) else 1

sys.exit(asyncio.run(main()))
EOF
  ); SIP_RC=$?
  echo "$SIP_CHECK_OUT"
  if [ "$SIP_RC" -eq 0 ]; then
    pass "trunk 'jett' + dispatch rule 'jett' present"
  else
    fail "trunk 'jett' and/or dispatch rule 'jett' missing (see list above)"
  fi
else
  fail "livekit.env or venv python missing — cannot query LiveKit API"
fi

echo "== brain =="
BRAIN=$("$AGENT_DIR/venv/bin/python" -c "
import sys
sys.path.insert(0, '/opt/agent')
import live_agent as la
b = la.resolve_brain()
print((b.label + ' ' + b.model) if b else 'NONE')
" 2>/dev/null) || BRAIN="NONE"
if [ -n "$BRAIN" ] && [ "$BRAIN" != "NONE" ]; then
  pass "brain resolves: $BRAIN"
else
  fail "brain does not resolve (OPENROUTER_API_KEY missing?)"
fi

echo
if [ "$FAIL" -ne 0 ]; then
  echo "LINE HEALTH: FAIL"
  exit 1
fi
echo "LINE HEALTH: PASS"
