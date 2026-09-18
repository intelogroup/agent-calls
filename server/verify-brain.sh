#!/bin/bash
# Verify the live-agent brain wiring on the VM.
# Run as the `agent` user (it only reads /opt/agent/agent.env).
# Checks:
#   (a) live_agent.resolve_brain() picks a backend from agent.env, and
#   (b) the OpenRouter key authenticates against GET /v1/models (free;
#       no chat completion, no tokens burned).
# Prints only the HTTP status code, never the key.
set -euo pipefail

AGENT_DIR=/opt/agent
set -a
# shellcheck disable=SC1091
. "$AGENT_DIR/agent.env"
set +a
cd "$AGENT_DIR"

BRAIN=$("$AGENT_DIR/venv/bin/python" -c "
import live_agent as la
b = la.resolve_brain()
print((b.label + ' ' + b.model) if b else 'NONE')
")
echo "resolved brain: $BRAIN"
[ "$BRAIN" != "NONE" ] || { echo "ERROR: no brain key in agent.env"; exit 1; }

if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  CODE=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 25 \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    https://openrouter.ai/api/v1/models)
  echo "openrouter /models -> HTTP $CODE"
  [ "$CODE" = "200" ] || echo "WARNING: OpenRouter key check returned HTTP $CODE"

  # (c) one cheap test completion (a few tokens) to prove the model serves
  MODEL="${JETT_BRAIN_MODEL:-openrouter/free}"
  echo "test completion with model: $MODEL"
  T0=$(date +%s.%N)
  RESP=$(curl -sS --max-time 90 \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    -H "Content-Type: application/json" \
    -H "HTTP-Referer: https://github.com/intelogroup/agent-calls" \
    -H "X-Title: jett-proxy voice agent" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: brain online\"}],\"max_tokens\":8}" \
    https://openrouter.ai/api/v1/chat/completions) || RESP=""
  T1=$(date +%s.%N)
  LAT=$(echo "$T1 $T0" | awk '{printf "%.1f", $1 - $2}')
  TEXT=$(printf '%s' "$RESP" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    ch = (d.get('choices') or [{}])[0]
    print(ch.get('message', {}).get('content', '').strip() or json.dumps(d)[:200])
except Exception as e:
    print('PARSE_FAIL ' + str(e)[:100])
" 2>/dev/null)
  echo "test completion latency: ${LAT}s reply: $TEXT"
else
  echo "no OpenRouter key in agent.env; skipping API checks (Meta fallback may apply)"
fi
