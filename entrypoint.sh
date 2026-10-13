#!/bin/bash
# Entrypoint for the durable voice-worker image.
# Two modes:
#   DRY_TEST=1  self-contained test: local livekit-server w/ generated creds,
#               worker pointed at it. No Oracle, no tunnel, no secrets.
#   (unset)     production: optional Oracle SSH tunnel (needs TUNNEL_SSH_KEY),
#               then the worker with env-provided credentials.
set -euo pipefail

if [ "${DRY_TEST:-0}" = "1" ]; then
  echo "[entrypoint] DRY_TEST mode: local livekit-server, generated credentials"
  LK_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
  cat > /tmp/livekit-dry.yaml <<EOF
port: 7880
bind_addresses:
  - "0.0.0.0"
keys:
  devkey: $LK_SECRET
log_level: info
EOF
  /opt/agent/bin/livekit-server --config /tmp/livekit-dry.yaml \
    > /tmp/livekit-dry.log 2>&1 &
  LK_PID=$!
  for _ in $(seq 1 30); do
    (echo > /dev/tcp/127.0.0.1/7880) 2>/dev/null && break
    sleep 1
  done
  if ! (echo > /dev/tcp/127.0.0.1/7880) 2>/dev/null; then
    echo "[entrypoint] ERROR: livekit-server did not start"
    cat /tmp/livekit-dry.log
    exit 1
  fi
  echo "[entrypoint] livekit-server up (pid $LK_PID)"
  export LIVEKIT_URL="ws://127.0.0.1:7880"
  export LIVEKIT_API_KEY="devkey"
  export LIVEKIT_API_SECRET="$LK_SECRET"
  # Unambiguous test identity in logs (server is local-only, so it can
  # never receive production dispatches regardless of the name).
  sed -i 's/^AGENT_NAME = "jett-proxy"$/AGENT_NAME = "jett-proxy-drytest"/' \
    /opt/agent/live_agent.py
  grep -q 'AGENT_NAME = "jett-proxy-drytest"' /opt/agent/live_agent.py
  echo "[entrypoint] dry-test agent name: jett-proxy-drytest"
fi

# --- production Oracle tunnel (skipped in dry-test) ---
if [ -n "${TUNNEL_SSH_KEY:-}" ] && [ "${DRY_TEST:-0}" != "1" ]; then
  echo "[entrypoint] starting Oracle SSH tunnel (7880/7881)"
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  printf '%s\n' "$TUNNEL_SSH_KEY" > /root/.ssh/tunnel_key
  chmod 600 /root/.ssh/tunnel_key
  ssh-keyscan -H 129.159.189.244 >> /root/.ssh/known_hosts 2>/dev/null || true
  (
    while true; do
      ssh -N -i /root/.ssh/tunnel_key \
        -o StrictHostKeyChecking=yes -o BatchMode=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -L 7880:localhost:7880 -L 7881:localhost:7881 \
        ubuntu@129.159.189.244 \
        || echo "[tunnel] ssh exited ($(date -u +%FT%TZ)), retrying in 10s"
      sleep 10
    done
  ) &
  echo "[entrypoint] tunnel supervisor started"
fi

echo "[entrypoint] starting worker, LIVEKIT_URL=${LIVEKIT_URL:-ws://localhost:7880}"
exec python3 /opt/agent/live_agent.py start
