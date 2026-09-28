#!/bin/bash
# Container entrypoint: install the bus deploy key (from Fly secret), then
# start the LiveKit agent worker. Secrets arrive as Fly secrets, never files.
set -euo pipefail

if [ -n "${BUS_KEY_PEM:-}" ]; then
  mkdir -p ~/.ssh
  printf '%s\n' "$BUS_KEY_PEM" > ~/.ssh/bus_key
  chmod 600 ~/.ssh/bus_key
  # unset from the environment so child process listings don't carry it
  unset BUS_KEY_PEM
fi

exec python /opt/agent/live_agent.py start
