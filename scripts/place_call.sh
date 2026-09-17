#!/bin/bash
# Place a SIP call and play a prerecorded WAV as the microphone input.
# Reads destination from /tmp/dest.txt, audio from $GITHUB_WORKSPACE/message-8k.wav
set -euo pipefail

if [ "${DEBUG_CALL:-0}" = "1" ]; then set -x; fi

DEST="$(cat /tmp/dest.txt)"
WAV="$GITHUB_WORKSPACE/message-8k.wav"
FIFO=/tmp/sipcmd

echo "=== call params ==="
echo "dest: $DEST"
ls -l "$WAV"
file "$WAV" || true

rm -f "$FIFO"
mkfifo "$FIFO"

# Start baresip with the fifo as stdio, capture everything
baresip -f "$HOME/.baresip" <> "$FIFO" > baresip-run.log 2>&1 &
BPID=$!
echo "baresip pid: $BPID"

exec 3<>"$FIFO"
sleep 12  # registration window

echo "=== registration state ==="
grep -a -i -m5 -E "regist" baresip-run.log || echo "(no register lines yet)"

echo "dialing $DEST"
echo "dial $DEST" >&3
sleep 25  # ring + playback + margin

echo "hanging up"
echo "hangup" >&3
sleep 2
echo "quit" >&3 || true
sleep 2
kill "$BPID" 2>/dev/null || true
wait "$BPID" 2>/dev/null || true

echo "=== call summary (signaling/media lines) ==="
grep -a -E -m30 -i "call|invite|bye|cancel|audio|rtp|regist" baresip-run.log || true
echo "=== end summary ==="

if grep -a -qi "established" baresip-run.log; then
  echo "CALL_ESTABLISHED"
else
  echo "CALL_FAILED: no 'established' in baresip log"
  exit 1
fi
