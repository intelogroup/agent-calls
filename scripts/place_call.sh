#!/bin/bash
# Place a SIP call and play the prerecorded message, then hang up.
# Usage: place_call.sh <sip-uri>
set -u

TO="$1"
WAV="$GITHUB_WORKSPACE/message-8k.wav"

DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WAV" | cut -d. -f1)"
[ -z "$DUR" ] && DUR=10

mkfifo /tmp/sipcmd
baresip -f ~/.baresip <> /tmp/sipcmd > /tmp/baresip.log 2>&1 &
BARESIP_PID=$!
exec 3<>/tmp/sipcmd

echo "waiting for SIP registration..."
sleep 12
echo "dial $TO" >&3
echo "calling $TO - playing ~${DUR}s message"
sleep $((DUR + 10))
echo "hangup" >&3
sleep 3
kill "$BARESIP_PID" 2>/dev/null || true

echo "--- baresip log (tail) ---"
grep -a -iE 'register|invite|established|bye|error|failed' /tmp/baresip.log | tail -20 || tail -20 /tmp/baresip.log
