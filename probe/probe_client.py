#!/usr/bin/env python3
"""PROBE (2026-10-15): synthetic caller for the DEPLOYED jett-voice worker.

Runs on the Oracle VM (self-hosted runner), same host as the LiveKit
server. Creates a room with the agent dispatch embedded (the proven
outbound-proof shape) but NEVER creates a SIP participant: the "callee"
is this script joining over WebRTC and playing a known line. Job
metadata marks it outbound so the Deepgram bridge is skipped and the
LOCAL pipeline (Phonon-2 STT + Pocket TTS) serves the call — the stack
under test. No PSTN, no real phone number is dialed: metadata "to" is
+10000000000 and nothing ever dials it.

Prints one PROBE_RESULT {json} line with timing + RMS receipts.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import math
import subprocess
import sys
import time
import urllib.request
import wave

sys.path.insert(0, "/tmp/probe-libs")
from livekit import rtc  # noqa: E402

WAV_PATH = sys.argv[1] if len(sys.argv) > 1 else "caller.wav"
KNOWN_LINE = ("Hello, this is a test call from the probe. "
              "Can you hear me clearly now?")
LOUD_RMS = 0.004


def parse_livekit_yaml():
    raw = subprocess.run(["sudo", "cat", "/opt/agent/livekit.yaml"],
                         capture_output=True, text=True).stdout
    api_key = api_secret = None
    port = 7880
    in_keys = False
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("port:") and not ln.startswith((" ", "\t")):
            try:
                port = int(s.split(":", 1)[1].strip())
            except Exception:
                pass
        if s == "keys:":
            in_keys = True
            continue
        if in_keys:
            if ln.startswith((" ", "\t")) and ":" in s \
                    and not s.startswith("#"):
                k, v = s.split(":", 1)
                api_key, api_secret = k.strip(), v.strip()
                in_keys = False
            elif s:
                in_keys = False
    assert api_key and api_secret, "livekit keys not found"
    return api_key, api_secret, port


def b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def make_jwt(api_key, api_secret, claims):
    header = {"alg": "HS256", "typ": "JWT"}
    msg = b64(json.dumps(header).encode()) + "." \
        + b64(json.dumps(claims).encode())
    sig = hmac.new(api_secret.encode(), msg.encode(),
                   hashlib.sha256).digest()
    return msg + "." + b64(sig)


def twirp(port, jwt, service, method, body, timeout=60):
    url = "http://127.0.0.1:%d/twirp/%s/%s" % (port, service, method)
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + jwt)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def frame_rms(data):
    # livekit frames arrive as an int16 memoryview already; bytes()
    # normalizes any memoryview format to raw bytes first.
    mv = memoryview(bytes(data)).cast("h")
    if not mv:
        return 0.0
    acc = 0.0
    for v in mv:
        acc += v * v
    return math.sqrt(acc / len(mv)) / 32768.0


async def main():
    api_key, api_secret, port = parse_livekit_yaml()
    now = int(time.time())
    room_name = "jett-probe-%d" % now
    meta = json.dumps({
        "direction": "outbound",
        "to": "+10000000000",
        "brief": ("PROBE TEST, not a real call. The person who answers "
                  "is an automated probe playing a test message. "
                  "Greet them, listen to the message, confirm you "
                  "heard it clearly, and keep it short."),
    })
    admin_jwt = make_jwt(api_key, api_secret, {
        "iss": api_key, "sub": "probe-admin", "nbf": now - 10,
        "exp": now + 3600,
        "video": {"roomCreate": True, "roomAdmin": True,
                  "room": room_name}})
    r = twirp(port, admin_jwt, "livekit.RoomService", "CreateRoom",
              {"name": room_name,
               "agents": [{"agentName": "jett-proxy", "metadata": meta}]})
    print("room created:", r.get("name") or room_name, flush=True)

    join_jwt = make_jwt(api_key, api_secret, {
        "iss": api_key, "sub": "tester", "name": "Probe Tester",
        "nbf": now - 10, "exp": now + 900,
        "video": {"roomJoin": True, "room": room_name,
                  "canPublish": True, "canSubscribe": True,
                  "canPublishData": True}})

    frames = []          # (t_rel, rms, raw_bytes, sample_rate)
    state = {"agent_seen": False, "sr": 48000}
    t0 = time.monotonic()

    def rel():
        return time.monotonic() - t0

    room = rtc.Room()

    async def consume(track):
        stream = rtc.AudioStream(track)
        async for ev in stream:
            f = ev.frame
            state["sr"] = f.sample_rate
            frames.append((rel(), frame_rms(f.data),
                           bytes(f.data), f.sample_rate))

    @room.on("track_subscribed")
    def _on_sub(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            state["agent_seen"] = True
            print("agent audio track subscribed at %.2fs from %s"
                  % (rel(), participant.identity), flush=True)
            asyncio.ensure_future(consume(track))

    await room.connect("ws://127.0.0.1:%d" % port, join_jwt)
    print("tester connected at %.2fs" % rel(), flush=True)

    # Publish the mic track IMMEDIATELY (silence first), like a real
    # answered phone leg whose media exists before the greeting:
    # runs 1-2 published only after the greeting and the session's
    # input pipeline never bound the late track for turn purposes.
    wf0 = wave.open(WAV_PATH, "rb")
    SR_IN = wf0.getframerate()
    RAW = wf0.readframes(wf0.getnframes())
    wf0.close()
    CHUNK = SR_IN // 10  # 100 ms of int16 mono
    pending = bytearray()
    source = rtc.AudioSource(SR_IN, 1)
    mic_track = rtc.LocalAudioTrack.create_audio_track("caller", source)
    await room.local_participant.publish_track(
        mic_track, rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE))
    print("mic track published at %.2fs (sr=%d)" % (rel(), SR_IN),
          flush=True)

    async def feeder():
        zeros = b"\x00" * (CHUNK * 2)
        try:
            while True:
                if len(pending) >= CHUNK * 2:
                    data = bytes(pending[:CHUNK * 2])
                    del pending[:CHUNK * 2]
                else:
                    data = zeros
                await source.capture_frame(rtc.AudioFrame(
                    data=data, sample_rate=SR_IN, num_channels=1,
                    samples_per_channel=CHUNK))
                await asyncio.sleep(CHUNK / SR_IN)
        except asyncio.CancelledError:
            pass

    feeder_task = asyncio.ensure_future(feeder())

    def loud_frames(after=0.0, before=None):
        return [t for (t, rms, _d, _s) in frames
                if rms > LOUD_RMS and t >= after
                and (before is None or t <= before)]

    async def wait_loud(after, timeout):
        start = rel()
        while rel() - start < timeout:
            hits = loud_frames(after=after)
            if hits:
                return hits[0]
            await asyncio.sleep(0.1)
        return None

    async def wait_quiet(after, quiet_s, timeout):
        start = rel()
        while rel() - start < timeout:
            hits = loud_frames(after=after)
            if hits and rel() - hits[-1] >= quiet_s:
                return hits[-1]
            if not hits and rel() - start >= quiet_s:
                return None
            await asyncio.sleep(0.1)
        hits = loud_frames(after=after)
        return hits[-1] if hits else None

    # --- greeting -------------------------------------------------
    t_greet_first = await wait_loud(0.0, 45)
    t_greet_last = None
    if t_greet_first is not None:
        t_greet_last = await wait_quiet(0.0, 2.0, 30)
    print("greeting first audio:", t_greet_first, "last:", t_greet_last,
          flush=True)

    # --- speak the known line (feeder plays it, then silence) -------
    t_pub_start = rel()
    pending.extend(RAW)
    dur = len(RAW) // 2 / SR_IN
    await asyncio.sleep(dur + 0.3)
    t_pub_end = rel()
    print("caller line spoken %.2fs..%.2fs" % (t_pub_start, t_pub_end),
          flush=True)

    # --- reply ------------------------------------------------------
    t_reply_first = await wait_loud(t_pub_end, 45)
    t_reply_last = None
    if t_reply_first is not None:
        t_reply_last = await wait_quiet(t_pub_end, 3.0, 40)
    print("reply first audio:", t_reply_first, "last:", t_reply_last,
          flush=True)

    def seg_rms(lo, hi):
        sel = [d for (t, _r, d, _s) in frames if lo <= t <= hi]
        if not sel:
            return 0.0, 0
        blob = b"".join(sel)
        return frame_rms(blob), len(blob) // 2

    g_rms, g_n = seg_rms(0.0, t_pub_start)
    r_rms, r_n = seg_rms(t_pub_end, rel())
    # save the agent audio for the record
    try:
        out = wave.open("agent_audio.wav", "wb")
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(state["sr"])
        out.writeframes(b"".join(d for (_t, _r, d, _s) in frames))
        out.close()
    except Exception as e:
        print("agent audio save failed:", e, flush=True)

    result = {
        "room": room_name,
        "agent_track_seen": state["agent_seen"],
        "greeting_first_audio_s": t_greet_first,
        "greeting_last_audio_s": t_greet_last,
        "greeting_rms": round(g_rms, 4),
        "caller_line": KNOWN_LINE,
        "publish_start_s": round(t_pub_start, 2),
        "publish_end_s": round(t_pub_end, 2),
        "reply_first_audio_s": t_reply_first,
        "reply_after_line_end_s": (round(t_reply_first - t_pub_end, 2)
                                   if t_reply_first is not None
                                   else None),
        "reply_last_audio_s": t_reply_last,
        "reply_rms": round(r_rms, 4),
        "frames_recorded": len(frames),
    }
    print("PROBE_RESULT " + json.dumps(result), flush=True)
    try:
        await room.disconnect()
    except Exception:
        pass
    ok = (state["agent_seen"] and t_greet_first is not None
          and g_rms > 0.01 and t_reply_first is not None
          and r_rms > 0.01)
    print("PROBE_CLIENT_VERDICT " + ("PASS" if ok else "FAIL"),
          flush=True)
    feeder_task.cancel()
    return 0 if ok else 2


sys.exit(asyncio.run(main()))
