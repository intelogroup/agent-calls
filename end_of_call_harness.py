"""Fake-transport harness for end-of-call dry runs (2026-11-02).

Drives deepgram_bridge._run_session with a scripted fake Deepgram
websocket and a fake LiveKit room, so a whole bridge conversation —
farewells included — plays out with NO network, NO LiveKit server,
NO SIP leg and NO Telnyx anything. The human on the line "stays
connected" (caller_gone is never set) unless a scenario sets it:
exactly Jim's reported failure shape, the person said bye but the
call must be ended from the agent side.

Module under test is chosen by EOC_MODULE_DIR (default: this dir),
so the same suite runs against production code and against a
scratch fixed copy:

    EOC_MODULE_DIR=/tmp/eoc-fix <venv>/python test_end_of_call.py
"""
import asyncio
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_DIR = os.environ.get("EOC_MODULE_DIR", _HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _MOD_DIR not in sys.path:
    sys.path.insert(0, _MOD_DIR)  # last insert wins: module under test first


class FakeLocalParticipant:
    def __init__(self, room):
        self._room = room

    async def publish_track(self, track):
        self._room.published.append(track)

    async def unpublish_track(self, sid):
        self._room.unpublished.append(sid)


class FakeRoom:
    """Just the surface _run_session touches, plus a record of any
    teardown verb the code under test invokes on the room."""

    def __init__(self):
        self.local_participant = FakeLocalParticipant(self)
        self.remote_participants = {}
        self.published = []
        self.unpublished = []
        self.disconnect_calls = 0

    async def disconnect(self):
        self.disconnect_calls += 1


class FakeWS:
    """Scripted Deepgram side of the websocket. `script` is a list
    of (delay_s, payload_dict) JSON messages; binary payloads may be
    given as bytes. After the script runs out, the socket stays open
    and silent — the conversation is 'live' until something ends it.
    Everything the bridge sends is recorded in .sent."""

    def __init__(self, script):
        self._script = list(script)
        self.sent = []
        self._block = asyncio.Event()

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._script:
            delay, payload = self._script.pop(0)
            if delay:
                await asyncio.sleep(delay)
            if isinstance(payload, (bytes, bytearray)):
                return bytes(payload)
            return json.dumps(payload)
        await self._block.wait()  # open, silent socket
        raise StopAsyncIteration


class _FakeAudioStream:
    """Stands in for rtc.AudioStream over the caller's track: the
    caller is present but says nothing more (no frames)."""

    def __init__(self, *a, **kw):
        self._block = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._block.wait()
        raise StopAsyncIteration


class _FakeAudioSource:
    """rtc.AudioSource stand-in (avoids real FFI handles in tests)."""

    def __init__(self, *a, **kw):
        self.frames = 0

    async def capture_frame(self, frame):
        self.frames += 1

    def clear_queue(self):
        pass


class _FakeLocalAudioTrack:
    sid = "fake-agent-mic"

    @classmethod
    def create_audio_track(cls, name, source):
        return cls()


def _msg(t, **kw):
    d = {"type": t}
    d.update(kw)
    return d


def settings_applied():
    return _msg("SettingsApplied")


def user_says(text):
    return _msg("ConversationText", role="user", content=text)


def agent_says(text):
    return _msg("ConversationText", role="assistant", content=text)


def function_call(name, fid="fn-1", arguments=None):
    return _msg("FunctionCallRequest",
                functions=[{"id": fid, "name": name,
                            "arguments": arguments or {}}])


async def run_fake_bridge_session(script, *, patience=5.0,
                                  caller_gone_after=None,
                                  mode="untrusted", outbound=True,
                                  consult_enabled=True,
                                  file_consult=None):
    """Run deepgram_bridge._run_session against the fakes.

    Returns a dict:
      outcome: "ended" (session returned on its own) or
               "still_open" (had to be cancelled at `patience`)
      elapsed: seconds the session ran
      sent:    raw messages the bridge sent to Deepgram
      events:  log_event names the bridge emitted
      room:    the FakeRoom (teardown verbs recorded on it)
    """
    import deepgram_bridge as dgb

    room = FakeRoom()
    ws = FakeWS(script)
    caller_gone = asyncio.Event()
    events = []

    async def append_lines(rel, lines, label):
        return True

    def log_event(name, **fields):
        events.append(name)

    if file_consult is None and consult_enabled:
        async def file_consult(question, priority="normal"):
            return "filed", {"request": {"id": "abcdef123456"}}

    orig = (dgb.rtc.AudioStream, dgb.rtc.AudioSource,
            dgb.rtc.LocalAudioTrack)
    dgb.rtc.AudioStream = _FakeAudioStream
    dgb.rtc.AudioSource = _FakeAudioSource
    dgb.rtc.LocalAudioTrack = _FakeLocalAudioTrack
    try:
        task = asyncio.ensure_future(dgb._run_session(
            ws, room, object(),  # caller_track unused: AudioStream faked
            call_id="eoc-probe",
            prompt=dgb.build_prompt("Probe brief.", mode=mode,
                                    outbound=outbound),
            greeting="",
            listen_model="nova-3", think_model="gpt-4o-mini",
            append_lines=append_lines, log_event=log_event,
            caller_gone=caller_gone, stats={},
            file_consult=file_consult,
            consult_enabled=consult_enabled, mode=mode))
        if caller_gone_after is not None:
            async def _gone():
                await asyncio.sleep(caller_gone_after)
                caller_gone.set()
            asyncio.ensure_future(_gone())
        try:
            await asyncio.wait_for(asyncio.shield(task),
                                   timeout=patience)
            outcome, elapsed = "ended", patience
        except asyncio.TimeoutError:
            outcome = "still_open"
            elapsed = patience
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        (dgb.rtc.AudioStream, dgb.rtc.AudioSource,
         dgb.rtc.LocalAudioTrack) = orig
    return {"outcome": outcome, "elapsed": elapsed, "sent": ws.sent,
            "events": events, "room": room}


def function_responses(sent):
    """Decoded FunctionCallResponse messages from a fake session."""
    out = []
    for raw in sent:
        if isinstance(raw, str):
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            if d.get("type") == "FunctionCallResponse":
                out.append(d)
    return out


# The scripted farewells. Each is the Deepgram event stream of a
# conversation that is OVER: the human says their goodbye, the
# agent says its closing line, and the agent's brain attempts the
# end_call function a correctly configured agent would have.
FAREWELL_SCENARIOS = [
    ("callee says 'bye'",
     "bye"),
    ("callee says 'goodbye'",
     "goodbye"),
    ("callee says 'thanks, that's all, bye'",
     "Thanks, that's all I needed. Bye."),
    ("callee says 'ok bye' as a full turn",
     "Ok bye then."),
    ("agent closes, callee agrees and says bye",
     "Nope, that's everything. Bye!"),
    ("callee says 'I have to go'",
     "Sorry, I have to go now."),
]


def farewell_script(user_line, *, closing="Got it — thanks for "
                      "your time. Goodbye!"):
    return [
        (0.05, settings_applied()),
        (0.10, user_says("Hi, just calling about the appointment.")),
        (0.10, agent_says("Of course — it's confirmed for Tuesday.")),
        (0.10, user_says(user_line)),
        (0.10, agent_says(closing)),
        (0.10, function_call("end_call")),
    ]


def silence_script():
    """A completed exchange, then the human goes quiet and stays
    on the line: no farewell, no hangup, no further events."""
    return [
        (0.05, settings_applied()),
        (0.10, user_says("Is the appointment confirmed?")),
        (0.10, agent_says("Yes, Tuesday at eleven. Anything else?")),
    ]
