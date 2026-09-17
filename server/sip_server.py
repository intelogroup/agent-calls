#!/usr/bin/env python3
"""
Inbound SIP voice agent — runs 24/7 on a server, answers calls, holds a
spoken conversation.

Flow per call:
  INVITE -> 100/180/200 -> ACK -> voice loop -> BYE
  voice loop: greet -> listen (RTP/PCMU in) -> STT (faster-whisper)
              -> LLM (llama.cpp) -> TTS (kokoro, espeak-ng fallback)
              -> speak (RTP/PCMU out) -> repeat

Config via environment (see agent.env.example). Only one concurrent call;
a second caller gets 486 Busy.
"""
import hashlib
import os
import re
import select
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

# ---------------------------------------------------------------- config

SIP_USER = os.environ["AGENT_SIP_USER"]
SIP_PASS = os.environ["AGENT_SIP_PASS"]
SIP_DOMAIN = os.environ.get("AGENT_SIP_DOMAIN", "sip.linphone.org")
SIP_PORT = int(os.environ.get("AGENT_SIP_PORT", "5060"))
AGENT_NAME = os.environ.get("AGENT_NAME", "Jett")
RTP_PORT = int(os.environ.get("AGENT_RTP_PORT", "10000"))
PUBLIC_IP = os.environ.get("AGENT_PUBLIC_IP", "")  # set on the server; else local IP
WHISPER_MODEL = os.environ.get("AGENT_WHISPER_MODEL", "tiny.en")
LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "/opt/agent/models/llm.gguf")
TTS_ENGINE = os.environ.get("AGENT_TTS_ENGINE", "kokoro")  # kokoro | espeak
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "8"))
MAX_CALL_SEC = int(os.environ.get("AGENT_MAX_CALL_SEC", "300"))
GREETING = os.environ.get("AGENT_GREETING",
                          f"Hey, it's {AGENT_NAME}. What do you want to talk about?")
MISSED = "Sorry, I didn't catch that. Could you say it again?"
GOODBYE = f"Alright, talk later. This was {AGENT_NAME}."

LOCAL_IP = PUBLIC_IP or socket.gethostbyname(socket.gethostname())
SIP_URI = f"sip:{SIP_USER}@{SIP_DOMAIN}"
FROM_URI = f"sip:{SIP_USER}@{SIP_DOMAIN}"

# ---------------------------------------------------------------- logging

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------------------------------------------------------- SIP utils

def parse_status(line):
    m = re.match(r"SIP/2\.0\s+(\d+)", line)
    return int(m.group(1)) if m else None

def parse_headers(lines):
    h = {}
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1)
            h[k.strip().lower()] = v.strip()
    return h

def read_sip_message(sock_file):
    """Read one SIP message framed by Content-Length. Returns (start_line, headers, body)."""
    start = sock_file.readline().decode("utf-8", "replace").strip()
    if not start:
        return None
    lines = []
    while True:
        ln = sock_file.readline().decode("utf-8", "replace")
        if ln in ("\r\n", "\n", ""):
            break
        lines.append(ln.strip())
    headers = parse_headers(lines)
    body = b""
    n = int(headers.get("content-length", "0") or 0)
    while len(body) < n:
        chunk = sock_file.read(n - len(body))
        if not chunk:
            break
        body += chunk
    return start, headers, body

def digest_response(user, password, realm, nonce, method, uri, qop=None,
                    nc="00000001", cnonce="abcdef"):
    ha1 = hashlib.md5(f"{user}:{realm}:{password}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    if qop:
        return hashlib.md5(
            f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()).hexdigest()
    return hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()

def parse_www_authenticate(value):
    parts = {}
    for m in re.finditer(r'(\w+)=(?:"([^"]*)"|([^\s,]+))', value):
        parts[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    return parts

BRANCH = "z9hG4bKsrv"
CALL_ID_REG = f"reg-{int(time.time())}@agent"

def build_register(cseq, auth=None):
    contact = f"<sip:{SIP_USER}@{LOCAL_IP}:{SIP_PORT};transport=tcp>"
    lines = [
        f"REGISTER sip:{SIP_DOMAIN} SIP/2.0",
        f"Via: SIP/2.0/TCP {LOCAL_IP}:{SIP_PORT};branch={BRANCH}{cseq};rport",
        "Max-Forwards: 70",
        f"From: \"{AGENT_NAME}\" <{FROM_URI}>;tag=agtag1",
        f"To: <{FROM_URI}>",
        f"Call-ID: {CALL_ID_REG}",
        f"CSeq: {cseq} REGISTER",
        f"Contact: {contact}",
        "Expires: 600",
    ]
    if auth:
        lines.append(f"Authorization: {auth}")
    lines += ["Content-Length: 0", "", ""]
    return "\r\n".join(lines)

def build_auth_header(challenge, method, uri):
    realm = challenge.get("realm", SIP_DOMAIN)
    nonce = challenge.get("nonce", "")
    qop = challenge.get("qop", "")
    qop = qop.split(",")[0].strip() if qop else ""
    resp = digest_response(SIP_USER, SIP_PASS, realm, nonce, method, uri,
                           qop if qop else None)
    h = (f'Digest username="{SIP_USER}", realm="{realm}", nonce="{nonce}", '
         f'uri="{uri}", response="{resp}", algorithm=MD5')
    if qop:
        h += f', qop={qop}, nc=00000001, cnonce=abcdef'
    opaque = challenge.get("opaque")
    if opaque:
        h += f', opaque="{opaque}"'
    return h

def build_sdp():
    return (
        "v=0\r\n"
        f"o={SIP_USER} 1 1 IN IP4 {LOCAL_IP}\r\n"
        "s=voice-agent\r\n"
        f"c=IN IP4 {LOCAL_IP}\r\n"
        "t=0 0\r\n"
        f"m=audio {RTP_PORT} RTP/AVP 0\r\n"
        "a=rtpmap:0 PCMU/8000\r\n"
        "a=sendrecv\r\n"
    )

def build_response(request_start, headers, code, reason, extra=(), body=""):
    method = request_start.split()[0]
    via = headers.get("via", "")
    from_h = headers.get("from", "")
    to_h = headers.get("to", "")
    call_id = headers.get("call-id", "")
    cseq = headers.get("cseq", "")
    if "tag=" not in to_h:
        to_h = to_h + ";tag=agto1"
    lines = [
        f"SIP/2.0 {code} {reason}",
        f"Via: {via}",
        f"From: {from_h}",
        f"To: {to_h}",
        f"Call-ID: {call_id}",
        f"CSeq: {cseq}",
    ]
    for k, v in extra:
        lines.append(f"{k}: {v}")
    if body:
        lines.append("Content-Type: application/sdp")
        lines.append(f"Content-Length: {len(body)}")
    else:
        lines.append("Content-Length: 0")
    lines += ["", ""]
    msg = "\r\n".join(lines)
    if body:
        msg += body
    return msg

# ---------------------------------------------------------------- RTP / audio

def pcmu_decode(data):
    out = np.empty(len(data), dtype=np.int16)
    for i, b in enumerate(data):
        u = (~b) & 0xFF
        val = ((u & 0x0F) << 3) + 0x84
        val <<= (u & 0x70) >> 4
        out[i] = (0x84 - val) if (u & 0x80) else (val - 0x84)
    return out

def pcmu_encode(pcm):
    BIAS = 0x84
    out = bytearray(len(pcm))
    for i, s in enumerate(pcm):
        s = int(s)
        sign = 0x80 if s < 0 else 0
        if s < 0:
            s = -s
        if s > 32635:
            s = 32635
        s += BIAS
        exp = 7
        for e in range(7, -1, -1):
            if s >= (1 << (e + 4)):
                exp = e
                break
        mant = (s >> (exp + 3)) & 0x0F
        out[i] = (~(sign | (exp << 4) | mant)) & 0xFF
    return bytes(out)

class RtpEndpoint:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.seq = 1
        self.ts = 0
        self.ssrc = 0xA6E77A11

    def send_pcm(self, pcm8k, dest):
        for off in range(0, len(pcm8k), 160):
            frame = pcm8k[off:off + 160]
            if len(frame) < 160:
                frame = np.pad(frame, (0, 160 - len(frame)))
            payload = pcmu_encode(frame.astype(np.int16))
            hdr = struct.pack(">BBHII", 0x80, 0, self.seq, self.ts, self.ssrc)
            try:
                self.sock.sendto(hdr + payload, dest)
            except OSError:
                pass
            self.seq = (self.seq + 1) & 0xFFFF
            self.ts = (self.ts + 160) & 0xFFFFFFFF
            time.sleep(0.02)

    def recv_frame(self, timeout=0.05):
        r, _, _ = select.select([self.sock], [], [], timeout)
        if not r:
            return None, None
        try:
            data, addr = self.sock.recvfrom(2048)
        except OSError:
            return None, None
        if len(data) < 12:
            return None, None
        return pcmu_decode(data[12:]), addr

    def drain(self):
        while True:
            pcm, _ = self.recv_frame(0)
            if pcm is None:
                break

# ---------------------------------------------------------------- speech: STT / LLM / TTS

_whisper = None
_llm = None
_kokoro = None

def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        log("loading whisper", WHISPER_MODEL)
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper

def transcribe(pcm8k):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(pcm8k.astype(np.int16).tobytes())
    try:
        segs, _ = get_whisper().transcribe(path, language="en")
        return " ".join(s.text for s in segs).strip()
    finally:
        os.unlink(path)

def get_llm():
    global _llm
    if _llm is None:
        if not os.path.exists(LLM_MODEL):
            log("LLM model missing at", LLM_MODEL, "- using fallback replies")
            return None
        from llama_cpp import Llama
        log("loading LLM", LLM_MODEL)
        _llm = Llama(model_path=LLM_MODEL, n_ctx=1024, n_threads=4,
                     verbose=False)
    return _llm

SYSTEM_PROMPT = (
    f"You are {AGENT_NAME}, a friendly voice assistant talking on a phone call. "
    "Reply in at most two short spoken sentences. No lists, no markdown, "
    "no emojis. Sound natural and conversational."
)

def think(user_text, history):
    llm = get_llm()
    if llm is None:
        return f"I heard you say: {user_text}. My thinking brain isn't loaded yet, but I'm listening."
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += history[-6:]
    msgs.append({"role": "user", "content": user_text})
    out = llm.create_chat_completion(messages=msgs, max_tokens=90,
                                     temperature=0.7)
    reply = out["choices"][0]["message"]["content"].strip()
    return reply[:400]

def get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro import KPipeline
        log("loading kokoro (slow, one time)")
        _kokoro = KPipeline(lang_code="a")
    return _kokoro

def tts_kokoro(text):
    import torch, soundfile as sf
    chunks = []
    for _, _, audio in get_kokoro()(text, voice="af_heart"):
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("kokoro produced no audio")
    a24 = torch.cat(chunks, dim=0).numpy()
    a8 = a24[::3]  # 24 kHz -> 8 kHz
    peak = np.max(np.abs(a8)) or 1.0
    return (a8 / peak * 30000).astype(np.int16)

def tts_espeak(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        subprocess.run(["espeak-ng", "--stdout", "-v", "en", "-s", "175",
                        text], stdout=open(path, "wb"), check=True,
                       timeout=30)
        import soundfile as sf
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        idx = (np.arange(int(len(audio) * 8000 / sr)) * sr / 8000).astype(int)
        idx = np.clip(idx, 0, len(audio) - 1)
        a8 = audio[idx]
        peak = np.max(np.abs(a8)) or 1.0
        return (a8 / peak * 30000).astype(np.int16)
    finally:
        os.unlink(path)

def synthesize(text):
    """TTS with fallback: kokoro -> espeak-ng."""
    if TTS_ENGINE == "kokoro":
        try:
            return tts_kokoro(text)
        except Exception as e:
            log("kokoro failed, falling back to espeak-ng:", e)
    return tts_espeak(text)

# ---------------------------------------------------------------- call handling

class Call:
    def __init__(self, invite_start, headers, body):
        self.call_id = headers.get("call-id", "")
        self.from_h = headers.get("from", "")
        self.to_h = headers.get("to", "")
        self.via = headers.get("via", "")
        self.cseq = int(headers.get("cseq", "1 INVITE").split()[0])
        self.remote_rtp = self._parse_sdp(body)
        self.ended = threading.Event()
        self.ack = threading.Event()
        self.rtp = RtpEndpoint(RTP_PORT)

    def _parse_sdp(self, body):
        txt = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
        ip, port = None, None
        for ln in txt.splitlines():
            ln = ln.strip()
            if ln.startswith("c=IN IP4"):
                ip = ln.split()[-1]
            elif ln.startswith("m=audio"):
                port = int(ln.split()[1])
        return (ip, port or 0)

class SipAgent:
    def __init__(self):
        self.sock = None
        self.sock_file = None
        self.lock = threading.Lock()
        self.active_call = None
        self.registered = threading.Event()
        self._stop = False

    # -- connection -------------------------------------------------
    def connect(self):
        s = socket.create_connection((SIP_DOMAIN, SIP_PORT), timeout=15)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        self.sock_file = s.makefile("rb")
        log("TCP connected to", SIP_DOMAIN, SIP_PORT)

    def send(self, msg):
        with self.lock:
            self.sock.sendall(msg.encode("utf-8"))

    # -- registration -----------------------------------------------
    def do_register(self):
        cseq = 1
        self.send(build_register(cseq))
        while True:
            start, headers, _ = read_sip_message(self.sock_file)
            if start is None:
                raise ConnectionError("server closed during REGISTER")
            code = parse_status(start)
            if code == 401 and "www-authenticate" in headers:
                ch = parse_www_authenticate(headers["www-authenticate"])
                auth = build_auth_header(ch, "REGISTER", f"sip:{SIP_DOMAIN}")
                cseq += 1
                self.send(build_register(cseq, auth))
                continue
            if code == 200:
                log("REGISTER_OK")
                self.registered.set()
                return
            if code == 403:
                log("REGISTER_FAILED 403 — check AGENT_SIP_USER/AGENT_SIP_PASS")
                return
            log("REGISTER unexpected:", start)

    def register_refresher(self):
        while not self._stop:
            time.sleep(540)  # refresh before the 600s expiry
            if self._stop:
                break
            try:
                self.registered.clear()
                self.do_register()
            except Exception as e:
                log("re-register failed:", e)
                self.reconnect()

    def reconnect(self):
        log("reconnecting...")
        try:
            self.sock.close()
        except Exception:
            pass
        time.sleep(5)
        self.connect()
        self.do_register()

    # -- reader ------------------------------------------------------
    def reader(self):
        while not self._stop:
            try:
                msg = read_sip_message(self.sock_file)
            except Exception as e:
                log("reader error:", e)
                self.reconnect()
                continue
            if msg is None:
                log("server closed connection")
                self.reconnect()
                continue
            start, headers, body = msg
            code = parse_status(start)
            if code is not None:
                continue  # responses to our requests are handled inline
            method = start.split()[0]
            if method == "INVITE":
                threading.Thread(target=self.handle_invite,
                                 args=(start, headers, body),
                                 daemon=True).start()
            elif method == "ACK":
                call = self.active_call
                if call and headers.get("call-id") == call.call_id:
                    call.ack.set()
            elif method == "BYE":
                call = self.active_call
                self.send(build_response(start, headers, 200, "OK"))
                if call and headers.get("call-id") == call.call_id:
                    log("remote BYE")
                    call.ended.set()
            elif method == "CANCEL":
                self.send(build_response(start, headers, 200, "OK"))
            elif method == "OPTIONS":
                self.send(build_response(start, headers, 200, "OK"))
            else:
                log("unhandled method:", method)

    # -- inbound call -------------------------------------------------
    def handle_invite(self, start, headers, body):
        if self.active_call is not None:
            self.send(build_response(start, headers, 486, "Busy Here"))
            log("rejected second call with 486")
            return
        call = Call(start, headers, body)
        self.active_call = call
        log("incoming INVITE from", headers.get("from", "?")[:60])
        self.send(build_response(start, headers, 100, "Trying"))
        time.sleep(0.4)
        self.send(build_response(start, headers, 180, "Ringing"))
        time.sleep(0.8)
        sdp = build_sdp()
        self.send(build_response(start, headers, 200, "OK",
                                 extra=[("Contact",
                                         f"<sip:{SIP_USER}@{LOCAL_IP}:{SIP_PORT};transport=tcp>")],
                                 body=sdp))
        if not call.ack.wait(timeout=12):
            log("no ACK, dropping call")
            self.active_call = None
            return
        log("CALL_ANSWERED")
        try:
            self.voice_loop(call, headers)
        except Exception as e:
            log("voice loop error:", e)
        finally:
            self.end_call(call, headers)

    def end_call(self, call, headers):
        if self.active_call is call:
            self.active_call = None
        if not call.ended.is_set():
            bye = self.build_bye(call, headers)
            try:
                self.send(bye)
                log("BYE_SENT")
            except Exception:
                pass
        call.rtp.sock.close()

    def build_bye(self, call, headers):
        cseq = call.cseq + 10
        lines = [
            f"BYE {FROM_URI} SIP/2.0",
            f"Via: SIP/2.0/TCP {LOCAL_IP}:{SIP_PORT};branch={BRANCH}bye;rport",
            "Max-Forwards: 70",
            f"From: \"{AGENT_NAME}\" <{FROM_URI}>;tag=agto1",
            f"To: {call.from_h}",
            f"Call-ID: {call.call_id}",
            f"CSeq: {cseq} BYE",
            "Content-Length: 0",
            "", "",
        ]
        return "\r\n".join(lines)

    # -- voice loop ----------------------------------------------------
    def record_utterance(self, call):
        """Record until 1.4s silence after speech (or limits). Returns int16 @8k or None."""
        call.rtp.drain()
        frames = []
        speech_seen = False
        silence_start = None
        start_t = time.time()
        first_speech_deadline = start_t + 9
        while not call.ended.is_set():
            pcm, _ = call.rtp.recv_frame(0.05)
            now = time.time()
            if now - start_t > 15:
                break
            if pcm is None:
                if speech_seen and silence_start and now - silence_start > 1.4:
                    break
                if not speech_seen and now > first_speech_deadline:
                    return None
                continue
            energy = float(np.mean(np.abs(pcm.astype(np.float32))))
            if energy > 350:  # speech
                speech_seen = True
                silence_start = None
            elif speech_seen:
                if silence_start is None:
                    silence_start = now
            frames.append(pcm)
            if speech_seen and silence_start and now - silence_start > 1.4:
                break
        if not speech_seen or call.ended.is_set():
            return None
        audio = np.concatenate(frames)
        return audio[-15 * 8000:]  # cap 15s

    def voice_loop(self, call, headers):
        dest = call.remote_rtp
        if not dest[0] or not dest[1]:
            log("no remote RTP in SDP, hanging up")
            return
        log("remote RTP:", dest)
        history = []
        self.speak(call, GREETING)
        t0 = time.time()
        empties = 0
        for turn in range(MAX_TURNS):
            if call.ended.is_set() or time.time() - t0 > MAX_CALL_SEC:
                break
            log(f"--- turn {turn + 1}: listening")
            pcm = self.record_utterance(call)
            if pcm is None:
                empties += 1
                if empties >= 2:
                    break
                self.speak(call, MISSED)
                continue
            empties = 0
            text = transcribe(pcm)
            log("heard:", text[:120])
            if not text:
                self.speak(call, MISSED)
                continue
            reply = think(text, history)
            log("reply:", reply[:120])
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": reply})
            self.speak(call, reply)
        self.speak(call, GOODBYE)

    def speak(self, call, text):
        if call.ended.is_set() or not text.strip():
            return
        try:
            pcm = synthesize(text)
        except Exception as e:
            log("TTS failed:", e)
            return
        log(f"speaking {len(pcm) / 8000:.1f}s")
        call.rtp.drain()
        call.rtp.send_pcm(pcm, call.remote_rtp)

    # -- main -----------------------------------------------------------
    def run(self):
        self.connect()
        self.do_register()
        if not self.registered.is_set():
            log("registration failed, exiting")
            sys.exit(1)
        threading.Thread(target=self.register_refresher, daemon=True).start()
        # warm up speech models in background (slow on first load)
        threading.Thread(target=self.warmup, daemon=True).start()
        log("agent live — waiting for calls")
        self.reader()

    def warmup(self):
        try:
            get_whisper()
            log("whisper ready")
        except Exception as e:
            log("whisper warmup failed:", e)
        try:
            if TTS_ENGINE == "kokoro":
                get_kokoro()
                log("kokoro ready")
        except Exception as e:
            log("kokoro warmup failed:", e)

if __name__ == "__main__":
    for v in ("AGENT_SIP_USER", "AGENT_SIP_PASS"):
        if v not in os.environ:
            sys.exit(f"missing env {v}")
    SipAgent().run()
