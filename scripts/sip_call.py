#!/usr/bin/env python3
"""Minimal SIP caller over TCP: STUN for public IP, INVITE (digest auth), RTP playback of a wav as PCMU, BYE.

Usage: sip_call.py <user> <dest_uri> <wav_path>
Password via SIP_CALL_PASS env (never logged). Prints CALL_ESTABLISHED on success.
"""
import hashlib
import os
import queue
import random
import re
import select
import socket
import string
import struct
import sys
import threading
import time
import wave

DOMAIN = "sip.linphone.org"
USER = sys.argv[1]
DEST = sys.argv[2]
WAV_PATH = sys.argv[3]
PASS = os.environ["SIP_CALL_PASS"]
DEBUG = os.environ.get("DEBUG_CALL", "0") == "1"

LOCAL_IP = "10.0.0.1"  # placeholder; server learns real IP via rport/received
PUBLIC_IP = None  # filled via STUN (no REGISTER: avoids self-fork loop-back)


def rnd(n):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def dbg(*a):
    if DEBUG:
        print("DBG:", *a, flush=True)


# ---------------------------------------------------------------- SIP transport
class SipStack:
    def __init__(self):
        self.sock = socket.create_connection((DOMAIN, 5060), timeout=10)
        self.buf = b""
        self.inbox = queue.Queue()
        self.running = True
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()
        self.cseq = 0

    def _read_loop(self):
        try:
            while self.running:
                r, _, _ = select.select([self.sock], [], [], 1.0)
                if not r:
                    continue
                data = self.sock.recv(65535)
                if not data:
                    break
                self.buf += data
                while True:
                    msg = self._try_parse()
                    if msg is None:
                        break
                    self.inbox.put(msg)
        except Exception as e:  # noqa: BLE001
            dbg("reader error:", e)

    def _try_parse(self):
        head_end = self.buf.find(b"\r\n\r\n")
        if head_end < 0:
            return None
        head = self.buf[:head_end].decode(errors="replace")
        lines = head.split("\r\n")
        first = lines[0]
        headers = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        clen = int(headers.get("content-length", "0") or "0")
        total = head_end + 4 + clen
        if len(self.buf) < total:
            return None
        body = self.buf[head_end + 4: total].decode(errors="replace")
        self.buf = self.buf[total:]
        return (first, headers, body)

    def send(self, text):
        dbg("SEND:\n" + text)
        self.sock.sendall(text.encode())

    def next_msg(self, timeout):
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------- auth
def parse_challenge(msg):
    _, headers, _ = msg
    for key in ("www-authenticate", "proxy-authenticate"):
        val = headers.get(key)
        if val and val.lower().startswith("digest"):
            params = dict(re.findall(r'(\w+)="([^"]*)"', val))
            params["_hdr"] = ("Authorization" if key == "www-authenticate"
                              else "Proxy-Authorization")
            return params
    return None


def auth_header(params, user, password, method, uri):
    realm, nonce = params["realm"], params["nonce"]
    qop = params.get("qop", "auth").split(",")[0].strip()
    nc, cnonce = "00000001", rnd(8)
    ha1 = md5(f"{user}:{realm}:{password}")
    ha2 = md5(f"{method}:{uri}")
    resp = md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")
    hdr = params["_hdr"]
    parts = (f'{hdr}: Digest username="{user}", realm="{realm}", nonce="{nonce}", '
             f'uri="{uri}", response="{resp}", algorithm=MD5, '
             f'cnonce="{cnonce}", opaque="{params.get("opaque", "")}", '
             f'qop={qop}, nc={nc}')
    return parts


# ---------------------------------------------------------------- requests
stack = SipStack()
from_tag = rnd(8)


def request(method, uri, headers_extra="", body="", auth_params=None, to_tag=None,
            to_uri=None):
    stack.cseq += 1
    branch = "z9hG4bK" + rnd(12)
    call_id = request.call_id
    to_line = f"To: <{to_uri or uri}>" + (f";tag={to_tag}" if to_tag else "")
    msg = (f"{method} {uri} SIP/2.0\r\n"
           f"Via: SIP/2.0/TCP {LOCAL_IP}:5060;branch={branch};rport\r\n"
           f"Max-Forwards: 70\r\n"
           f"From: <sip:{USER}@{DOMAIN}>;tag={from_tag}\r\n"
           f"{to_line}\r\n"
           f"Call-ID: {call_id}\r\n"
           f"CSeq: {stack.cseq} {method}\r\n"
           f"Contact: <sip:{USER}@{LOCAL_IP}:5060;transport=tcp>\r\n")
    if auth_params:
        msg += auth_header(auth_params, USER, PASS, method, uri) + "\r\n"
    if headers_extra:
        msg += headers_extra
    if body:
        msg += f"Content-Length: {len(body.encode())}\r\nContent-Type: application/sdp\r\n"
    else:
        msg += "Content-Length: 0\r\n"
    msg += "\r\n" + body
    stack.send(msg)
    return stack.cseq


request.call_id = rnd(16)


def wait_response(cseq_expected, timeout=15):
    """Wait for a SIP response; auto-answer incoming requests meanwhile."""
    end = time.time() + timeout
    while time.time() < end:
        msg = stack.next_msg(timeout=max(0.1, end - time.time()))
        if msg is None:
            continue
        first, headers, body = msg
        dbg("RECV:", first)
        if first.startswith("INVITE "):
            # forked copy of our own INVITE coming back; decline it
            stack.send(f"SIP/2.0 486 Busy Here\r\n"
                       f"To: {headers.get('to', '')};tag={rnd(6)}\r\n"
                       f"From: {headers.get('from', '')}\r\n"
                       f"Call-ID: {headers.get('call-id', '')}\r\n"
                       f"CSeq: {headers.get('cseq', '')}\r\n"
                       f"Via: {headers.get('via', '')}\r\n"
                       f"Content-Length: 0\r\n\r\n")
            continue
        if first.startswith("OPTIONS "):
            stack.send(f"SIP/2.0 200 OK\r\n"
                       f"To: {headers.get('to', '')};tag={rnd(6)}\r\n"
                       f"From: {headers.get('from', '')}\r\n"
                       f"Call-ID: {headers.get('call-id', '')}\r\n"
                       f"CSeq: {headers.get('cseq', '')}\r\n"
                       f"Via: {headers.get('via', '')}\r\n"
                       f"Content-Length: 0\r\n\r\n")
            continue
        if first.startswith("SIP/2.0"):
            return msg
    return None


def stun_public_ip(server="stun.l.google.com", port=19302, timeout=5):
    """RFC 5389 binding request: learn our reflexive IPv4 without REGISTERing.

    We deliberately do NOT REGISTER with the SIP server: registering the
    caller under the callee's address-of-record makes the proxy fork our own
    INVITE back to us (486 loop-back noise). STUN gives us the public IP for
    the SDP offer without creating a server-side contact binding. The
    outbound INVITE is still digest-authenticated (407 challenge), which the
    proxy accepts without a prior REGISTER.
    """
    tid = bytes(random.getrandbits(8) for _ in range(12))
    req = b"\x00\x01\x00\x00\x21\x12\xa4\x42" + tid
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(req, (server, port))
        data, _ = s.recvfrom(2048)
    finally:
        s.close()
    if len(data) < 20 or data[0:2] != b"\x01\x01":
        raise RuntimeError("bad STUN response")
    if data[8:20] != tid:
        raise RuntimeError("STUN transaction id mismatch")
    off = 20
    while off + 4 <= len(data):
        atype = int.from_bytes(data[off:off + 2], "big")
        alen = int.from_bytes(data[off + 2:off + 4], "big")
        val = data[off + 4:off + 4 + alen]
        # XOR-MAPPED-ADDRESS (0x0020), IPv4 family (0x01)
        if atype == 0x0020 and len(val) >= 8 and val[1] == 0x01:
            xaddr = int.from_bytes(val[4:8], "big") ^ 0x2112A442
            return socket.inet_ntoa(xaddr.to_bytes(4, "big"))
        off += 4 + ((alen + 3) // 4) * 4
    raise RuntimeError("no XOR-MAPPED-ADDRESS in STUN response")


# ---------------------------------------------------------------- 1. PUBLIC IP via STUN (no REGISTER)
print("== STUN ==", flush=True)
try:
    PUBLIC_IP = stun_public_ip()
except Exception as e:  # noqa: BLE001
    print(f"STUN_FAILED: {e}")
    sys.exit(1)
print("public ip via STUN:", PUBLIC_IP, flush=True)
print("STUN_OK", flush=True)

# ---------------------------------------------------------------- 2. INVITE
rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rtp_sock.bind(("0.0.0.0", 0))
rtp_port = rtp_sock.getsockname()[1]
sdp_ip = PUBLIC_IP or LOCAL_IP
sdp = ("\r\n".join([
    "v=0",
    f"o={USER} 1 1 IN IP4 {sdp_ip}",
    "s=call",
    f"c=IN IP4 {sdp_ip}",
    "t=0 0",
    f"m=audio {rtp_port} RTP/AVP 0",
    "a=rtpmap:0 PCMU/8000",
    ""]) + "\r\n")

print("== INVITE ==", flush=True)
print("SDP_OFFER:", flush=True)
for ln in sdp.split("\r\n"):
    print(f"  sdp> {ln}", flush=True)
print(f"RTP_SOCK_BOUND: {rtp_sock.getsockname()}", flush=True)
t0 = time.time()
cseq = request("INVITE", DEST, body=sdp)
# Wait for a final response, tolerating 100/180 provisionals.
# Push-binding gate (2026-09-25): the proxy can only wake a sleeping phone if
# it holds a push-token binding for the callee, which it signals with
# "110 Push sent" (RFC 8599) ~100ms after 100 Trying on the authed INVITE.
# No 110 within PUSH_WAIT_S of 100 Trying means no binding exists, so no
# INVITE can ever wake the phone -- fail fast with exit code 2 and an
# actionable verdict instead of burning 50s on the inevitable 408 timeout.
PUSH_WAIT_S = 3.0
final = None
authed_cseq = None   # CSeq of the digest-authed INVITE
trying_at = None     # when 100 Trying arrived for the authed INVITE
saw_push_sent = False
end = time.time() + 60
while time.time() < end:
    # Poll in short slices while inside the push-watch window so we can bail
    # the moment it closes without a 110.
    timeout = max(0.5, end - time.time())
    if trying_at is not None and not saw_push_sent:
        timeout = min(timeout, 0.5)
    resp = wait_response(cseq, timeout=timeout)
    if resp is None:
        if (trying_at is not None and not saw_push_sent
                and time.time() - trying_at >= PUSH_WAIT_S):
            break  # push window closed with no 110 -> verdict below
        continue
    first, headers, body = resp
    code = int(first.split()[1])
    print("invite response:", first.split(" ", 2)[1], first.split(" ", 2)[2] if len(first.split(" ", 2)) > 2 else "", flush=True)
    if code in (401, 407):
        chal = parse_challenge(resp)
        cseq = request("INVITE", DEST, body=sdp, auth_params=chal)
        authed_cseq = cseq
        trying_at = None
        saw_push_sent = False
        continue
    if code < 200:
        if code == 100 and authed_cseq is not None and trying_at is None:
            trying_at = time.time()
        elif code == 110:
            saw_push_sent = True
            print("PUSH_SENT: proxy fired a push notification to wake the callee", flush=True)
        continue
    final = resp
    break

if trying_at is not None and not saw_push_sent and final is None:
    print("VERDICT: NO_PUSH_BINDING", flush=True)
    print(f"NO_PUSH_BINDING: {DEST} has no push-token binding at the proxy, so the",
          flush=True)
    print("proxy cannot wake the phone. Open Linphone on the phone to re-register,",
          flush=True)
    print("then retry the call.", flush=True)
    # CANCEL the outstanding INVITE (same CSeq number per RFC 3261) so the
    # proxy stops hunting for an unreachable callee. Exit 2 lets the workflow
    # tell "phone unwakeable" apart from other call failures.
    stack.send(
        f"CANCEL {DEST} SIP/2.0\r\n"
        f"Via: SIP/2.0/TCP {LOCAL_IP}:5060;branch=z9hG4bK{rnd(12)};rport\r\n"
        f"Max-Forwards: 70\r\n"
        f"From: <sip:{USER}@{DOMAIN}>;tag={from_tag}\r\n"
        f"To: <{DEST}>\r\n"
        f"Call-ID: {request.call_id}\r\n"
        f"CSeq: {authed_cseq} CANCEL\r\n"
        f"Content-Length: 0\r\n\r\n")
    print("CANCEL_SENT", flush=True)
    stack.close()
    sys.exit(2)

if final is None or int(final[0].split()[1]) >= 300:
    print("INVITE_FAILED")
    sys.exit(1)

print("CALL_ESTABLISHED", flush=True)
fheaders, sdp_answer = final[1], final[2]
to_tag = re.search(r"tag=([^;,\s]+)", fheaders.get("to", "") or "")
to_tag = to_tag.group(1) if to_tag else ""
peer_contact = fheaders.get("contact", "")
m = re.search(r"<([^>]+)>", peer_contact or "")
peer_uri = m.group(1) if m else DEST
# RTP target from SDP answer
rip = re.search(r"c=IN IP4 ([0-9.]+)", sdp_answer)
rport = re.search(r"m=audio (\d+)", sdp_answer)
rtp_ip = rip.group(1) if rip else None
rtp_port_peer = int(rport.group(1)) if rport else 0
print(f"peer rtp: {rtp_ip}:{rtp_port_peer}", flush=True)
print("SDP_ANSWER:", flush=True)
for ln in (sdp_answer or "").split("\r\n"):
    print(f"  sdp> {ln}", flush=True)

# ACK (no auth needed in-dialog for flexisip usually)
stack.send(f"ACK {peer_uri} SIP/2.0\r\n"
           f"Via: SIP/2.0/TCP {LOCAL_IP}:5060;branch=z9hG4bK{rnd(12)};rport\r\n"
           f"From: <sip:{USER}@{DOMAIN}>;tag={from_tag}\r\n"
           f"To: <{DEST}>;tag={to_tag}\r\n"
           f"Call-ID: {request.call_id}\r\n"
           f"CSeq: {stack.cseq} ACK\r\n"
           f"Content-Length: 0\r\n\r\n")
t_ack = time.time()

# Our SSRC, shared by the sender and the inbound telemetry listener below.
our_ssrc = random.randint(0, 2**32 - 1)

# ---------------------------------------------------------------- telemetry: inbound RTP/RTCP listener
inbound = {"rtp": 0, "rtcp": 0, "first_at": None, "first_from": None, "rr_for_us": 0}
stop_listen = threading.Event()

def rtp_listener():
    rtp_sock.settimeout(0.5)
    while not stop_listen.is_set():
        try:
            data, addr = rtp_sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break
        now = time.time()
        pt = data[1] if len(data) > 1 else -1
        is_rtcp = 200 <= pt <= 204
        if inbound["first_at"] is None:
            inbound["first_at"] = now
            inbound["first_from"] = addr
            print(f"INBOUND_FIRST kind={'RTCP' if is_rtcp else 'RTP'} from={addr[0]}:{addr[1]} size={len(data)} t_ack+{now - t_ack:.2f}s", flush=True)
        if is_rtcp:
            inbound["rtcp"] += 1
            if our_ssrc.to_bytes(4, "big") in data:
                inbound["rr_for_us"] += 1
                print(f"INBOUND_RTCP_RR_FOR_US from={addr[0]}:{addr[1]} size={len(data)}", flush=True)
        else:
            inbound["rtp"] += 1

# ---------------------------------------------------------------- 3. RTP playback
def pcm_to_ulaw(pcm_bytes):
    out = bytearray()
    for i in range(0, len(pcm_bytes), 2):
        s = struct.unpack_from("<h", pcm_bytes, i)[0]
        sign = 0
        if s < 0:
            s = -s
            sign = 0x80
        if s > 32635:
            s = 32635
        s += 0x84
        exp = 0
        tmp = s >> 7
        while tmp > 1:
            tmp >>= 1
            exp += 1
        mantissa = (s >> (exp + 3)) & 0x0F
        out.append((~(sign | (exp << 4) | mantissa)) & 0xFF)
    return bytes(out)


def rtp_sender():
    try:
        with wave.open(WAV_PATH, "rb") as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2
            assert w.getframerate() == 8000, "wav must be 8kHz"
            frames = w.readframes(w.getnframes())
    except Exception as e:  # noqa: BLE001
        print("WAV_ERROR", e, flush=True)
        return
    payload = pcm_to_ulaw(frames)
    seq, ts, ssrc = random.randint(0, 65535), random.randint(0, 2**32 - 1), our_ssrc
    n = 0
    # Play twice: a push-woken phone often misses the first second while its
    # audio session comes up.
    for repeat in range(2):
        for off in range(0, len(payload), 160):
            chunk = payload[off:off + 160]
            if len(chunk) < 160:
                chunk += b"\xff" * (160 - len(chunk))  # ulaw silence pad
            hdr = struct.pack(">BBHII", 0x80, 0x00, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
            try:
                rtp_sock.sendto(hdr + chunk, (rtp_ip, rtp_port_peer))
            except OSError as e:
                print("RTP_SEND_ERROR", e, flush=True)
                return
            seq += 1
            ts += 160
            n += 1
            time.sleep(0.02)
        if repeat == 0:
            print("RTP_REPEAT", flush=True)
            time.sleep(0.5)  # gap between plays
    print(f"RTP_DONE packets={n}", flush=True)
    t_n.append(n)


t_n = []
if rtp_ip and rtp_port_peer:
    # Give a push-woken callee a moment to bring its audio path up before
    # the message starts.
    print("waiting 2s for callee audio to settle...", flush=True)
    time.sleep(2)
    lt = threading.Thread(target=rtp_listener, daemon=True)
    lt.start()
    t = threading.Thread(target=rtp_sender, daemon=True)
    t.start()
    t.join(timeout=60)
    time.sleep(2)  # let tail audio play out
    stop_listen.set()
    lt.join(timeout=3)
    n_sent = t_n[0] if t_n else 0
    if inbound["first_at"] is not None:
        first = f"{inbound['first_from'][0]}:{inbound['first_from'][1]} @t_ack+{inbound['first_at'] - t_ack:.2f}s"
    else:
        first = "none"
    print(f"TELEMETRY sent={n_sent} inbound_rtp={inbound['rtp']} inbound_rtcp={inbound['rtcp']} rr_for_our_ssrc={inbound['rr_for_us']} first_inbound={first}", flush=True)
else:
    print("NO_RTP_TARGET", flush=True)
    time.sleep(5)

# ---------------------------------------------------------------- 4. BYE
print("== BYE ==", flush=True)
code, _ = transact("BYE", peer_uri, to_tag=to_tag, to_uri=DEST)
print("BYE_SENT", flush=True)
stack.close()
print("DONE", flush=True)
