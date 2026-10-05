#!/usr/bin/env python3
"""Mock SIP proxy for dry-testing scripts/sip_call.py locally.

Listens on TCP 127.0.0.1:5060 and plays one scripted scenario per run:
  decline  - 407 -> 100, 110 Push sent, 183, 603 Decline   (reproduces run 37293669253)
  nopush   - 407 -> 100 only (no 110 within the client's 3s window)
  success  - 407 -> 100, 180, 200 OK w/ SDP answer; expects ACK, RTP, BYE

Digest auth is validated for real (user=testuser, pass from MOCK_SIP_PASS).
Writes a JSON event log to the path in MOCK_EVENTS.
"""
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "decline"
EVENTS_PATH = os.environ.get("MOCK_EVENTS", "/tmp/mock-events.json")
PASSWORD = os.environ.get("MOCK_SIP_PASS", "testpass")
USER = "testuser"
REALM = "127.0.0.1"

events = []
ev_lock = threading.Lock()
rtp_stats = {"packets": 0, "bad_pt": 0, "first": None}
rtp_done_emitted = False


def emit_rtp_done():
    """Record the final RTP summary exactly once (dialog may end before the
    listener's recv timeout fires)."""
    global rtp_done_emitted
    with ev_lock:
        if rtp_done_emitted:
            return
        rtp_done_emitted = True
        events.append({"kind": "rtp_done", "t": round(time.time(), 3),
                       "packets": rtp_stats["packets"],
                       "bad_pt": rtp_stats["bad_pt"],
                       "first": rtp_stats["first"]})


def ev(**kw):
    with ev_lock:
        kw["t"] = round(time.time(), 3)
        events.append(kw)


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def read_msg(conn):
    """Read one SIP message (headers + Content-Length body) from TCP."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65535)
        if not chunk:
            return None
        buf += chunk
    head_end = buf.find(b"\r\n\r\n")
    head = buf[:head_end].decode(errors="replace")
    headers = {}
    for ln in head.split("\r\n")[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    clen = int(headers.get("content-length", "0") or "0")
    total = head_end + 4 + clen
    while len(buf) < total:
        chunk = conn.recv(65535)
        if not chunk:
            break
        buf += chunk
    body = buf[head_end + 4:total].decode(errors="replace")
    return head.split("\r\n")[0], headers, body


def send(conn, text):
    conn.sendall(text.encode())


def copy_headers(h):
    out = {}
    for k in ("via", "from", "to", "call-id", "cseq"):
        if k in h:
            out[k] = h[k]
    return out


def response(first, h, extra_headers="", body=""):
    lines = [first]
    ch = copy_headers(h)
    lines.append(f"Via: {ch['via']}")
    lines.append(f"From: {ch['from']}")
    to = ch["to"]
    lines.append(f"To: {to}")
    lines.append(f"Call-ID: {ch['call-id']}")
    lines.append(f"CSeq: {ch['cseq']}")
    if extra_headers:
        lines.append(extra_headers)
    blen = len(body.encode())
    lines.append(f"Content-Length: {blen}")
    lines.append("")
    lines.append(body)
    return "\r\n".join(lines)


def check_digest(h):
    """Validate Proxy-Authorization digest against PASSWORD. Returns (ok, why)."""
    auth = h.get("proxy-authorization", "")
    if not auth.lower().startswith("digest"):
        return False, "no proxy-authorization"
    p = dict(re.findall(r'(\w+)="([^"]*)"', auth))
    for k, v in re.findall(r'(\w+)=([^\s",]+)', auth):  # qop=auth, nc=00000001 (unquoted)
        p.setdefault(k, v)
    try:
        ha1 = md5(f"{p['username']}:{p['realm']}:{PASSWORD}")
        ha2 = md5(f"INVITE:{p['uri']}")
        expect = md5(f"{ha1}:{p['nonce']}:{p['nc']}:{p['cnonce']}:{p['qop']}:{ha2}")
    except KeyError as e:
        return False, f"missing param {e}"
    if p.get("username") != USER:
        return False, "wrong username"
    return (p["response"] == expect), "digest mismatch" if p["response"] != expect else "ok"


def challenge_407(h):
    nonce = "mocknonce123"
    return response(
        "SIP/2.0 407 Proxy Authentication Required", h,
        extra_headers=(f'Proxy-Authenticate: Digest realm="{REALM}", nonce="{nonce}", '
                       f'opaque="mockopaque", algorithm=MD5, qop="auth"'))


def provisional(h, code, reason):
    return response(f"SIP/2.0 {code} {reason}", h)


def handle(conn):
    invite_branch = None
    authed = False
    rtp_sock = None
    try:
        while True:
            msg = read_msg(conn)
            if msg is None:
                break
            first, h, body = msg
            ev(kind="recv", first=first[:60])
            if first.startswith("INVITE"):
                branch = re.search(r"branch=([^;,\s]+)", h.get("via", ""))
                branch = branch.group(1) if branch else ""
                if "proxy-authorization" not in h:
                    invite_branch = branch
                    ev(kind="invite_noauth", branch=branch)
                    send(conn, challenge_407(h))
                    continue
                ok, why = check_digest(h)
                ev(kind="digest", ok=ok, why=why)
                if not ok:
                    send(conn, response("SIP/2.0 403 Forbidden", h))
                    break
                authed = True
                invite_branch = branch  # the pending (authed) transaction CANCEL must reference
                ev(kind="invite_authed", branch=branch,
                   branch_match=True)
                send(conn, provisional(h, 100, "Trying"))
                if SCENARIO == "decline":
                    send(conn, provisional(h, 110, "Push sent"))
                    send(conn, provisional(h, 183, "Session Progress"))
                    time.sleep(1.0)
                    send(conn, response("SIP/2.0 603 Decline", h,
                                        extra_headers="To: %s;tag=srvdecline" % h["to"]))
                    ev(kind="sent_603")
                elif SCENARIO == "nopush":
                    ev(kind="sent_100_only")
                    # stay silent: client's 3s push window must expire
                elif SCENARIO == "success":
                    send(conn, provisional(h, 180, "Ringing"))
                    rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    rtp_sock.bind(("127.0.0.1", 0))
                    rtp_port = rtp_sock.getsockname()[1]
                    ev(kind="rtp_listen", port=rtp_port)
                    sdp = ("\r\n".join([
                        "v=0", "o=mock 1 1 IN IP4 127.0.0.1", "s=mock",
                        "c=IN IP4 127.0.0.1", "t=0 0",
                        f"m=audio {rtp_port} RTP/AVP 0",
                        "a=rtpmap:0 PCMU/8000", ""]) + "\r\n")
                    to_tagged = h["to"] + ";tag=srv200ok"
                    resp = response("SIP/2.0 200 OK", h,
                                    extra_headers=(f"To: {to_tagged}\r\n"
                                                   f"Contact: <sip:mock@127.0.0.1:5060;transport=tcp>\r\n"
                                                   "Content-Type: application/sdp"),
                                    body=sdp)
                    # response() already emits To: from copy_headers; fix the tag
                    resp = resp.replace(f"To: {h['to']}\r\n",
                                        f"To: {to_tagged}\r\n", 1)
                    send(conn, resp)
                    ev(kind="sent_200")
                    # receive RTP in background
                    threading.Thread(target=rtp_loop, args=(rtp_sock,), daemon=True).start()
            elif first.startswith("ACK"):
                ev(kind="recv_ack")
            elif first.startswith("CANCEL"):
                branch = re.search(r"branch=([^;,\s]+)", h.get("via", ""))
                branch = branch.group(1) if branch else ""
                ev(kind="recv_cancel", branch=branch,
                   branch_matches_invite=(branch == invite_branch))
                send(conn, response("SIP/2.0 200 OK", h))
            elif first.startswith("BYE"):
                to = h.get("to", "")
                cseq = h.get("cseq", "")
                ev(kind="recv_bye", to_tag=("tag=" in to), cseq=cseq)
                send(conn, response("SIP/2.0 200 OK", h))
                break
    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        ev(kind="conn_error", err=str(e))
    finally:
        emit_rtp_done()  # dialog over: freeze the RTP counters now
        try:
            conn.close()
        except OSError:
            pass
        with ev_lock:
            with open(EVENTS_PATH, "w") as f:
                json.dump(events, f, indent=1)


def rtp_loop(sock):
    sock.settimeout(5)
    try:
        while True:
            data, _ = sock.recvfrom(2048)
            pt = data[1] & 0x7F if len(data) > 1 else -1
            if pt != 0:
                rtp_stats["bad_pt"] += 1
            rtp_stats["packets"] += 1
            if rtp_stats["first"] is None:
                rtp_stats["first"] = {"size": len(data), "pt": pt}
                ev(kind="rtp_first", size=len(data), pt=pt)
    except socket.timeout:
        pass
    finally:
        emit_rtp_done()
        sock.close()


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 5060))
    srv.listen(1)
    ev(kind="listening", scenario=SCENARIO)
    conn, _ = srv.accept()
    handle(conn)
    srv.close()


if __name__ == "__main__":
    main()
