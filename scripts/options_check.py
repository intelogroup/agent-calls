#!/usr/bin/env python3
"""Pre-dial callee registration check over TCP (digest-authenticated).

Sends SIP OPTIONS to the destination AOR and reports whether the callee
has a live binding at the proxy. Lets the workflow fail fast -- before
the expensive TTS/call attempt -- when the phone is not registered.

Usage: options_check.py <dest_uri>   (e.g. sip:intelogroup@sip.linphone.org)
Auth (optional but recommended): SIP_CHECK_USER / SIP_CHECK_PASS env vars.
  Without auth the proxy may 407 the probe, which yields UNKNOWN.
Exit codes: 0 = REGISTERED (proceed), 1 = NOT_REGISTERED (abort),
            2 = UNKNOWN (proceed with warning).
"""
import hashlib
import os
import random
import re
import socket
import string
import sys

DOMAIN = "sip.linphone.org"
ATTEMPTS = 3
TIMEOUT = 8

USER = os.environ.get("SIP_CHECK_USER", "")
PASS = os.environ.get("SIP_CHECK_PASS", "")


def rnd(n):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def parse_auth(header_value):
    """Pull Digest challenge params. Mirrors sip_call.py's parse_challenge."""
    params = dict(re.findall(r'(\w+)="([^"]*)"', header_value))
    return params


def auth_header(challenge, method, uri, proxy_auth):
    """Build the auth header. Mirrors sip_call.py's auth_header exactly."""
    p = parse_auth(challenge)
    realm, nonce = p["realm"], p["nonce"]
    qop = p.get("qop", "auth").split(",")[0].strip()
    nc, cnonce = "00000001", rnd(8)
    ha1 = md5(f"{USER}:{realm}:{PASS}")
    ha2 = md5(f"{method}:{uri}")
    resp = md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")
    hdr = "Proxy-Authorization" if proxy_auth else "Authorization"
    return (f'{hdr}: Digest username="{USER}", realm="{realm}", nonce="{nonce}", '
            f'uri="{uri}", response="{resp}", algorithm=MD5, '
            f'cnonce="{cnonce}", opaque="{p.get("opaque", "")}", '
            f'qop={qop}, nc={nc}')


def build_options(dest, aor, auth_value=None):
    branch, callid, tag = "z9hG4bK" + rnd(12), rnd(16), rnd(8)
    # From must be the authenticated account: Flexisip 403s probes whose
    # From URI is not a real local user.
    from_uri = f"sip:{USER}@{DOMAIN}" if USER else f"sip:predial-check@{DOMAIN}"
    lines = [
        f"OPTIONS {dest} SIP/2.0",
        f"Via: SIP/2.0/TCP 10.9.9.9:5060;branch={branch};rport",
        "Max-Forwards: 70",
        f"From: <{from_uri}>;tag={tag}",
        f"To: <sip:{aor}>",
        f"Call-ID: {callid}",
        "CSeq: 1 OPTIONS",
        f"Contact: <{from_uri}>;transport=tcp",
        "User-Agent: predial-check/1.0",
    ]
    if auth_value:
        lines.append(auth_value)
    lines += ["Content-Length: 0", "", ""]
    return "\r\n".join(lines), callid


def transact(message):
    """Send one SIP message over a fresh TCP connection, return raw response."""
    s = socket.create_connection((DOMAIN, 5060), timeout=10)
    try:
        s.sendall(message.encode())
        s.settimeout(TIMEOUT)
        data = s.recv(8192)
    finally:
        s.close()
    return data.decode(errors="replace")


def status_and_challenge(raw):
    head = raw.split("\r\n\r\n")[0]
    lines = head.split("\r\n")
    first = lines[0] if lines else ""
    parts = first.split()
    code = None
    if len(parts) >= 2 and parts[0].startswith("SIP/2"):
        try:
            code = int(parts[1])
        except ValueError:
            pass
    challenge, proxy = None, False
    for ln in lines[1:]:
        if ln.lower().startswith("proxy-authenticate:"):
            challenge, proxy = ln.split(":", 1)[1].strip(), True
            break
        if ln.lower().startswith("www-authenticate:") and challenge is None:
            challenge = ln.split(":", 1)[1].strip()
    return code, first, challenge, proxy


def probe_once(dest, aor):
    msg, _ = build_options(dest, aor)
    raw = transact(msg)
    code, first, challenge, proxy = status_and_challenge(raw)
    print("raw:", first, flush=True)
    if code in (401, 407) and challenge and USER and PASS:
        auth = auth_header(challenge, "OPTIONS", dest, proxy_auth=proxy)
        msg2, _ = build_options(dest, aor, auth)
        raw2 = transact(msg2)
        code2, first2, _, _ = status_and_challenge(raw2)
        print("authed raw:", first2, flush=True)
        return code2
    return code


def verdict(v, detail):
    print(f"VERDICT: {v} ({detail})", flush=True)
    sys.exit({"REGISTERED": 0, "NOT_REGISTERED": 1, "UNKNOWN": 2}[v])


def main():
    if len(sys.argv) < 2:
        print("usage: options_check.py <dest_uri>", flush=True)
        sys.exit(2)
    dest = sys.argv[1].strip()
    aor = dest[4:] if dest.lower().startswith("sip:") else dest
    aor = aor.split(";")[0]
    if not USER or not PASS:
        print("note: no SIP_CHECK_USER/PASS, probe is unauthenticated", flush=True)
    last = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            code = probe_once(dest, aor)
        except Exception as e:  # noqa: BLE001 - transport failure
            print(f"attempt {attempt}: transport error: {e}", flush=True)
            code = None
        print(f"attempt {attempt}: code={code}", flush=True)
        if code is None:
            continue
        last = code
        if 100 <= code < 200 or code in (200, 486, 487):
            verdict("REGISTERED", f"final {code}")
        if code in (404, 410, 480, 483, 503, 604):
            verdict("NOT_REGISTERED", f"final {code}")
        # anything else (401/403/407/408/500/...) is inconclusive: retry
    if last is None:
        verdict("UNKNOWN", "no SIP response after retries")
    verdict("UNKNOWN", f"inconclusive final code {last} after retries")


main()
