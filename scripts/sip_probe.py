#!/usr/bin/env python3
"""Minimal SIP REGISTER probe over TCP. Prints the raw server response.
Usage: sip_probe.py <user> ; password from SIP_PROBE_PASS env (never logged)."""
import os
import random
import socket
import string
import sys

USER = sys.argv[1]
DOMAIN = "sip.linphone.org"
PASS = os.environ.get("SIP_PROBE_PASS", "")
BRANCH = "z9hG4bK" + "".join(random.choices(string.ascii_letters + string.digits, k=12))
CALLID = "".join(random.choices(string.ascii_letters + string.digits, k=16))
TAG = "".join(random.choices(string.ascii_letters + string.digits, k=8))

reg = (
    f"REGISTER sip:{DOMAIN} SIP/2.0\r\n"
    f"Via: SIP/2.0/TCP 10.0.0.1:5060;branch={BRANCH};rport\r\n"
    f"Max-Forwards: 70\r\n"
    f"From: <sip:{USER}@{DOMAIN}>;tag={TAG}\r\n"
    f"To: <sip:{USER}@{DOMAIN}>\r\n"
    f"Call-ID: {CALLID}\r\n"
    f"CSeq: 1 REGISTER\r\n"
    f"Contact: <sip:{USER}@10.0.0.1:5060;transport=tcp>\r\n"
    f"Expires: 600\r\n"
    f"User-Agent: sip-probe/1.0\r\n"
    f"Content-Length: 0\r\n\r\n"
)

print("connecting TCP to", DOMAIN, "5060 ...", flush=True)
s = socket.create_connection((DOMAIN, 5060), timeout=10)
print("TCP connected, sending REGISTER", flush=True)
s.sendall(reg.encode())
s.settimeout(10)
try:
    data = s.recv(65535)
except socket.timeout:
    print("RESULT: TIMEOUT (no SIP response)")
    sys.exit(2)
print("RESULT: got", len(data), "bytes")
print(data.decode(errors="replace")[:2000])
