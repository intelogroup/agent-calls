#!/usr/bin/env python3
"""Regression dry-test for scripts/sip_call.py (no real calls placed).

Exercises a copy of sip_call.py against mock_sip_server.py (local TCP
127.0.0.1:5060) across the three production paths:
  decline  407 -> 100, 110 Push sent, 183, 603 Decline  (exit 1, INVITE_FAILED)
  nopush   407 -> 100 only, no 110 within 3s            (exit 2, NO_PUSH_BINDING)
  success  407 -> 100, 180, 200 OK w/ SDP               (exit 0, BYE_SENT, DONE)

Usage: test_sip_call.py [sip_call.py path] [scenarios]
Defaults to ../sip_call.py and all three scenarios. Requires ffmpeg
(for the 8kHz test wav) and python3. UDP loopback blocked (some sandboxes)
downgrades the RTP packet-flow assertion to a send-attempt check.

For each scenario it starts mock_sip_server.py, runs the client exactly like
the workflow's 'Place call' step does, applies the workflow's verdict logic
(rc==2 -> NO_PUSH_BINDING error path; else grep CALL_ESTABLISHED), and reports.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import tempfile
TESTDIR = tempfile.mkdtemp(prefix="sip-drytest-")

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(os.path.join(HERE, "..", "sip_call.py"))
SCENARIOS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["decline", "nopush", "success"]
assert os.path.isfile(SRC), f"client not found: {SRC}"

def udp_works():
    import socket
    s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s1.bind(("127.0.0.1", 0))
        s2.bind(("127.0.0.1", 0))
        s2.settimeout(1)
        s1.sendto(b"probe", ("127.0.0.1", s2.getsockname()[1]))
        s2.recvfrom(16)
        return True
    except OSError:
        return False
    finally:
        s1.close()
        s2.close()


UDP_OK = udp_works()

PASS = "testpass"


def make_test_copy(src):
    with open(src) as f:
        c = f.read()
    assert 'DOMAIN = "sip.linphone.org"' in c, "unexpected DOMAIN line"
    c = c.replace('DOMAIN = "sip.linphone.org"', 'DOMAIN = "127.0.0.1"  # TEST STUB')
    old = "    PUBLIC_IP = stun_public_ip()"
    assert old in c, "unexpected STUN call site"
    c = c.replace(old, '    PUBLIC_IP = "127.0.0.1"  # TEST STUB: no STUN in dry test')
    dst = os.path.join(TESTDIR, "sip_call_test.py")
    with open(dst, "w") as f:
        f.write(c)
    return dst


def make_wav():
    wav = os.path.join(TESTDIR, "test-8k.wav")
    if not os.path.exists(wav):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=1",
             "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le", wav],
            check=True)
    return wav


def run_scenario(client_py, wav, scenario):
    events_path = os.path.join(TESTDIR, f"events-{scenario}.json")
    if os.path.exists(events_path):
        os.remove(events_path)
    env = dict(os.environ, MOCK_SIP_PASS=PASS, MOCK_EVENTS=events_path,
               SIP_CALL_PASS=PASS, DEBUG_CALL="1")
    mock = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "mock_sip_server.py"), scenario],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.7)  # let the mock bind
    log_path = os.path.join(TESTDIR, f"sip-call-{scenario}.log")
    try:
        with open(log_path, "w") as log:
            p = subprocess.run(
                [sys.executable, client_py, "testuser",
                 "sip:testuser@127.0.0.1", wav],
                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        rc = p.returncode
    finally:
        mock.wait(timeout=15)
    with open(log_path) as f:
        out = f.read()
    with open(events_path) as f:
        events = json.load(f)
    # --- workflow 'Place call' verdict logic (from call.yml) ---
    if rc == 2:
        verdict = "NO_PUSH_BINDING (workflow would ::error:: + ntfy + exit 2)"
    elif "CALL_ESTABLISHED" in out:
        verdict = "CALL_ESTABLISHED (workflow step passes)"
    else:
        verdict = f"STEP FAILS (rc={rc}, no CALL_ESTABLISHED -> grep exits 1)"
    return rc, out, events, verdict


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    return cond


def main():
    client_py = make_test_copy(SRC)
    wav = make_wav()
    print(f"client under test: {SRC}")
    print(f"test copy:         {client_py}")
    all_ok = True

    if "decline" in SCENARIOS:
        print("\n== scenario: decline (603 after 183, reproduces run 37293669253) ==")
        rc, out, evs, verdict = run_scenario(client_py, wav, "decline")
        print(f"  exit={rc} | {verdict}")
        all_ok &= check("exit code 1", rc == 1, f"got {rc}")
        all_ok &= check("INVITE_FAILED printed", "INVITE_FAILED" in out)
        all_ok &= check("no CALL_ESTABLISHED", "CALL_ESTABLISHED" not in out)
        all_ok &= check("digest auth validated by mock",
                        any(e.get("kind") == "digest" and e.get("ok") for e in evs))

    if "nopush" in SCENARIOS:
        print("\n== scenario: nopush (no 110 Push sent -> fail fast) ==")
        rc, out, evs, verdict = run_scenario(client_py, wav, "nopush")
        print(f"  exit={rc} | {verdict}")
        all_ok &= check("exit code 2", rc == 2, f"got {rc}")
        all_ok &= check("NO_PUSH_BINDING printed", "NO_PUSH_BINDING" in out)
        all_ok &= check("CANCEL_SENT printed", "CANCEL_SENT" in out)
        cancel = next((e for e in evs if e.get("kind") == "recv_cancel"), None)
        all_ok &= check("mock saw CANCEL", cancel is not None)
        if cancel:
            all_ok &= check("CANCEL branch == INVITE branch (RFC 3261)",
                            cancel.get("branch_matches_invite") is True,
                            f"cancel={cancel.get('branch')}")

    if "success" in SCENARIOS:
        print("\n== scenario: success (200 OK -> RTP -> BYE) ==")
        rc, out, evs, verdict = run_scenario(client_py, wav, "success")
        print(f"  exit={rc} | {verdict}")
        all_ok &= check("exit code 0", rc == 0, f"got {rc}")
        all_ok &= check("CALL_ESTABLISHED printed", "CALL_ESTABLISHED" in out)
        if UDP_OK:
            all_ok &= check("RTP_DONE printed", "RTP_DONE" in out)
        else:
            # Sandbox blocks UDP loopback: rtp_sender hits EPERM on sendto and
            # returns after printing RTP_SEND_ERROR. Production runners do RTP
            # fine (attempt 2 today: RTP_DONE packets=530).
            all_ok &= check("RTP attempted (sandbox blocks UDP)",
                            "RTP_SEND_ERROR" in out)
        all_ok &= check("BYE_SENT printed", "BYE_SENT" in out)
        all_ok &= check("DONE printed (clean shutdown)", "DONE" in out)
        all_ok &= check("no Traceback", "Traceback" not in out,
                        [l for l in out.splitlines() if "Error" in l][:2])
        rtp = next((e for e in evs if e.get("kind") == "rtp_done"), None)
        if UDP_OK:
            all_ok &= check("mock received RTP PT=0 packets",
                            rtp is not None and rtp.get("packets", 0) > 100 and rtp.get("bad_pt") == 0,
                            str(rtp))
        else:
            print("  [SKIP] RTP packet flow: sandbox blocks UDP loopback "
                  "(client attempted send; verified on GitHub runner in prod)")
        bye = next((e for e in evs if e.get("kind") == "recv_bye"), None)
        all_ok &= check("mock saw BYE with To tag + BYE CSeq",
                        bye is not None and bye.get("to_tag") and "BYE" in bye.get("cseq", ""),
                        str(bye))

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
