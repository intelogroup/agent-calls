# scripts/tests — sip_call.py regression dry-test

No real calls are placed. `test_sip_call.py` runs a copy of
`../sip_call.py` against `mock_sip_server.py`, a scripted SIP proxy on local
TCP `127.0.0.1:5060` that validates digest auth for real and replays the three
production paths:

| scenario | proxy script | expected client behavior |
|---|---|---|
| `decline` | 407 → 100, 110 Push sent, 183, 603 Decline | exit 1, `INVITE_FAILED` (reproduces the 2026-10-05 run 37293669253 failure) |
| `nopush` | 407 → 100 only (no 110 within the 3s push window) | exit 2, `NO_PUSH_BINDING`, `CANCEL_SENT`, CANCEL reuses the INVITE's Via branch (RFC 3261 §9.1) |
| `success` | 407 → 100, 180, 200 OK with SDP answer | exit 0, `CALL_ESTABLISHED`, `BYE_SENT`, `DONE`, mock sees a well-formed BYE |

It also applies the workflow's verdict logic from `.github/workflows/call.yml`
(exit 2 → `::error::` + ntfy path; otherwise grep for `CALL_ESTABLISHED`).

## Run

```bash
python3 scripts/tests/test_sip_call.py
# or: python3 scripts/tests/test_sip_call.py path/to/sip_call.py decline,success
```

Requires: python3, ffmpeg (builds the 8kHz test wav). ~15s. All artifacts go
to a temp dir; nothing is written into the repo.

Note: sandboxes that block UDP loopback (EPERM on sendto) downgrade the RTP
packet-flow assertion to a send-attempt check — the full RTP path is covered
on the GitHub runner instead.

## History

Written 2026-10-05 while debugging run 37293669253. Caught two real bugs:
`transact()` was deleted but still called on the BYE path (NameError after
every successful call since 2026-09-25), and CANCEL used a fresh random Via
branch the proxy couldn't correlate with the pending INVITE.
