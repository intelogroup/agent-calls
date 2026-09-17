# agent-calls

Free setup for the agent to place **prerecorded voice calls** to the user —
no paid telephony, no external numbers.

## How it works

1. A GitHub Actions workflow (`Call`) runs on `ubuntu-latest` runners, which
   have unrestricted internet access (the agent sandbox blocks raw SIP).
2. The runner synthesizes the message text with **Kokoro TTS** (local,
   open weights; `espeak-ng` fallback) and converts it to 8 kHz mono WAV.
3. `baresip` registers the Linphone SIP account and dials the user, playing
   the WAV as the microphone source. It hangs up after the message finishes.

## Triggering a call

The agent triggers it via the GitHub API (`workflow_dispatch`) with inputs:

- `message` — text to speak
- `to` — SIP address (default `sip:intelogroup@sip.linphone.org`)

## Secrets (repository → Settings → Secrets → Actions)

- `SIP_USERNAME` — Linphone username
- `SIP_PASSWORD` — Linphone SIP password

Nothing else is needed. The SIP domain (`sip.linphone.org`) is not sensitive
and lives in the workflow file.
