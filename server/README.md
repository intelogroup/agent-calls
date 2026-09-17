# Inbound voice agent — call Jett anytime

A 24/7 server that registers a SIP identity, answers incoming calls, and
holds a spoken conversation: listen → transcribe (faster-whisper) →
think (Qwen 2.5 1.5B via llama.cpp) → speak (Kokoro, espeak-ng fallback).

## What you need (the 3 things only you can do)

1. **Oracle Cloud Free Tier account** — oracle.com/cloud/free. Email + card
   for verification (never charged on the free tier). This is the always-on box.
2. **A VM**: Ampere A1, Ubuntu 24.04, 4 OCPU / 24 GB RAM (inside free limits).
   - Subnet security list: allow ingress **UDP port 10000** (RTP audio) and
     **TCP port 22** (SSH) from anywhere. Egress: all.
   - Add this SSH public key so the assistant can deploy for you:
     ```
     ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOSv8vb0slmLAo62ta1sMlXlyqFY15uau2zznOONbaSB hatch
     ```
   - Note the VM's **public IP**.
3. **A second free Linphone account for the agent** — in the Linphone app:
   add account → create a new free account (e.g. username `jett-agent`).
   The agent registers as this identity; you dial it like any contact.

Then send the assistant: the VM's public IP + the agent's SIP username/password.

## What the assistant does from there

```bash
scp -r server/ root@<VM-IP>:/opt/agent/
ssh root@<VM-IP> "bash /opt/agent/setup.sh"   # installs everything, ~20-40 min
# fill /opt/agent/agent.env, then:
systemctl start agent.service
```

## Files

- `sip_server.py` — the daemon: TCP SIP registration (with re-register +
  auto-reconnect), inbound INVITE handling, RTP/PCMU send/recv, voice loop
  with energy-based end-of-speech detection, Whisper STT, llama.cpp LLM,
  Kokoro/espeak TTS. One call at a time; second caller gets 486 Busy.
- `setup.sh` — provisioning: system packages, venv, pip deps, model downloads.
- `requirements.txt` — faster-whisper, llama-cpp-python, kokoro, torch, etc.
- `agent.service` — systemd unit (auto-restart).
- `agent.env.example` — config template.

## Config (`/opt/agent/agent.env`)

| Var | Meaning |
|---|---|
| `AGENT_SIP_USER` / `AGENT_SIP_PASS` | agent's Linphone account |
| `AGENT_PUBLIC_IP` | VM public IP (used in SDP) |
| `AGENT_RTP_PORT` | UDP port for audio (default 10000) |
| `AGENT_TTS_ENGINE` | `kokoro` (nicer) or `espeak` (instant, robotic) |
| `AGENT_MAX_TURNS` / `AGENT_MAX_CALL_SEC` | call length guards |

## Honest expectations

- Each reply takes **tens of seconds** (transcribe + think + speak on a small
  CPU VM). Voice-message pace, not a phone-call pace.
- Logs: `journalctl -u agent.service -f`.
- If the LLM model fails to load, the agent says so out loud and keeps
  listening (degraded but honest).
