# Jett's voice proxy — call Jett anytime

One SIP line, one worker. Dial `sip:jett@129.159.189.244` and talk to
Jett's voice proxy: real-time voice (LiveKit handles VAD / barge-in /
turn-taking), a Muse Spark brain briefed on Jett's notes (`JETT.md`), and a
warm Kokoro voice.

The proxy answers from its brief when confident — and for anything needing
the real Jett's live memory, tools, or judgment it calls `consult_jett()`,
which drops the question into the private `intelogroup/agent-call-bus` repo
(`calls/live/<call-id>/in.jsonl`) and speaks Jett's reply back when it lands
(`out.jsonl`). See `PROTOCOL.md` in that repo.

## Stack (1 GB VM, no local LLM)

- `livekit-server` — SFU, localhost signaling (`:7880`)
- `livekit-sip` — SIP↔WebRTC bridge (public SIP `:5060`, RTP `12000-12100/udp`)
- `redis-server` — shared bus for the above
- `live_agent.py` — the proxy worker (LiveKit Agents pipeline):
  faster-whisper `tiny.en` STT → Muse Spark LLM → kokoro-onnx TTS
- `setup-sip.py` — creates the inbound trunk (`jett`) + dispatch rule
  (room prefix `jett-`, auto-dispatches agent `jett-proxy`)

## Files

- `live_agent.py` — the proxy worker. Loads repo-root `JETT.md` as its brief.
- `setup.sh` — provisioning: binaries, venv, models, systemd units, bus key.
- `setup-sip.py` — idempotent SIP trunk + dispatch rule creation.
- `jett-proxy.service` — systemd unit for the worker.
- `requirements.txt` — python deps.
- `agent.env.example` — config template.
- `sip_server.py` — **RETIRED / superseded.** The old DIY half-duplex
  inbound server (raw SIP + local Qwen). Kept for reference only; the
  LiveKit proxy above replaces it.

## Config (`/opt/agent/agent.env` + `/opt/agent/livekit.env`)

| Var | Meaning |
|---|---|
| `META_API_KEY` | Meta Model API key (live brain). Worker idles gracefully without it. |
| `META_MODEL` | model name, default `muse-spark-1.3` (auto-discovery via `/v1/models`) |
| `LIVEKIT_URL` | `ws://localhost:7880` |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | generated server-side into `livekit.env` |
| `BUS_DIR` / `BUS_REPO_SSH` | agent-call-bus checkout for `consult_jett` |
| `CONSULT_TIMEOUT_SEC` | how long to wait for Jett's reply (default 180) |
| `AGENT_MAX_CALL_SEC` | call length guard (default 600) |

## Firewall (OCI security list)

Inbound needed: **TCP+UDP 5060** (SIP), **UDP 12000–12100** (SIP media).
Nothing else public: LiveKit's `:7880`/RTC ports are localhost-only in this
SIP-only setup.

## Logs

`journalctl -u jett-proxy.service -f`, `journalctl -u livekit-sip.service -f`.

## Post-deploy checklist (operator)

1. **Deploy key**: setup.sh generates `/home/agent/.ssh/bus_key` and prints
   the public key in the deploy log. Register it with WRITE access:
   `gh-api api POST /repos/intelogroup/agent-call-bus/keys
   '{"title":"jett-proxy-bus","key":"ssh-ed25519 AAAA…","read_only":false}'`
   (manual fallback: repo Settings → Deploy keys). Without it, `consult_jett`
   degrades honestly ("can't reach Jett").
2. **Brain key**: store `META_API_KEY` as a GitHub Actions secret, re-run
   deploy (writes `agent.env`, restarts worker). Until then the worker
   answers calls with a spoken "brain not connected" notice — never silence.
3. **Verify**: `systemctl is-active` on redis/livekit-server/livekit-sip/
   jett-proxy; place a test call to `sip:jett@129.159.189.244` from Linphone
   on cellular data (hospital Wi-Fi blocks UDP media — see root README).
4. **Consult test**: ask something only Jett would know; watch
   `calls/live/<call-id>/in.jsonl` appear in agent-call-bus; reply in
   `out.jsonl` and confirm the caller hears it.
