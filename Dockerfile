# Durable image for Jett's interactive voice worker (LiveKit agent).
#
# Replaces the hand-built ephemeral Fly machine (PID 1 = sleep infinity,
# everything installed via exec, wiped on restart). Bakes at build time:
#   - python deps (pinned, integration-tested 2026-09-18)
#   - faster-whisper tiny.en (STT)
#   - kokoro-onnx int8 model + voices (TTS)
# Runtime secrets come ONLY from env (Fly secrets), never baked in:
#   LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
#   OPENROUTER_API_KEY / META_API_KEY, TUNNEL_SSH_KEY (optional Oracle tunnel)
#
# DRY_TEST=1 boots a fully self-contained test: a local livekit-server with
# generated credentials, worker pointed at it. No Oracle, no tunnel,
# no secrets, no production traffic possible.

FROM python:3.12-slim

ENV AGENT_DIR=/opt/agent \
    PYTHONUNBUFFERED=1 \
    HF_HUB_CACHE=/opt/agent/models/hf-cache \
    PIP_NO_CACHE_DIR=1

# libsndfile1: soundfile wheel needs it. curl: model downloads.
# openssh-client: optional Oracle tunnel at runtime. ca-certificates: TLS.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 curl openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/agent

# --- python deps first (own layer, changes rarely) ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --- agent code (mirrors deploy-vm.yml layout: /opt/agent) ---
COPY live_agent.py /opt/agent/live_agent.py
COPY JETT.md /opt/agent/JETT.md

# --- models, baked so no runtime download ---
RUN mkdir -p /opt/agent/models && \
    curl -fL --retry 3 -o /opt/agent/models/kokoro-v1.0.int8.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx && \
    curl -fL --retry 3 -o /opt/agent/models/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin && \
    python3 -c "from faster_whisper import WhisperModel; \
WhisperModel('tiny.en', device='cpu', compute_type='int8'); \
print('whisper tiny.en cached')" && \
    python3 -c "from kokoro_onnx import Kokoro; \
Kokoro('/opt/agent/models/kokoro-v1.0.int8.onnx', '/opt/agent/models/voices-v1.0.bin'); \
print('kokoro warmup OK')"

# --- livekit-server binary: used ONLY in DRY_TEST mode ---
RUN LK_TAG=$(curl -fsSL https://api.github.com/repos/livekit/livekit/releases/latest \
      | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4) && \
    echo "livekit-server tag: $LK_TAG" && \
    ASSET_URL=$(curl -fsSL "https://api.github.com/repos/livekit/livekit/releases/tags/$LK_TAG" \
      | grep -o '"browser_download_url": *"[^"]*linux_amd64\.tar\.gz"' | head -1 | cut -d'"' -f4) && \
    [ -n "$ASSET_URL" ] || { echo "ERROR: no livekit-server asset"; exit 1; } && \
    curl -fSL --retry 3 "$ASSET_URL" -o /tmp/lk.tgz && \
    mkdir -p /opt/agent/bin && \
    tar xzf /tmp/lk.tgz -C /opt/agent/bin livekit-server && \
    chmod +x /opt/agent/bin/livekit-server && rm -f /tmp/lk.tgz && \
    /opt/agent/bin/livekit-server --version

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
