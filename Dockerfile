# Jett voice proxy (LiveKit agent worker) on Fly.io.
# Mirrors the Oracle VM worker: faster-whisper tiny.en STT + silero VAD,
# OpenRouter brain chain, kokoro-onnx TTS. Connects OUTBOUND to the
# LiveKit server on Oracle (wss://livekit-api.clixen.app); no inbound ports.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    AGENT_DIR=/opt/agent \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=0

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    ffmpeg git openssh-client curl build-essential espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/agent

COPY requirements.txt .
RUN pip install -q --no-cache-dir -r requirements.txt

# Bake model assets into the image (no volume, no runtime downloads).
RUN mkdir -p /opt/agent/models && \
    curl -skL --retry 3 -o /opt/agent/models/kokoro-v1.0.int8.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx && \
    curl -skL --retry 3 -o /opt/agent/models/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin && \
    ls -la /opt/agent/models/ && \
    test -s /opt/agent/models/kokoro-v1.0.int8.onnx && \
    test -s /opt/agent/models/voices-v1.0.bin

# Pre-warm caches so the first call has no download latency.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8'); print('whisper cached')" && \
    python -c "from livekit.plugins import silero; silero.VAD.load(); print('silero cached')"

COPY live_agent.py JETT.md entrypoint.sh /opt/agent/
RUN chmod +x /opt/agent/entrypoint.sh

ENTRYPOINT ["/opt/agent/entrypoint.sh"]
