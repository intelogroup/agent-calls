#!/bin/bash
# Provision the voice-agent VM (Ubuntu 24.04) as the 24/7 inbound voice agent.
# Run as root (or with sudo). Idempotent-ish: safe to re-run.
#
# RAM-aware: on small boxes (<2 GB RAM) it installs the slim stack —
# whisper tiny, Qwen2.5-0.5B, espeak-ng, plus a 2 GB swapfile —
# instead of the full stack (Qwen2.5-1.5B + kokoro).
set -euo pipefail

AGENT_DIR=/opt/agent
MODEL_DIR=$AGENT_DIR/models
SERVICE_USER=agent

MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
MEM_GB=$((MEM_KB / 1024 / 1024))
echo "RAM detected: ${MEM_GB} GB"
if [ "$MEM_GB" -lt 2 ]; then
  SMALL=1
  echo "small-box mode: slim models, espeak-ng, +2GB swap"
else
  SMALL=0
  echo "full mode"
fi

if [ "$SMALL" = 1 ] && [ ! -f /swapfile ]; then
  echo "== 2GB swapfile =="
  fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo "/swapfile none swap sw 0 0" >> /etc/fstab
  echo OK
fi

echo "== system packages =="
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev ffmpeg espeak-ng \
    build-essential cmake curl git > /dev/null
echo OK

id -u $SERVICE_USER >/dev/null 2>&1 || useradd -m -s /bin/bash $SERVICE_USER
mkdir -p $MODEL_DIR $AGENT_DIR
chown -R $SERVICE_USER:$SERVICE_USER $AGENT_DIR

echo "== python venv =="
if [ ! -x $AGENT_DIR/venv/bin/python ]; then
    python3 -m venv $AGENT_DIR/venv
fi
$AGENT_DIR/venv/bin/pip -q install --upgrade pip
if [ "$SMALL" = 1 ]; then
  # slim: no torch/kokoro (espeak-ng handles TTS)
  $AGENT_DIR/venv/bin/pip -q install faster-whisper llama-cpp-python \
      soundfile numpy huggingface_hub
else
  $AGENT_DIR/venv/bin/pip -q install -r $AGENT_DIR/requirements.txt
fi
echo OK

echo "== whisper model (tiny.en) =="
sudo -u $SERVICE_USER $AGENT_DIR/venv/bin/python -c "
from faster_whisper import WhisperModel
WhisperModel('tiny.en', device='cpu', compute_type='int8')
print('whisper cached')"
echo OK

if [ "$SMALL" = 1 ]; then
  echo "== LLM model, small box (Qwen2.5-0.5B Q4_K_M) =="
  HF_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
  HF_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
else
  echo "== LLM model (Qwen2.5-1.5B Q4_K_M) =="
  HF_REPO="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
  HF_FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"
fi
if [ ! -f $MODEL_DIR/llm.gguf ]; then
    sudo -u $SERVICE_USER $AGENT_DIR/venv/bin/hf download \
        $HF_REPO $HF_FILE --local-dir $MODEL_DIR >/dev/null
    mv $MODEL_DIR/$HF_FILE $MODEL_DIR/llm.gguf
fi
echo OK

if [ "$SMALL" = 1 ]; then
  echo "small box: skipping kokoro warmup (espeak-ng will be used for TTS)"
else
  echo "== kokoro warmup (slow, one time) =="
  sudo -u $SERVICE_USER $AGENT_DIR/venv/bin/python -c "
from kokoro import KPipeline
KPipeline(lang_code='a')
print('kokoro cached')" || echo "kokoro warmup failed - espeak-ng fallback will be used"
  echo OK
fi

echo "== systemd service =="
cp $AGENT_DIR/agent.service /etc/systemd/system/agent.service
systemctl daemon-reload
systemctl enable agent.service
echo OK

echo
echo "DONE. Next:"
echo "  1. Fill in $AGENT_DIR/agent.env (SIP creds, public IP)"
echo "  2. systemctl start agent.service"
echo "  3. journalctl -u agent.service -f"
