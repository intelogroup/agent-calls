#!/bin/bash
# Provision an Oracle Cloud Ampere A1 VM (Ubuntu 24.04) as the 24/7 voice agent.
# Run as root (or with sudo). Idempotent-ish: safe to re-run.
set -euo pipefail

AGENT_DIR=/opt/agent
MODEL_DIR=$AGENT_DIR/models
SERVICE_USER=agent

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
$AGENT_DIR/venv/bin/pip -q install -r $AGENT_DIR/requirements.txt
echo OK

echo "== whisper model (tiny.en) =="
sudo -u $SERVICE_USER $AGENT_DIR/venv/bin/python -c "
from faster_whisper import WhisperModel
WhisperModel('tiny.en', device='cpu', compute_type='int8')
print('whisper cached')"
echo OK

echo "== LLM model (Qwen2.5-1.5B Q4_K_M) =="
if [ ! -f $MODEL_DIR/llm.gguf ]; then
    sudo -u $SERVICE_USER $AGENT_DIR/venv/bin/hf download \
        Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf \
        --local-dir $MODEL_DIR >/dev/null
    mv $MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf $MODEL_DIR/llm.gguf
fi
echo OK

echo "== kokoro warmup (slow, one time) =="
sudo -u $SERVICE_USER $AGENT_DIR/venv/bin/python -c "
from kokoro import KPipeline
KPipeline(lang_code='a')
print('kokoro cached')" || echo "kokoro warmup failed - espeak-ng fallback will be used"
echo OK

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
