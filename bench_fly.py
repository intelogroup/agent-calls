import os, time
import numpy as np

def t():
    return time.perf_counter()

print("spec_bench start", flush=True)
print("spec_bench cpu_count", os.cpu_count(), flush=True)
try:
    ci = open("/proc/cpuinfo").read()
    print("spec_bench cpu_model", ci.split("model name")[1].split(":")[1].split("\n")[0].strip(), flush=True)
except Exception as e:
    print("spec_bench cpu_model ?", e, flush=True)
print("spec_bench loadavg", [round(x, 2) for x in os.getloadavg()], flush=True)

# ---- TTS: production Kokoro path ----
from kokoro_onnx import Kokoro
t0 = t()
k = Kokoro("/opt/agent/models/kokoro-v1.0.int8.onnx",
           "/opt/agent/models/voices-v1.0.bin")
print("spec_bench kokoro_load_s", round(t() - t0, 2), flush=True)
sents = [
    "Hey, it's Jett's line. What do you want to talk about?",
    "I can hear you loud and clear, Jim. What can I do for you?",
    "Cambridge Health Alliance says they have no flu shot on record for you, "
    "and they want you to call or use the link to schedule before the "
    "October thirty-first deadline.",
]
for s in sents:
    t0 = t()
    samples, sr = k.create(s, voice="af_heart", speed=1.0, lang="en-us")
    dt = t() - t0
    dur = len(samples) / sr
    print(f"spec_bench tts chars={len(s)} synth_s={dt:.3f} audio_s={dur:.2f} "
          f"rtf={dt / dur:.3f} sr={sr}", flush=True)

# ---- STT test input: a Kokoro-spoken question ----
q = ("Hey Jett, can you hear me well, and what is on my calendar "
     "this Thursday afternoon?")
samples, sr = k.create(q, voice="af_heart", speed=1.0, lang="en-us")
print("spec_bench stt_input_audio_s", round(len(samples) / sr, 2), flush=True)
n = int(len(samples) * 16000 / sr)
audio16 = np.interp(np.linspace(0, len(samples), n, endpoint=False),
                    np.arange(len(samples)), samples).astype(np.float32)

# ---- STT: production faster-whisper path, streaming simulation ----
from faster_whisper import WhisperModel
t0 = t()
m = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("spec_bench whisper_load_s", round(t() - t0, 2), flush=True)

def transcribe(a):
    segs, _ = m.transcribe(a, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()

step = int(16000 * 0.35)
i = 0
while i < len(audio16):
    buf = audio16[: i + step]
    i += step
    t0 = t()
    txt = transcribe(buf)
    dt = t() - t0
    print(f"spec_bench stt_step buf_s={len(buf) / 16000:.2f} decode_s={dt:.3f} "
          f"interim={txt[:40]!r}", flush=True)
t0 = t()
final = transcribe(audio16)
print(f"spec_bench stt_final decode_s={t() - t0:.3f} text={final!r}", flush=True)
print("spec_bench done", flush=True)
