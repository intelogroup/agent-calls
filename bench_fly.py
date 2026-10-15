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

# ---- TTS: production Pocket path (via live_agent's warm engine) ----
import live_agent as la
engine = la.PocketTTS()
t0 = t()
la._PocketEngine.get()
print("spec_bench pocket_load_s", round(t() - t0, 2), flush=True)
sents = [
    "Hey, it's Jett's line. What do you want to talk about?",
    "I can hear you loud and clear, Jim. What can I do for you?",
    "Cambridge Health Alliance says they have no flu shot on record for you, "
    "and they want you to call or use the link to schedule before the "
    "October thirty-first deadline.",
]
for s in sents:
    t0 = t()
    samples, sr = engine._synth(s)
    dt = t() - t0
    dur = len(samples) / sr
    print(f"spec_bench tts chars={len(s)} synth_s={dt:.3f} audio_s={dur:.2f} "
          f"rtf={dt / dur:.3f} sr={sr}", flush=True)

# ---- STT test input: a Pocket-spoken question ----
q = ("Hey Jett, can you hear me well, and what is on my calendar "
     "this Thursday afternoon?")
samples, sr = engine._synth(q)
print("spec_bench stt_input_audio_s", round(len(samples) / sr, 2), flush=True)
n = int(len(samples) * 16000 / sr)
audio16 = np.interp(np.linspace(0, len(samples), n, endpoint=False),
                    np.arange(len(samples)), samples).astype(np.float32)

# ---- STT: production Phonon-2 path (vendor LiveSession) ----
import fermion
from fermion._speech.live import LiveSession, warm_up
t0 = t()
speech = fermion.load_speech("phonon-2", quiet=True)
print("spec_bench phonon_load_s", round(t() - t0, 2), flush=True)
warm_up(speech)

partials = []
sess = LiveSession(speech, on_partial=lambda tx: partials.append(tx))
step = int(16000 * 0.05)
fed = 0
for i in range(0, len(audio16), step):
    block = audio16[i:i + step]
    fed += len(block)
    t0 = t()
    sess.feed(block)
    dt = t() - t0
    if partials:
        print(f"spec_bench stt_step fed_s={fed / 16000:.2f} decode_s={dt:.3f} "
              f"interim={partials[-1][:40]!r}", flush=True)
        partials.clear()
t0 = t()
final = sess.finish()
print(f"spec_bench stt_final decode_s={t() - t0:.3f} text={final!r}", flush=True)
print("spec_bench done", flush=True)
