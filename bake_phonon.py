#!/usr/bin/env python3
"""Build-time bake for Phonon-2 (the worker's only STT, 2026-10-15).

Forces the vendor weights into the image's caches so the worker
never downloads at boot, verifies the unpacked container against
the sha256 pinned in the model's own config.json /
packed_manifest.json (as verified in the phonon2 bench), and proves
the full load + decode path works inside the image. Run by the
Dockerfile only; not part of the runtime.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

# From FermionResearch/Phonon-2 config.json (artifact.container_sha256)
# and packed_manifest.json — the values the 2026-10-15 bench verified.
CONTAINER_SHA256 = \
    "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"
CONTAINER_BYTES = 177438361

import fermion

speech = fermion.load_speech("phonon-2", quiet=True)

candidates = []
for root in (Path.home() / ".cache" / "fermion",
             Path(os.environ.get("HF_HUB_CACHE", ""))):
    if str(root) and root.exists():
        candidates += list(root.rglob("model.fermion"))
if not candidates:
    raise SystemExit("phonon bake: model.fermion not found in caches")
container = candidates[0]
h = hashlib.sha256()
with open(container, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
digest = h.hexdigest()
size = container.stat().st_size
print(f"phonon bake: container {container} {size} bytes sha256={digest}")
assert digest == CONTAINER_SHA256, "container sha256 MISMATCH"
assert size == CONTAINER_BYTES, "container size MISMATCH"

text, decode_s, _ = speech.transcribe_array(
    np.zeros(16000, dtype=np.float32))
print(f"phonon bake OK: silence decode {decode_s:.2f}s text={text!r}")
