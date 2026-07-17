#!/usr/bin/env python3
"""
Silero VAD v6 sidecar for MeetingTranscriber.

Protocol (all over stdin/stdout):
  - prints "READY" once the model is loaded
  - then reads raw float32 mono 16 kHz PCM from stdin in 512-sample frames
    (2048 bytes) and writes one speech probability (0..1) per frame as a
    text line to stdout.

Fully offline: model weights ship inside the silero-vad pip package.
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from silero_vad import load_silero_vad  # noqa: E402

FRAME = 512          # samples per frame @ 16 kHz (what Silero expects)
FRAME_BYTES = FRAME * 4

torch.set_num_threads(1)  # tiny model — one thread is fastest

model = load_silero_vad()
if hasattr(model, "reset_states"):
    model.reset_states()

sys.stdout.write("READY\n")
sys.stdout.flush()

stdin = sys.stdin.buffer
while True:
    data = stdin.read(FRAME_BYTES)
    if not data or len(data) < FRAME_BYTES:
        break
    chunk = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
    with torch.no_grad():
        prob = float(model(chunk, 16000).item())
    sys.stdout.write(f"{prob:.4f}\n")
    sys.stdout.flush()
