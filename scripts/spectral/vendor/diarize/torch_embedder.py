"""WeSpeaker embeddings through THIS PROJECT's PyTorch model.

NOT UPSTREAM CODE. Added by MeetingTranscriber, 2026-08-03, as the replacement
for `wespeakerruntime` (ONNX) — see the marked modification in `embeddings.py`.
It is kept in its own file rather than inlined so the upstream files stay as
close to verbatim as possible: `embeddings.py` differs from upstream by an import
and one constructor call, and everything else in the vendored tree is untouched.

WHY THIS EXISTS
---------------
Upstream's `wespeakerruntime` downloads its own ONNX copy of WeSpeaker
ResNet34-LM from a hardcoded URL. This project already ships that exact model as
PyTorch (`pyannote/wespeaker-voxceleb-resnet34-LM`, the weights
`wespeaker-service.py` loads), so the ONNX path would add a runtime network call
the offline requirement forbids, a second format of identical weights, and
`onnxruntime` as a dependency — for no benefit.

MEASURED, not assumed:
  * cosine between the PyTorch vector and the ONNX vector for the SAME span is
    **1.0000** (three spans); pairwise cosines agree to four decimals.
  * the full upstream pipeline returns IDENTICAL results with either backend —
    same speaker count and same segment count on four real recordings, including
    a 5-speaker meeting where both said 5.
  * it is ~3x faster end to end, because upstream calls onnxruntime once per
    sub-window through a temp WAV.

THE CONTRACT is deliberately upstream's, not a nicer one: `Speaker()` takes no
useful arguments and `extract_embedding(path)` takes a file path, because that is
what `embeddings.py`'s loop already does. Widening it would mean editing more
upstream lines.
"""
from __future__ import annotations

import math
import os

import numpy as np
import soundfile as sf

# Must be set before pyannote is imported anywhere in the process. pyannote 4.x
# ships telemetry ENABLED by default, posting to https://otel.pyannote.ai — which
# the 100 %-offline requirement forbids outright, and which `HF_HUB_OFFLINE` does
# NOT cover (proved with a DNS probe, 2026-07-30). ASSIGNED, never `setdefault`:
# the requirement is absolute, so an inherited environment must not switch it on.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
SR = 16_000


class Speaker:
    """Drop-in replacement for `wespeakerruntime.Speaker`, backed by PyTorch.

    Loaded once per process and reused: the pipeline calls
    `extract_embedding` once per sub-window, which is hundreds of calls on a long
    recording.
    """

    def __init__(self, *args, **kwargs) -> None:  # upstream passes lang="en"
        import torch
        from pyannote.audio.pipelines.speaker_verification import (
            PretrainedSpeakerEmbedding,
        )

        self._torch = torch
        # CPU deliberately. This whole engine is CPU-only by design (upstream's
        # selling point), the model is ~26 MB, and a full 98 s meeting embeds in
        # ~3 s — so MPS would buy nothing while adding an allocator to fuse.
        self._model = PretrainedSpeakerEmbedding(MODEL, device=torch.device("cpu"))

    def extract_embedding(self, path: str) -> np.ndarray:
        """256-dim embedding for one audio file, resampled to 16 kHz mono.

        The resample is NOT optional politeness: feeding this model 44.1 kHz
        audio was measured in this project to produce a vector that scores ~0.036
        against the SAME speaker's 16 kHz vector — a silent identity failure
        rather than an error (see `wespeaker-service.py`).
        """
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        wav = data.mean(axis=1)
        if rate != SR:
            from scipy.signal import resample_poly

            g = math.gcd(int(rate), SR)
            wav = resample_poly(wav, SR // g, int(rate) // g).astype(np.float32)
        tensor = self._torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
        return np.asarray(self._model(tensor.unsqueeze(0))).reshape(-1)
