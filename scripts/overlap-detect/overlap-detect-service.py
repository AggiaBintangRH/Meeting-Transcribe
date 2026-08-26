#!/usr/bin/env python3
"""Overlap DETECTION — where did two people speak at once, and nothing more.

WHY THIS IS ITS OWN SERVICE, not a mode of `pyannote-service.py`.
Under the MOSS and spectral diarization engines the pyannote PIPELINE is never
loaded — that is what makes every pyannote-dependent path a natural no-op there
— and those are exactly the engines that need this, because both assign one
speaker per instant and can never mark overlap themselves. Adding a mode to a
process that does not exist in those sessions would mean loading the whole
pipeline (clustering, embeddings, ~1.17 GB) to use one 32 MB network. So this
follows the project's own rule instead: one service per model, one log per
service.

WHAT IT LOADS. Only the `segmentation/` sub-model of the
`speaker-diarization-community-1` checkpoint this app already ships. Nothing is
downloaded. Its head is a POWERSET over speaker combinations, so "two speakers
active" is a class it predicts directly rather than something inferred by
thresholding separate per-speaker curves — which is why the decode below goes
through `Powerset.to_multilabel` and not a `> 0.5` per column (that mistake reads
0 speakers everywhere, measured).

MEASURED on the owner's M4 before this was offered in Settings at all:
  * loads in ~0 s, 32 MB;
  * scans at ~160x realtime — a 43-minute meeting in 16 s;
  * verified in BOTH directions: 16.2 % of a clip that genuinely contains
    overlap, 0.0 % of a clean one, and 0.1-1.8 % of the owner's real meetings.
    It does not simply fire on room noise.

WHAT IT DOES NOT DO. It never recovers the words. During genuine overlap a
single microphone carries one mixed waveform, and every separation attempt in
this project (PixIT, MossFormer2, DiCoW) was measured and removed. A region here
means "the text in this row is approximate", which is the whole value.

Protocol, one JSON object per line on stdin:
    {"cmd": "detect", "audio": "/path.wav"}   -> {"type":"result","audio":...,
                                                  "regions":[[start,end],...]}
Replies: {"type":"status","text":"LOADED"} once ready, {"type":"error","text":..}
on failure. Errors never kill the process — a failed detection must not cost the
user their transcript.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
import traceback

# THREE dirname calls: this file is one folder under scripts/, so two would
# resolve HF_HOME to scripts/models. The trap is documented in CLAUDE.md and has
# cost a debugging round before.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(BASE, "models", "hub"))

# ASSIGNED, never setdefault: pyannote 4.x ships telemetry ON by default and the
# offline requirement is absolute, so an inherited environment must not be able
# to switch it back on. Same rule as pyannote-service.py and wespeaker-service.py.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

#: Frames shorter than this are not worth marking — a row-level tag that fires on
#: 40 ms of crosstalk would mark most of a lively meeting and mean nothing.
MIN_REGION_SEC = 0.20

#: The segmentation network's own window. Fixed by the checkpoint, not a choice.
WINDOW_SEC = 10.0

#: Hard ceiling on this process's GPU allocation, in GB. Inert while the model
#: runs on CPU (it is tiny), armed anyway for the day that changes — the same
#: reasoning spectral-service.py records.
#:
#: HALF THE MACHINE, derived rather than typed, like every other MPS sidecar
#: here. It was the literal 32.0 until 2026-08-26, which on a 16 GB Mac is twice
#: the physical RAM: the cap is armed as
#: `set_per_process_memory_fraction(min(CAP / ceiling, 1.0))`, and macOS reports
#: ~13.0 GB as the ceiling there, so `32.0 / 13.0` clamped the `min` to 1.0 —
#: the whole allocator, uncapped.
#:
#: ⚠ THAT WAS HARMLESS HERE AND IS STILL WORTH FIXING. This model runs on CPU,
#: so there is no MPS allocation to cap either way; the bug was real in
#: nemo-service.py, which does run on MPS. A fuse kept "armed for the day that
#: changes" has to be armed CORRECTLY, or the day it matters it will not act —
#: and that day arrives as a one-line device change nobody connects to this
#: constant.
def _half_the_physical_ram_gb(fallback: float = 32.0) -> float:
    """Half this machine's physical RAM, in GB — the rule stated above, computed.

    Verbatim from the other MPS sidecars. Standalone by the same decision that
    made every sidecar standalone (owner, 2026-07-28): the copies ARE the point,
    and `layout/mps-fuse-is-sized-to-the-machine` is what makes copying safe —
    it discovers every caller of `set_per_process_memory_fraction` rather than
    naming files, so a new one cannot be forgotten the way this file was.

    `sysconf` rather than `torch.mps.recommended_max_memory()` because this is a
    module-level constant and torch is not imported yet in every service that
    carries one. On this 64 GB M4 it returns exactly 32.0 — byte-for-byte the
    value that shipped — so the change is a verified no-op here and acts only on
    smaller machines.

    Falls back to the historical constant if the query fails. A fuse that cannot
    size itself must not become a fuse that refuses everything.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        half = total / (1024 ** 3) / 2.0
        return round(half, 1) if half > 0 else fallback
    except (ValueError, OSError, AttributeError):  # noqa: BLE001
        return fallback


MPS_MEMORY_CAP_GB = _half_the_physical_ram_gb()


def emit(payload: dict) -> None:
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def log(message: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def fail(message: str) -> None:
    emit({"type": "error", "text": message})
    sys.exit(1)


def brief_traceback() -> str:
    lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
    return " | ".join(lines[-3:])


def find_segmentation() -> str:
    """The segmentation weights inside the community-1 snapshot we already ship."""
    pattern = os.path.join(BASE, "models", "hub",
                           "models--pyannote--speaker-diarization-community-1",
                           "snapshots", "*", "segmentation", "pytorch_model.bin")
    hits = sorted(glob.glob(pattern))
    if not hits:
        fail("The pyannote segmentation model was not found — run "
             "download-best-models.sh (it is part of speaker-diarization-community-1).")
    return hits[-1]


def load_audio_16k(path: str):
    """Decode with soundfile, resample with scipy — never hand a path to a library.

    THE FFMPEG LESSON, for the fourth time in this project. `torchaudio.load` is a
    torchcodec wrapper since TorchAudio 2.9, torchcodec links FFmpeg dylibs
    through a single Homebrew LC_RPATH, and `build.sh` PRUNES torchcodec from both
    bundled interpreters — so a client Mac has no decoder at all on that path.
    soundfile carries its own libsndfile and is already in the bundle.
    """
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    rate = int(rate)
    if rate != 16_000:
        from scipy.signal import resample_poly

        g = math.gcd(rate, 16_000)
        data = resample_poly(data, 16_000 // g, rate // g).astype("float32")
    return np.ascontiguousarray(data, dtype="float32")


def main() -> None:
    try:
        import numpy as np
        import torch
        from pyannote.audio import Model
        from pyannote.audio.utils.powerset import Powerset
    except Exception:  # noqa: BLE001
        fail(f"overlap-detect import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    if torch.backends.mps.is_available():
        try:
            ceiling = torch.mps.recommended_max_memory() / (1024 ** 3)
            if ceiling > 0:
                torch.mps.set_per_process_memory_fraction(
                    min(MPS_MEMORY_CAP_GB / ceiling, 1.0))
        except Exception:  # noqa: BLE001
            log("WARNING could not cap MPS memory")

    weights = find_segmentation()
    log(f"loading pyannote segmentation from {weights}")
    try:
        model = Model.from_pretrained(weights)
        model.eval()
        powerset = Powerset(len(model.specifications.classes),
                            model.specifications.powerset_max_classes)
    except Exception:  # noqa: BLE001
        fail(f"segmentation load failed: {brief_traceback()}")

    log(f"overlap detector ready — CPU, window {WINDOW_SEC:.0f}s, "
        f"minimum region {MIN_REGION_SEC:.2f}s")
    emit({"type": "status", "text": "LOADED"})

    def detect(path: str):
        """Regions, in RECORDING seconds, where >= 2 speakers are active."""
        import torch as _t

        wav = load_audio_16k(path)
        win = int(WINDOW_SEC * 16_000)
        regions: list[list[float]] = []
        run_start = None
        total_frames = 0
        overlap_frames = 0

        for offset in range(0, max(1, len(wav)), win):
            chunk = wav[offset:offset + win]
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            with _t.no_grad():
                logits = model(_t.from_numpy(chunk).unsqueeze(0))
                multilabel = powerset.to_multilabel(logits)[0].numpy()
            active = multilabel.sum(axis=1)
            step = WINDOW_SEC / multilabel.shape[0]
            total_frames += len(active)
            overlap_frames += int((active >= 2).sum())
            base = offset / 16_000
            for i, n in enumerate(active):
                t = base + i * step
                if n >= 2 and run_start is None:
                    run_start = t
                elif n < 2 and run_start is not None:
                    # Closed here rather than at the window edge, so a region that
                    # spans two windows is not split into two short ones.
                    if t - run_start >= MIN_REGION_SEC:
                        regions.append([round(run_start, 3), round(t, 3)])
                    run_start = None
        if run_start is not None:
            end = len(wav) / 16_000
            if end - run_start >= MIN_REGION_SEC:
                regions.append([round(run_start, 3), round(end, 3)])
        share = overlap_frames / total_frames * 100 if total_frames else 0.0
        return regions, share

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:  # noqa: BLE001
            emit({"type": "error", "text": f"Bad job line: {line[:120]}"})
            continue

        if job.get("cmd") != "detect":
            emit({"type": "error",
                  "text": f"overlap-detect serves only cmd=detect (got {job.get('cmd')!r})"})
            continue

        audio = job.get("audio", "")
        if not os.path.exists(audio):
            emit({"type": "error", "text": f"Audio not found: {audio}"})
            continue

        started = time.time()
        try:
            regions, share = detect(audio)
            took = time.time() - started
            log(f"detect done in {took:.1f}s — {len(regions)} region(s), "
                f"{share:.1f}% of the recording")
            emit({"type": "result", "audio": audio, "regions": regions})
        except Exception:  # noqa: BLE001 — report and keep serving
            emit({"type": "error", "text": f"Overlap detection failed: {brief_traceback()}"})


if __name__ == "__main__":
    main()
