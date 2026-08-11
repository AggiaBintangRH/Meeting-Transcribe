#!/usr/bin/env python3
"""
CAM++ speaker diarization sidecar for MeetingTranscriber.

The SIXTH diarization engine (2026-08-11), beside pyannote, MOSS, spectral, NeMo
and DiariZen. Engines are now
`pyannote | moss | spectral | nemo | diarizen | campplus`.

WHAT IT IS, AND WHAT IT IS NOT
------------------------------
CAM++ is a SPEAKER EMBEDDING model, not a diarizer. There is no "CAM++
diarization" to install: what ships here is a pipeline BUILT around it, in the
same three stages every embedding-based diarizer uses —

    Silero VAD  ->  CAM++ embeddings over sliding sub-windows  ->  clustering

That is the same SHAPE as the `spectral` engine and deliberately NOT the same
implementation; see "WHY NOT REUSE SPECTRAL'S PIPELINE" below.

CHECKPOINT: `Wespeaker/wespeaker-voxceleb-campplus-LM` — **Apache 2.0**, 66 MB,
512-dim, trained on VoxCeleb with large-margin fine-tuning.

  ⚠ THE LICENCE DECIDED THE CHECKPOINT, exactly as it did for DiariZen. Of the
  CAM++ weights on the hub, the popular ones are the ModelScope Chinese
  variants; `mlx-community/campplus_multilingual_16k_advanced` is MLX-native and
  would have been the cheapest to load, and it declares **NO LICENCE AT ALL**.
  This app is sold to a client, so "no licence" is not a permissive licence — it
  is all rights reserved. The Apache-2.0 VoxCeleb checkpoint is also the right
  one on the merits here: the meeting language is English.

  It is loaded in PYTORCH rather than MLX on purpose. mlx-audio ships a CAMPPlus
  class (`tts/models/chatterbox/s3gen/xvector.py`) and it was considered first —
  but it defaults to a 192-dim embedding for a different checkpoint, and mapping
  938 torch tensors including BatchNorm running statistics onto it by hand is
  precisely the silent-failure exercise that cost a day on the Fun-ASR sidecar
  the same week. In torch the checkpoint loads with ZERO missing keys.

VENDORED: `vendor/wespeaker/models/{campplus,pooling_layers}.py`, verbatim from
github.com/wenet-e2e/wespeaker (Apache 2.0, LICENSE beside them). Vendored
rather than pip-installed for the reason `build.sh:110` records: it strips
`git+` refs out of the frozen requirements, so a pip install would silently
vanish from the packaged `.app` — the exact failure that shipped a broken DiCoW
to a client machine on 2026-07-27.

NO NEW INTERPRETER, and no new pip dependency: torch, torchaudio, scipy,
soundfile and silero-vad are all already in the main `.venv`. Unlike NeMo,
DiCoW and DiariZen (~1.5 GB of portable Python each) this costs the bundle only
the 66 MB checkpoint.

NOT PYANNOTE-BASED, which is worth stating because every other torch sidecar
here is: this process never imports pyannote, so it needs neither the
`PYANNOTE_METRICS_ENABLED` telemetry opt-out nor the torchcodec avoidance those
files carry. Audio is decoded with soundfile and handed to Silero as a TENSOR,
never as a path — the ffmpeg lesson, which has now bitten this project five
times and is designed out here rather than patched.

MEASURED ON THIS M4 (2026-08-11), CPU:

  file                          truth   result            time
  Meeting5People.wav (98 s)       5     5 speakers        1.7 s  (58x RT)
  Overlap123.wav (39 s)           3     3 speakers        1.1 s  (35x RT)
  meeting-2026-07-30 (67.1 min)   ?     3 spk / 99 turns  87.5 s (46x RT)

  Peak RSS 6.5 GB on the 67-minute file, essentially all of it the affinity
  matrix and its eigendecomposition (see MAX_WINDOWS).

  THE LONG-FILE ROW IS THE INTERESTING ONE. On that same recording `spectral`
  auto-counts **20 speakers / 1869 turns** and pyannote emits a single
  **51.5-minute** turn — the disagreement CLAUDE.md records as open question 2.
  This engine lands on neither extreme: 99 turns, longest 86 s. That is a fifth
  independent measurement, and it is recorded as EVIDENCE, not as a verdict —
  there is still no ground truth for that file, and its room carried music and
  loudspeaker audio, which the ATND findings show is acoustically voice-like.

WHY NOT REUSE SPECTRAL'S PIPELINE, since the shape is the same: its speaker
COUNT ESTIMATION is the part CLAUDE.md documents as broken (GMM-BIC returning 20
speakers on a 67-minute meeting, and 13 on a 3-person clip), while its
clustering is fine. Building a second engine on a stage already known to fail
would inherit the one defect worth avoiding. This uses eigengap on the
normalized Laplacian instead — NME-SC in shape, the same family NeMo uses — and
on the long file above it does not blow up.

TUNING, AND ITS EVIDENCE
------------------------
`PRUNE_PERCENTILE` is the one knob that decides the speaker count. Swept on the
two recordings whose true count is KNOWN, at win 2.0 / hop 1.0:

    pct    Meeting5People (5)   Overlap123 (3)
    0.86         4                    3
    0.90         4                    3
    0.92-0.97    5 ✓                  3 ✓
    0.98         6                    3

A SIX-VALUE PLATEAU, and 0.95 sits near its middle rather than on an edge — the
same standard DiariZen's `ahc_threshold` was judged by, and the reason this is a
measured choice rather than a lucky one. Below the plateau it MERGES a speaker
away; above it it INVENTS one. If a future measurement is ever ambiguous between
two values, take the LOWER one: this project treats fabrication as the worse
direction everywhere else (the ASR hallucination gates, DiariZen's
`min_cluster_size`), and here the lower value is the one that cannot invent.

Protocol — IDENTICAL to the spectral/pyannote services, because one Swift caller
drives them all:
  stdout: {"type":"status","text":"LOADED"}
          {"type":"result","audio":path,
           "segments":[{"start":…,"end":…,"label":"SPEAKER_00"}]}
          {"type":"error","text":...}
          A segment carries ONLY start/end/label — never id, name or conf. This
          process has never seen a profile store and must not look as though it
          could: identity belongs to wespeaker-service.py, per the 2026-07-28
          split. That split is what lets this engine have saved profiles,
          renaming and `spk` for free.
          The "audio" ECHO IS LOAD-BEARING: it is how the app knows which file to
          hand the identity stage next, with no second bookkeeping map.
          result/error carry "stream":"remote" for remote jobs; the field is
          OMITTED for office jobs. A PURE ROUTING ECHO — nothing here is
          per-stream.
          START/END ARE NOT ROUNDED — the identity stage slices the waveform with
          these floats, and rounding would move each slice by up to ~8 samples
          and change the embedding. Rounding happens once, in
          `AudioRecorder.composeTurns`.
  stdin (one JSON per line):
          {"cmd":"final","audio":path,"num_speakers":0}
          optional: "min_speakers", "max_speakers", and
          "stream":"office"|"remote" — ABSENT MEANS OFFICE.
          `exclusive` and `cluster_threshold` are pyannote's knobs and have no
          meaning here (this engine is inherently exclusive — one label per
          instant — and its clustering has no distance threshold). Accepted and
          ignored rather than rejected, so one caller can drive either engine.
          EOF exits.

WHOLE-FILE PASS ONLY, like spectral / NeMo / DiariZen. There is no live/chunk
branch and its absence is asserted by `campplus/no-live-chunk-branch`. Chunking
would not merely be slower, it would be INCOHERENT: this engine counts and
clusters globally, so a 30 s window's "Speaker 1" would have nothing to do with
the next window's.

THIS PROCESS IS STATELESS: nothing persists between jobs except the loaded VAD
and embedder. There is no `reset` — the only session state, the profile stores,
belongs to the WeSpeaker service.

Fully offline: HF_HOME points at the project models/ folder.
"""
import json
import math
import os
import sys
import time
import traceback

# This file lives at scripts/<service>/<file>.py (one folder per service,
# owner 2026-07-29), so the project root — which owns models/ — is THREE levels
# up, not two. With two, HF_HOME/HF_HUB_CACHE silently resolve to
# `scripts/models`, py_compile passes, and it fails at the first model load.
# Bundled, that root is Contents/Resources.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# The vendored WeSpeaker tree, inserted BEFORE the import below. Its own folder
# under this service, never a shared `scripts/vendor/` — the MOSS lesson: two
# services pointing at one tree works today and breaks silently months later
# when that folder moves.
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

MODEL_REPO = "Wespeaker/wespeaker-voxceleb-campplus-LM"
SR = 16_000

# Sub-window geometry. 2.0 s carries enough speech for a stable CAM++ vector
# while still resolving a short turn; 1.0 s hop gives 50 % overlap so a speaker
# change is never straddled by every window at once.
WINDOW_SEC = 2.0
HOP_SEC = 1.0
# A window shorter than this is not embedded at all — CAM++ needs ~0.5 s to say
# anything meaningful, and a 0.2 s vector is noise that clusters as its own
# speaker.
MIN_WINDOW_SEC = 0.5

# THE ONE KNOB THAT DECIDES THE SPEAKER COUNT. See TUNING in the module
# docstring for the sweep this came from and why 0.95 rather than an edge value.
PRUNE_PERCENTILE = 0.95

# Never look for more than this many speakers when counting automatically. Not a
# capability limit — a sanity bound on the eigengap search, which will happily
# find a "gap" at 40 on noisy far-field audio (exactly spectral's 20-speaker
# failure). The owner's real question is 12+ people, so this is set well above
# it and above NeMo's own `max_num_speakers` of 20.
MAX_SPEAKERS = 20

# Cap on sub-windows per job. The clustering builds an N x N affinity matrix and
# eigendecomposes it, which is O(N^3): MEASURED here, `eigh` costs 0.12 s at
# N=1000, 0.86 s at 2000 and 8.7 s at 4000. A 67-minute meeting is N≈4000 and
# already accounts for most of this engine's 6.5 GB peak; a 3-hour one would be
# N≈10800, a 0.93 GB matrix and ~3 minutes in `eigh` alone.
#
# So beyond this cap the HOP IS WIDENED rather than the audio truncated. A long
# meeting loses time RESOLUTION (turn boundaries land on a coarser grid) and
# keeps every second of its content — the opposite trade from dropping audio,
# which would silently lose speakers entirely.
MAX_WINDOWS = 6000


def emit(payload: dict) -> None:
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


def fail(message: str) -> None:
    emit({"type": "error", "text": message})
    sys.exit(1)


def brief_traceback() -> str:
    lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
    return " | ".join(lines[-3:])


def log(message: str) -> None:
    """Timing/diagnostic line → logs/campplus.log via stderr.

    Kept off stdout deliberately: stdout is the app's JSON-lines protocol and an
    extra line there would be a parse error, not a log entry.
    """
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def load_audio_16k(path: str):
    """Decode to 16 kHz mono float32 with soundfile — NEVER through a path API.

    THE FFMPEG LESSON, designed out rather than patched. Whisper hit it
    (`37c16bac9`), pyannote hit it through torchcodec (2026-07-30), spectral hit
    it through `silero_vad.read_audio` (2026-08-04), and DiariZen hit it through
    `torchaudio.load` (2026-08-10). Every one of those was a library handed a
    FILE PATH and reaching for FFmpeg dylibs that the packaged `.app` does not
    carry. Nothing in this sidecar is ever given a path except this function.

    The 16 kHz resample is NOT optional politeness: feeding a WeSpeaker-family
    model 44.1 kHz audio was measured in this project to produce a vector
    scoring ~0.036 against the SAME speaker's 16 kHz vector — a silent identity
    failure rather than an error.
    """
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if rate != SR:
        from scipy.signal import resample_poly

        g = math.gcd(int(rate), SR)
        wav = resample_poly(wav, SR // g, int(rate) // g).astype(np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32)


def load_campplus():
    """The vendored CAM++ with the Apache-2.0 VoxCeleb-LM weights.

    `strict=False` is deliberate and its ONE expected leftover is asserted: the
    checkpoint carries `projection.weight`, the ArcMargin classification head
    used in training, which an embedding extractor must not apply. Everything
    else must match — so a genuinely wrong checkpoint still fails loudly here
    rather than producing plausible-looking garbage vectors, which is the
    failure mode this file's Fun-ASR sibling was written after.
    """
    import glob

    import torch
    from wespeaker.models.campplus import CAMPPlus

    pattern = os.path.join(os.environ["HF_HOME"], "hub",
                           "models--" + MODEL_REPO.replace("/", "--"),
                           "snapshots", "*", "avg_model.pt")
    found = sorted(glob.glob(pattern))
    if not found:
        raise FileNotFoundError(
            f"{MODEL_REPO} is not in {os.environ['HF_HOME']}/hub — "
            "run download-best-models.sh")

    state = torch.load(found[-1], map_location="cpu")
    state = state.get("state_dict", state)
    model = CAMPPlus(feat_dim=80, embed_dim=512)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"CAM++ checkpoint is missing {len(missing)} parameters "
            f"(first: {missing[:3]}) — wrong checkpoint for this architecture")
    unexpected = [k for k in unexpected if not k.startswith("projection.")]
    if unexpected:
        raise RuntimeError(
            f"CAM++ checkpoint has {len(unexpected)} unexpected parameters "
            f"(first: {unexpected[:3]}) — wrong checkpoint for this architecture")
    model.eval()
    return model


def global_fbank(wav):
    """80-bin Kaldi fbank for the WHOLE recording, with GLOBAL mean subtraction.

    GLOBAL CMN IS LOAD-BEARING, not a detail — it was the single change that
    made this engine work. Computing the mean per SUB-WINDOW (the obvious
    implementation, and the first one tried) normalises away the very channel
    and level information a 2 s window needs to be comparable with its
    neighbours: measured on Meeting5People.wav it returned **44 speakers** at a
    0.40 threshold and never fewer than 8, against a truth of 5. With one mean
    over the whole file the same audio returns 5.

    Framed at 10 ms, so frame index == centisecond — which is what lets
    `embed_window` slice by time without a second index.
    """
    import torch
    import torchaudio.compliance.kaldi as kaldi

    tensor = torch.from_numpy(wav).unsqueeze(0) * (1 << 15)
    feats = kaldi.fbank(tensor, num_mel_bins=80, frame_length=25, frame_shift=10,
                        dither=0.0, sample_frequency=SR, window_type="hamming",
                        use_energy=False)
    return feats - feats.mean(dim=0, keepdim=True)


def embed_window(model, feats, start_sec: float, end_sec: float):
    """One L2-normalised 512-dim CAM++ vector, or None if the window is too short."""
    import numpy as np
    import torch

    lo, hi = int(start_sec * 100), int(end_sec * 100)  # 10 ms frames
    segment = feats[lo:hi]
    if segment.shape[0] < int(MIN_WINDOW_SEC * 100):
        return None
    with torch.no_grad():
        out = model(segment.unsqueeze(0))
    out = out[-1] if isinstance(out, tuple) else out
    vector = out.squeeze(0).numpy()
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm


def speech_regions(vad, wav):
    """(start, end) seconds of speech, from Silero VAD.

    Given a TENSOR, never a path — `silero_vad.read_audio` is the exact call
    that made the spectral engine undecodable in the packaged `.app`.
    """
    import torch
    from silero_vad import get_speech_timestamps

    stamps = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR,
                                   min_speech_duration_ms=250,
                                   min_silence_duration_ms=250)
    return [(s["start"] / SR, s["end"] / SR) for s in stamps]


def cluster(vectors, num_speakers: int = 0, min_speakers: int = 0,
            max_speakers: int = 0):
    """Spectral clustering with an eigengap speaker count. Returns integer labels.

    NME-SC in shape — the family NeMo's NME-SC and 3D-Speaker's own recipe both
    belong to — and deliberately NOT the GMM-BIC count estimator the `spectral`
    engine uses, which CLAUDE.md documents returning 20 speakers on a 67-minute
    meeting and 13 on a 3-person clip.

    Row-wise percentile pruning is what makes the eigengap readable: a raw
    cosine affinity over far-field audio is dense and its spectrum has no clear
    gap, so every row keeps only its strongest links and the rest are attenuated
    (not zeroed — a hard zero can disconnect the graph and manufacture a
    component per island, which reads as an extra speaker).
    """
    import numpy as np

    affinity = (vectors @ vectors.T + 1.0) / 2.0  # cosine -> [0, 1]
    np.fill_diagonal(affinity, 0.0)
    for row in range(affinity.shape[0]):
        cut = np.quantile(affinity[row], PRUNE_PERCENTILE)
        affinity[row][affinity[row] < cut] *= 0.01
    affinity = (affinity + affinity.T) / 2.0

    degree = affinity.sum(axis=1)
    degree[degree == 0] = 1e-8
    laplacian = np.eye(len(affinity)) - affinity / np.sqrt(np.outer(degree, degree))
    values, eigenvectors = np.linalg.eigh(laplacian)

    if num_speakers > 0:
        k = num_speakers
    else:
        ceiling = min(max_speakers or MAX_SPEAKERS, MAX_SPEAKERS, len(values) - 1)
        floor = max(min_speakers, 1)
        if ceiling < 2:
            k = 1
        else:
            gaps = np.diff(values[:ceiling + 1])
            # Skip gap 0: the first eigenvalue of a connected graph is ~0 and its
            # gap to the second is always the largest, which would answer "1
            # speaker" for every meeting.
            k = int(np.argmax(gaps[1:]) + 2)
        k = max(k, floor)
    k = max(1, min(k, len(vectors)))

    embedding = eigenvectors[:, :k]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embedding = embedding / norms

    if k == 1:
        return np.zeros(len(vectors), dtype=int), 1

    from scipy.cluster.vq import kmeans2

    # seed fixed so one recording gives one answer: a diarization that changed
    # its labels between two runs of the same file would make every measurement
    # in this project's notes unreproducible.
    _, labels = kmeans2(embedding, k, minit="++", seed=0, iter=50)
    return labels, k


def diarize(model, vad, path: str, num_speakers: int = 0, min_speakers: int = 0,
            max_speakers: int = 0):
    """The whole-file pass: VAD -> windows -> CAM++ -> clustering -> turns."""
    import numpy as np

    wav = load_audio_16k(path)
    regions = speech_regions(vad, wav)
    if not regions:
        return []

    speech_sec = sum(e - s for s, e in regions)
    hop = HOP_SEC
    estimated = int(speech_sec / hop) if hop else 0
    if estimated > MAX_WINDOWS:
        hop = speech_sec / MAX_WINDOWS
        log(f"{speech_sec / 60:.1f} min of speech would need {estimated} windows; "
            f"widening the hop {HOP_SEC:.2f}s -> {hop:.2f}s to stay under "
            f"{MAX_WINDOWS} (coarser boundaries, no audio dropped)")

    feats = global_fbank(wav)
    spans, vectors = [], []
    for start, end in regions:
        cursor = start
        while cursor < end - MIN_WINDOW_SEC:
            stop = min(cursor + WINDOW_SEC, end)
            vector = embed_window(model, feats, cursor, stop)
            if vector is not None:
                spans.append((cursor, stop))
                vectors.append(vector)
            cursor += hop

    if len(vectors) < 2:
        # One window cannot be clustered, but it IS speech and dropping it would
        # be over-deletion. It is one speaker by definition.
        return [{"start": s, "end": e, "label": "SPEAKER_00"} for s, e in spans]

    labels, _ = cluster(np.array(vectors), num_speakers, min_speakers, max_speakers)

    return merge_windows(spans, labels)


def merge_windows(spans, labels):
    """Collapse the labelled sliding windows into non-intersecting turns.

    A MODULE-LEVEL FUNCTION rather than a few lines inside `diarize` so the
    invariant below can be pinned without loading a 66 MB checkpoint:
    `campplus/turns-never-intersect` calls this directly with synthetic
    overlapping windows.
    """
    # Collapse consecutive same-label windows into turns. Windows overlap by
    # design, so touching spans are merged on time rather than on index.
    #
    # THE TURNS THIS RETURNS MUST NEVER INTERSECT, and that is not a tidiness
    # rule — it is the property the whole engine is placed by. This diarizer
    # assigns exactly ONE label per window, so it cannot detect overlap; two
    # intersecting turns therefore cannot mean "two people spoke at once", they
    # can only be an artefact. The app does not know that: `overlapRegions()`
    # infers overlap purely from turns that intersect by >= 0.4 s, and would tag
    # the transcript with simultaneous speech that never happened — fabrication,
    # the direction this project treats as the worst one.
    #
    # Emitting each window's span verbatim DID exactly that. Consecutive windows
    # are WINDOW_SEC wide at HOP_SEC spacing, so they share WINDOW_SEC - HOP_SEC
    # of audio (1.0 s at the shipped constants); when the label changes across
    # that pair, the two turns overlap by precisely that much. Measured on
    # `Overlap123.wav` before the fix: four intersecting pairs, every one of them
    # 1.000 s, at every speaker change.
    turns = []
    for (start, end), label in zip(spans, labels):
        label = int(label)
        if turns and turns[-1]["_label"] == label and start <= turns[-1]["end"] + 1e-6:
            turns[-1]["end"] = max(turns[-1]["end"], end)
            continue
        if turns and start < turns[-1]["end"]:
            # A LABEL CHANGE INSIDE THE SHARED SPAN. The true switch is somewhere
            # in it and nothing here can locate it more precisely — both windows
            # cover that audio and they disagree. So the boundary goes at the
            # MIDPOINT, which minimises the worst-case error and is the same rule
            # `WordAttribution.snapRanges` applies to a pause on the Swift side.
            # Clamped so a turn can never be trimmed behind its own start.
            boundary = max(turns[-1]["start"], (turns[-1]["end"] + start) / 2)
            turns[-1]["end"] = boundary
            start = max(start, boundary)
        if end > start:
            turns.append({"start": start, "end": end, "_label": label})
    # A window whose whole span was consumed by the clamp above leaves a
    # zero-width turn. Dropping it removes no audio from any speaker — the
    # neighbour it was trimmed against covers exactly that span.
    return [{"start": t["start"], "end": t["end"],
             "label": f"SPEAKER_{t['_label']:02d}"}
            for t in turns if t["end"] - t["start"] > 1e-6]


def main() -> None:
    try:
        import numpy  # noqa: F401
        import torch
    except Exception:  # noqa: BLE001
        fail(f"torch/numpy import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    # CPU deliberately, and measured rather than assumed: the whole pass is
    # 46-58x realtime on CPU, the model is 66 MB, and MPS would add an allocator
    # to fuse for a stage that is not the bottleneck (the eigendecomposition is,
    # and that is numpy on CPU either way).
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4))))

    try:
        from silero_vad import load_silero_vad

        vad = load_silero_vad()
    except Exception:  # noqa: BLE001
        fail(f"Silero VAD load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    try:
        model = load_campplus()
    except Exception:  # noqa: BLE001
        fail(f"CAM++ load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    # Warmed HERE rather than on the first job, so LOADED means loaded. On this
    # engine the first job IS the stop pass, so a load failure discovered then
    # would land after the meeting, with nothing left to re-run — the exact
    # defect the 2026-08-10 NeMo audit fixed.
    import numpy as np

    warm = np.zeros(int(1.5 * SR), dtype=np.float32)
    embed_window(model, global_fbank(warm), 0.0, 1.5)

    log(f"loaded {MODEL_REPO} (CAM++ 512-dim, Apache 2.0) + Silero VAD · "
        f"window {WINDOW_SEC}s hop {HOP_SEC}s pct {PRUNE_PERCENTILE}")
    emit({"type": "status", "text": "LOADED"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:  # noqa: BLE001
            emit({"type": "error", "text": f"Bad job line: {line[:120]}"})
            continue

        # Which identity space this job belongs to. ABSENT MEANS OFFICE, so a
        # single-stream app drives this sidecar exactly as it drives pyannote.
        # Routing information ONLY: nothing here behaves differently for remote.
        stream = job.get("stream")
        is_remote = stream == "remote"
        echo = {"stream": stream} if stream else {}

        # Whole-file passes and nothing else. Refused loudly, because a
        # global-clustering engine answering a windowed request would return
        # labels that look continuous and are not.
        mode = job.get("cmd", "final")
        if mode != "final":
            emit({"type": "error",
                  "text": f"campplus has only a whole-file pass (got cmd={mode!r})",
                  **echo})
            continue

        audio = job.get("audio", "")
        if not os.path.exists(audio):
            emit({"type": "error", "text": f"Audio not found: {audio}", **echo})
            continue

        # A WAV whose header says ZERO FRAMES is not "a meeting where nobody
        # spoke" — it is a recording whose `data` chunk size was never written,
        # because the app went away before `AVAudioFile` was released. The
        # samples are still in the file; only the count is missing. Without this
        # the VAD finds no speech in an empty array and the pass returns
        # `segments: []` — a whole meeting reported as having no speakers, with
        # no message anywhere. That silence is the direction this project treats
        # as dangerous, and it is why spectral grew the same guard.
        try:
            import soundfile as sf

            frames = sf.info(audio).frames
        except Exception:  # noqa: BLE001 — an unreadable header is its own answer
            frames = -1
        if frames == 0:
            emit({"type": "error",
                  "text": "This recording's WAV header says 0 frames, so nothing "
                          "can read it — the audio is almost certainly still in "
                          "the file and the size field was never written (the app "
                          "was quit or killed mid-recording). Repair it with "
                          "scripts/tools/repair-wav-header.py, then run the pass "
                          "again.",
                  **echo})
            continue

        num_speakers = int(job.get("num_speakers", 0))
        min_speakers = int(job.get("min_speakers", 0))
        max_speakers = int(job.get("max_speakers", 0))

        started = time.time()
        try:
            segments = diarize(model, vad, audio, num_speakers,
                               min_speakers, max_speakers)
            log(f"final done in {time.time() - started:.1f}s — {len(segments)} turns, "
                f"{len({s['label'] for s in segments})} speakers"
                f"{' [remote]' if is_remote else ''}"
                f"{f' (count pinned to {num_speakers})' if num_speakers else ''}")
            emit({"type": "result", "audio": audio, "segments": segments, **echo})
        except Exception:  # noqa: BLE001 — job error: report, keep serving
            # The stream is echoed so the app fails only THAT stream's gate — a
            # remote failure must never abort the office transcript.
            emit({"type": "error",
                  "text": f"Diarization failed: {brief_traceback()}", **echo})


if __name__ == "__main__":
    main()
