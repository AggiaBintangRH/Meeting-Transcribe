#!/usr/bin/env python3
"""Speaker DIARIZATION sidecar — the SPECTRAL engine (persistent, stateless).

ONE SERVICE PER ROLE. This is the third diarizer in the project, alongside
`pyannote/pyannote-service.py` (pyannote community-1) and
`moss-diar/moss-diar-service.py` (MOSS speaker-attributed ASR). Like the pyannote
one, it answers WHO-SPOKE-WHEN with pipeline-LOCAL labels (`SPEAKER_00`,
`SPEAKER_01`, …) and has no idea who those people are. Identity — the WeSpeaker
embeddings and the two persistent profile stores — lives in
`scripts/wespeaker/wespeaker-service.py` and nowhere else.

THE ENGINE is the vendored, Apache-2.0 `diarize` package
(github.com/FoxNoseTech/diarize @ 4f25d27, `vendor/diarize/`): Silero VAD →
sliding-window WeSpeaker embeddings → GMM-BIC speaker counting → spectral
clustering → temporal (Viterbi) label smoothing. Its ONE modification is
`vendor/diarize/torch_embedder.py`, which swaps upstream's `wespeakerruntime`
(ONNX, downloads its own weights) for this project's existing PyTorch
`pyannote/wespeaker-voxceleb-resnet34-LM` — measured cosine 1.0000 against the
ONNX vector and identical pipeline output on four real recordings.

FINAL ONLY — THERE IS DELIBERATELY NO LIVE / PER-CHUNK PATH IN v1.
The engine counts speakers globally (GMM-BIC over every embedding in the file)
and then clusters globally, so a 30 s window would be counted and clustered on
its own and its labels would mean nothing across windows — the same failure MOSS
has when it is called per chunk (documented at length in CLAUDE.md: two different
people both numbered `S01` because the chunk boundary fell on the speaker change).
Adding a live branch "for symmetry" would therefore ship labels that look
continuous and are not. A job carrying any `cmd` other than `final` is REFUSED
loudly rather than served by a whole-file pass pretending to be a window.

THIS PROCESS IS STATELESS: nothing persists between jobs except the loaded VAD
and embedder. There is no `reset`, because the only session state — the profile
stores — belongs to the WeSpeaker service.

Protocol:
  stdout: {"type":"status","text":"LOADED"}
          {"type":"result","audio":path,
           "segments":[{"start":…,"end":…,"label":"SPEAKER_00"}]}
          {"type":"error","text":...}
          A segment carries ONLY start/end/label — never id, name or conf. That
          is the pyannote/wespeaker split made structural for this engine too:
          identity cannot leak out of a pipeline stage, so a pipeline can never
          be the thing that writes a voice into a profile store.
          The "audio" ECHO IS LOAD-BEARING: it is how the app knows which file to
          hand the identity stage next, with no second bookkeeping map.
          result/error carry "stream":"remote" for remote jobs; the field is
          OMITTED for office jobs, so the office wire format is identical to the
          pyannote service's. It is a PURE ROUTING ECHO — nothing in this process
          is per-stream.
          START/END ARE NOT ROUNDED. The pipeline's own floats go out verbatim
          because the identity stage slices the waveform with them; rounding here
          would move each slice by up to ~8 samples and change the embedding.
          Rounding to 3 dp happens once, where it always has: at the point the
          app-visible turn is composed (`AudioRecorder.composeTurns`).
  stdin (one JSON per line):
          {"cmd":"final","audio":path,"num_speakers":0}
          optional: "min_speakers", "max_speakers", and
          "stream":"office"|"remote" — ABSENT MEANS OFFICE.
          `exclusive` and `cluster_threshold` are pyannote's knobs and have no
          meaning here (this engine is inherently exclusive — one speaker per
          instant — and its clustering has no threshold). They are accepted and
          ignored rather than rejected, so one caller can drive either engine.
          EOF exits.

Fully offline: HF_HOME points at the project models/ folder, PYANNOTE_METRICS_ENABLED
turns off pyannote 4.x's default-on telemetry (the embedder is a pyannote model),
and audio is decoded here with soundfile so torchcodec — and its Homebrew-only
ffmpeg — is never reached. The last two are not tidiness; see the comments on each.
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

# pyannote.audio 4.x ships OpenTelemetry tracing that is ON BY DEFAULT
# (pyannote/audio/telemetry/config.yaml: `metrics_enabled: true`) and posts to
# https://otel.pyannote.ai/v1/traces from every Model init — which is exactly
# what this engine's embedder is (`PretrainedSpeakerEmbedding` in
# vendor/diarize/torch_embedder.py). A span carries the pyannote version, a
# random session uuid and the audio duration: no audio and no text, but still a
# network call describing a client meeting, which client hard requirement #1
# (100 % offline) forbids outright. `HF_HUB_OFFLINE` does NOT cover it — that
# flag only gates the Hugging Face hub (proved with a DNS probe, 2026-07-30).
#
# Assigned, not `setdefault`: the requirement is absolute, so an inherited
# environment must not be able to turn it back on. `torch_embedder.py` assigns it
# too, deliberately — but this process must not depend on the import order of a
# vendored file to stay offline, so it is set here, first, unconditionally.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

# The vendored engine lives under THIS service, and each service inserts its OWN
# vendor dir (the MOSS rule: two byte-identical trees mean pointing at the other
# one works today and breaks silently the month it moves).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

# Hard ceiling on this process's GPU allocation, in GB — armed in main().
#
# A fuse, not a budget. macOS reports a "recommended max working set" (51.8 GB on
# the owner's 64 GB M4) and PyTorch's MPS allocator grows all the way to it
# WITHOUT EVER RAISING; MOSS proved that is not theoretical (one oversized call
# reached ~50 GB and the Mac had to be rescued by hand).
#
# This engine's embedder runs on CPU on purpose (see torch_embedder.py), so the
# fuse is inert TODAY. It is armed anyway: moving the embedder to MPS is a
# one-line edit, and a fuse that has to be remembered at that moment is a fuse
# that will not be there. HALF THE MACHINE, derived below rather than typed —
# the same rule every other MPS-capable sidecar uses.
def _mps_cap_gb(fallback: float = 32.0) -> float:
    """The MPS fuse for THIS machine, in GB: half the physical RAM, floored at 10.

    ⚠ WHY NOT THE LITERAL 32.0. That number WAS "half the machine", for the 64 GB
    M4 every figure in this project was measured on. On a 16 GB Mac it is TWICE
    the physical RAM, and because the cap is armed as
    `set_per_process_memory_fraction(min(CAP / ceiling, 1.0))` the `min` clamped
    to 1.0 — the whole allocator, uncapped. Captured in the wild on the client's
    Mac, which printed it three times:

        MPS memory capped at 32 GB of the 11.8 GB macOS reports as available

    ⚠ AND WHY THE FLOOR OF 10, which is the half of this that was LEARNED THE
    HARD WAY. "Half the physical RAM" alone is a rule calibrated on a 64 GB
    machine, and it does not survive being scaled down. It leaves 19.8 GB of
    macOS's 51.8 GB recommendation spare at 64 GB, but only 3.8 GB of 11.8 at
    16 GB — and the OTHER sidecars need a roughly FIXED ~4 GB, not a proportion.
    So on a small machine the proportional rule squeezes out work that fits.

    Measured, on the client's 16 GB Mac: with the cap at half (8.0 GB), NeMo
    raised `MPS backend out of memory (MPS allocated: 8.23 GiB ... max allowed:
    8.00 GiB)` — on a machine where macOS itself was willing to give 11.8 GB, and
    where NeMo had run FINE two days earlier, uncapped, in 5.7 s. The fuse was
    stricter than the operating system and broke working functionality. That is
    the over-deletion direction this project ranks worst, in a new costume.

    The floor is 10.0 rather than a proportion because the number that matters is
    absolute: it must clear NeMo's measured 8.23 GB with margin, and stay under
    MOSS's measured 18.35 GB pool so the engine that actually took a Mac down on
    2026-08-21 is still refused. It changes NOTHING on a machine of 20 GB or more
    — this 64 GB M4 still gets exactly 32.0.

    ⚠ THE COST, owner-accepted with it stated (2026-08-26): 10 GB of MPS pool plus
    ~4 GB of other sidecars plus macOS is tight on a 16 GB machine, so a long NeMo
    meeting may still fail — its peak is 13.33 GB at 67 minutes — or the machine
    may slow down. The owner asked for exactly this: let it run, and let the lag be
    the evidence for a hardware upgrade rather than an invisible refusal. The
    failure stays BOUNDED either way, which is the whole point: PyTorch raises at
    the cap instead of growing without limit.

    `sysconf` rather than `torch.mps.recommended_max_memory()` because this is a
    module-level constant and torch is not imported yet in every service that
    carries one.

    Falls back to the historical constant if the query fails. A fuse that cannot
    size itself must not become a fuse that refuses everything.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        half = total / (1024 ** 3) / 2.0
        return round(max(half, 10.0), 1) if half > 0 else fallback
    except (ValueError, OSError, AttributeError):  # noqa: BLE001
        return fallback


MPS_MEMORY_CAP_GB = _mps_cap_gb()


def emit(payload: dict) -> None:
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def fail(message: str) -> None:
    emit({"type": "error", "text": message})
    sys.exit(1)


def brief_traceback() -> str:
    lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
    return " | ".join(lines[-3:])


def log(message: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def install_soundfile_vad_reader() -> str:
    """Make the vendored VAD decode with soundfile instead of torchaudio.

    THE FFMPEG LESSON, A THIRD TIME — and here it is a real defect in the engine
    we vendored, not a style preference.

    `vendor/diarize/vad.py` calls `silero_vad.read_audio(path)`. On the installed
    torchaudio (2.11) that function is `torchaudio.load(path)`, which as of
    TorchAudio 2.9 is a **torchcodec** wrapper; on failure it falls back to
    `torchcodec.decoders.AudioDecoder` and, failing that, raises. torchcodec's
    `libtorchcodec_core*` dylibs link libavcodec/libavformat/libavutil through a
    single `LC_RPATH` of `/opt/homebrew/opt/ffmpeg/lib`, and the 2026-07-30
    pyannote audit PRUNES torchcodec from both bundled interpreters — so on a
    client Mac this call has no decoder at all and VAD dies at the first job,
    exactly as pyannote's `Audio.crop` did before `load_waveform` was written.

    Neither of the two other path-taking stages has this problem, checked rather
    than assumed: `utils.get_audio_duration` uses `sf.info` (its torchaudio
    fallback is reached only if soundfile itself fails), and both
    `embeddings.extract_embeddings` and `torch_embedder.extract_embedding` use
    `sf.read`. The VAD is the only hole.

    The fix is applied HERE, in the sidecar, rather than by editing the vendored
    tree: `vad.py` imports `read_audio` from `silero_vad` *inside* the function,
    so rebinding the attribute on the package is picked up at call time and the
    upstream files stay verbatim (they differ from upstream by exactly one import
    and one constructor call, and that is worth keeping true).

    Resampling is scipy's `resample_poly`, the same one `torch_embedder.py`
    already uses for this model, instead of `torchaudio.transforms.Resample`.
    Every WAV this app hands the engine is 16 kHz mono, so the branch is not even
    taken in practice; measured on `recordings/Meeting5People.wav` (16 kHz mono)
    the shim's tensor is bit-identical to the original reader's.

    Returns a short description for the log.
    """
    import numpy as np
    import silero_vad
    import soundfile as sf
    import torch

    def read_audio(path, sampling_rate: int = 16000):
        # 1-D for mono, and NOT `always_2d=True` + `.mean(axis=1)` — that pair
        # forces a `(N, 1)` array and then allocates a SECOND full-length array to
        # collapse it back, for a result identical to the first by definition.
        # It was one copy of the entire recording (4 bytes/sample) bought for
        # nothing; see the memory note in `embeddings.py` for the other one.
        # Multichannel input still collapses, so the contract is unchanged.
        data, rate = sf.read(str(path), dtype="float32")
        wav = data if data.ndim == 1 else data.mean(axis=1)
        rate = int(rate)
        if rate != sampling_rate:
            from scipy.signal import resample_poly

            g = math.gcd(rate, int(sampling_rate))
            wav = resample_poly(wav, int(sampling_rate) // g, rate // g).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))

    silero_vad.read_audio = read_audio
    return "VAD decoding routed through soundfile (no torchcodec, no ffmpeg)"


# ---------------------------------------------------------------- main
def main() -> None:
    try:
        import torch

        reader_note = install_soundfile_vad_reader()
        from diarize import diarize as run_diarize
    except Exception:  # noqa: BLE001
        fail(f"spectral engine import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    # Arm the fuse BEFORE anything loads weights, so even a bad load cannot run
    # away. Wrapped because these APIs are version-dependent and a missing one
    # must not cost us diarization: an uncapped run is what we have always had,
    # so failing to cap is a warning, never fatal.
    if torch.backends.mps.is_available():
        try:
            ceiling_gb = torch.mps.recommended_max_memory() / (1024 ** 3)
            if ceiling_gb > 0:
                torch.mps.set_per_process_memory_fraction(
                    min(MPS_MEMORY_CAP_GB / ceiling_gb, 1.0))
                log(f"MPS memory capped at {MPS_MEMORY_CAP_GB:.0f} GB "
                    f"of the {ceiling_gb:.1f} GB macOS reports as available "
                    f"(inert while the embedder is on CPU — armed for the day it is not)")
        except Exception:  # noqa: BLE001
            log("WARNING could not cap MPS memory")

    log(reader_note)
    log(f"HF_HOME={os.environ.get('HF_HOME')}")

    # Warm the two models NOW rather than on the first job, so LOADED means
    # loaded: a 30 s stop-pass gate must not also be paying for a lazy import.
    # Both are small (Silero ~2 MB, WeSpeaker ~26 MB) and CPU-resident.
    log("loading Silero VAD + WeSpeaker ResNet34-LM (CPU)")
    try:
        from silero_vad import load_silero_vad

        load_silero_vad()
        from diarize.torch_embedder import Speaker

        Speaker()
    except Exception:  # noqa: BLE001
        fail(f"spectral model load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    log("spectral engine ready — turns only, no identity in this process, "
        "final pass only")
    emit({"type": "status", "text": "LOADED"})

    def wire_segments(result):
        """Pipeline-LOCAL turns on the wire: start, end, label. Nothing else.

        No id, no name, no conf — this process has never seen a profile store and
        must not look as though it could. See the module docstring on why the
        times are not rounded.
        """
        return [{"start": seg.start, "end": seg.end, "label": seg.speaker}
                for seg in result.segments]

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

        # v1 serves whole-file passes and nothing else — see the module
        # docstring. Refused loudly, because a global-clustering engine answering
        # a windowed request would return labels that look continuous and are not.
        mode = job.get("cmd", "final")
        if mode != "final":
            emit({"type": "error",
                  "text": f"spectral has only a whole-file pass in v1 (got cmd={mode!r})",
                  **echo})
            continue

        audio = job.get("audio", "")
        if not os.path.exists(audio):
            emit({"type": "error", "text": f"Audio not found: {audio}", **echo})
            continue

        # A WAV whose header says ZERO FRAMES is not "a meeting where nobody
        # spoke" — it is a recording whose `data` chunk size was never written,
        # because the app went away before `AVAudioFile` was released. The samples
        # are still in the file; only the count is missing. Measured 2026-08-05:
        # 13 such recordings (~1.7 GB) on the owner's machine, one holding 34.1
        # minutes of speech.
        #
        # Without this, VAD finds no speech in an empty array and the pass returns
        # `segments: []` — a whole meeting reported as having no speakers, with no
        # message anywhere. pyannote already fails loudly on the same file (its
        # pipeline raises), so this is the one engine that was silent, and silence
        # is the direction this project treats as dangerous.
        try:
            import soundfile as sf

            frames = sf.info(audio).frames
        except Exception:  # noqa: BLE001 — an unreadable header is its own answer
            frames = -1
        if frames == 0:
            emit({"type": "error",
                  "text": "This recording's WAV header says 0 frames, so nothing "
                          "can read it — the audio is almost certainly still in the "
                          "file and the size field was never written (the app was "
                          "quit or killed mid-recording). Repair it with "
                          "scripts/tools/repair-wav-header.py, then run the pass "
                          "again.",
                  **echo})
            continue

        # 0 = auto (GMM-BIC decides). min/max only bound the auto search and are
        # upstream's own defaults when absent.
        num_speakers = int(job.get("num_speakers", 0))
        kwargs = {"num_speakers": num_speakers} if num_speakers > 0 else {}
        if int(job.get("min_speakers", 0)) > 0:
            kwargs["min_speakers"] = int(job["min_speakers"])
        if int(job.get("max_speakers", 0)) > 0:
            kwargs["max_speakers"] = int(job["max_speakers"])

        started = time.time()
        try:
            result = run_diarize(audio, **kwargs)
            segments = wire_segments(result)
            log(f"final done in {time.time() - started:.1f}s — "
                f"{len(segments)} turns, {len({s['label'] for s in segments})} speakers"
                f"{' [remote]' if is_remote else ''}")
            emit({"type": "result", "audio": audio, "segments": segments, **echo})
        except Exception:  # noqa: BLE001 — job error: report, keep serving
            # The stream is echoed so the app fails only THAT stream's gate —
            # a remote failure must never abort the office transcript.
            emit({"type": "error",
                  "text": f"Diarization failed: {brief_traceback()}", **echo})


if __name__ == "__main__":
    main()
