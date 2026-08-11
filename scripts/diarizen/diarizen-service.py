#!/usr/bin/env python3
"""Speaker DIARIZATION sidecar — the DiariZen engine (persistent, stateless).

ONE SERVICE PER ROLE. The FIFTH diarizer here, beside `pyannote-service.py`,
`spectral-service.py`, `moss-diar-service.py` and `nemo-service.py`. Like the
first two and NeMo it answers WHO-SPOKE-WHEN with pipeline-LOCAL labels and has
no idea who those people are — identity lives in `wespeaker-service.py` alone.

THE ENGINE is DiariZen (BUT Speech@FIT), checkpoint `BUT-FIT/diarizen-meeting-base`:
a WavLM-Base+ encoder with Conformer layers producing END-TO-END neural
segmentation (EEND), then WeSpeaker embeddings and agglomerative clustering.

That makes it architecturally UNLIKE the other four. pyannote, spectral and NeMo
all segment first and embed second; DiariZen's segmentation is a neural model
trained on far-field single-channel MEETINGS (AMI, AISHELL-4, AliMeeting) — the
same conditions as the owner's ATND array, which is the whole reason it is here.

LICENSING, and it is the reason THIS checkpoint and no other:
  * The code (this repo's `vendor/`) is MIT.
  * `BUT-FIT/diarizen-meeting-base` is **MIT** — verified from its own model card.
  * Every OTHER DiariZen checkpoint (`wavlm-large-s80-md`, `-mlc`, `-v2`,
    `-origin`, `wavlm-base-s80-md`) is **CC BY-NC 4.0**, non-commercial, because
    they add RAMC / MSDWild / DIHARD-3 to the training mix. This project ships to
    a paying client, so those are unusable. DO NOT swap the checkpoint for a
    "better" one without re-reading its licence — five of the six are forbidden.
  * The repo's blanket `MODEL_LICENSE` says NC for "the weights"; the per-model
    tags disagree and are more specific. Both files are vendored so the claim can
    be re-checked rather than remembered.

FINAL ONLY — no live/per-chunk path, for the same reason as spectral and NeMo:
clustering is global, so a 30 s window's labels would mean nothing across windows.
Any `cmd` other than `final` is REFUSED loudly.

Protocol (byte-compatible with the pyannote, spectral and NeMo services):
  stdout: {"type":"status","text":"LOADED"}
          {"type":"result","audio":path,
           "segments":[{"start":…,"end":…,"label":"0"}]}
          {"type":"error","text":...}
          Segments carry ONLY start/end/label — never id, name or conf.
          The "audio" ECHO is load-bearing: it tells the app which file to hand
          the identity stage, with no second bookkeeping map.
          result/error carry "stream":"remote" for remote jobs and OMIT the key
          for office ones, so the office wire is identical to pyannote's.
          START/END ARE NOT ROUNDED — the identity stage slices the waveform with
          them, and rounding here would move each slice and change the embedding.
  stdin:  {"cmd":"final","audio":path,"num_speakers":0}
          optional "stream":"office"|"remote" — ABSENT MEANS OFFICE.
          EOF exits.

IT REUSES THE WESPEAKER CHECKPOINT THIS APP ALREADY SHIPS
(`pyannote/wespeaker-voxceleb-resnet34-LM`), because `HF_HOME` points at the
project's own `models/`. No second copy of a 26 MB embedder, and the vectors are
the same ones `wespeaker-service.py` scores against.
"""
import json
import os
import sys
import time
import traceback

# ---------------------------------------------------------------- THE WIRE
WIRE = sys.stdout

# scripts/<service>/<file>.py → the project root is THREE levels up. With two,
# HF_HOME silently resolves to `scripts/models` and the first job re-downloads
# every weight into a folder nobody looks in. Bundled, that root is Resources.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HOME"] = os.path.join(BASE, "models")
os.environ["HF_HUB_OFFLINE"] = "1"

# MPS NEEDS THIS, and it is not optional on the pinned torch.
#
# Upstream pins torch 2.1.1, whose MPS backend has no `aten::_weight_norm_interface`
# — the op WavLM's weight-normalised convolutions use. Without the flag the first
# job dies with NotImplementedError; with it, that one op runs on CPU and the rest
# stays on MPS. This is PyTorch's own documented remedy, not a workaround invented
# here.
#
# ASSIGNED, not `setdefault`: an inherited empty value would re-break the engine.
# Measured on this M4 over `Meeting5People.wav` — CPU 209.2 s (0.5x realtime),
# MPS with this flag 7.7 s (12.8x), and the two produce IDENTICAL output
# (4 speakers, 11 turns), which is what makes MPS verified here rather than
# assumed (the MOSS precedent, not DiCoW's).
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_ID = "BUT-FIT/diarizen-meeting-base"

# The ONE deviation from the checkpoint's own clustering config, and it is
# measured rather than chosen (2026-08-10).
#
# `min_cluster_size` is how few segments a cluster may hold before it is
# DISCARDED. Upstream's 30 threw away the quietest speaker on the only recording
# in this project whose answer is actually known.
#
#     min_cluster_size   Meeting5People (truth 5)   July slice   1-person monologue
#            30                    4                    2               2
#            15                    5  <-- correct        2               2
#             5                    6                    5               -
#
# So: one file corrected, two unchanged, nothing made worse. That is why 15.
#
# IT IS NOT "SMALLER IS BETTER", AND THE ROW FOR 5 IS THERE TO SAY SO. The
# parameter is monotonic in the SPEAKER COUNT, not in accuracy: at 5 it invents
# people — six on a five-person recording, five on a slice holding about two.
#
# The two failure directions are not equally bad in a transcript. Too high loses a
# speaker: their words survive under someone else's name. Too low SPLITS one
# person into two names, and the reader sees a conversation that never happened
# between two people who are one. This project treats fabrication as the worse
# direction everywhere else (the ASR hallucination gates), and it applies here.
# If a future measurement is ambiguous between two values, take the HIGHER one.
#
# Everything else is left at the checkpoint's defaults, including `min_speakers`
# and `max_speakers`: both were measured INERT on every file here (removing the
# 2..8 bounds entirely changed no answer), so overriding them would be a change
# with no evidence behind it. THE COUNT ITSELF IS ALWAYS AUTOMATIC — there is no
# setting for it and none is sent (owner, 2026-08-10).
MIN_CLUSTER_SIZE = 15

# The vendored source. `pip install`ed into `.venv-diarizen` non-editable, from
# THIS tree, because an editable install freezes as a `git+` ref and `build.sh`
# strips those — the exact failure that shipped a broken DiCoW to a client on
# 2026-07-27. The tree is here so the bundled interpreter can be built from it.
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def install_wire() -> str:
    """Take fd 1 for the JSON protocol and send every other writer to stderr.

    DIARIZEN PRINTS TO STDOUT, AND THAT DESTROYS THIS PROTOCOL. Measured on the
    first drive of this sidecar (2026-08-10): a single job put `Loaded
    configuration: {...}`, `self.embedding: …`, `Extracting segmentations.`,
    `Extracting Embeddings.` and `Clustering.` on fd 1, interleaved with our JSON.
    A line-by-line decoder dies on the first of them.

    NeMo has the identical defect and the identical remedy — pyannote, spectral
    and MOSS all log to stderr, so this is a per-service mitigation rather than a
    shared rule, and each copy names the writers it was written against.

    AT THE FILE-DESCRIPTOR LEVEL, not `sys.stdout`, on purpose: upstream's bare
    `print()`, pyannote's progress bars, tqdm, and C-level writes from libsndfile
    are separate writers, and rebinding a Python name reaches only one of them.
    Duplicating fd 1 to a private handle and pointing fd 1 at fd 2 moves all of
    them into stderr, which is `logs/diarizen.log`, where they belong.

    MUST RUN BEFORE DIARIZEN IS IMPORTED, for the same reason as NeMo's: a handler
    bound to fd 1 at import time would hold the wire for the life of the process.
    """
    global WIRE
    WIRE = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return "wire isolated on a private fd; every other stdout writer -> stderr"


def emit(payload: dict) -> None:
    WIRE.write(json.dumps(payload) + "\n")
    WIRE.flush()


def log(message: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def fail(message: str) -> None:
    emit({"type": "error", "text": message})
    log(message)
    sys.exit(1)


def brief_traceback() -> str:
    return " | ".join(traceback.format_exc().strip().splitlines()[-3:])


def main() -> None:
    # FIRST, before any heavy import — see `install_wire`. DiariZen prints on load.
    wire_note = install_wire()
    try:
        import torch

        from diarizen.pipelines.inference import DiariZenPipeline
    except Exception:  # noqa: BLE001
        fail(f"DiariZen import failed: {brief_traceback()} — run "
             "download-best-models.sh (this engine needs .venv-diarizen)")

    if not os.path.isdir(VENDOR):
        fail(f"the vendored DiariZen source is missing at {VENDOR}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        pipeline = DiariZenPipeline.from_pretrained(MODEL_ID)
        # Re-instantiated from the checkpoint's OWN parameters with one field
        # replaced, rather than a dict written here: everything upstream tuned on
        # AMI/AliMeeting stays exactly as they left it, and a future checkpoint
        # bringing different values keeps them.
        clustering = dict(pipeline.PIPELINE_PARAMS["clustering"])
        clustering["min_cluster_size"] = MIN_CLUSTER_SIZE
        pipeline.instantiate({"clustering": clustering})
        if device != "cpu":
            # The constructor hardcodes `cuda:0 if available else cpu` and takes
            # no device argument, so the move happens here. It is a pyannote
            # Pipeline underneath, which is what makes `.to()` available at all.
            pipeline.to(torch.device(device))
    except Exception:  # noqa: BLE001
        fail(f"DiariZen model load failed: {brief_traceback()} — "
             f"expected {MODEL_ID} under {os.environ['HF_HOME']}")

    log(wire_note)
    log(f"device={device} (MPS fallback armed for aten::_weight_norm_interface)")
    log(f"model={MODEL_ID} (MIT; the other five DiariZen checkpoints are CC BY-NC)")
    log(f"min_speakers={pipeline.min_speakers} max_speakers={pipeline.max_speakers} "
        f"min_cluster_size={MIN_CLUSTER_SIZE} (upstream 30 — see its declaration)")
    log("DiariZen engine ready — turns only, no identity in this process, "
        "final pass only")
    emit({"type": "status", "text": "LOADED"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:  # noqa: BLE001
            emit({"type": "error", "text": f"bad job line: {line[:120]}"})
            continue

        is_remote = job.get("stream") == "remote"
        echo = {"stream": "remote"} if is_remote else {}

        mode = job.get("cmd", "final")
        if mode != "final":
            emit({"type": "error",
                  "text": f"DiariZen has only a whole-file pass (got cmd={mode!r})",
                  **echo})
            continue

        audio = job.get("audio", "")
        if not os.path.exists(audio):
            emit({"type": "error", "text": f"Audio not found: {audio}", **echo})
            continue

        # A WAV whose header says ZERO FRAMES is not a silent meeting — it is a
        # recording whose `data` size was never written because the app went away
        # mid-recording. The samples are still there. Verbatim spectral and NeMo
        # semantics: silence is the direction this project treats as dangerous.
        try:
            import soundfile as sf

            frames = sf.info(audio).frames
        except Exception:  # noqa: BLE001
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

        started = time.time()
        try:
            result = pipeline(audio)
            segments = [
                {"start": turn.start, "end": turn.end, "label": str(speaker)}
                for turn, _, speaker in result.itertracks(yield_label=True)
            ]
            log(f"final done in {time.time() - started:.1f}s — {len(segments)} turns, "
                f"{len({s['label'] for s in segments})} speakers"
                f"{' [remote]' if is_remote else ''}")
            emit({"type": "result", "audio": audio, "segments": segments, **echo})
        except Exception:  # noqa: BLE001 — job error: report, keep serving
            # The stream is echoed so the app fails only THAT stream's gate; a
            # remote failure must never abort the office transcript.
            emit({"type": "error",
                  "text": f"Diarization failed: {brief_traceback()}", **echo})


if __name__ == "__main__":
    main()
