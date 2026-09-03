#!/usr/bin/env python3
"""
VibeVoice speaker diarization sidecar for MeetingTranscriber.

The SEVENTH diarization engine (2026-09-02), owner-requested after the client
asked for VibeVoice by name. Engines are now
`pyannote | moss | spectral | nemo | diarizen | campplus | vibevoice`.

WHAT IT IS, AND WHY IT IS A BATCH ENGINE
----------------------------------------
Same checkpoint as `vibevoice-asr` and `vibevoice-rt` — microsoft/VibeVoice-ASR-
Streaming-1.5B, MIT — in its THIRD role. The model is speaker-attributed ASR: it
emits `Speaker N:` runs inside the transcript, and those runs are the diarization.

⚠ IT RUNS ONCE OVER THE WHOLE RECORDING AT STOP, like spectral / NeMo /
DiariZen / CAM++, and NOT per chunk like MOSS. That is the one design decision
in this file and it was made against measurement rather than taste:

  * MOSS diarizes per chunk and must then STITCH labels across chunk seams,
    because its labels are anonymous per call. On the 43-minute 7-speaker
    recording that produced 77 per-window labels collapsing to 10-11 people
    against a truth of 7 — the worst result of any engine on that file.
  * VibeVoice keeps ONE running KV cache across the whole pass, so `Speaker 0`
    at minute 40 is the same decision as `Speaker 0` at minute 1. There are no
    seams to stitch, and stitching is what MOSS gets wrong.

A whole-file pass therefore costs nothing that a windowed one would save, and
avoids the failure mode of the only other engine shaped like this.

⚠ TIMES ARE DERIVED FROM THE CHUNK GRID, NOT REPORTED BY THE MODEL. The
streaming checkpoint emits text per chunk with no timestamps; each chunk covers
a known `chunk_duration` (2.9333 s here, read from the checkpoint). Where one
chunk holds several `Speaker N:` runs the window is split between them in
proportion to their text length — the same character-position estimate
`assignSentences` uses on the Swift side, and it is an ESTIMATE, stated here so
nobody reads these boundaries as measured.

  The consequence, so it is not discovered later: boundary precision is at best
  ~2.9 s, against pyannote's measured median error of 263 ms. For a transcript
  whose rows are sentences that is usually invisible; for anything that needs a
  precise turn edge it is not good enough. The non-streaming VibeVoice checkpoint
  reports real timestamps and is the upgrade path if this engine is ever kept.

⚠ AND THE SPEAKER COUNT CANNOT BE PINNED. `streaming_generate` takes no
`num_speakers` and the package contains none anywhere — checked by grep over the
whole vendored tree, not inferred. `num_speakers` therefore arrives on the wire
(every engine gets it) and is LOGGED AND IGNORED, exactly as `--language` is in
the other two roles. Measured auto counts on this project's known-answer files,
recorded so the number is not rediscovered:

    Overlap123.wav        truth 3  ->  3   ✓
    Meeting5People.wav    truth 5  ->  4   ✗   every other engine gets 5
    client ATND 02-52-34  truth 5  ->  1   ✗✗  five people merged into one

  The SPK control beside the record button is what rescues pyannote, spectral
  and CAM++ on that ATND file (4 -> 5). It cannot rescue this one.

MEMORY: 10.96 GB RSS, 9.8 GB MPS pool. Above the 11.8 GB ceiling of a 16 GB Mac
once the other sidecars are counted. Its own interpreter (`.venv-vibevoice`)
because transformers <5.0 conflicts with the MLX stack's 5.x.

Protocol — IDENTICAL to the spectral/campplus/pyannote services, because one
Swift caller drives them all:
  stdout: {"type":"status","text":"LOADED"}
          {"type":"result","audio":path,
           "segments":[{"start":…,"end":…,"label":"SPEAKER_00"}]}
          {"type":"error","text":...}
          A segment carries ONLY start/end/label — never id, name or conf. This
          process has never seen a profile store and must not look as though it
          could: identity belongs to wespeaker-service.py, per the 2026-07-28
          split. That split is what gives this engine saved profiles, renaming
          and `spk` for free.
  stdin:  one JSON job per line:
          {"cmd":"final","audio":"/path.wav"[,"num_speakers":N][,"stream":"remote"]}

⚠ WHOLE-FILE PASSES ONLY. A `chunk` job is REFUSED loudly rather than served:
this engine's labels are decided by one continuous cache over the whole pass, so
a windowed request would return labels that look continuous across windows and
are not — the failure MOSS actually has.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# This service's OWN vendored tree — byte-identical to the other two roles' and
# deliberately not shared. `layout/vibevoice-vendor-trees-are-own-and-identical`
# reads this literal by AST and hash-compares all three.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

#: `Speaker 12:` at the start of a run. The model writes the label with no
#: space before the colon and no space after it; both are tolerated here so a
#: checkpoint that formats differently degrades to "one speaker" rather than to
#: an exception.
SPEAKER_RE = re.compile(r"Speaker\s*(\d+)\s*:", re.IGNORECASE)


def log(message: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def fail(message: str) -> None:
    emit({"type": "error", "text": message})
    log(f"FATAL {message}")
    sys.exit(1)


def brief_traceback() -> str:
    return " | ".join(traceback.format_exc().strip().splitlines()[-3:])


def frame_config(model_path: str) -> dict:
    """Chunk and lookahead from the CHECKPOINT — upstream's own arithmetic.
    See the chunked service for why it is neither hard-coded nor imported from
    `demo/`; `layout/vibevoice-frame-config-matches-upstream` pins all copies."""
    with open(os.path.join(model_path, "preprocessor_config.json"),
              encoding="utf-8") as fh:
        cfg = json.load(fh)
    missing = [k for k in ("chunk_frames", "lookahead_frames") if k not in cfg]
    if missing:
        fail(f"{model_path} has no {', '.join(missing)} — it is not a STREAMING "
             "checkpoint, and this service can only drive a streaming one")
    sample_rate = int(cfg["target_sample_rate"])
    frame_seconds = cfg["speech_tok_compress_ratio"] / sample_rate
    return {
        "sample_rate": sample_rate,
        "chunk_duration": cfg["chunk_frames"] * frame_seconds,
        "text_audio_delay": cfg["lookahead_frames"] * frame_seconds,
    }


def segments_from_chunks(chunks: list[str], chunk_duration: float) -> list[dict]:
    """Turn per-chunk text into speaker spans on the chunk grid.

    Each chunk covers `[i*chunk_duration, (i+1)*chunk_duration]`. A chunk holding
    one `Speaker N:` run becomes one span over that whole window. A chunk holding
    several is split between them IN PROPORTION TO TEXT LENGTH — an estimate, and
    the same one `assignSentences` makes on the Swift side.

    ⚠ A run that CONTINUES from the previous chunk carries no label of its own;
    it inherits the last speaker seen. That is what makes a sentence spanning a
    chunk boundary stay with one speaker instead of being handed to whoever the
    grid happened to start.

    Adjacent spans with the same label are merged, so a speaker who talks for a
    minute is ONE turn rather than twenty consecutive ones.
    """
    spans: list[dict] = []
    current = 0  # inherited when a chunk opens mid-run
    for index, text in enumerate(chunks):
        start = index * chunk_duration
        end = start + chunk_duration
        if not (text or "").strip():
            continue
        # Split the chunk into (speaker, text) runs. `re.split` with one capture
        # group yields [before, id, text, id, text, …]; `before` is the tail of
        # the previous chunk's speaker and keeps `current`.
        parts = SPEAKER_RE.split(text)
        runs: list[tuple[int, str]] = []
        head = parts[0].strip()
        if head:
            runs.append((current, head))
        for i in range(1, len(parts) - 1, 2):
            current = int(parts[i])
            body = parts[i + 1].strip()
            if body:
                runs.append((current, body))
        if not runs:
            continue
        total = sum(len(t) for _, t in runs) or 1
        cursor = start
        for speaker, body in runs:
            width = (end - start) * (len(body) / total)
            spans.append({"start": cursor, "end": cursor + width, "label": speaker})
            cursor += width
        # Float drift over hundreds of chunks would otherwise walk the last span
        # off the grid; snap it back to the window edge it belongs to.
        spans[-1]["end"] = end

    merged: list[dict] = []
    for span in spans:
        if merged and merged[-1]["label"] == span["label"]:
            merged[-1]["end"] = span["end"]
        else:
            merged.append(dict(span))
    return [{"start": round(s["start"], 3),
             "end": round(s["end"], 3),
             "label": f"SPEAKER_{s['label']:02d}"} for s in merged]


def main() -> None:
    model_path = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            model_path = args[i + 1]
    if not model_path:
        fail("--model is required")

    fc = frame_config(model_path)
    log(f"loading VibeVoice diarization from {model_path}")
    t0 = time.time()
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration)
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        if processor.tokenizer.text_chunk_end_id is None:
            fail(f"{model_path} is not a streaming checkpoint: its tokenizer has "
                 "no <|text_chunk_end|>, so no chunk would ever end")
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path, attn_implementation="sdpa")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(torch.float32).to(device).eval()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail(f"model load failed: {brief_traceback()} — run download-best-models.sh")

    log(f"loaded on {device} in {time.time() - t0:.1f}s — chunk "
        f"{fc['chunk_duration']:.4f}s, whole-file passes only")
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

        stream = job.get("stream")
        echo = {"stream": stream} if stream else {}

        if job.get("cmd", "final") != "final":
            emit({"type": "error",
                  "text": "VibeVoice diarizes the whole recording in one pass and "
                          "cannot answer a windowed request: its labels come from "
                          "ONE continuous cache, so per-window labels would look "
                          "continuous across windows without being so.",
                  **echo})
            continue

        audio = job.get("audio")
        if not audio or not os.path.exists(audio):
            emit({"type": "error", "text": f"Audio not found: {audio}", **echo})
            continue

        # ACCEPTED AND IGNORED, said out loud. Every engine is sent the SPK
        # control's value; this model has no parameter to receive it.
        pinned = job.get("num_speakers") or 0
        if pinned:
            log(f"num_speakers={pinned} received and IGNORED — this model has no "
                "such parameter, so the count is always automatic")

        started = time.time()
        try:
            # SAMPLES, NEVER A PATH — upstream's demo decodes with ffmpeg, which
            # is not bundled. The trap this project has hit six times.
            data, rate = sf.read(audio, dtype="float32", always_2d=True)
            mono = data[:, 0]
            target = fc["sample_rate"]
            if rate != target:
                import scipy.signal as ss
                mono = ss.resample_poly(mono, target, rate).astype(np.float32)
            if mono.size == 0:
                emit({"type": "error",
                      "text": "The recording decodes to zero frames. If the app "
                              "was force-quit while recording, repair the header "
                              "with scripts/tools/repair-wav-header.py.", **echo})
                continue

            chunks: list[str] = []
            for _idx, _total, text in model.streaming_generate(
                    audio_tensor=torch.from_numpy(mono),
                    tokenizer=processor.tokenizer,
                    chunk_duration=fc["chunk_duration"],
                    text_audio_delay=fc["text_audio_delay"],
                    sample_rate=target,
                    max_new_tokens_per_chunk=256,
                    temperature=0.0):
                chunks.append(text)

            segments = segments_from_chunks(chunks, fc["chunk_duration"])
            speakers = len({s["label"] for s in segments})
            log(f"final done in {time.time() - started:.1f}s — {len(segments)} turns, "
                f"{speakers} speakers{' [remote]' if stream == 'remote' else ''}")
            emit({"type": "result", "audio": audio, "segments": segments, **echo})
        except Exception:  # noqa: BLE001 — job error: report, keep serving
            emit({"type": "error",
                  "text": f"diarization failed: {brief_traceback()}", **echo})


if __name__ == "__main__":
    main()
