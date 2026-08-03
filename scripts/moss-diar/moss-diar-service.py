#!/usr/bin/env python3
"""
MOSS-Transcribe-Diarize sidecar for MeetingTranscriber — the DIARIZATION ROLE.

Speaker-attributed ASR: ONE 0.9B model returns the transcript, the speaker
labels and the timestamps together, instead of an ASR pass plus a separate
diarization pass. Upstream: https://github.com/OpenMOSS/MOSS-Transcribe-Diarize

ONE SERVICE PER ROLE, completed here (phase 2, 2026-07-31). MOSS is selectable in
TWO places — as the chunked ASR model, and as the diarization engine — and in
"MOSS as ASR + MOSS as diarizer" mode BOTH run at once. Until phase 1 they were
one script, so two live processes wrote into one log. This file serves the
DIARIZATION role only (`ChunkedASRService.Config.mossDiarization()` →
logs/moss-diar.log); the ASR role runs scripts/moss-asr/moss-asr-service.py →
logs/moss-asr.log. Do not merge the two back together, and do not point either
role at the other's file: the log pair would drift (see
ChunkedASRService.Config.logName).

THE `-2` FILE-TRANSCRIBE FRAME AND `emit_file` ARE DELIBERATELY ABSENT — this is
the ONE behavioural difference from the ASR file, and it is not an oversight to
be "restored for symmetry" (the whisper-service.py precedent: a file must not
carry a branch it is never sent). Proven at the call sites: `transcribeFile` is
reached from exactly three places, and every one of them goes through
`modelLoader.chunkedASR` — AudioRecorder+OverlapRepair.swift (guarded by
`guard let chunked = modelLoader.chunkedASR`, and the whole repair pass is
skipped outright when the diarization engine is MOSS) and
AudioRecorder+RemoteStream.swift (its `chunked` parameter is bound from
`modelLoader.chunkedASR` at both AudioRecorder.swift call sites). The diarization
process is held in `mossDiarService`, which is only ever fed audio frames and
FLUSH. So a `-2` frame can never arrive here, and a branch that cannot run is a
branch that can only rot. Pinned by sidecar-tests.py
`moss/diar-has-no-file-branch`.

Protocol — byte-identical framing to every other ASR sidecar MINUS the `-2`
frame, so the SAME Swift client (`ChunkedASRService`) drives them all:
  stdout: JSON lines
          {"type":"status","text":"LOADED"}
          {"type":"final","text":"Hello there. Hi.","segments":[
              {"start":0.0,"end":4.1,"speaker":"S01","text":"Hello there."},
              {"start":4.4,"end":5.2,"speaker":"S02","text":"Hi."}]}
          {"type":"error","text":"..."}
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes)
            n == 0 : FLUSH — transcribe buffered chunk, emit final, reset
            n == -1: exit

`text` is the segments' text joined in order with the tags stripped. In this role
the caller keeps `segments` and discards `text` — but the key is STRUCTURALLY
MANDATORY and stays POPULATED: `ChunkedASRService.Message.text` is non-optional,
so a `final` without it fails JSONDecoder, the line is silently skipped, the
pendingChunkWindows FIFO never drains and the 180 s watchdog fires on every
chunk. Emitting `""` would decode, but it would forfeit the byte-identical
equivalence proof against the ASR copy and break the emit_final/FLUSH comparisons
in `moss/asr-matches-moss` for no gain. `segments` is ADDITIVE and optional —
every consumer that predates it still decodes a MOSS final. Times in `segments`
are CHUNK-LOCAL seconds; the app offsets them by the window start. `speaker` is
the model's RAW label ("S01"); it is anonymous PER CALL — S01 in chunk N is not
S01 in chunk N+1, and nothing here pretends otherwise.

No "conf" and no "words", ever: this model reports neither, and absent means
"not measured" (house convention), never "low".

Args: --model <hf-repo>   [--language <code>]  (accepted and ignored — MOSS has
no language argument; the flag exists only so the shared Swift client can never
fail by passing one.)

Fully offline: HF_HUB_OFFLINE=1, weights read from the project models/ folder.
"""
import argparse
import json
import os
import struct
import sys
import tempfile
import time
import traceback
import wave

# This file lives at scripts/moss-diar/ (one folder per service, owner 2026-07-29),
# so the project root — which owns models/ — is THREE levels up, not two.
# Bundled, that root is Contents/Resources.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set BEFORE transformers is imported. The app normally passes these in
# (PythonRuntime.sidecarEnvironment); the defaults keep a hand-run sidecar
# pointed at the project's models/ folder. Pinning HF_HUB_CACHE too pre-empts the
# stale dynamic-module cache (models/modules/transformers_modules) that produced
# DiCoW's confusing `FileNotFoundError: .../decoding.py` — this model is loaded
# with trust_remote_code as well, so it is exposed to exactly the same failure.
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(BASE, "models", "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# The VENDORED helper is the one that runs — in dev and in the packaged .app
# alike. The `.venv` also has it pip-installed from git, but that install is
# inert provenance: build.sh:110 strips `git+` lines out of the frozen
# requirements, so the bundled interpreter would never get it.
# scripts/moss-diar/vendor/ rides into the bundle with the rest of scripts/
# (build.sh [B3]). It sits NEXT TO this service, not in a shared scripts/vendor
# — each service owns its own vendored third-party code (owner, 2026-07-29).
#
# THE SILENT TRAP: scripts/moss-asr/vendor/ holds a byte-identical copy for the
# ASR role, so pointing this line at the other service's folder WOULD WORK TODAY
# and break only when one of them moves. Pinned by sidecar-tests.py
# `moss/asr-vendor-is-own-and-identical`, which reads this literal by AST.
sys.path.insert(0, os.path.join(BASE, "scripts", "moss-diar", "vendor"))

REPO = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
SR = 16_000
MIN_CHUNK_SEC = 0.25      # ignore blips shorter than this
MAX_BUFFER_SEC = 300      # safety cap
MAX_NEW_TOKENS = 2048     # upstream app/cli.py's default

# RMS below which a chunk is not worth a model call. Value and rationale copied
# verbatim from AudioRecorder.remoteSilenceRMS: 0.004 ≈ −48 dBFS — a digital
# silence sits at or near 0.0 and even room noise through a codec stays far
# below this, while ordinary speech is an order of magnitude above it.
#
# This gate is not an optimisation. MEASURED on this model: 30 s of digital
# silence makes it answer as an LLM rather than transcribe —
#   [00.00..10.00] S01 I'm sorry, I can't assist with that request. I'm a
#   Qwen-1 model developed by Qwen-Omni...
# — which would land in the transcript as real speech. It does NOT do this when
# silence is merely the leading part of a chunk that also contains speech, so
# skipping the all-silent chunk closes the failure at its trigger.
#
# KEPT IN THE DIARIZATION ROLE ON PURPOSE. This is a property of the MODEL, not
# of a role, and the caller applies no gate of its own: `flushMossDiarChunk`
# (AudioRecorder+Moss.swift) forwards the office chunk boundary with no RMS check
# whatsoever, so an all-silent chunk reaches THIS process directly. Nothing
# upstream would stop it.
SILENCE_RMS = 0.004

# Hard ceiling on what this process may allocate on the GPU, in GB.
#
# NOT an optimisation and NOT a budget — it is a fuse. macOS reports a
# "recommended max working set" (51.8 GB on the owner's 64 GB M4) and PyTorch's
# MPS allocator will grow all the way to it WITHOUT EVER RAISING, so a single
# oversized call takes most of the machine's RAM and the system starts swapping
# while the app looks merely slow.
#
# MEASURED 2026-07-31: a one-pass transcription of 10 MINUTES of audio reached
# ~50 GB and had to be killed by hand; nothing in this process bounded it. Audio
# costs ~13.2 context tokens per second, so cost climbs with the LENGTH of a
# single call — which is exactly what `chunked.intervalSec` controls.
#
# The value is HALF the machine's RAM, chosen so the fuse can only ever blow on a
# genuine runaway. MEASURED 2026-07-31 with the cap armed: five consecutive 30 s
# chunks plateau at 18.35 GB of MPS pool and STAY there (no leak); a 60 s chunk
# peaked at 8.22 GB and a 120 s chunk — the longest `chunked.intervalSec` offers
# — at 14.70 GB. Resident process memory stayed flat at 5.65 GB throughout, so
# most of that pool is the allocator's reusable cache rather than real pressure.
#
# A first attempt set this to 20 GB and was WRONG: 18.35 GB observed against a
# 20 GB ceiling is 1.65 GB of headroom, i.e. a fuse that could blow mid-meeting
# on ordinary audio. The number has to sit far above the measured plateau and far
# below the 51.8 GB runaway, and 32 GB is the only value that does both.
MPS_MEMORY_CAP_GB = 32.0


def emit(kind: str, text: str) -> None:
    try:
        sys.stdout.write(json.dumps({"type": kind, "text": text}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


def emit_final(text: str, segments) -> None:
    """Emit a chunk final carrying the model's own speaker segmentation.

    `segments` is always present on a MOSS final (possibly empty) — a skipped or
    empty chunk sends `{"text":"","segments":[]}` rather than nothing at all, so
    the app's pendingChunkWindows FIFO always drains. Silence on a FLUSH would
    misalign every later chunk's window.

    `text` is kept AND POPULATED even though this role's caller discards it: the
    key is structurally mandatory (`Message.text` is non-optional in Swift), and
    an empty one would decode but forfeit the byte-identical comparison against
    the ASR copy. See the module docstring.
    """
    try:
        sys.stdout.write(json.dumps({"type": "final", "text": text,
                                     "segments": segments}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


# NO `emit_file` HERE, and no `-2` branch in the read loop below — deliberate,
# with the call-site proof written out in the module docstring. Do not add it back
# for symmetry with scripts/moss-asr/.


def log(message: str) -> None:
    """Liveness/debug logging → logs/moss-diar.log via stderr.

    Its OWN file, not the ASR role's logs/moss-asr.log: the two roles run as
    SEPARATE processes and can be live at the same time. Two writers on one file
    is the 2026-07-15 mistake, and splitting the roles is what makes it live
    again here. Note that logs/moss-diarization.log is a THIRD, unrelated file —
    the Swift side writes that one (AudioRecorder+Moss.mossLog).
    """
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def fail(message: str) -> None:
    emit("error", message)
    sys.exit(1)


def brief_traceback() -> str:
    lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
    return " | ".join(lines[-3:])


# --- gates ------------------------------------------------------------------
# Pure, module-level and tested without a model load (sidecar-tests.py moss/*):
# both guard OBSERVED failures, and both are deliberately narrow. Over-deletion
# is the dangerous direction here — deleted text leaves no trace in the
# transcript, only in this log (see CLAUDE.md, "Hallucination gates").


def silence_skip_reason(samples, threshold: float = SILENCE_RMS):
    """Why this chunk must not reach the model at all, or None to transcribe it.

    All-silent input makes the model answer as a chatbot (see SILENCE_RMS), so
    the cheapest and safest place to stop it is before the call.
    """
    import numpy as np
    if samples is None or len(samples) == 0:
        return "empty buffer"
    array = np.asarray(samples, dtype=np.float32)
    level = float(np.sqrt(np.mean(array * array)))
    if level < threshold:
        return f"near-silent (rms {level:.5f} < {threshold:.5f})"
    return None


# Backstop for a refusal the silence gate did not catch. Kept as narrow as it can
# be made: a human being CAN say "I'm sorry" or "I can't help with that", so the
# only things matched are the model naming itself, plus the exact refusal opening
# observed — and that one only at the START of the whole output, so the same
# words spoken mid-meeting survive. The generic CANNED_HALLUCINATIONS set the
# mlx-audio services carry (qwen3/ granite/ voxtral/) is deliberately NOT copied
# here — nor imported, since nothing is shared between sidecars: it is tuned for
# subtitle-trained models emitting "Thank you." on silence, which is a different
# failure with a different false-positive risk.
REFUSAL_MARKERS = ("i'm a qwen", "i am a qwen", "developed by qwen")
REFUSAL_PREFIXES = ("i'm sorry, i can't assist with that request",
                    "i am sorry, i can't assist with that request",
                    "i can't assist with that request")


def refusal_drop_reason(text: str):
    """Why this whole-chunk text is an LLM refusal, or None to keep it.

    KEPT IN THE DIARIZATION ROLE, and it matters MORE here than in the ASR role,
    which is why it still returns ("", []) rather than merely blanking text: the
    refusal came back AS A TIMESTAMPED SEGMENT. Left through, that segment would
    become a SpeakerTurn in `mossTurns` — a speaker span for words nobody spoke —
    and in "other model as ASR + MOSS as diarizer" mode it would additionally
    mis-split the REAL ASR text across a fabricated speaker boundary. Dropping the
    segments is the whole point of the gate in this role; dropping only the text
    would leave the damage behind.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    for marker in REFUSAL_MARKERS:
        if marker in lowered:
            return f"model self-identification ({marker!r})"
    for prefix in REFUSAL_PREFIXES:
        if lowered.startswith(prefix):
            return f"refusal opening ({prefix!r})"
    return None


def truncation_warning(generated: int, cap: int = MAX_NEW_TOKENS):
    """Why this chunk's transcript is probably cut short, or None if it is whole.

    Upstream's `generate_transcription` returns `generated_tokens` and we used to
    throw it away. Reaching `cap` means generation STOPPED MID-TRANSCRIPT, and
    that loss is otherwise INVISIBLE: `parse_transcript` drops an incomplete
    trailing segment without raising (measured 2026-07-31 — a 3-segment
    transcript cut mid-text parses to 2 segments, no exception). Over-deletion
    with no trace is the failure mode this codebase logs everything to avoid, so
    this line is the only evidence that would ever exist.

    It costs MORE in this role than in the ASR one: a truncated tail here is not
    just missing words, it is missing SPEAKER TURNS, so the ASR model's real text
    for that stretch gets split by an incomplete set of spans.

    Not expected to fire at the default settings. Measured on this M4 over the
    owner's own recording, MOSS generates ~7-9 tokens per second of audio:
    a 30 s chunk used 194-265 of 2048 (7.7-10.6x headroom) and 78 s used 574
    (3.6x). The `chunked.intervalSec` picker goes up to 120 s, which is where
    the headroom narrows to roughly 2x — and upstream's own README tells you to
    raise `max_new_tokens` for long audio. That is the case this guards.
    """
    if generated >= cap:
        return (f"generation hit the {cap}-token cap ({generated} tokens) — the END "
                "of this chunk is missing from the transcript. Use a shorter chunk "
                "interval, or raise MAX_NEW_TOKENS (upstream README advises this "
                "for long audio).")
    return None


def speaker_index(label: str) -> int:
    """"S01" → 1. 0 when the label is not one the parser can have produced.

    The parser only ever emits `S` + digits (`_parse_speaker`), so the fallback
    is unreachable through the normal path; it exists so a malformed label can
    never raise in the middle of a chunk.

    NOT DEAD CODE — a RETAINED TEST CONTRACT. Nothing in this file calls it, so a
    dead-code sweep would remove it; `sidecar-tests.py moss/parse-to-wire-shape`
    calls `module.speaker_index(...)` on EVERY entry of MOSS_COPIES, so deleting
    it here fails the suite. Leave it.
    """
    digits = "".join(ch for ch in (label or "") if ch.isdigit())
    return int(digits) if digits else 0


def wire_segments(parsed) -> list:
    """Upstream TranscriptSegment objects → the JSON shape the app decodes.

    Times stay CHUNK-LOCAL exactly as the model produced them; the app adds the
    window start (the `offsetTurns` precedent). Empty-text segments are dropped —
    a speaker span with no words is nothing the transcript can show.
    """
    out = []
    for seg in parsed:
        text = (seg.text or "").strip()
        if not text:
            continue
        out.append({"start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "speaker": seg.speaker,
                    "text": text})
    return out


def joined_text(segments) -> str:
    """The `final`'s plain `text`: the segments' text in order, tags stripped."""
    return " ".join(s["text"] for s in segments).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=REPO)
    # Accepted and ignored — see the module docstring.
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()

    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        fail(f"numpy import failed: {brief_traceback()}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception:  # noqa: BLE001
        fail(f"transformers/torch import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    try:
        from moss_transcribe_diarize import (
            build_transcription_messages,
            generate_transcription,
            parse_transcript,
        )
    except Exception:  # noqa: BLE001
        fail(f"vendored moss_transcribe_diarize import failed: {brief_traceback()} "
             "— scripts/moss-diar/vendor/moss_transcribe_diarize is missing or broken")

    def write_temp_wav(audio) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return path

    # ---- device -----------------------------------------------------------
    # DELIBERATE DEVIATION from upstream: `resolve_device("auto")` only knows
    # cuda and cpu, so on a Mac it silently returns CPU — 19.9 s per 30 s chunk
    # (1.51x realtime) against 6.4 s on MPS (4.67x). Measured on the owner's M4
    # over his own recording, MPS output was identical to CPU to 2 decimal places
    # on every timestamp, so MPS is verified sound FOR THIS MODEL. (DiCoW's
    # "MPS unverified" note is about DiCoW's custom remote code, not this.)
    # float32 on both: the checkpoint is loaded with dtype="auto" and cast here,
    # matching upstream's CPU behaviour.
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Arm the fuse BEFORE the weights land, so even a bad load cannot run away.
    # See MPS_MEMORY_CAP_GB for why this exists (a 10-minute call reached ~50 GB
    # with nothing to stop it). Wrapped because these APIs are version-dependent
    # and a missing one must not cost us the model: an uncapped run is what we
    # have always had, so failing to cap is a warning, never a fatal error.
    if device.type == "mps":
        try:
            ceiling_gb = torch.mps.recommended_max_memory() / (1024 ** 3)
            if ceiling_gb > 0:
                fraction = min(MPS_MEMORY_CAP_GB / ceiling_gb, 1.0)
                torch.mps.set_per_process_memory_fraction(fraction)
                log(f"MPS memory capped at {MPS_MEMORY_CAP_GB:.0f} GB "
                    f"of the {ceiling_gb:.1f} GB macOS reports as available")
        except Exception:  # noqa: BLE001
            log(f"WARNING could not cap MPS memory ({brief_traceback()}) — "
                "a very long chunk could allocate without bound")
    dtype = torch.float32
    log(f"runtime: MOSS-Transcribe-Diarize on {device.type} float32, model {args.model}")

    started = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                     dtype="auto")
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                                  fix_mistral_regex=True)
        model = model.to(dtype=dtype).to(device).eval()
    except Exception:  # noqa: BLE001
        fail(f"MOSS model load failed ({args.model}): {brief_traceback()}")
    log(f"model loaded in {time.time() - started:.1f}s")

    def transcribe_path(path: str):
        """(text, segments) for one audio file. Never raises for gate reasons."""
        # DELIBERATE DEVIATION #2: the upstream DEFAULT_PROMPT is used VERBATIM,
        # Chinese and all (build_transcription_messages' default). It is the
        # formatting instruction that produces the `[start][Sxx]text[end]` shape
        # `parse_transcript` reads, the owner's direction was to follow the
        # upstream inference code, and measured output on his ENGLISH recording
        # came back in correct English. Do NOT "translate" or reword it.
        messages = build_transcription_messages(path)
        result = generate_transcription(model, processor, messages,
                                        max_new_tokens=MAX_NEW_TOKENS,
                                        do_sample=False, device=device, dtype=dtype)
        raw = (result.get("text") or "").strip()
        truncated = truncation_warning(int(result.get("generated_tokens") or 0))
        if truncated:
            log(f"WARNING {truncated}")
        else:
            log(f"generated {int(result.get('generated_tokens') or 0)}/{MAX_NEW_TOKENS} tokens")
        segments = wire_segments(parse_transcript(raw))
        text = joined_text(segments)
        reason = refusal_drop_reason(text)
        if reason:
            log(f"drop {reason}: {text[:120]!r}")
            return "", []
        return text, segments

    # Warmup: one real generate over ~1 s of quiet TONE, not silence — silence
    # would hit the gate above (and, ungated, the refusal). So LOADED means the
    # MPS kernels are compiled and the first real chunk is not the slow one.
    try:
        t = np.arange(SR) / SR
        warm = (0.05 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
        warm_path = write_temp_wav(warm)
        try:
            warm_started = time.time()
            transcribe_path(warm_path)
            log(f"warmup generate done in {time.time() - warm_started:.1f}s")
        finally:
            os.unlink(warm_path)
    except Exception:  # noqa: BLE001
        fail(f"MOSS warmup failed ({args.model}): {brief_traceback()}")

    def transcribe(audio):
        """(text, segments) for one buffered chunk."""
        path = write_temp_wav(audio)
        try:
            return transcribe_path(path)
        finally:
            os.unlink(path)

    emit("status", "LOADED")

    buffer = np.zeros(0, dtype=np.float32)
    stdin = sys.stdin.buffer

    while True:
        header = stdin.read(4)
        if not header or len(header) < 4:
            break
        (n,) = struct.unpack("<i", header)

        if n == -1:
            break
        # NO `n == -2` (FILE-TRANSCRIBE) branch: this process is only ever fed
        # audio frames and FLUSH. See the module docstring for the call-site proof.
        if n == 0:  # FLUSH — transcribe this chunk
            secs = buffer.size / SR
            skip = None
            if buffer.size < int(MIN_CHUNK_SEC * SR):
                skip = f"only {secs:.2f}s buffered"
            else:
                skip = silence_skip_reason(buffer)
            if skip:
                log(f"FLUSH skipped — {skip}")
                emit_final("", [])
            else:
                log(f"FLUSH received — transcribing {secs:.1f}s chunk")
                chunk_started = time.time()
                try:
                    text, segments = transcribe(buffer)
                    speakers = sorted({s["speaker"] for s in segments})
                    log(f"chunk done in {time.time() - chunk_started:.1f}s "
                        f"({len(text)} chars, {len(segments)} segments, "
                        f"speakers {speakers})")
                    emit_final(text, segments)
                except Exception:  # noqa: BLE001
                    log("chunk FAILED")
                    emit("error", f"Chunk transcription failed: {brief_traceback()}")
            buffer = np.zeros(0, dtype=np.float32)
            continue

        data = stdin.read(n * 4)
        if not data or len(data) < n * 4:
            break
        buffer = np.concatenate([buffer, np.frombuffer(data, dtype=np.float32)])
        if buffer.size > MAX_BUFFER_SEC * SR:
            buffer = buffer[-MAX_BUFFER_SEC * SR:]


if __name__ == "__main__":
    main()
