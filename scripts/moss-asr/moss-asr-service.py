#!/usr/bin/env python3
"""
MOSS-Transcribe-Diarize sidecar for MeetingTranscriber — the CHUNKED-ASR ROLE.

Speaker-attributed ASR: ONE 0.9B model returns the transcript, the speaker
labels and the timestamps together, instead of an ASR pass plus a separate
diarization pass. Upstream: https://github.com/OpenMOSS/MOSS-Transcribe-Diarize

ONE SERVICE PER ROLE (2026-07-31). MOSS is selectable in TWO places — as the
chunked ASR model, and as the diarization engine — and before the split BOTH
launched one script, so two processes of the same file wrote into one log. This
file serves the ASR role only; the diarization role runs
scripts/moss-diar/moss-diar-service.py and writes logs/moss-diar.log. Do not merge
the two back together, and do not point either role at the other's file: the log
pair would drift (see ChunkedASRService.Config.logName).

Its own sidecar from the start, rather than a branch inside the then-shared
chunked-asr-service.py: that file was MLX-only, this is PyTorch/transformers on
MPS, and "Whisper as ASR + MOSS as the diarizer" needs MOSS in a process the ASR
selection does not control. One sidecar per engine is the house pattern, and this
file was its template for the ASR models — the shared file was split into
whisper/ qwen3/ granite/ voxtral/ and deleted (2026-07-29..30).

NOTHING may be dropped from this file as "the diarizer does not need it" — every
branch below is driven in the ASR role, and each is verified live in the Swift
call sites:
  * the `-2` FILE-TRANSCRIBE frame + emit_file: driven by REMOTE chunks
    (AudioRecorder+RemoteStream.swift) and by overlap-repair re-ASR
    (AudioRecorder+OverlapRepair.swift, which skips only when the DIARIZATION
    ENGINE is MOSS — with MOSS as ASR and pyannote diarizing, it runs).
  * `segments` on `final`: in MOSS+MOSS mode there is no second MOSS process, so
    THIS process alone feeds the diarization rows via onChunkSegments.
  * both gates (silence + refusal): they guard MEASURED failures of the model
    itself, not of a role.
The DIARIZATION-role file is the one WITHOUT the `-2` branch; that asymmetry is
pinned from both sides by sidecar-tests.py `moss/diar-has-no-file-branch`, which
fails if this file ever loses it.

Protocol — byte-identical framing to every other ASR sidecar, so the SAME Swift
client (`ChunkedASRService`) drives them all:
  stdout: JSON lines
          {"type":"status","text":"LOADED"}
          {"type":"final","text":"Hello there. Hi.","segments":[
              {"start":0.0,"end":4.1,"speaker":"S01","text":"Hello there."},
              {"start":4.4,"end":5.2,"speaker":"S02","text":"Hi."}]}
          {"type":"file_result","id":N,"text":"...","segments":[...]}
                     `segments` is OMITTED on file_error and present on
                     file_result — same shape as the FLUSH `final`, so the
                     stop-time full pass can rebuild speakers as well as words.
          {"type":"file_error","id":N,"text":"..."}
          {"type":"error","text":"..."}
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes)
            n == 0 : FLUSH — transcribe buffered chunk, emit final, reset
            n == -1: exit
            n == -2: FILE-TRANSCRIBE — one int32-length-prefixed UTF-8 JSON body
                     {"id":N,"path":"..."} follows; emit file_result/file_error.

`text` is the segments' text joined in order with the tags stripped, so the
existing onChunkTranscript contract holds verbatim. `segments` is ADDITIVE and
optional — every consumer that predates it still decodes a MOSS final. Times in
`segments` are CHUNK-LOCAL seconds; the app offsets them by the window start.
`speaker` is the model's RAW label ("S01"); it is anonymous PER CALL — S01 in
chunk N is not S01 in chunk N+1, and nothing here pretends otherwise.

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

# This file lives at scripts/moss-asr/ (one folder per service, owner 2026-07-29),
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
# scripts/moss-asr/vendor/ rides into the bundle with the rest of scripts/
# (build.sh [B3]). It sits NEXT TO this service, not in a shared scripts/vendor
# — each service owns its own vendored third-party code (owner, 2026-07-29).
#
# THE SILENT TRAP: scripts/moss-diar/vendor/ holds a byte-identical copy for the
# diarization role, so pointing this line at the other service's folder WOULD WORK
# TODAY and break only when one of them moves. Pinned by sidecar-tests.py
# `moss/asr-vendor-is-own-and-identical`, which reads this literal by AST.
sys.path.insert(0, os.path.join(BASE, "scripts", "moss-asr", "vendor"))

REPO = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
SR = 16_000
MIN_CHUNK_SEC = 0.25      # ignore blips shorter than this
MAX_BUFFER_SEC = 300      # safety cap
MAX_NEW_TOKENS = 5120     # the model's OWN generation_config.json default
#
# RAISED 2026-08-05, from 2048. The old value was copied from upstream's
# `app/cli.py` default and treated here as if it were a limit of the model. It is
# not: the checkpoint's own `generation_config.json` says **5120**, that is also
# upstream's SERVER default, and upstream's README says outright "For longer
# multi-speaker audio, raise `max_new_tokens`" (its example is 65536).
#
# What 2048 actually cost, measured by driving this sidecar directly on a real
# recording: a 240 s window generated exactly 2048/2048 tokens — TRUNCATED, with
# the end of that chunk silently missing from the transcript (the gate below is
# what makes it not silent). At 5120 the same window finishes at 2159/5120 (42 %)
# and a 300 s window at 2738/5120 (53 %). So the "MOSS cannot take a long window"
# conclusion recorded earlier in CLAUDE.md was a property of THIS CONSTANT, not
# of the model.
#
# The window size (`mossFullPassWindowSec`) is deliberately NOT changed with it —
# raising the ceiling is one-directional and can only reduce truncation, while
# enlarging the window trades fewer speaker seams against peak RAM and total pass
# time, neither of which has been measured properly yet.

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
    """
    try:
        sys.stdout.write(json.dumps({"type": "final", "text": text,
                                     "segments": segments}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def emit_file(kind: str, req_id, text: str, segments=None) -> None:
    """Emit a file-transcribe result carrying its request id (remote / repair).

    `segments` is ADDITIVE and OPTIONAL, and both halves of that matter. It is
    omitted entirely when None, so `file_error` and every caller that predates it
    put the identical three keys on the wire and an older app still decodes a
    reply from a newer sidecar.

    IT EXISTS FOR THE STOP-TIME FULL PASS (2026-08-18). MOSS transcribes and
    labels in ONE call, and this frame used to throw the labels away
    (`text, _ = transcribe_path(path)`), so a full pass driven through it produced
    a re-transcribed meeting with NOBODY in it — strictly worse than the live
    chunks it replaced. Remote and overlap repair do not read the key and are
    unaffected; they ask this frame a question about words, not about people.
    """
    payload = {"type": kind, "id": req_id, "text": text}
    if segments is not None:
        payload["segments"] = segments
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def log(message: str) -> None:
    """Liveness/debug logging → logs/moss-asr.log via stderr.

    Its OWN file, not logs/moss-diar.log: the diarization role runs
    scripts/moss-diar/moss-diar-service.py as a SEPARATE process and keeps that
    log. Two writers on one file is the 2026-07-15 mistake, and splitting the
    roles is what makes it live again here.
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
    """Why this whole-chunk text is an LLM refusal, or None to keep it."""
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
             "— scripts/moss-asr/vendor/moss_transcribe_diarize is missing or broken")

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
    # float32 on both, and the old justification here was WRONG (corrected
    # 2026-08-05): it said this "matches upstream's CPU behaviour", but upstream
    # forces float32 only when the device IS cpu (`model_runner.py:_ensure_loaded`)
    # and we are on MPS, where it uses bf16. So this IS a deviation.
    #
    # It stays, on measurement rather than on the wrong reason. bf16 was tried:
    # text, timestamps and speakers came back byte-identical, and the weights do
    # halve (3.62 -> 1.73 GB) — but RSS does not move at all (5.68 GB either way),
    # because the allocator pool dominates, not the weights. Verified-identical
    # output is not a reason to change dtype when the benefit does not survive
    # measurement.
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
        if n == -2:  # FILE-TRANSCRIBE — independent of the live streaming buffer
            hdr2 = stdin.read(4)
            if not hdr2 or len(hdr2) < 4:
                break
            (m,) = struct.unpack("<i", hdr2)
            body = stdin.read(m)
            if not body or len(body) < m:
                break
            req_id = None
            try:
                req = json.loads(body.decode("utf-8"))
                req_id = req.get("id")
                path = req.get("path", "")
                if not path or not os.path.exists(path):
                    emit_file("file_error", req_id, f"file not found: {path}")
                    continue
                log(f"file-transcribe id={req_id}: {path}")
                text, segments = transcribe_path(path)
                log(f"file-transcribe id={req_id} done ({len(text)} chars, "
                    f"{len(segments)} segment(s))")
                # Segments ride along now — see `emit_file`. The FLUSH path has
                # always carried them; this frame discarded them, which is what
                # made a MOSS full pass impossible rather than merely refused.
                emit_file("file_result", req_id, text, segments)
            except Exception:  # noqa: BLE001
                emit_file("file_error", req_id,
                          f"file transcription failed: {brief_traceback()}")
            continue
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
