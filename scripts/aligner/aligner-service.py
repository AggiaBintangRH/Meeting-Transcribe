#!/usr/bin/env python3
"""
Forced-alignment sidecar for MeetingTranscriber (persistent).

ONE SERVICE PER MODEL (owner decision, 2026-07-29). The Qwen3-ForcedAligner used
to be loaded INSIDE whichever chunked ASR sidecar was running, so its 1.2 GB sat
in the ASR process and every chunk's alignment ran on the ASR's own thread,
before the `final` was emitted. This file is that aligner, extracted into its own
process — behaviour of the alignment itself is unchanged (the gates, the `src`
pairing and every log line below are the originals), but the TIMING is not:

  WORD TIMESTAMPS ARE NOW ASYNCHRONOUS. The app shows the chunk's text the
  instant the ASR sidecar returns it, with the estimated character-proportional
  speaker split, and asks this service for the words separately. When the words
  arrive (usually well under a second later) the app rebuilds the affected rows
  word-exactly. The transcript never waits on alignment, and an aligner that is
  slow, wedged or dead costs nothing but the refinement.

FULLY STANDALONE, deliberately: nothing here is imported from any other sidecar
and there is no shared protocol module — the same choice, with the same accepted
cost, recorded in CLAUDE.md ("One sidecar per ASR model").

Protocol:
  stdout: JSON lines
          {"type":"status","text":"LOADED"}            after the model is up
          {"type":"align_result","id":7,
           "words":[{"text","start","end","src"},...],"dur":30.0}
          {"type":"align_result","id":7}               gates rejected the
                     alignment — "words"/"dur" are ABSENT, never [] or null.
                     Absent means "no trustworthy times for this chunk", which
                     the app renders as the estimated split it already showed.
          {"type":"align_error","id":7,"text":"file not found: ..."}
          {"type":"error","text":"Aligner load failed ..."}   startup only
  stdin:  length-prefixed frames:
          [int32 n]
            n == -2: ALIGN — one int32-length-prefixed UTF-8 JSON body follows
                     {"id":N,"path":"/tmp/.../align-chunk.wav","text":"..."}
                     The text is the chunk transcript VERBATIM: `src` indices are
                     computed against `text.split()` and are meaningless against
                     any other string, which is why the app also re-checks that
                     the segment's text is unchanged before applying a reply.
            n == -1: exit

Args: --model <hf-repo>   (mlx-community/Qwen3-ForcedAligner-0.6B-bf16)

`language="English"` is hardcoded, exactly as it was inside the ASR sidecars —
this project is English-only (CLAUDE.md, owner 2026-07-21).

stderr → logs/aligner.log. Its OWN file: two writers on one log is a mistake
this project already made and fixed (2026-07-15).
"""
import argparse
import json
import os
import struct
import sys
import traceback

# This file lives at scripts/<service>/<file>.py (one folder per service,
# owner 2026-07-29), so the project root — which owns models/ — is THREE levels
# up, not two. Bundled, that root is Contents/Resources.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SR = 16_000

# Forced alignment force-fits whatever text it is handed onto the audio, so
# hallucinated tail text (seen from Whisper) gets stamped past the end of the
# chunk. Anything ending later than this is not a real word time.
ALIGN_END_TOLERANCE_SEC = 0.5
# If more than this fraction of items land past the end, the whole alignment
# is untrustworthy — emit no times at all rather than a truncated guess.
ALIGN_MAX_PAST_END_FRACTION = 0.10

# numpy at module level (not inside main) so the pure functions below can be
# imported by sidecar-tests.py without running the service. A missing numpy is
# reported through fail() in main(), the same way every other sidecar reports it.
try:
    import numpy as np
    NUMPY_ERROR = None
except Exception:  # noqa: BLE001
    np = None
    NUMPY_ERROR = traceback.format_exc().splitlines()[-1]


def emit(kind: str, text: str) -> None:
    try:
        sys.stdout.write(json.dumps({"type": kind, "text": text}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


def emit_align(req_id, words=None, dur: float = None) -> None:
    """Emit one alignment reply, carrying its request id.

    "words"/"dur" are present ONLY when alignment ran and passed every gate.
    A rejected alignment is an align_result WITHOUT them — not an align_error,
    and never an empty list: rejection is a normal, expected outcome (the gates
    exist to produce it), whereas an error means the request itself failed.
    Absent means absent, the same convention `conf` follows on the ASR wire.
    """
    payload = {"type": "align_result", "id": req_id}
    if words is not None:
        payload["words"] = words
        payload["dur"] = dur
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def emit_align_error(req_id, text: str) -> None:
    try:
        sys.stdout.write(json.dumps(
            {"type": "align_error", "id": req_id, "text": text}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def log(message: str) -> None:
    """Liveness/debug logging → logs/aligner.log via stderr."""
    import time
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def fail(message: str) -> None:
    emit("error", message)
    sys.exit(1)


def brief_traceback() -> str:
    lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
    return " | ".join(lines[-3:])


def load_audio_16k(path: str) -> "np.ndarray":
    """Decode a WAV to mono float32 at 16 kHz WITHOUT ffmpeg.

    mlx_whisper.transcribe() shells out to ffmpeg ONLY when handed a string
    path; given a numpy array it does no external decode. Clean dev machines
    and the packaged .app have no ffmpeg (it is not a pip package and we do
    not bundle it), so we decode with soundfile — which ships its own
    libsndfile — and resample here instead of relying on a system binary.
    """
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:                       # collapse any stereo to mono
        audio = audio.mean(axis=1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        audio = resample_poly(audio, SR // g, int(sr) // g)
    return np.ascontiguousarray(audio, dtype=np.float32)


def pair_source_indices(align_proc, log, text: str, items):
    """Map every aligned item back to its index in text.split(), or None.

    The aligner's tokenizer DROPS punctuation-only tokens ("a — b" yields 2
    items, not 3), so a naive item↔word index would silently shift every
    later word into the wrong speaker row. This replays the model's own
    tokenize_space_lang() per source word and cross-checks the result
    against encode_timestamp()'s word_list (the ground truth for what the
    items will be). Any inconsistency ⇒ None ⇒ no timestamps at all; a
    wrong index is worse than no alignment.

    `align_proc` and `log` are explicit parameters rather than closure captures
    (they were locals of the ASR sidecar's `main()`), so this function is a
    plain import for tests instead of an AST extraction. The BODY is unchanged.
    """
    expected = []  # [(token, source word index)]
    for src_index, word in enumerate(text.split()):
        cleaned = align_proc.clean_token(word)
        if not cleaned:
            continue  # pure punctuation — consumes no item
        for token in align_proc.split_segment_with_chinese(cleaned):
            expected.append((token, src_index))

    word_list, _ = align_proc.encode_timestamp(text, "English")
    if [tok for tok, _ in expected] != list(word_list):
        log(f"align: pairing mismatch vs tokenizer "
            f"({len(expected)} paired vs {len(word_list)} tokens)")
        return None
    if len(expected) != len(items):
        log(f"align: item count {len(items)} != token count {len(expected)}")
        return None
    for (token, _), item in zip(expected, items):
        if item.text != token:
            log(f"align: token drift ({item.text!r} != {token!r})")
            return None
    return [src_index for _, src_index in expected]


def align_chunk(aligner, align_proc, log, audio: "np.ndarray", text: str):
    """Return a list of {"text","start","end","src"} for this chunk, or None.

    Never raises: the transcript is the product, timestamps are a bonus.
    """
    if aligner is None or not text:
        return None
    import time
    started = time.time()
    try:
        dur = audio.size / SR
        result = aligner.generate(audio, text, language="English")
        items = list(result)
        if not items:
            log("align: no items returned")
            return None

        src_indices = pair_source_indices(align_proc, log, text, items)
        if src_indices is None:
            return None  # pair_source_indices already logged why

        # Reject force-fitted tail text (hallucinated words stamped past
        # the end of the audio) — and bail entirely if there are many.
        limit = dur + ALIGN_END_TOLERANCE_SEC
        past_end = sum(1 for it in items if it.end_time > limit)
        if past_end > max(1, int(len(items) * ALIGN_MAX_PAST_END_FRACTION)):
            log(f"align: REJECTED — {past_end}/{len(items)} items end past "
                f"{dur:.1f}s chunk")
            return None
        if past_end:
            log(f"align: dropped {past_end} item(s) ending past {dur:.1f}s")

        words = [
            {"text": it.text,
             "start": round(float(it.start_time), 3),
             "end": round(float(it.end_time), 3),
             "src": src_indices[i]}
            for i, it in enumerate(items)
            if it.end_time <= limit
        ]
        if not words:
            log("align: everything past the end — no words emitted")
            return None
        log(f"align: {len(words)} words in {time.time() - started:.2f}s "
            f"({dur / max(time.time() - started, 1e-6):.0f}x realtime)")
        return words
    except Exception:  # noqa: BLE001
        log(f"align FAILED (transcript unaffected): {brief_traceback()}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Qwen3-ForcedAligner repo")
    args = parser.parse_args()

    if np is None:
        fail(f"numpy import failed: {NUMPY_ERROR}")

    # Loaded BEFORE status LOADED, so a broken/missing aligner surfaces in the
    # app's loading overlay through fail() instead of mid-recording.
    log(f"aligner: {args.model}")
    try:
        import time as _t
        from mlx_audio.stt import load as load_align
        _t0 = _t.time()
        aligner = load_align(args.model)
        # The model carries the very tokenizer used to build the item list;
        # reusing it means our pairing can never drift from the model's.
        align_proc = getattr(aligner, "aligner_processor", None)
        if align_proc is None:
            from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import (
                ForceAlignProcessor,
            )
            align_proc = ForceAlignProcessor()
        log(f"aligner loaded in {_t.time() - _t0:.1f}s")
    except Exception:  # noqa: BLE001
        fail(f"Aligner load failed ({args.model}): {brief_traceback()}")

    emit("status", "LOADED")

    stdin = sys.stdin.buffer

    while True:
        header = stdin.read(4)
        if not header or len(header) < 4:
            break
        (n,) = struct.unpack("<i", header)

        if n == -1:
            break
        if n != -2:
            # Unknown opcode: the stream is framed, so there is no way to
            # resynchronise — say so and stop rather than read garbage as audio.
            log(f"unknown frame {n} — closing")
            break

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
            text = req.get("text", "")
            if not path or not os.path.exists(path):
                emit_align_error(req_id, f"file not found: {path}")
                continue
            audio = load_audio_16k(path)
            dur = audio.size / SR
            log(f"align id={req_id}: {dur:.1f}s, {len(text.split())} words")
            words = align_chunk(aligner, align_proc, log, audio, text)
            emit_align(req_id, words, dur if words is not None else None)
        except Exception:  # noqa: BLE001
            emit_align_error(req_id, f"alignment failed: {brief_traceback()}")


if __name__ == "__main__":
    main()
