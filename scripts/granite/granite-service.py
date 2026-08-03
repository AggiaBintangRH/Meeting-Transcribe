#!/usr/bin/env python3
"""
Granite Speech 4.1 2B NAR chunked ASR sidecar for MeetingTranscriber (persistent).

ONE SIDECAR PER ASR MODEL (owner decision, 2026-07-29). This file is a VERBATIM
EXTRACTION of the mlx-audio branch that used to live inside
chunked-asr-service.py, specialised to Granite — not a redesign. Same wire bytes,
same text, same canned-phrase gate, same logs. With Qwen3 and Voxtral split out
alongside it, chunked-asr-service.py is gone.

THIS MODEL, measured on the owner's M4 (see CLAUDE.md):
  * ~5.6 s per 30 s chunk — inside the chunk budget, a little behind Qwen3 (4.3 s)
    and Whisper (4.2 s).
  * FIVE languages: en, fr, de, es, pt. NO Japanese, and this count has now been
    wrong in BOTH directions, so read it off the checkpoint rather than trusting
    any note: models/hub/…granite-speech-4.1-2b-nar-mlx/…/README.md front-matter
    says `language: [en, fr, de, es, pt]`. The "six, Japanese IS included" note
    that stood here until 2026-07-31 came from ibm-granite's *base* Granite Speech
    card — a different checkpoint from the **NAR** variant we actually ship.
    No Indonesian, no Malay.
  * `generate()` HAS NO LANGUAGE PARAMETER. The Swift side knows this —
    `GraniteSpeechModel.languageArgument` returns nil for every code — so
    `--language` never arrives at this service at all, and the language picker is
    DISABLED for Granite rather than populated: a control that does nothing is
    worse than none. The flag is still accepted below (and would still be swallowed
    harmlessly) purely so a stale caller cannot make argparse kill the load.
  * LOWERCASE OUTPUT is a known Granite quirk, not a bug in this file: it writes
    "thank you", not "Thank you." The canned gate below matches case-insensitively
    for exactly that reason.
  * GRANITE HALLUCINATES CANNED CAPTIONS. The observed 'Thank you.' on a
    near-silent chunk — on THIS model, not Whisper — is why canned_drop_reason
    exists at all. The note claiming only Whisper did this was wrong.
  * No confidence signal of any kind (see emit_final).

FULLY STANDALONE, deliberately: nothing here is imported from another sidecar and
there is no shared protocol module. The owner chose full separation over a shared
definition of the wire bytes, accepting that a protocol change means editing every
service. What makes that safe is not discipline but a test: sidecar-tests.py
`whisper/protocol-matches-chunked` drives this file, qwen3-service.py,
voxtral-service.py and whisper-service.py through the SAME payload builders and
the SAME real FLUSH branch, and fails loudly if their reply shapes diverge.

DELIBERATELY ABSENT: `whisper_drop_reason`, `whisper_chunk_confidence` and the
WHISPER_NO_SPEECH_MAX / WHISPER_HALLUCINATION_MAX_WORDS / WHISPER_COMPRESSION_MAX
constants. mlx-audio exposes neither `no_speech_prob` nor `compression_ratio` nor
`avg_logprob`, so none of that rule can even be evaluated here — and copying it in
would ADD gates in the over-deletion direction, which is the dangerous one (see
CLAUDE.md, "Hallucination gates — a rule that has now failed in BOTH directions").
The canned-phrase gate below is the ONLY hallucination check this runtime has,
which matters more for Granite than for any other model here: it is the one
observed producing a canned caption.

Protocol — byte-identical framing to the other ASR sidecars, so the SAME Swift
client (`ChunkedASRService`) drives them all:
  stdout: JSON lines {"type":"status"|"final"|"error","text":...}
          status LOADED after model init
          NO "conf" ever — this runtime reports no confidence at all; see
          emit_final. Absent means "not measured", never "low".
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes)
            n == 0 : FLUSH — transcribe buffered chunk, emit final, reset
            n == -1: exit
            n == -2: FILE-TRANSCRIBE — one int32-length-prefixed UTF-8 JSON body
                     {"id":N,"path":"/tmp/.../track.wav"} follows; transcribe that
                     file with the loaded model, emit file_result/file_error.
                     Fully additive — never touches the live streaming buffer.

Args: --model <hf-repo> [--language <code>]   (--language is inert here — see above)
Audio is passed to the model as a temp WAV for maximum compatibility with the
mlx-audio implementation.

NO ALIGNMENT HERE (2026-07-29). The Qwen3-ForcedAligner used to load in this
process and stamp per-word times onto each chunk before the final was emitted,
which made word timestamps SYNCHRONOUS — the transcript waited on them. It is
now its own sidecar (scripts/aligner/aligner-service.py), asked separately by
the app, so a final NEVER carries "words"/"dur" and there is no --align-model
flag any more.
"""
import argparse
import json
import os
import struct
import sys
import tempfile
import traceback
import wave

# This file lives at scripts/<service>/<file>.py (one folder per service,
# owner 2026-07-29), so the project root — which owns models/ — is THREE levels
# up, not two. Bundled, that root is Contents/Resources.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SR = 16_000
MIN_CHUNK_SEC = 0.25   # ignore blips shorter than this
MAX_BUFFER_SEC = 300   # safety cap


def emit(kind: str, text: str) -> None:
    try:
        sys.stdout.write(json.dumps({"type": kind, "text": text}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


def emit_final(text: str, conf: float = None) -> None:
    """Emit a chunk final, optionally carrying the chunk's ASR confidence.

    Kept separate from emit() so the existing (kind, text) contract — and the
    Swift decoder that reads "text" — stays untouched. "conf" is present only
    when the runtime actually reports a confidence (Whisper). An absent "conf"
    means the model exposes no such signal, which is not the same as low
    confidence — so it is omitted rather than sent as 0.

    THIS SERVICE NEVER PASSES ONE: Granite runs through mlx-audio, which exposes no
    no_speech, no compression_ratio and no avg_logprob, so there is nothing to
    derive a confidence from and the key is always absent. The parameter is kept
    (identical to every other ASR sidecar's) because the wire format is shared and
    is compared byte-for-byte across services.

    NO "words"/"dur": word timestamps come from the aligner sidecar now, on a
    reply of their own. A final has carried them for the last time.
    """
    payload = {"type": "final", "text": text}
    if conf is not None:
        payload["conf"] = round(float(conf), 3)
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def emit_file(kind: str, req_id, text: str, conf: float = None) -> None:
    """Emit a file-transcribe result carrying its request id (overlap repair)."""
    payload = {"type": kind, "id": req_id, "text": text}
    if conf is not None:
        payload["conf"] = round(float(conf), 3)
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def log(message: str) -> None:
    """Liveness/debug logging → logs/granite.log via stderr.

    Its OWN log file, not logs/chunked-asr.log: with the services split, two
    sidecars could be alive at once and two writers on one file is a mistake this
    project already made and fixed (2026-07-15).
    """
    import time
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def fail(message: str) -> None:
    emit("error", message)
    sys.exit(1)


def brief_traceback() -> str:
    lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
    return " | ".join(lines[-3:])


# --- canned-phrase gate (every mlx-audio model) ------------------------------
# The note that "only Whisper hallucinates this way" was wrong: the owner hit
# 'Thank you.' on Granite too, intermittently, on the same kind of near-silent
# chunk. Granite and Qwen3 run through mlx-audio, which reports no no_speech or
# compression_ratio, so the Whisper gate cannot be reused at all — the only
# signals available are the text and how long the audio was.
#
# Density alone is NOT safe here: a real one-word reply in an otherwise quiet
# 30 s chunk looks identical to a hallucination by that measure, and deleting it
# would repeat the mistake that cost real sentences earlier. So this gate is
# deliberately narrow — it fires only on text that is BOTH a known canned
# caption AND implausibly sparse for its duration. A genuine "Okay." survives
# because it is not in this set; 'Thank you.' across 30 s of silence does not.
#
# These are the closing captions of subtitled video, which is what the whole
# model family is trained on. Matching ignores case and trailing punctuation —
# and the case-insensitivity is load-bearing for THIS model: Granite writes
# lowercase ("thank you"), so a case-sensitive set would let its hallucination
# straight through.
CANNED_HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thank you so much",
    "thanks for watching", "thank you for watching", "thanks for listening",
    "please subscribe", "subscribe", "you", "bye", "bye bye", "goodbye",
    "see you next time", "the end", "music", "applause", "silence",
}

# Same VALUES as the shared sidecar used (4.0 / 0.5), RENAMED. They were called
# WHISPER_MIN_DENSITY_DURATION / WHISPER_MIN_WORDS_PER_SEC there because Whisper's
# own gate happened to reuse them — but the rule below is the mlx-audio gate, and
# there is no Whisper anywhere in this file. A `WHISPER_` prefix here would
# suggest this threshold follows Whisper's, which it does not: it belongs to the
# canned-phrase rule and moves only with it.
#
# Words/second floor for text longer than the duration floor: a hallucination
# regardless of anything else. Real speech runs ~2-3 words/s; the classic failure
# is 30 s of silence read as "Thank you." (0.07 words/s). The 4 s floor protects
# a genuine short reply ("Okay." in a 2 s chunk).
CANNED_MIN_DURATION_SEC = 4.0
CANNED_MIN_WORDS_PER_SEC = 0.5


def canned_drop_reason(text: str, duration: float):
    """Why this whole-chunk text is a canned hallucination, or None to keep it.

    Model-agnostic: uses only the text and the audio duration, so it works for
    the mlx-audio models that expose no confidence numbers.
    """
    stripped = text.strip().strip(".!?,;:").lower()
    if not stripped or stripped not in CANNED_HALLUCINATIONS:
        return None
    words = len(stripped.split())
    if duration >= CANNED_MIN_DURATION_SEC and words < duration * CANNED_MIN_WORDS_PER_SEC:
        return (f"canned hallucination ({words}w in {duration:.0f}s = "
                f"{words / duration:.2f}/s)")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    # ACCEPTED BUT INERT. Granite's generate() has no language parameter, so the
    # Swift side never sends this flag (`GraniteSpeechModel.languageArgument`
    # returns nil for every code, and the picker is disabled). It stays declared
    # so a stale caller that does send it gets its argument parsed and ignored
    # instead of killing the model load with an argparse error mid-meeting; the
    # kwargs below then hit the TypeError path that has always existed for
    # exactly this model.
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()

    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        fail(f"numpy import failed: {brief_traceback()}")

    language = None if args.language in ("", "auto") else args.language

    def write_temp_wav(audio: "np.ndarray") -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return path

    # ---- runtime: mlx-audio -------------------------------------------------
    log(f"runtime: mlx-audio, model {args.model}")
    try:
        from mlx_audio.stt import load
    except Exception:  # noqa: BLE001
        fail(f"mlx-audio import failed: {brief_traceback()}")

    try:
        model = load(args.model)
    except Exception:  # noqa: BLE001
        fail(f"Model load failed ({args.model}): {brief_traceback()}")
    log("model loaded")

    kwargs = {"language": language} if language else {}

    def transcribe_path(path: str):
        """(text, None) — mlx-audio models report no confidence at all.

        The None is not a placeholder for "we didn't bother": Granite exposes
        neither no_speech nor avg_logprob, so there is no signal to derive one
        from. Fabricating a number for parity with the Whisper sidecar would put
        a confidence in front of the user that the model never claimed. The UI
        shows nothing at all instead.
        """
        try:
            result = model.generate(path, **kwargs)
        except TypeError:  # model doesn't take a language kwarg — Granite's case
            result = model.generate(path)
        text = (result.text or "").strip()
        # These models expose no confidence numbers, so the canned-phrase
        # gate is the only hallucination check available to them. Observed
        # on THIS model: an occasional 'Thank you.' on a near-silent chunk.
        import soundfile as sf
        duration = sf.info(path).duration
        reason = canned_drop_reason(text, duration)
        if reason:
            log(f"drop {reason}: {text!r}")
            return "", None
        return text, None

    def transcribe(audio: "np.ndarray"):
        """(text, confidence-or-None) for one buffered chunk."""
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
        if n == -2:  # FILE-TRANSCRIBE — additive, independent of the live buffer
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
                text, conf = transcribe_path(path)
                log(f"file-transcribe id={req_id} done ({len(text)} chars)")
                emit_file("file_result", req_id, text, conf)
            except Exception:  # noqa: BLE001
                emit_file("file_error", req_id,
                          f"file transcription failed: {brief_traceback()}")
            continue
        if n == 0:  # FLUSH — transcribe this chunk
            if buffer.size >= int(MIN_CHUNK_SEC * SR):
                import time
                secs = buffer.size / SR
                log(f"FLUSH received — transcribing {secs:.1f}s chunk")
                started = time.time()
                try:
                    text, conf = transcribe(buffer)
                    log(f"chunk done in {time.time() - started:.1f}s ({len(text)} chars"
                        + (f", conf={conf:.3f}" if conf is not None else "") + ")")
                    emit_final(text, conf)
                except Exception:  # noqa: BLE001
                    log("chunk FAILED")
                    emit("error", f"Chunk transcription failed: {brief_traceback()}")
            else:
                log(f"FLUSH ignored — only {buffer.size / SR:.2f}s buffered")
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
