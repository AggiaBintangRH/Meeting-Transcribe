#!/usr/bin/env python3
"""
Whisper large-v3 chunked ASR sidecar for MeetingTranscriber (persistent).

ONE SIDECAR PER ASR MODEL (owner decision, 2026-07-29). This file was the first
of that split: it is a VERBATIM EXTRACTION of the Whisper branch that used to
live inside the shared chunked-asr-service.py, not a redesign. Same wire bytes,
same text, same hallucination gates, same confidence, same logs. Qwen3 / Granite /
Voxtral followed on 2026-07-30 (scripts/qwen3/, scripts/granite/,
scripts/voxtral/), and the shared file was deleted once each had been proven
byte-identical to it on real audio.

FULLY STANDALONE, deliberately: nothing here is imported from another sidecar and
there is no shared protocol module. The owner chose full separation over a shared
definition of the wire bytes, accepting that a protocol change means editing every
service. What makes that safe is not discipline but a test: sidecar-tests.py
`whisper/protocol-matches-chunked` drives this file AND all three mlx-audio
services through the SAME payload builders and the SAME real FLUSH branch, and
fails loudly if their reply shapes ever diverge.

DELIBERATELY ABSENT: `canned_drop_reason` / `CANNED_HALLUCINATIONS`. That gate
belongs to the mlx-audio services only — the Whisper branch never called it.
Copying it in would ADD a gate Whisper does not have today, in the over-deletion
direction, which is the dangerous one (see CLAUDE.md, "Hallucination gates — a
rule that has now failed in BOTH directions").

Protocol — byte-identical framing to every other ASR sidecar, so the SAME Swift
client (`ChunkedASRService`) drives them all:
  stdout: JSON lines {"type":"status"|"final"|"error","text":...}
          status LOADED after model init
          final and file_result may additionally carry "conf" — a 0..1 pooled
          per-token probability for the chunk. Whisper is the only runtime in
          this project that reports one at all; absent means "not measured",
          never "low".
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes)
            n == 0 : FLUSH — transcribe buffered chunk, emit final, reset
            n == -1: exit
            n == -2: FILE-TRANSCRIBE — one int32-length-prefixed UTF-8 JSON body
                     {"id":N,"path":"/tmp/.../track.wav"} follows; transcribe that
                     file with the loaded model, emit file_result/file_error.
                     Fully additive — never touches the live streaming buffer.

Args: --model <hf-repo> [--language <code>]
      [--task transcribe|translate] [--auto-detect-language]
      [--initial-prompt TEXT] [--best-of N]
      [--no-speech-threshold F] [--logprob-threshold F]
      [--compression-threshold F] [--hallucination-silence-sec F]
Audio is passed to the model as a temp WAV, then decoded here — mlx_whisper is
NEVER handed a string path (see load_audio_16k).

NO ALIGNMENT HERE (2026-07-29). The Qwen3-ForcedAligner used to load in this
process and stamp per-word times onto each chunk before the final was emitted,
which made word timestamps SYNCHRONOUS — the transcript waited on them. It is
now its own sidecar (scripts/aligner/aligner-service.py), asked separately by
the app, so a final NEVER carries "words"/"dur" and there is no --align-model
flag any more.

DECODING OPTIONS (2026-07-29) — every default reproduces today's behaviour
exactly. Before this change the call was
`mlx_whisper.transcribe(audio, path_or_hf_repo=..., language=...)`; with no
flag given, `whisper_transcribe_kwargs()` builds a dict that is VALUE-identical
to that call, because the thresholds it states explicitly (2.4 / -1.0 / 0.6,
task "transcribe", word_timestamps False) are mlx_whisper's own defaults and the
three optional knobs are OMITTED rather than sent as 0. A run at default settings
must stay byte-identical to a run before this change; `sidecar-tests.py`
`whisper/option-defaults-are-todays-behaviour` is the regression guard.

NOT exposed: beam_size. mlx_whisper raises NotImplementedError for it — see the
comment above whisper_transcribe_kwargs.

Two of the options carry a risk the user has to be able to see afterwards, so
both are logged, not merely honoured:

  * --initial-prompt is prior CONTEXT, not an instruction. Whisper is a
    text-continuation model and its own decoder feeds the prompt tokens back in
    (`decode_options["prompt"] = all_tokens[prompt_reset_since:]`), so on unclear
    audio it can ECHO the prompt's wording into the transcript — i.e. produce
    text nobody said. The prompt's own tokens are stripped from the output;
    generated echoes are not. Logged once at startup AND as a short
    `prompt active (...)` marker on EVERY chunk, so a suspicious transcript can
    be traced back to it in logs/whisper.log.

  * --hallucination-silence-sec is read by mlx_whisper only INSIDE its
    `if word_timestamps:` block, so it does nothing at all unless word
    timestamps are on. Setting it therefore FORCES word_timestamps=True, which
    reverses a measured project decision (CLAUDE.md: "Whisper's native
    word_timestamps — MEASURED and REJECTED"). It is forced loudly rather than
    silently no-opped. Whisper's word times stay INTERNAL to that skip — they
    are never emitted; word times come from the aligner sidecar, which is a
    separate process (see "NO ALIGNMENT HERE" above).
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
    """Liveness/debug logging → logs/whisper.log via stderr.

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


# --- Whisper hallucination gate --------------------------------------------
# Whisper invents canned text ("Thank you", "Thanks for watching") on silence:
# it was trained on subtitled video, where quiet stretches carry closing
# captions. Whisper's own no_speech+logprob rule does not catch it, because the
# model is CONFIDENT about the hallucination (measured on real silence:
# no_speech 0.67, logprob -0.26). Whisper-only — the mlx-audio models do not
# fail this way.
#
# Kept as a pure function so the rule can be tested against transcript captured
# from real meetings instead of being re-guessed. It has cost transcript twice
# already, in both directions: first hallucinations got through, then the fix
# for them deleted real speech.
#
# TWO LAYERS EXIST ON PURPOSE — this is NOT duplication to be tidied away:
#   layer 1  Whisper's OWN no_speech_threshold (default 0.6, now user-movable
#            via --no-speech-threshold). It lives inside decoding: combined with
#            logprob_threshold it makes Whisper skip a 30 s window as silence.
#            Whichever way it is moved, it is the model's decision, taken before
#            we ever see the segment.
#   layer 2  whisper_drop_reason() below, at WHISPER_NO_SPEECH_MAX = 0.5, over
#            the segments Whisper DID emit. It exists precisely because layer 1
#            does not catch confident hallucinations (measured on real silence:
#            no_speech 0.67, logprob -0.26 — under 0.5 would still not have been
#            enough on its own, which is why the word-count rule was added).
# The numbers differ (0.5 vs 0.6) because they judge different things at
# different times. Do not "unify" them and do not make layer 2 follow the flag:
# layer 2 is the gate whose thresholds have already failed in both directions
# and are pinned by tests.
WHISPER_NO_SPEECH_MAX = 0.5
# The no_speech rule applies ONLY at or below this word count. Gating on
# no_speech alone was wrong and silently deleted real sentences — measured on
# the owner's recordings:
#   0.86  "If we have a one-minute speech to deliver,"
#   0.55  "into the picture this framework will not only help us speak..."
#   0.55  "yeah amazon prime also serves into that needs based um..."
# So no_speech does NOT separate hallucination from speech; LENGTH does. The
# canned hallucinations are short ("Thank you.", "you"); a 13-word sentence
# about Whole Foods never is. Long text is now kept even when Whisper is unsure
# anyone spoke.
WHISPER_HALLUCINATION_MAX_WORDS = 4
WHISPER_COMPRESSION_MAX = 2.4       # Whisper's own repetition-loop threshold
# Words/second floor for a segment longer than the duration floor: a
# hallucination regardless of Whisper's confidence. Real speech runs ~2-3
# words/s; the classic failure is 30 s of silence read as "Thank you."
# (0.07 words/s). The 4 s floor protects a genuine short reply ("Okay.").
WHISPER_MIN_DENSITY_DURATION = 4.0
WHISPER_MIN_WORDS_PER_SEC = 0.5


def whisper_drop_reason(text: str, no_speech: float, compression: float,
                        duration: float):
    """Why this Whisper segment should be discarded, or None to keep it."""
    words = len(text.split())
    if no_speech > WHISPER_NO_SPEECH_MAX and words <= WHISPER_HALLUCINATION_MAX_WORDS:
        return f"hallucination (no_speech={no_speech:.2f}, {words}w)"
    if compression > WHISPER_COMPRESSION_MAX:
        return f"repetition (compression={compression:.2f})"
    if duration >= WHISPER_MIN_DENSITY_DURATION and words < duration * WHISPER_MIN_WORDS_PER_SEC:
        return (f"hallucination ({words} words in {duration:.0f}s = "
                f"{words / duration:.2f}/s)")
    return None


# --- transcript confidence (Whisper only) -----------------------------------
# Whisper reports `avg_logprob` per segment: the mean log-probability of the
# tokens it emitted there. Pooling those into ONE number per chunk is a weighted
# average IN THE LOG DOMAIN (never an average of the exponentiated values —
# that would over-weight the confident segments), weighted by how much text each
# segment actually contributed, then mapped back with exp().
#
# What the result means: the geometric-mean per-token probability of the words
# that survived the hallucination gate. It is NOT a percent-correct and not a
# WER estimate — a fluently-worded mishearing scores high. Word count stands in
# for token count; Whisper does not report token counts per segment, and the two
# are proportional enough for a relative-confidence read.
#
# Pure and module-level so it is testable with no model load: the pooling rule is
# the part that can be silently wrong, and a wrong number shown to the user as
# confidence is worse than no number at all.
#
# Empty input returns None, never 0.0 or 1.0. A chunk whose every segment was
# dropped has nothing to be confident about, and inventing a value there is
# exactly the failure mode the whole "absent field" convention exists to prevent.
def whisper_chunk_confidence(kept):
    """Pool per-segment (word_count, avg_logprob) into one 0..1 confidence.

    kept: [(word_count:int, avg_logprob:float)] for the segments KEPT in the
          transcript (dropped hallucinations contribute nothing, by construction
          — the caller appends only after the gate has decided).
    Returns a float in (0, 1], or None when there is nothing to pool.
    """
    import math
    pairs = [(w, lp) for w, lp in kept if w > 0 and lp is not None]
    total_words = sum(w for w, _ in pairs)
    if not pairs or total_words <= 0:
        return None
    pooled = sum(w * lp for w, lp in pairs) / total_words
    # avg_logprob is occasionally a hair above 0 on very short, very certain
    # segments; clamp so the displayed number can never read "1.03".
    return min(1.0, math.exp(pooled))


# --- decoding options ------------------------------------------------------
# mlx_whisper.transcribe()'s OWN defaults, restated here so the "defaults equal
# today's behaviour" rule is checkable in one place instead of inferred from the
# library. Stating them explicitly on the call is value-identical to omitting
# them; the three knobs that are NOT restated (initial_prompt, best_of,
# hallucination_silence_threshold) all default to None in mlx_whisper, and
# passing 0 is NOT the same as None — 0 would mean "sample 0 candidates" / "a
# 0 s silence threshold". Those are therefore OMITTED, which is how None is
# expressed here.
#
# ❌ beam_size is DELIBERATELY NOT EXPOSED, and this is not an oversight:
# mlx_whisper cannot do beam search at all. decoding.py:437 raises
#     NotImplementedError("Beam search decoder is not yet implemented")
# the moment `beam_size is not None`. Measured 2026-07-29: with --beam-size 2
# the sidecar died in its warmup and the app would have shown a load failure
# instead of starting the meeting. There is no threshold, fallback or version
# flag that makes it work here — mlx_whisper only has GreedyDecoder. A knob
# that can only fail is worse than no knob, so there is none. If a future
# mlx_whisper ships a beam decoder, add `beam_size` back here and to
# WhisperOptions; until then do not "restore" it.
WHISPER_DEFAULT_TASK = "transcribe"
WHISPER_DEFAULT_NO_SPEECH_THRESHOLD = 0.6
WHISPER_DEFAULT_LOGPROB_THRESHOLD = -1.0
WHISPER_DEFAULT_COMPRESSION_THRESHOLD = 2.4


def whisper_transcribe_kwargs(model: str, language, *,
                              task: str = WHISPER_DEFAULT_TASK,
                              initial_prompt: str = "",
                              best_of: int = 0,
                              no_speech_threshold: float = WHISPER_DEFAULT_NO_SPEECH_THRESHOLD,
                              logprob_threshold: float = WHISPER_DEFAULT_LOGPROB_THRESHOLD,
                              compression_threshold: float = WHISPER_DEFAULT_COMPRESSION_THRESHOLD,
                              hallucination_silence_sec: float = 0.0) -> dict:
    """Build the kwargs for mlx_whisper.transcribe() from the CLI settings.

    Pure and module-level ON PURPOSE: this is the one place where a settings
    value becomes a decoding decision, and the failure it can produce (a knob
    that silently changes the transcript, or a sentinel 0 reaching the decoder
    as a real value) is invisible in the output. `sidecar-tests.py` drives this
    function directly — no model load — for exactly that reason.

    Sentinels: best_of / hallucination_silence_sec of 0 mean "off" and are
    dropped, so mlx_whisper sees its own None. `initial_prompt` of "" likewise.
    `condition_on_previous_text` is deliberately NOT exposed and stays
    at Whisper's default True — the original call's comment says so, and it is
    what recovers connective phrases across chunk boundaries.

    hallucination_silence_sec > 0 additionally forces word_timestamps=True: in
    mlx_whisper's source the threshold is only read inside `if word_timestamps:`,
    so without it the setting would be accepted and do nothing. Whisper's word
    times are used only for that internal skip — they are never emitted (word
    times belong to the aligner sidecar, on its own reply).
    """
    kwargs = {
        "path_or_hf_repo": model,
        "language": language,
        "task": task,
        "no_speech_threshold": float(no_speech_threshold),
        "logprob_threshold": float(logprob_threshold),
        "compression_ratio_threshold": float(compression_threshold),
        "word_timestamps": False,
    }
    prompt = (initial_prompt or "").strip()
    if prompt:
        kwargs["initial_prompt"] = prompt
    if best_of and int(best_of) > 0:
        kwargs["best_of"] = int(best_of)
    if hallucination_silence_sec and float(hallucination_silence_sec) > 0:
        kwargs["hallucination_silence_threshold"] = float(hallucination_silence_sec)
        kwargs["word_timestamps"] = True
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="auto")
    # Decoding options. Every default below reproduces today's behaviour; see
    # whisper_transcribe_kwargs and the module docstring.
    parser.add_argument("--task", default=WHISPER_DEFAULT_TASK,
                        choices=["transcribe", "translate"],
                        help="translate = output English regardless of what was spoken")
    parser.add_argument("--auto-detect-language", action="store_true",
                        help="ignore --language and let Whisper detect it per chunk")
    parser.add_argument("--initial-prompt", default="",
                        help="prior context (names, jargon). CAN be echoed into "
                             "the transcript — logged on every chunk")
    parser.add_argument("--best-of", type=int, default=0,
                        help="0 = Whisper's default None")
    parser.add_argument("--no-speech-threshold", type=float,
                        default=WHISPER_DEFAULT_NO_SPEECH_THRESHOLD)
    parser.add_argument("--logprob-threshold", type=float,
                        default=WHISPER_DEFAULT_LOGPROB_THRESHOLD)
    parser.add_argument("--compression-threshold", type=float,
                        default=WHISPER_DEFAULT_COMPRESSION_THRESHOLD)
    parser.add_argument("--hallucination-silence-sec", type=float, default=0.0,
                        help="0 = OFF. Any value > 0 FORCES word_timestamps=True "
                             "(mlx_whisper reads it only there)")
    args = parser.parse_args()

    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        fail(f"numpy import failed: {brief_traceback()}")

    # --auto-detect-language wins over --language: the app can send both (the
    # language picker keeps its value while auto-detect is on), and auto-detect
    # is the more specific instruction. None ⇒ Whisper detects per chunk.
    language = None if args.language in ("", "auto") else args.language
    if args.auto_detect_language:
        language = None

    transcribe_kwargs = whisper_transcribe_kwargs(
        args.model, language,
        task=args.task,
        initial_prompt=args.initial_prompt,
        best_of=args.best_of,
        no_speech_threshold=args.no_speech_threshold,
        logprob_threshold=args.logprob_threshold,
        compression_threshold=args.compression_threshold,
        hallucination_silence_sec=args.hallucination_silence_sec,
    )
    prompt_words = len(transcribe_kwargs.get("initial_prompt", "").split())

    # Log every option that is NOT at its default, so a transcript that looks
    # wrong can be explained from logs/whisper.log alone. At default settings
    # this prints the "defaults" line and nothing else — the run is then
    # byte-identical to the pre-options sidecar.
    changed = []
    for key, default in (("task", WHISPER_DEFAULT_TASK),
                         ("no_speech_threshold", WHISPER_DEFAULT_NO_SPEECH_THRESHOLD),
                         ("logprob_threshold", WHISPER_DEFAULT_LOGPROB_THRESHOLD),
                         ("compression_ratio_threshold", WHISPER_DEFAULT_COMPRESSION_THRESHOLD)):
        if transcribe_kwargs[key] != default:
            changed.append(f"{key}={transcribe_kwargs[key]!r} (default {default!r})")
    for key in ("best_of", "hallucination_silence_threshold"):
        if key in transcribe_kwargs:
            changed.append(f"{key}={transcribe_kwargs[key]!r} (default off)")
    if args.auto_detect_language:
        changed.append("language=auto-detect")
    log("decoding: defaults" if not changed
        else "decoding: " + ", ".join(changed))

    if prompt_words:
        log(f"initial prompt ACTIVE ({prompt_words} words): "
            f"{transcribe_kwargs['initial_prompt']!r}")
        log("initial prompt RISK: Whisper continues text, so on unclear audio it "
            "can EMIT WORDS FROM THE PROMPT that nobody said. Each chunk below "
            "logs 'prompt active' — check them before trusting affected text.")
    if transcribe_kwargs.get("hallucination_silence_threshold") is not None:
        log(f"hallucination_silence_threshold="
            f"{transcribe_kwargs['hallucination_silence_threshold']}s ⇒ "
            "word_timestamps=True FORCED — mlx_whisper reads that threshold only "
            "inside its `if word_timestamps:` block, so the setting would "
            "otherwise be accepted and do nothing.")
        log("NOTE: this turns on Whisper's own word timing, which this project "
            "measured against the Qwen3 aligner and did NOT adopt. Those word "
            "times are used only for Whisper's internal skip — they are never "
            "emitted; word timestamps come from the aligner sidecar, which is a "
            "separate process this service knows nothing about.")

    def write_temp_wav(audio: "np.ndarray") -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return path

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

    # ---- runtime: mlx-whisper ----------------------------------------------
    # Apple's official MLX runtime; made for the mlx-community/whisper-large-v3-mlx
    # repo, tokenizer bundled.
    log(f"runtime: mlx-whisper, model {args.model}")
    try:
        import mlx_whisper
    except Exception:  # noqa: BLE001
        fail(f"mlx-whisper import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    def transcribe_path(path: str):
        """(text, confidence) — confidence is None if nothing was kept."""
        # condition_on_previous_text stays ON (Whisper's default, not exposed as
        # a setting) so the model has sentence context and recovers connective
        # phrases. Everything else comes from transcribe_kwargs, which equals
        # this call's original arguments unless the user moved a knob.
        # Hand mlx_whisper a decoded array, never the path, so it never calls
        # ffmpeg (absent on clean Macs and in the bundle — see load_audio_16k).
        if prompt_words:
            # Per-chunk marker: the prompt can put its own wording into THIS
            # chunk's text, so the evidence has to sit next to the chunk, not
            # only in a startup line scrolled far above it.
            log(f"prompt active ({prompt_words} words) — text may be echoed from it")
        result = mlx_whisper.transcribe(load_audio_16k(path), **transcribe_kwargs)
        segments = result.get("segments")
        if not segments:
            # No per-segment detail ⇒ no avg_logprob to pool ⇒ no confidence.
            return (result.get("text") or "").strip(), None
        kept = []
        scored = []   # (word_count, avg_logprob) for the KEPT segments only
        for s in segments:
            text = (s.get("text") or "").strip()
            dur = float(s.get("end", 0.0)) - float(s.get("start", 0.0))
            reason = whisper_drop_reason(
                text,
                no_speech=s.get("no_speech_prob", 0.0),
                compression=s.get("compression_ratio", 0.0),
                duration=dur)
            if reason:
                log(f"drop {reason}: {text!r}")
                continue
            kept.append(text)
            # STRICTLY after the drop decision, and reading the same segment
            # fields it already reads: the confidence describes the transcript
            # the user sees, so a dropped hallucination must not colour it.
            # Read-only w.r.t. the gate — no threshold or verdict changes here.
            logprob = s.get("avg_logprob")
            scored.append((len(text.split()),
                           None if logprob is None else float(logprob)))
        return " ".join(kept).strip(), whisper_chunk_confidence(scored)

    # Warmup with 0.5s silence — forces the model to load NOW so the
    # loading overlay reflects reality and the first chunk is fast.
    try:
        warmup = write_temp_wav(np.zeros(SR // 2, dtype=np.float32))
        try:
            transcribe_path(warmup)
        finally:
            os.unlink(warmup)
    except Exception:  # noqa: BLE001
        fail(f"Whisper model load failed ({args.model}): {brief_traceback()}")
    log("model loaded (mlx-whisper warmup done)")

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
