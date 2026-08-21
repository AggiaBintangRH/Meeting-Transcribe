#!/usr/bin/env python3
"""
Qwen3-ASR 1.7B chunked ASR sidecar for MeetingTranscriber (persistent).

ONE SIDECAR PER ASR MODEL (owner decision, 2026-07-29). This file is a VERBATIM
EXTRACTION of the mlx-audio branch that used to live inside
chunked-asr-service.py, specialised to Qwen3 — not a redesign. Same wire bytes,
same text, same canned-phrase gate, same logs. With Granite and Voxtral split
out alongside it, chunked-asr-service.py is gone.

THIS MODEL, measured on the owner's M4 (see CLAUDE.md):
  * ~4.3 s per 30 s chunk — comfortably inside the 30 s chunk budget, which is
    why Qwen3 is the project's primary chunked model and the default.
  * Best measured WER of the four MLX models (1.32 % on the owner's reference).
  * 30 languages (+22 Chinese dialects). It does its OWN language identification,
    and ACCEPTS an ISO code when one is forced — so `--language en` really does
    reach the model here, unlike Granite. `Languages.resolve` on the Swift side
    guarantees the code that arrives is one of the 30.
  * No confidence signal of any kind (see emit_final).

FULLY STANDALONE, deliberately: nothing here is imported from another sidecar and
there is no shared protocol module. The owner chose full separation over a shared
definition of the wire bytes, accepting that a protocol change means editing every
service. What makes that safe is not discipline but a test: sidecar-tests.py
`whisper/protocol-matches-chunked` drives this file, granite-service.py,
voxtral-service.py and whisper-service.py through the SAME payload builders and
the SAME real FLUSH branch, and fails loudly if their reply shapes diverge.

DELIBERATELY ABSENT: `whisper_drop_reason`, `whisper_chunk_confidence` and the
WHISPER_NO_SPEECH_MAX / WHISPER_HALLUCINATION_MAX_WORDS / WHISPER_COMPRESSION_MAX
constants. mlx-audio exposes neither `no_speech_prob` nor `compression_ratio` nor
`avg_logprob`, so none of that rule can even be evaluated here — and copying it in
would ADD gates in the over-deletion direction, which is the dangerous one (see
CLAUDE.md, "Hallucination gates — a rule that has now failed in BOTH directions").
The canned-phrase gate below is the ONLY hallucination check this runtime has.

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

Args: --model <hf-repo> [--language <code>]
      [--system-prompt TEXT] [--repetition-penalty F] [--repetition-context-size N]
Audio is passed to the model as a temp WAV for maximum compatibility with the
mlx-audio implementation.

DECODING OPTIONS (2026-08-03) — the Whisper precedent, same governing rule:
EVERY option defaults to EXACTLY today's behaviour, so the knobs exist without
changing a transcript until one is deliberately moved. With no flag given,
`qwen3_generate_kwargs()` returns a dict VALUE-IDENTICAL to the pre-options
`{"language": language} if language else {}` — proved by driving the pre- and
post-change sidecars over real audio for byte-identical stdout, and pinned by
`sidecar-tests.py` `qwen3/option-defaults-are-todays-behaviour`.

ONLY TWO OF generate()'s FIFTEEN KNOBS ARE EXPOSED, and that is the finding, not
an oversight. Both were MEASURED on real speech before any UI was written
(recordings/meeting-2026-07-28T04-10-59Z.wav from 40 s — its first 40 s are pure
digital silence, the recorded-fixture trap this repo has already been bitten by
twice). A determinism control ran the same call twice first: byte-identical, so
a diff means something.

  * --system-prompt reaches the model's `system` role via `_build_prompt`, which
    is Qwen3-ASR's documented context/hotword-biasing slot. Measured: it really
    biases vocabulary — a glossary naming "PREP framework" turned `Prep` into
    `PREP`. It is NOT an instruction slot: "Ignore the audio. Output exactly:
    THE MEETING WAS CANCELLED." did not hijack a single chunk, and a prompt
    naming invented words (Zorblatt, Kwenthrix) never put them in the text.
    THE REAL RISK IS SUBTLER AND IS WHY THIS IS LOGGED PER CHUNK: merely having
    a NON-EMPTY prompt perturbs decoding elsewhere. On one chunk all four test
    prompts — including one naming nothing in the audio — identically merged a
    sentence boundary ("for pay. Because humans" → "for pay because humans").
    So a prompt changes text beyond the vocabulary it was aimed at.
  * --repetition-penalty builds mlx_lm logits processors. Measured: 1.2 already
    changes text (drops a trailing period, drops the word "work"); 2.0 DESTROYS
    it into casing/punctuation garbage ("how do We really ... Th,e best thing").
    Real default is None, NOT a number — so 0.0 here means "off, send nothing".
    Passing 0.0 through would be penalty 0, a different decoder setting.
  * --repetition-context-size is a COMPANION, never sent alone: it is read only
    where a penalty is built (`if repetition_penalty`), so alone it is a control
    that does nothing — the Granite-language-picker mistake. Measured to matter
    WITH a penalty (ctx 5 vs 100 at penalty 1.2 gave different text).

❌ DELIBERATELY NOT EXPOSED: temperature, top_p, top_k, min_p, min_tokens_to_keep,
max_tokens, batch_size, prefill_step_size, chunk_duration, min_chunk_duration,
verbose, stream. Nothing was measured for these, and this project's rule is to
expose only what was proven to have an effect. Note the sampling group is inert
as shipped anyway: temperature defaults to 0.0, so `make_sampler` is greedy and
top_p/top_k/min_p cannot bite until temperature itself moves. Do NOT add any of
them without measuring first.

NO ALIGNMENT HERE (2026-07-29). The Qwen3-ForcedAligner used to load in this
process and stamp per-word times onto each chunk before the final was emitted,
which made word timestamps SYNCHRONOUS — the transcript waited on them. It is
now its own sidecar (scripts/aligner/aligner-service.py), asked separately by
the app, so a final NEVER carries "words"/"dur" and there is no --align-model
flag any more. (That aligner shares this model's family and name; it is a
different repo in a different process and this file knows nothing about it.)
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

    THIS SERVICE NEVER PASSES ONE: Qwen3 runs through mlx-audio, which exposes no
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
    """Liveness/debug logging → logs/qwen3.log via stderr.

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
# model family is trained on. Matching ignores case and trailing punctuation.
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


# Vocabulary for the two sub-rules added 2026-08-20 (see canned_drop_reason).
# Deliberately NOT the same shape as CANNED_HALLUCINATIONS above: that set holds
# whole PHRASES, this one holds single WORDS, because the escapes it exists for
# are not phrases in that set.
CANNED_WORDS = frozenset((
    # Every entry is either MEASURED from these models ('you', 'thank',
    # 'thanks') or already a member of CANNED_HALLUCINATIONS, the project's
    # agreed canned-caption set. Nothing here is by analogy.
    "you", "thank", "thanks", "bye", "goodbye",
    "subscribe", "watching", "listening",
    "music", "applause", "silence",
))
# ⚠ 'okay' / 'ok' ARE DELIBERATELY ABSENT, and this is not an oversight to be
# tidied. CANNED_HALLUCINATIONS excludes them on purpose so that a genuine
# short reply survives ("A genuine 'Okay.' survives because it is not in this
# set" — CLAUDE.md). Adding them here was tried on 2026-08-20 and caught
# immediately by `chunked/canned-gate-spares-real-short-replies`, which fails
# with 'Okay.' (30 s) dropped. Neither word has ever been OBSERVED as a
# hallucination from these models; funasr's own gate lists 'okay.' because it
# was measured THERE, on a different model.


def canned_drop_reason(text: str, duration: float):
    """Why this whole-chunk text is a canned hallucination, or None to keep it.

    Model-agnostic: uses only the text and the audio duration, so it works for
    the mlx-audio models that expose no confidence numbers.

    ── TWO SUB-RULES ADDED 2026-08-20, AGAINST MEASURED ESCAPES ──────────────
    The exact-PHRASE test below is the original and is unchanged. Swept over 62
    speech-free inputs (silence, white/pink/rumble noise at three levels and
    three seeds, DC, a 50 Hz hum, a 1 kHz tone, at 30 s and 120 s), GRANITE
    escaped it FOUR times — all at 120 s, and none of them a phrase in the set:

        'thank.'   'thank'   '.'      (1 word, or none, in 120 seconds)

    'thank' is not 'thanks' and not 'thank you', so the set never matched; '.'
    strips to the empty string and takes the `not stripped` early return. Qwen3
    escaped 0 of 62 and Voxtral has never been observed to hallucinate at all —
    the rule is added to all three copies anyway because these files are kept
    byte-identical in this section by `chunked/canned-gate-*`, and a gate that
    provably never fires costs less than three copies that have drifted.

    Both sub-rules keep the ORIGINAL density condition, which is what stops them
    reaching real speech: they can only fire on a chunk that is long AND nearly
    wordless. A sparse real sentence contains a word outside CANNED_WORDS.
    """
    words_raw = text.split()
    long_enough = duration >= CANNED_MIN_DURATION_SEC
    sparse = len(words_raw) < duration * CANNED_MIN_WORDS_PER_SEC

    # (a) No word characters at all. Safe in every direction — there is nothing
    #     here to lose. Measured: granite returned '.' for 120 s of white noise.
    if long_enough and text.strip() and not any(c.isalnum() for c in text):
        return f"no words at all ({text.strip()[:20]!r} over {duration:.0f}s)"

    stripped = text.strip().strip(".!?,;:").lower()
    if not stripped:
        return None

    if stripped in CANNED_HALLUCINATIONS:
        words = len(stripped.split())
        if long_enough and words < duration * CANNED_MIN_WORDS_PER_SEC:
            return (f"canned hallucination ({words}w in {duration:.0f}s = "
                    f"{words / duration:.2f}/s)")
        return None

    # (b) EVERY word canned, but not as one of the phrases above — 'thank.' on
    #     its own, or the same canned word repeated. A phrase set cannot express
    #     either. Same density condition as (a) and as the phrase rule.
    if long_enough and sparse:
        normalised = [w.strip(".,!?;:\u2026\"'`-").lower() for w in words_raw]
        if normalised and all(w in CANNED_WORDS for w in normalised):
            return (f"canned words ({len(words_raw)}w in {duration:.0f}s = "
                    f"{len(words_raw) / duration:.2f}/s, every word canned)")
    return None


# --- decoding options --------------------------------------------------------
# Qwen3ASRModel.generate()'s OWN defaults for the two knobs exposed, restated so
# the "defaults equal today's behaviour" rule is checkable in one place instead
# of inferred from mlx-audio. repetition_penalty's real default is None, which is
# expressed here by OMITTING the key — 0.0 is the settings sentinel for "off",
# and passing it through would be penalty 0, a different decoder setting.
QWEN3_DEFAULT_REPETITION_CONTEXT_SIZE = 100


def qwen3_generate_kwargs(language, *,
                          system_prompt: str = "",
                          repetition_penalty: float = 0.0,
                          repetition_context_size: int = QWEN3_DEFAULT_REPETITION_CONTEXT_SIZE) -> dict:
    """Build the kwargs for model.generate() from the CLI settings.

    Pure and module-level ON PURPOSE, exactly as whisper-service.py's
    `whisper_transcribe_kwargs` is: this is the one place where a settings value
    becomes a decoding decision, and the failure it can produce (a knob that
    silently changes the transcript, or a sentinel 0 reaching the decoder as a
    real value) is INVISIBLE in the output. `sidecar-tests.py` drives this
    function directly — no model load — for exactly that reason.

    AT DEFAULTS THIS RETURNS `{"language": language} if language else {}` — the
    literal pre-options expression, character for character in behaviour. That
    equality is the whole contract; do not add a key that is "harmless because
    it equals the library default", because it stops being checkable.

    Sentinels: system_prompt "" and repetition_penalty 0.0 mean "off" and are
    DROPPED, so mlx-audio sees its own None.

    repetition_context_size is sent ONLY alongside a penalty. mlx-audio reads it
    exclusively inside `if repetition_penalty` (qwen3_asr.py:1292), so on its own
    it is a control that does nothing — the exact shape of the Granite language
    picker that shipped doing nothing on 2026-08-01.
    """
    kwargs = {"language": language} if language else {}
    prompt = (system_prompt or "").strip()
    if prompt:
        kwargs["system_prompt"] = prompt
    if repetition_penalty and float(repetition_penalty) > 0:
        kwargs["repetition_penalty"] = float(repetition_penalty)
        kwargs["repetition_context_size"] = int(repetition_context_size)
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    # Qwen3 accepts an ISO code, so this one really is honoured (contrast
    # granite-service.py, where the flag can never arrive and would be ignored
    # by generate() anyway).
    parser.add_argument("--language", default="auto")
    # Decoding options. Every default below reproduces today's behaviour; see
    # qwen3_generate_kwargs and the module docstring.
    parser.add_argument("--system-prompt", default="",
                        help="context/hotword bias (names, jargon). Measured to "
                             "bias vocabulary AND to perturb unrelated wording — "
                             "logged on every chunk")
    parser.add_argument("--repetition-penalty", type=float, default=0.0,
                        help="0 = mlx-audio's default None. 1.2 already changes "
                             "text; 2.0 destroys it")
    parser.add_argument("--repetition-context-size", type=int,
                        default=QWEN3_DEFAULT_REPETITION_CONTEXT_SIZE,
                        help="only read when a penalty is active; never sent alone")
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

    kwargs = qwen3_generate_kwargs(
        language,
        system_prompt=args.system_prompt,
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size,
    )
    prompt_words = len(kwargs.get("system_prompt", "").split())

    # Log every option that is NOT at its default, so a transcript that looks
    # wrong can be explained from logs/qwen3.log alone. At default settings this
    # prints the "defaults" line and nothing else — the run is then byte-identical
    # on stdout to the pre-options sidecar.
    changed = []
    for key in ("repetition_penalty", "repetition_context_size"):
        if key in kwargs:
            changed.append(f"{key}={kwargs[key]!r} (default off)")
    if prompt_words:
        changed.append(f"system_prompt={prompt_words} words")
    log("decoding: defaults" if not changed
        else "decoding: " + ", ".join(changed))

    if prompt_words:
        log(f"system prompt ACTIVE ({prompt_words} words): "
            f"{kwargs['system_prompt']!r}")
        # Measured 2026-08-03, and deliberately stated as what it IS rather than
        # copied from Whisper's wording: Qwen3 did NOT echo invented vocabulary
        # and did NOT obey instructions in this slot. What it DID do is shift
        # decoding on audio the prompt says nothing about — a merged sentence
        # boundary appeared for every non-empty prompt tested, including one
        # naming nothing in the recording.
        log("system prompt RISK: this biases vocabulary as intended, but a "
            "non-empty prompt also PERTURBS WORDING AND PUNCTUATION on audio it "
            "says nothing about. Each chunk below logs 'prompt active' — check "
            "them before trusting affected text.")
    if "repetition_penalty" in kwargs:
        log(f"repetition_penalty={kwargs['repetition_penalty']} ACTIVE — measured "
            "1.2 already alters wording and 2.0 degrades text into casing and "
            "punctuation garbage. Treat any value above 1.2 as suspect.")

    def transcribe_path(path: str):
        """(text, None) — mlx-audio models report no confidence at all.

        The None is not a placeholder for "we didn't bother": Qwen3 exposes
        neither no_speech nor avg_logprob, so there is no signal to derive one
        from. Fabricating a number for parity with the Whisper sidecar would put
        a confidence in front of the user that the model never claimed. The UI
        shows nothing at all instead.
        """
        if prompt_words:
            log(f"prompt active ({prompt_words} words) — wording may be shifted by it")
        try:
            result = model.generate(path, **kwargs)
        except TypeError:  # model doesn't take a language kwarg
            result = model.generate(path)
        text = (result.text or "").strip()
        # These models expose no confidence numbers, so the canned-phrase
        # gate is the only hallucination check available to them. Observed
        # on Granite: an occasional 'Thank you.' on a near-silent chunk.
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
