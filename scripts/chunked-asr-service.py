#!/usr/bin/env python3
"""
Chunked ASR sidecar for MeetingTranscriber (persistent).

Loads the selected chunked model ONCE, then transcribes rolling chunks of
the meeting as they are flushed by the app (every N seconds, aligned to
VAD silence). Higher-accuracy pass than the realtime Nemotron captions.

Protocol:
  stdout: JSON lines {"type":"status"|"final"|"error","text":...}
          status LOADED after model init
          final may additionally carry "words" (per-word timestamps) and
          "dur" (chunk length in seconds) when --align-model is given and
          alignment succeeded; both keys are simply absent otherwise.
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes)
            n == 0 : FLUSH — transcribe buffered chunk, emit final, reset
            n == -1: exit
            n == -2: FILE-TRANSCRIBE — one int32-length-prefixed UTF-8 JSON body
                     {"id":N,"path":"/tmp/.../track.wav"} follows; transcribe that
                     file with the loaded model, emit file_result/file_error.
                     Fully additive — never touches the live streaming buffer.

Args: --model <hf-repo> [--language <code>] [--align-model <hf-repo>]
Audio is passed to the model as a temp WAV for maximum compatibility
across Qwen3 / Whisper / Voxtral mlx-audio implementations.

--align-model enables optional forced alignment (Qwen3-ForcedAligner) over
the flushed chunk, so each final can carry per-word start/end times. Absent
⇒ alignment is fully disabled and the output is byte-identical to before.
Alignment never affects the transcript: any failure just drops "words".
"""
import argparse
import json
import os
import struct
import sys
import tempfile
import traceback
import wave

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SR = 16_000
MIN_CHUNK_SEC = 0.25   # ignore blips shorter than this
MAX_BUFFER_SEC = 300   # safety cap

# Forced alignment force-fits whatever text it is handed onto the audio, so
# hallucinated tail text (seen from Whisper) gets stamped past the end of the
# chunk. Anything ending later than this is not a real word time.
ALIGN_END_TOLERANCE_SEC = 0.5
# If more than this fraction of items land past the end, the whole alignment
# is untrustworthy — emit no times at all rather than a truncated guess.
ALIGN_MAX_PAST_END_FRACTION = 0.10


def emit(kind: str, text: str) -> None:
    try:
        sys.stdout.write(json.dumps({"type": kind, "text": text}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


def emit_final(text: str, words=None, dur: float = None) -> None:
    """Emit a chunk final, optionally carrying per-word timestamps.

    Kept separate from emit() so the existing (kind, text) contract — and the
    Swift decoder that reads "text" — stays untouched. "words"/"dur" are only
    present when alignment ran AND passed every gate.
    """
    payload = {"type": "final", "text": text}
    if words is not None:
        payload["words"] = words
        payload["dur"] = dur
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def emit_file(kind: str, req_id, text: str) -> None:
    """Emit a file-transcribe result carrying its request id (overlap repair)."""
    try:
        sys.stdout.write(json.dumps({"type": kind, "id": req_id, "text": text}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def log(message: str) -> None:
    """Liveness/debug logging → logs/chunked-asr.log via stderr."""
    import time
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def fail(message: str) -> None:
    emit("error", message)
    sys.exit(1)


def brief_traceback() -> str:
    lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
    return " | ".join(lines[-3:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--align-model", default=None,
                        help="Qwen3-ForcedAligner repo; omit to disable alignment")
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

    # ---- runtime dispatch: one runtime per model family --------------------
    if "whisper" in args.model.lower():
        # Whisper → mlx-whisper (Apple's official MLX runtime; made for the
        # mlx-community/whisper-large-v3-mlx repo, tokenizer bundled).
        log(f"runtime: mlx-whisper, model {args.model}")
        try:
            import mlx_whisper
        except Exception:  # noqa: BLE001
            fail(f"mlx-whisper import failed: {brief_traceback()} — "
                 "run download-best-models.sh")

        # Whisper hallucinates canned text ("Thank you", "Thanks for watching")
        # on silence — it was trained on subtitled video where quiet stretches
        # carry closing captions. Its own no_speech/logprob rule does NOT catch
        # this, because it is CONFIDENT about the hallucination (measured on real
        # silence: no_speech=0.67, logprob=-0.26, so the "high no_speech AND low
        # logprob" gate keeps it). But no_speech_prob alone separates cleanly:
        # real speech on the owner's recordings sat at 0.001-0.121, the "Thank
        # you." hallucination at 0.672. So drop any segment above this on its own.
        # compression_ratio catches the separate repetition-loop failure.
        # This is Whisper-only; the mlx-audio models never hallucinate this way.
        WHISPER_NO_SPEECH_MAX = 0.5
        WHISPER_COMPRESSION_MAX = 2.4   # Whisper's own repetition-loop threshold
        # A segment longer than this holding fewer than DENSITY words/second is a
        # hallucination regardless of what Whisper's own confidence says. Real
        # speech runs ~2-3 words/s; the classic failure is a 30 s stretch of
        # silence transcribed as the two words "Thank you." (0.07 words/s). The
        # no_speech gate above misses it when Whisper is confident (seen at
        # no_speech 0.88 on a 30 s chunk), so density is the backstop that does
        # not depend on Whisper's confidence at all. The 4 s floor protects a
        # genuine short reply ("Okay.") from being judged on density.
        WHISPER_MIN_DENSITY_DURATION = 4.0
        WHISPER_MIN_WORDS_PER_SEC = 0.5

        def transcribe_path(path: str) -> str:
            # Keep Whisper's default decoding — condition_on_previous_text stays
            # ON so the model has sentence context (recovers connective phrases).
            result = mlx_whisper.transcribe(
                path, path_or_hf_repo=args.model, language=language
            )
            segments = result.get("segments")
            if not segments:
                return (result.get("text") or "").strip()
            kept = []
            for s in segments:
                ns = s.get("no_speech_prob", 0.0)
                cr = s.get("compression_ratio", 0.0)
                text = (s.get("text") or "").strip()
                dur = float(s.get("end", 0.0)) - float(s.get("start", 0.0))
                words = len(text.split())
                if ns > WHISPER_NO_SPEECH_MAX:
                    log(f"drop hallucination (no_speech={ns:.2f}): {text!r}")
                    continue
                if cr > WHISPER_COMPRESSION_MAX:
                    log(f"drop repetition (compression={cr:.2f}): {text!r}")
                    continue
                if (dur >= WHISPER_MIN_DENSITY_DURATION
                        and words < dur * WHISPER_MIN_WORDS_PER_SEC):
                    log(f"drop hallucination ({words} words in {dur:.0f}s = "
                        f"{words / dur:.2f}/s): {text!r}")
                    continue
                kept.append(text)
            return " ".join(kept).strip()

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
    else:
        # Qwen3 / Voxtral → mlx-audio
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

        def transcribe_path(path: str) -> str:
            try:
                result = model.generate(path, **kwargs)
            except TypeError:  # model doesn't take a language kwarg
                result = model.generate(path)
            return (result.text or "").strip()

    def transcribe(audio: "np.ndarray") -> str:
        path = write_temp_wav(audio)
        try:
            return transcribe_path(path)
        finally:
            os.unlink(path)

    # ---- optional forced aligner -------------------------------------------
    # Loaded here, BEFORE status LOADED, so a broken/missing aligner surfaces in
    # the app's loading overlay through fail() instead of mid-recording.
    aligner = None
    align_proc = None
    if args.align_model:
        log(f"aligner: {args.align_model}")
        try:
            import time as _t
            from mlx_audio.stt import load as load_align
            _t0 = _t.time()
            aligner = load_align(args.align_model)
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
            fail(f"Aligner load failed ({args.align_model}): {brief_traceback()}")

    def pair_source_indices(text: str, items):
        """Map every aligned item back to its index in text.split(), or None.

        The aligner's tokenizer DROPS punctuation-only tokens ("a — b" yields 2
        items, not 3), so a naive item↔word index would silently shift every
        later word into the wrong speaker row. This replays the model's own
        tokenize_space_lang() per source word and cross-checks the result
        against encode_timestamp()'s word_list (the ground truth for what the
        items will be). Any inconsistency ⇒ None ⇒ no timestamps at all; a
        wrong index is worse than no alignment.
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

    def align_chunk(audio: "np.ndarray", text: str):
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

            src_indices = pair_source_indices(text, items)
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
                text = transcribe_path(path)
                log(f"file-transcribe id={req_id} done ({len(text)} chars)")
                emit_file("file_result", req_id, text)
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
                    text = transcribe(buffer)
                    log(f"chunk done in {time.time() - started:.1f}s ({len(text)} chars)")
                    words = align_chunk(buffer, text) if text else None
                    emit_final(text, words, secs if words is not None else None)
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
