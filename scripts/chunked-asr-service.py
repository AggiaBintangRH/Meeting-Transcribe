#!/usr/bin/env python3
"""
Chunked ASR sidecar for MeetingTranscriber (persistent).

Loads the selected chunked model ONCE, then transcribes rolling chunks of
the meeting as they are flushed by the app (every N seconds, aligned to
VAD silence). Higher-accuracy pass than the realtime Nemotron captions.

Protocol:
  stdout: JSON lines {"type":"status"|"final"|"error","text":...}
          status LOADED after model init
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
Audio is passed to the model as a temp WAV for maximum compatibility
across Qwen3 / Whisper / Voxtral mlx-audio implementations.
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


def emit(kind: str, text: str) -> None:
    try:
        sys.stdout.write(json.dumps({"type": kind, "text": text}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


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

        def transcribe_path(path: str) -> str:
            # Keep Whisper's default decoding — condition_on_previous_text stays
            # ON so the model has sentence context (recovers connective phrases).
            # The earlier "loops" were a repeating source clip, not model drift.
            result = mlx_whisper.transcribe(
                path, path_or_hf_repo=args.model, language=language
            )
            return (result.get("text") or "").strip()

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
                    emit("final", text)
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
