#!/usr/bin/env python3
"""
Nemotron 3.5 ASR streaming sidecar for MeetingTranscriber.

Loads mlx-community/nemotron-3.5-asr-streaming-0.6b (multilingual, 40 locales)
via mlx-audio and transcribes audio streamed from the app.

Protocol:
  stdout: JSON lines {"type": "status"|"partial"|"final", "text": ...}
          first line after model load contains READY
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes)
            n == 0 : FLUSH — transcribe current utterance as final, reset
            n == -1: exit

Config via argv: --language <auto|id-ID|en-US|...> --chunk-ms <80|160|560|1120>
Fully offline: HF_HOME points at the project models/ folder.
"""
import argparse
import json
import os
import struct
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _emit_raw(kind: str, text: str) -> None:
    try:
        sys.stdout.write(json.dumps({"type": kind, "text": text}) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


def fail(message: str) -> None:
    """Report a startup error to the app, then exit."""
    _emit_raw("error", message)
    sys.exit(1)


try:
    import numpy as np
except Exception as exc:  # noqa: BLE001
    fail(f"numpy import failed: {exc}. Run download-best-models.sh")

MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
SR = 16_000
MAX_BUFFER = 60 * SR          # cap utterance buffer at 60 s
PARTIAL_EVERY = int(1.5 * SR)  # emit a partial every 1.5 s of new audio

# Chunk-size setting → trained attention look-ahead [left, right]
ATT_CONTEXT = {80: [56, 0], 160: [56, 3], 560: [56, 6], 1120: [56, 13]}


def emit(kind: str, text: str) -> None:
    _emit_raw(kind, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="auto")
    parser.add_argument("--chunk-ms", type=int, default=160)
    args = parser.parse_args()

    att_context = ATT_CONTEXT.get(args.chunk_ms, [56, 3])
    language = None if args.language in ("", "auto") else args.language

    import traceback

    def brief_traceback() -> str:
        lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
        return " | ".join(lines[-3:])

    try:
        from mlx_audio.stt import load  # heavy import — keep inside main
    except Exception:  # noqa: BLE001
        fail(f"mlx-audio import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    try:
        model = load(MODEL)
    except Exception:  # noqa: BLE001
        fail(f"Nemotron model load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    def transcribe(audio: np.ndarray) -> str:
        kwargs = {}
        if language:
            kwargs["language"] = language
        try:
            return model.generate(audio, att_context_size=att_context, **kwargs).text
        except TypeError:
            pass  # build without att_context_size / array support — degrade below
        try:
            return model.generate(audio, **kwargs).text
        except Exception:
            # last resort: some builds only accept file paths
            import tempfile
            import wave
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                with wave.open(path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
                return model.generate(path, **kwargs).text
            finally:
                os.unlink(path)

    emit("status", "READY")

    buffer = np.zeros(0, dtype=np.float32)
    samples_since_partial = 0
    stdin = sys.stdin.buffer

    while True:
        header = stdin.read(4)
        if not header or len(header) < 4:
            break
        (n,) = struct.unpack("<i", header)

        if n == -1:
            break
        if n == 0:  # FLUSH — finalize utterance
            if buffer.size > SR // 4:  # ignore < 250 ms blips
                emit("final", transcribe(buffer).strip())
            buffer = np.zeros(0, dtype=np.float32)
            samples_since_partial = 0
            continue

        data = stdin.read(n * 4)
        if not data or len(data) < n * 4:
            break
        buffer = np.concatenate([buffer, np.frombuffer(data, dtype=np.float32)])
        if buffer.size > MAX_BUFFER:
            buffer = buffer[-MAX_BUFFER:]

        samples_since_partial += n
        if samples_since_partial >= PARTIAL_EVERY and buffer.size > SR // 2:
            samples_since_partial = 0
            emit("partial", transcribe(buffer).strip())


if __name__ == "__main__":
    main()
