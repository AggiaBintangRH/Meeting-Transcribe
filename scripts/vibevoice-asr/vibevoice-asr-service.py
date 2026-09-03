#!/usr/bin/env python3
"""
VibeVoice-ASR-Streaming chunked ASR sidecar for MeetingTranscriber.

microsoft/VibeVoice-ASR-Streaming-1.5B (MIT, 5.63 GB, 10 languages). Owner-
requested 2026-09-02 after the client asked for it by name.

⚠ THIS IS A SPEAKER-ATTRIBUTED ASR: one model emits text AND "who said it",
like MOSS. It therefore lands in the chunked ASR slot and its speaker labels
ride the transcript text as `Speaker N:` prefixes; it does NOT register as a
diarization engine. That is the MOSS-as-ASR shape, deliberately — see the
DiarizationTab notes on why a model that does both is still selected in one
slot at a time.

MEASURED ON THIS M4 (2026-09-02) BEFORE IT WAS BUILT, and recorded here so the
numbers are not re-derived — and so nobody reads the integration as an
endorsement:

  load                        5.2 s
  RSS                         10.96 GB          <-- the number that decides things
  MPS pool                    9.77-9.81 GB      flat across 30 s / 59 s / 98 s / 121 s
  speed                       4.6-5.3x realtime

  speaker count, auto:
    Overlap123.wav      truth 3   ->  3   ✓
    Meeting5People.wav  truth 5   ->  4   ✗   (all five app engines get 5)
    ATND 02-52-34       truth 5   ->  1   ✗✗  (five people merged into one)

⚠ THERE IS NO WAY TO PIN THE SPEAKER COUNT. `streaming_generate` takes no
`num_speakers` and the package contains no such parameter anywhere — checked,
not assumed. The one lever measured to fix every other engine on the client's
recordings does not exist here.

⚠ AND IT DOES NOT FIT THE CLIENT'S 16 GB MAC. 10.96 GB RSS against an 11.8 GB
macOS ceiling, before the other ~4 GB of sidecars. This is the MOSS situation
again (18.35 GB pool), and MOSS is what made that machine hang on 2026-08-21.
Built anyway at the owner's explicit direction, with the cost stated: "kita add
dulu saja nanti saya yang test".

Protocol — IDENTICAL to the other chunked ASR sidecars, because one Swift client
(`ChunkedASRService`) drives all of them:
  stdout: JSON lines {"type": "status"|"final"|"error", "text": ...}
          status LOADED after model init
          NO "conf" ever — this runtime reports no per-chunk confidence.
          Absent means "not measured", never "low".
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes)
            n == 0 : FLUSH — transcribe buffered chunk, emit final, reset
            n == -1: exit
            n == -2: FILE-TRANSCRIBE — one int32-length-prefixed UTF-8 JSON body
                     {"id":N,"path":"/tmp/.../track.wav"} follows; transcribe
                     that file, emit file_result/file_error. Fully additive.

Args: --model <path> [--language <code>]

⚠ --language IS ACCEPTED AND LOGGED, NEVER HONOURED. `streaming_generate` has no
language parameter — checked in the vendored source, not inferred from the model
card. It joins Granite, Voxtral, MOSS and Parakeet in the group whose picker is
selectable and says plainly that it will not change the transcript (the
2026-07-31 reversal). The code is forwarded anyway so the log records what was
asked for, and dies where the truth is.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import traceback

# One folder under scripts/, so THREE dirname calls to reach the project root.
# Two would resolve to scripts/ and put HF_HOME at scripts/models — the trap the
# folder-per-service move set on 2026-07-28, silent until the first model load.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The vendored package, OURS and under this service's own folder. `build.sh`
# strips `git+` refs out of the frozen requirements, so a pip install from
# GitHub silently vanishes from the packaged .app — the exact failure that
# shipped a broken DiCoW to a client machine on 2026-07-27. Inserted before the
# import so dev and the bundle run identical code.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

SR = 16_000
#: Below this the chunk is not sent to the model at all. Same floor and the same
#: reasoning as every other chunked sidecar: a sub-second buffer is a boundary
#: artefact, not speech.
MIN_CHUNK_SEC = 1.0


def log(message: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def emit(kind: str, text: str) -> None:
    sys.stdout.write(json.dumps({"type": kind, "text": text}) + "\n")
    sys.stdout.flush()


def emit_final(text: str, conf: float = None) -> None:
    """Emit a chunk final. `conf` is NEVER passed here — this runtime exposes no
    per-chunk confidence, and an absent key means "not measured" rather than
    "low". The parameter is kept because the wire format is shared with five
    other ASR sidecars and `whisper/protocol-matches-chunked` compares them."""
    payload = {"type": "final", "text": text}
    if conf is not None:
        payload["conf"] = conf
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def emit_file(kind: str, job_id: int, text: str) -> None:
    sys.stdout.write(json.dumps({"type": kind, "id": job_id, "text": text}) + "\n")
    sys.stdout.flush()


def fail(message: str) -> None:
    emit("error", message)
    log(f"FATAL {message}")
    sys.exit(1)


def brief_traceback() -> str:
    return " | ".join(traceback.format_exc().strip().splitlines()[-3:])


def read_exact(n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        part = sys.stdin.buffer.read(n - len(buf))
        if not part:
            return b""
        buf += part
    return buf


def main() -> None:
    model_path = None
    language = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            model_path = args[i + 1]
        elif a == "--language" and i + 1 < len(args):
            language = args[i + 1]

    if not model_path:
        fail("--model is required")
    if language:
        # See the module docstring: accepted, logged, never honoured.
        log(f"language={language} — ACCEPTED BUT NOT HONOURED: "
            "streaming_generate has no language parameter")

    log(f"loading VibeVoice ASR from {model_path}")
    t0 = time.time()
    try:
        # transformers 4.51.3 crashes formatting this config at INFO level
        # (`TypeError: Object of type dtype is not JSON serializable` from
        # `logger.info(f"Model config {config}")`, which builds the f-string
        # eagerly regardless of level). Measured 2026-09-02; silencing the
        # logger is the whole workaround.
        import transformers
        transformers.logging.set_verbosity_error()

        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration)
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        if processor.tokenizer.text_chunk_end_id is None:
            fail(f"{model_path} is not a streaming checkpoint: its tokenizer has "
                 "no <|text_chunk_end|>, so no chunk would ever end")

        # ⚠ dtype AFTER from_pretrained, not as a kwarg. Passing `dtype=` puts a
        # torch.dtype into the config dict, which is what the logger above then
        # fails to serialise. Load, then convert.
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path, attn_implementation="sdpa")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # float32 on MPS is upstream's own choice (demo/…_from_file.py:116 uses
        # bfloat16 ONLY for cuda). Not ours to second-guess without measuring
        # output equality, which has not been done.
        model = model.to(torch.float32).to(device).eval()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail(f"model load failed: {brief_traceback()} — run download-best-models.sh")

    frame_config = _frame_config(model_path)
    log(f"loaded on {device} in {time.time() - t0:.1f}s — "
        f"chunk {frame_config['chunk_duration']:.3f}s, "
        f"lookahead {frame_config['text_audio_delay']:.3f}s, "
        f"sr {frame_config['sample_rate']}")
    emit("status", "LOADED")

    buffer = np.zeros(0, dtype=np.float32)

    def transcribe(samples: np.ndarray) -> str:
        """Run the model over `samples` (16 kHz mono float32) and return text.

        ⚠ SAMPLES, NEVER A PATH. Upstream's demo calls
        `load_audio_use_ffmpeg(path)`, and ffmpeg is not a pip package and is not
        bundled — the trap this project has hit five times (Whisper, pyannote,
        spectral, DiariZen, MOSS). `streaming_generate` takes an audio tensor, so
        the decode stays here, in soundfile, and the .app needs no ffmpeg.
        """
        target = frame_config["sample_rate"]
        if SR != target:
            import scipy.signal as ss
            samples = ss.resample_poly(samples, target, SR).astype(np.float32)
        out = []
        for _idx, _total, text in model.streaming_generate(
                audio_tensor=torch.from_numpy(samples),
                tokenizer=processor.tokenizer,
                chunk_duration=frame_config["chunk_duration"],
                text_audio_delay=frame_config["text_audio_delay"],
                sample_rate=target,
                max_new_tokens_per_chunk=256,
                temperature=0.0):
            out.append(text)
        return "".join(out).strip()

    while True:
        header = read_exact(4)
        if not header:
            break
        (n,) = struct.unpack("<i", header)

        if n == -1:
            break

        if n == -2:  # FILE-TRANSCRIBE — additive, never touches `buffer`
            length_bytes = read_exact(4)
            if not length_bytes:
                break
            (blen,) = struct.unpack("<i", length_bytes)
            body = read_exact(blen)
            if not body:
                break
            job_id = -1
            try:
                request = json.loads(body.decode("utf-8"))
                job_id = int(request["id"])
                audio, rate = sf.read(request["path"], dtype="float32", always_2d=True)
                mono = audio[:, 0]
                if rate != SR:
                    import scipy.signal as ss
                    mono = ss.resample_poly(mono, SR, rate).astype(np.float32)
                emit_file("file_result", job_id, transcribe(mono))
            except Exception:  # noqa: BLE001
                emit_file("file_error", job_id, brief_traceback())
            continue

        if n == 0:  # FLUSH
            if buffer.size < int(MIN_CHUNK_SEC * SR):
                log(f"FLUSH skipped — {buffer.size / SR:.2f}s is under the "
                    f"{MIN_CHUNK_SEC:.1f}s floor")
                emit_final("")
                buffer = np.zeros(0, dtype=np.float32)
                continue
            t1 = time.time()
            try:
                text = transcribe(buffer)
            except Exception:  # noqa: BLE001
                log(f"transcribe failed: {brief_traceback()}")
                text = ""
            log(f"chunk done in {time.time() - t1:.1f}s ({len(text)} chars)")
            emit_final(text)
            buffer = np.zeros(0, dtype=np.float32)
            continue

        raw = read_exact(n * 4)
        if not raw:
            break
        buffer = np.concatenate([buffer, np.frombuffer(raw, dtype=np.float32)])


def _frame_config(model_path: str) -> dict:
    """Chunk length and lookahead come from the CHECKPOINT, never from us.

    Upstream's own note: "The chunk and lookahead are read from the checkpoint's
    preprocessor_config.json, so a checkpoint always runs at the chunk it was
    trained on." Hard-coding either would silently run the model off its training
    configuration the day a different checkpoint is dropped in.

    This is `demo/vibevoice_asr_streaming_inference_from_file.load_frame_config`
    reimplemented rather than imported: only `vibevoice/` is vendored, not
    `demo/`, and vendoring a whole demo directory to reach one six-line function
    would ship a gradio/fastapi entry point into an offline app. The arithmetic
    is pinned by `vibevoice/frame-config-matches-upstream` so the copy cannot
    drift from the original.

    Measured on the shipped checkpoint: 22 frames x 3200/24000 s = 2.9333 s
    chunk, 4 frames = 0.5333 s lookahead.
    """
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


if __name__ == "__main__":
    main()
