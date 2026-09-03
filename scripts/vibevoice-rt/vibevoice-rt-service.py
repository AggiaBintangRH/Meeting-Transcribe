#!/usr/bin/env python3
"""
VibeVoice-ASR-Streaming REALTIME sidecar for MeetingTranscriber.

The FOURTH realtime engine (2026-09-02), beside nemotron/, parakeet/ and
funasr/. Same checkpoint as `vibevoice-asr` — microsoft/VibeVoice-ASR-Streaming-
1.5B, MIT — in its OTHER role. It speaks EXACTLY the frame protocol
nemotron-service.py and parakeet-service.py speak, because one Swift client
(`RealtimeASRService`) drives all of them.

⚠ TWO SERVICES FOR ONE MODEL, and that is the MOSS precedent rather than
duplication for its own sake. The two roles have DIFFERENT protocols — `-2`
means REMOTE AUDIO here and FILE-TRANSCRIBE in the chunked service — and they
can be two live processes at once, which is exactly how MOSS ended up with two
writers on one log in 2026-07. Two folders, two services, two logs.

⚠ AND RUNNING BOTH AT ONCE COSTS TWICE THE MEMORY. Each process holds its own
copy of the weights: 10.96 GB RSS measured, so VibeVoice-as-realtime AND
VibeVoice-as-chunked is ~22 GB. That fits this 64 GB M4 and fits nothing
smaller. Nothing here refuses the combination — the owner's standing decision is
that every engine stays selectable and a slowdown is evidence for a hardware
budget, not something to hide behind a refusal.

THE STREAMING API IS GENUINELY INCREMENTAL, unlike Parakeet's. `init_streaming_
state()` + `streaming_generate_step()` advance a running KV cache one audio
window at a time, so a lane's cost does NOT grow with utterance length the way a
whole-buffer re-transcribe does. That is a better fit for this app's shape than
anything else here — and it is why the partial cadence below is the model's own
chunk rather than a number we picked.

MEASURED ON THIS M4 (2026-09-02), model load EXCLUDED — that exclusion matters,
a first run that counted the 5.2 s load reported 1.6x and was simply wrong:

    30 s audio -> 5.8 s   5.21x realtime    duty 19 % one lane, 38 % two
    60 s audio -> 11.7 s  5.15x realtime    duty 19 % one lane, 39 % two

⚠ CONSTANT ACROSS 30 s AND 60 s, and that is the whole design paying off. Every
other realtime engine here re-transcribes its lane's WHOLE buffer per partial, so
cost grows with utterance length and Nemotron needs `PARTIAL_DUTY` cadence
stretching to stay under budget at all. This one advances a cache, so a 60 s
utterance costs exactly twice a 30 s one and no more.

So it is the SLOWEST engine here per second of audio (Parakeet ~128x, Fun-ASR
~49x, Nemotron ~14x) and NOT the heaviest on duty: 38 % against Fun-ASR's 81 %
and Nemotron's 285 %. Judge it on the duty column, which is what decides whether
captions keep up.

⚠ WHAT IT COSTS INSTEAD IS MEMORY: 10.96 GB RSS, the same as the chunked role,
because it is the same weights. See the two-processes note above.

THE WINDOW ARITHMETIC IS UPSTREAM'S, from demo/vibevoice_asr_streaming_fastapi_
demo.py, which is the one path that feeds a live microphone:

    window = chunk + lookahead          (2.9333 s + 0.5333 s at 24 kHz)
    encode_speech(window) -> features -> streaming_generate_step(features, state)
    buffer = buffer[chunk:]             drop the advance, KEEP the lookahead —
                                        it is the next chunk's leading audio

Consuming rather than indexing is what bounds a long meeting: an index-only
cursor keeps every sample of the session alive and re-copies all of it on each
incoming frame. Upstream says so in its own comment; it is repeated here because
this is the line a "simplification" would remove.

⚠ THE LANE BUFFER IS 16 kHz AND THE MODEL WANTS 24 kHz. The resample happens
per WINDOW, not per incoming frame. Resampling each ~85 ms tap buffer separately
would put a filter discontinuity at every frame boundary — dozens per second —
and those land inside words. Resampling the assembled window instead puts at
most one at each window edge, where the lookahead already overlaps.

Protocol — IDENTICAL to nemotron/parakeet/funasr:
  stdout: JSON lines {"type": "status"|"partial"|"final", "text": ...}
          first line after model load contains READY
          remote-lane messages additionally carry "stream":"remote"; the field is
          OMITTED for office, so a single-stream session's output is
          byte-identical in shape to the other three engines'.
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes) — OFFICE
            n == 0 : FLUSH office — finish the utterance as final, reset
            n == -1: exit
            n == -2: REMOTE audio — [int32 m] then m float32 samples follow
            n == -3: FLUSH remote

Args: --model <path> [--language <code>]

⚠ --language IS ACCEPTED AND LOGGED, NEVER HONOURED — `streaming_generate_step`
has no language parameter, checked in the vendored source. Same group as
Granite, Voxtral, MOSS, Parakeet and this model's own chunked role.

⚠ NO HALLUCINATION GATE, DELIBERATELY, AND THIS IS NOT AN OVERSIGHT. The
project's rule is that a gate guards an OBSERVED failure: Whisper's canned
captions, Granite's, MOSS answering as a chatbot, Fun-ASR inventing sentences in
three languages. Nothing of the kind has been observed from this model, and a
gate against an unobserved failure is pure over-deletion risk — the direction
that leaves no trace in a transcript. If one is ever seen, add it against the
text that was actually produced.
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys
import time
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# This service's OWN vendored tree, under its own folder. Byte-identical to the
# chunked role's and deliberately not shared: either service pointing its
# sys.path at the OTHER's folder works today and breaks the day that folder
# moves — months later, in a release. `vibevoice/vendor-trees-are-own-and-identical`
# reads this literal by AST and hash-compares the two trees.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

#: The checkpoint, HARD-CODED like every other realtime sidecar's (parakeet,
#: funasr, nemotron all declare their own `MODEL`). `RealtimeASRService` passes
#: no `--model` to any of them — its `processArguments` are `--language` plus at
#: most one engine-specific flag — so requiring one here made this service the
#: only one that could not start. It said so, in `logs/vibevoice-rt.log`:
#: "FATAL --model is required".
MODEL = "microsoft/VibeVoice-ASR-Streaming-1.5B"

SR = 16_000
#: Cap a lane's buffer. Unlike the other realtime engines this is NOT about cost
#: — the KV cache advances per window, so a long utterance is not re-transcribed
#: — it bounds the memory a lane can hold if FLUSH never arrives.
MAX_BUFFER = 60 * SR


#: `Speaker 12:` at the start of a run, however the model spaces it.
SPEAKER_LABEL_RE = re.compile(r"\s*Speaker\s*\d+\s*:\s*", re.IGNORECASE)

#: `[Silence]` — the model's own annotation for a stretch with no speech. It is
#: GENERATED TEXT, not a special token: upstream strips only `<|…|>` markers, so
#: this one survives into the transcript.
#:
#: ⚠ ONE MARKER, BECAUSE ONE IS WHAT WAS OBSERVED. Measured 2026-09-02: 20 s of
#: digital silence produced 63 characters that were nothing but `[Silence]`, and
#: a 98 s real recording produced NONE at all. `[Music]`, `[Noise]` and the rest
#: are not in this pattern because this model has never been seen to emit them —
#: the same rule the hallucination gates follow, and for the same reason: a
#: pattern guarding an unobserved case can only delete real text.
SILENCE_MARKER_RE = re.compile(r"\s*\[Silence\]\s*", re.IGNORECASE)


def strip_speaker_labels(text: str) -> str:
    """Remove the model's own `Speaker N:` prefixes from transcript text.

    ⚠ THIS MODEL IS SPEAKER-ATTRIBUTED ASR AND WE USE IT AS ASR ONLY. Its
    diarization role was built and withdrawn on 2026-09-02 (see CLAUDE.md — it
    could not reproduce its own speaker count), so the row's speaker comes from
    the real diarizer, and the label inside the text is a SECOND naming of the
    same row that agrees with nothing:

        SPEAKER 5 · 01:24–01:28
        Speaker 0:Can help out with the low cost.     <- two names, one row

    Owner, 2026-09-02: "remove the speaker nya".

    ⚠ THE LABELS ARE NOT DISCARDED SILENTLY — they were never used. Nothing
    downstream reads them: `ChunkedASRService` decodes `text` and the aligner
    splits it into words. Stripping here means the wire carries what every other
    ASR sidecar's does, which is what makes VibeVoice interchangeable with them.

    A run boundary becomes a single space rather than nothing, so two speakers'
    sentences do not fuse into one word.

    ⚠ `[Silence]` GOES TOO, and a chunk that was ONLY silence therefore returns
    the EMPTY STRING — which is what stops it becoming a transcript row at all.
    Owner, 2026-09-02, from a real transcript that had rows reading
    `[Silence][Silence][Silence]…`. Every other ASR sidecar here returns "" for a
    silent chunk; this makes VibeVoice behave the same way rather than narrating
    the silence.

    ⚠ NOT A HALLUCINATION GATE, and the difference matters. The gates in the
    Whisper and mlx-audio services decide whether MODEL-INVENTED WORDS are real
    speech, and they are dangerous because they can delete a genuine sentence.
    This deletes a literal annotation the model prints instead of words — there
    is no sentence it could take with it.
    """
    return SILENCE_MARKER_RE.sub(" ", SPEAKER_LABEL_RE.sub(" ", text)).strip()


def log(message: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def emit(kind: str, text: str, remote: bool = False) -> None:
    payload = {"type": kind, "text": text}
    if remote:
        payload["stream"] = "remote"
    sys.stdout.write(json.dumps(payload) + "\n")
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


def _checkpoint_dir(model_path: str) -> str:
    """A local directory for `model_path`, which may be a REPO ID.

    ⚠ THE APP SENDS A REPO ID, NOT A PATH, and this cost a "model loading
    failed" on the first real launch. Every sidecar here is handed
    `ModelInfo.hfRepo`; only the hand-drives during development passed an
    absolute snapshot path, which is exactly why the bug survived three services
    being tested. Upstream's own `load_frame_config` branches the same way.

    Resolution is offline: `HF_HUB_OFFLINE=1` is set above, so this reads the
    cache that `download-best-models.sh` filled and never reaches the network.
    """
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download
    return snapshot_download(model_path)


def frame_config(model_path: str) -> dict:
    """Chunk and lookahead from the CHECKPOINT — see the chunked service's copy
    of this function for why it is not hard-coded and not imported from `demo/`.
    `vibevoice/frame-config-matches-upstream` pins both copies to upstream's
    arithmetic."""
    with open(os.path.join(_checkpoint_dir(model_path),
                           "preprocessor_config.json"),
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


class Lane:
    """One audio stream. Office and Remote each own one.

    The weights are SHARED (one process, one copy) and the AUDIO never is —
    separate buffers, separate KV caches, separate accumulated text. That
    separation IS the point of dual-stream capture; the Nemotron sidecar's
    docstring makes the same promise for the same reason.
    """

    def __init__(self, name: str, remote: bool) -> None:
        self.name = name
        self.remote = remote
        self.buffer = np.zeros(0, dtype=np.float32)
        self.state = None
        self.texts: list[str] = []

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.state = None
        self.texts = []


def main() -> None:
    language = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        # `--model` is ACCEPTED but not required: no realtime caller passes one,
        # and rejecting it outright would make a future caller's override an
        # argparse-style failure at session start rather than a no-op.
        if a == "--model" and i + 1 < len(args):
            globals()["MODEL"] = args[i + 1]
        elif a == "--language" and i + 1 < len(args):
            language = args[i + 1]
    model_path = MODEL
    if language:
        log(f"language={language} — ACCEPTED BUT NOT HONOURED: the streaming API "
            "has no language parameter")

    fc = frame_config(model_path)
    log(f"loading VibeVoice realtime from {model_path}")
    t0 = time.time()
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration)
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        if processor.tokenizer.text_chunk_end_id is None:
            fail(f"{model_path} is not a streaming checkpoint: its tokenizer has "
                 "no <|text_chunk_end|>, so no chunk would ever end")
        # dtype AFTER from_pretrained — passing `dtype=` puts a torch.dtype in the
        # config, which transformers 4.51.3 then fails to JSON-serialise while
        # logging it. Measured 2026-09-02.
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path, attn_implementation="sdpa")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(torch.float32).to(device).eval()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail(f"model load failed: {brief_traceback()} — run download-best-models.sh")

    target = fc["sample_rate"]
    chunk_16k = int(round(fc["chunk_duration"] * SR))
    window_16k = int(round((fc["chunk_duration"] + fc["text_audio_delay"]) * SR))
    log(f"loaded on {device} in {time.time() - t0:.1f}s — chunk "
        f"{fc['chunk_duration']:.3f}s, lookahead {fc['text_audio_delay']:.3f}s, "
        f"partial cadence is the model's own chunk")
    emit("status", "READY")

    office = Lane("office", remote=False)
    remote = Lane("remote", remote=True)

    def step(lane: Lane, window16: np.ndarray) -> str:
        """Encode one window and advance that lane's cache by one chunk."""
        import scipy.signal as ss
        window = ss.resample_poly(window16, target, SR).astype(np.float32)
        audio = torch.from_numpy(window).to(next(model.parameters()).device)
        features = model.encode_speech(audio.unsqueeze(0))
        if lane.state is None:
            lane.state = model.init_streaming_state(processor.tokenizer,
                                                    context_info=None)
        text, lane.state = model.streaming_generate_step(
            audio_features=features,
            streaming_state=lane.state,
            tokenizer=processor.tokenizer,
            max_new_tokens=256,
            temperature=0.0)
        return text

    def drain(lane: Lane, flush: bool) -> bool:
        """Consume whole windows; on flush, zero-pad the tail. Returns True if
        anything was produced."""
        produced = False
        while True:
            have = lane.buffer.size
            if have >= window_16k:
                window = lane.buffer[:window_16k]
            elif flush and have > 0:
                window = np.zeros(window_16k, dtype=np.float32)
                window[:have] = lane.buffer
            else:
                return produced
            try:
                text = step(lane, window)
            except Exception:  # noqa: BLE001
                log(f"{lane.name}: step failed: {brief_traceback()}")
                lane.buffer = lane.buffer[chunk_16k:] if have >= window_16k \
                    else np.zeros(0, dtype=np.float32)
                return produced
            if text:
                lane.texts.append(text)
                produced = True
            # Drop the advance, KEEP the lookahead — it is the next chunk's
            # leading audio (upstream's rule; see the module docstring).
            lane.buffer = lane.buffer[chunk_16k:] if have >= window_16k \
                else np.zeros(0, dtype=np.float32)
            if flush and lane.buffer.size == 0:
                return produced

    def feed(lane: Lane, samples: np.ndarray) -> None:
        lane.buffer = np.concatenate([lane.buffer, samples])
        if lane.buffer.size > MAX_BUFFER:
            lane.buffer = lane.buffer[-MAX_BUFFER:]
        if drain(lane, flush=False):
            emit("partial", strip_speaker_labels("".join(lane.texts)),
                 remote=lane.remote)

    def flush(lane: Lane) -> None:
        drain(lane, flush=True)
        emit("final", strip_speaker_labels("".join(lane.texts)),
             remote=lane.remote)
        lane.reset()

    while True:
        header = read_exact(4)
        if not header:
            break
        (n,) = struct.unpack("<i", header)

        if n == -1:
            break
        if n == 0:
            flush(office)
            continue
        if n == -3:
            flush(remote)
            continue
        if n == -2:
            length = read_exact(4)
            if not length:
                break
            (m,) = struct.unpack("<i", length)
            raw = read_exact(m * 4)
            if not raw:
                break
            feed(remote, np.frombuffer(raw, dtype=np.float32))
            continue

        raw = read_exact(n * 4)
        if not raw:
            break
        feed(office, np.frombuffer(raw, dtype=np.float32))


if __name__ == "__main__":
    main()
