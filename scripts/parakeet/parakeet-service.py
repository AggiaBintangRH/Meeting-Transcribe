#!/usr/bin/env python3
"""
Parakeet TDT 0.6b v3 realtime ASR sidecar for MeetingTranscriber.

The SECOND realtime engine (2026-08-11), beside scripts/nemotron/. Loads
mlx-community/parakeet-tdt-0.6b-v3 (CC BY-4.0, 2.3 GB, 25 languages) via
mlx-audio and transcribes audio streamed from the app. It speaks EXACTLY the
frame protocol nemotron-service.py speaks — one Swift client
(`RealtimeASRService`) drives both, the same way one `ChunkedASRService` drives
five ASR sidecars, so the wire is a contract rather than a coincidence
(`realtime/parakeet-protocol-matches-nemotron` in scripts/sidecar-tests.py
fails if the two ever drift).

Dual-stream (Office + Remote): ONE process, TWO independent audio LANES — same
design and same reasons as the Nemotron sidecar. The weights load once and are
shared; the AUDIO is never shared. Each lane owns its buffer, its partial
counter, its FLUSH and its MAX_BUFFER trim. Office and Remote samples are never
mixed, concatenated or summed — that separation IS the point of dual-stream
capture. Sharing the model is safe because nothing survives a call: every
partial/final re-transcribes that lane's whole buffer with a fresh
`model.generate()`.

Protocol:
  stdout: JSON lines {"type": "status"|"partial"|"final", "text": ...}
          first line after model load contains READY
          remote-lane messages additionally carry "stream":"remote"; the field
          is OMITTED for office messages, so a single-stream session's output
          is byte-identical to the Nemotron sidecar's.
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes) — OFFICE
            n == 0 : FLUSH office — transcribe current utterance as final, reset
            n == -1: exit
            n == -2: REMOTE audio — [int32 m] then m float32 samples follow
            n == -3: FLUSH remote

Config via argv: --language <auto|en|de|...>
  ⚠ ACCEPTED AND LOGGED, NEVER HONOURED — see `main()`. There is deliberately
  no --chunk-ms: this model has no attention-context setting to point one at.

MEASURED ON THIS M4 (2026-08-11), and recorded here so the numbers behind the
design decisions below are not re-derived from scratch:

  buffer   parakeet            nemotron
   10 s    0.083 s (121x RT)   0.708 s (14.1x)
   30 s    0.235 s (128x)      2.135 s (14.1x)
   60 s    0.462 s (130x)      4.315 s (13.9x)

  Load 3.5–4.0 s. Two ACTIVE lanes at the flat 1.5 s cadence: ~31 % duty
  (Nemotron, same shape: 285 %).

THREE decisions that follow from those numbers, each of which a future audit
would otherwise be tempted to "fix":

1. NO `PARTIAL_DUTY` CADENCE STRETCHING, unlike the Nemotron sidecar. That
   stretch exists there solely because a 30 s partial costs Nemotron 2.1 s, so
   a flat 1.5 s cadence fell behind realtime. Here the same partial costs
   0.235 s and the 31 % two-lane duty above was measured AT the flat cadence.
   Copying the stretch across would slow the caption down to solve a problem
   this engine does not have.

2. NO TOKEN MERGING. mlx-audio ships `merge_longest_contiguous` /
   `merge_longest_common_subsequence` for stitching incremental hypotheses, and
   they must NOT be used here. At 130x RT a whole-buffer re-transcribe is ~8 %
   of one lane's budget, so incremental decoding buys nothing measurable — and
   it imports a bug class this project has already paid for twice (the
   `TranscriptMerge` COMBINE duplication, 2026-07-28; DiCoW's overlapping
   recovered spans, 2026-07-14). Recorded as the LEVER, exactly like Nemotron's
   `PARTIAL_WINDOW` note: if duty ever tips (a third stream, a faster cadence,
   heavier hardware), this is the thing to reach for — with a measurement
   first.

3. NO HALLUCINATION GATE, AND NONE MUST BE ADDED. Measured before deciding:
   30 s of digital silence, 30 s of tiny noise, and a 3 s flush-sized silent
   buffer ALL returned the empty string — 0 characters, no canned caption, no
   invented sentence. Whisper's and Granite's gates exist because those models
   really do emit "Thank you." over silence; this one does not. A gate guarding
   a failure that does not occur is pure risk in the direction this project
   treats as the dangerous one: over-deletion leaves no trace in the transcript,
   only in the log. If a hallucination is ever OBSERVED here, measure it and
   write the gate against what was seen — do not port another model's.

`stream_generate()` is deliberately unused: it computes `total_samples` up
front and therefore cannot be fed a live microphone at all.

Fully offline: HF_HOME points at the project models/ folder.
"""
import argparse
import json
import os
import struct
import sys

# This file lives at scripts/<service>/<file>.py (one folder per service,
# owner 2026-07-29), so the project root — which owns models/ — is THREE levels
# up, not two. Bundled, that root is Contents/Resources. Getting this wrong is
# SILENT until the first model load: HF_HOME would resolve to scripts/models.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _emit_raw(kind: str, text: str, stream: str = None) -> None:
    # `stream` is appended only when set, so office lines keep the exact bytes
    # the Nemotron sidecar writes — same keys, same order. The Swift decoder is
    # shared, and so is the "absent means office" convention.
    message = {"type": kind, "text": text}
    if stream:
        message["stream"] = stream
    try:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


def log(message: str) -> None:
    """Timing/diagnostic line → logs/parakeet.log via stderr.

    Kept off stdout deliberately: stdout is the app's JSON-lines protocol and
    an extra line there would be a parse error, not a log entry.
    """
    import time
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def fail(message: str) -> None:
    """Report a startup error to the app, then exit."""
    _emit_raw("error", message)
    sys.exit(1)


try:
    import numpy as np
except Exception as exc:  # noqa: BLE001
    fail(f"numpy import failed: {exc}. Run download-best-models.sh")

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
SR = 16_000
MAX_BUFFER = 60 * SR          # cap utterance buffer at 60 s
PARTIAL_EVERY = int(1.5 * SR)  # emit a partial every 1.5 s of new audio

# A lane whose trailing window is this quiet emits no partial at all. Same
# threshold and same reasoning as the Nemotron sidecar, and the saving is
# ENGINE-INDEPENDENT: people take turns, so one lane is usually idle, and an
# idle lane transcribing nothing forever is waste no matter how fast the model
# is. FLUSH is deliberately NOT gated: a final over a quiet buffer is one cheap
# call, and silence there is the correct thing to confirm.
PARTIAL_SILENCE_RMS = 0.004


def emit(kind: str, text: str, stream: str = None) -> None:
    _emit_raw(kind, text, stream)


class Lane:
    """One audio stream's recognizer state — and nothing else's.

    Two Lane instances share the model object (stateless across calls) and
    share nothing else. There is deliberately no path by which one lane's
    samples can reach another's `buffer`.
    """

    def __init__(self, stream: str = None, partial_every: int = PARTIAL_EVERY) -> None:
        self.stream = stream  # None = office; "remote" tags the output lines
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_partial = 0
        # Per-lane rather than a module global, so `--partial-ms` cannot be
        # rebound halfway through a session and leave the two lanes running to
        # different cadences. Both lanes are built with the same value at
        # startup; nothing mutates it afterwards.
        self.partial_every = partial_every

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_partial = 0

    def append(self, samples: np.ndarray) -> None:
        self.buffer = np.concatenate([self.buffer, samples])
        if self.buffer.size > MAX_BUFFER:
            self.buffer = self.buffer[-MAX_BUFFER:]
        self.samples_since_partial += samples.size

    def wants_partial(self) -> bool:
        # FLAT cadence — no buffer-length stretch. See decision 1 in the module
        # docstring: at 130x RT the whole-buffer partial is cheap enough that
        # stretching would only make the caption slower.
        return (self.samples_since_partial >= self.partial_every
                and self.buffer.size > SR // 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="auto")
    # THE CAPTION CADENCE — how much NEW audio a lane must collect before it
    # spends a partial. Parakeet-only, and it exists because this engine is the
    # first one fast enough for the choice to be real: measured 0.235 s for a
    # 30 s buffer, so TWO active lanes cost ~0.47 s per tick and the duty is
    # simply 0.47 / cadence — ~31 % at the 1500 ms default, ~63 % at 750 ms.
    #
    # Nemotron deliberately has no equivalent flag. Its partial costs 2.135 s on
    # the same buffer, so its cadence is not a free choice; it is governed by
    # `PARTIAL_DUTY`, the stretch this sidecar does not have.
    #
    # The DEFAULT is 1500 ms = `PARTIAL_EVERY`, i.e. exactly the behaviour that
    # shipped before this flag existed — the house rule that a new setting must
    # change nothing until it is deliberately moved.
    parser.add_argument("--partial-ms", type=int, default=int(PARTIAL_EVERY / SR * 1000))
    args = parser.parse_args()

    # Clamped, not trusted. A stale or hand-edited value must not be able to put
    # a lane into a state the model cannot sustain: below ~250 ms two lanes
    # would exceed 100 % duty on a long buffer and fall permanently behind, and
    # above MAX_BUFFER a partial would never fire at all.
    partial_ms = max(250, min(args.partial_ms, 10_000))
    partial_every = int(partial_ms / 1000 * SR)
    if partial_ms != args.partial_ms:
        log(f"--partial-ms {args.partial_ms} out of range, clamped to {partial_ms}")

    # THE LANGUAGE FLAG IS INERT, AND THAT IS MEASURED, NOT ASSUMED.
    # `ParakeetTDT.generate()` has no `language` parameter — only `**kwargs`,
    # the same swallow-without-raising shape as Granite's and Voxtral's. Driven
    # directly with None, "fr", "de" and a nonsense "xx", it returned
    # byte-identical 264-character output every time and raised nothing.
    #
    # It is still ACCEPTED and LOGGED rather than refused at the Swift boundary:
    # the choice then travels the whole path and dies where the truth is, and
    # this log line is the record of what the user actually asked for. The
    # Settings tab says plainly that the setting will not change the transcript.
    # Do NOT "fix" this to honour the flag — there is nothing here to honour.
    language = None if args.language in ("", "auto") else args.language

    import traceback

    def brief_traceback() -> str:
        lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
        return " | ".join(lines[-3:])

    try:
        import mlx.core as mx
        from mlx_audio.stt import load  # heavy import — keep inside main
    except Exception:  # noqa: BLE001
        fail(f"mlx-audio import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    try:
        model = load(MODEL)
    except Exception:  # noqa: BLE001
        fail(f"Parakeet model load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    log(f"loaded {MODEL} · language={args.language} "
        f"(accepted and logged, NOT honoured — this model has no language "
        f"parameter; measured byte-identical output for every code)")

    def transcribe(audio: np.ndarray) -> str:
        # `mx.array` is REQUIRED, not a style choice: handed a numpy array,
        # generate() raises
        # `TypeError: Cannot interpret 'mlx.core.float32' as a data type`.
        # Measured 2026-08-11. Passing a temp WAV path instead would work and
        # is deliberately not done — a detour through the filesystem per
        # partial, for a call that already has the samples in memory.
        return model.generate(mx.array(audio)).text

    # Timing instrumentation, same shape as the Nemotron sidecar's so one eye
    # can read either log. The loop is single-threaded, so with two lanes active
    # they take turns: `wait` is the gap since the PREVIOUS generate() finished,
    # which is what makes that turn-taking visible. Two processes would not
    # help — MLX work serializes on the one GPU regardless.
    import time as _time
    last_generate_end = [_time.time()]

    def timed_transcribe(lane: "Lane", kind: str) -> str:
        # Both partial and final transcribe the whole buffer, so the live
        # caption shows the full utterance; cadence bounds the cost.
        secs = lane.buffer.size / SR
        started = _time.time()
        wait = started - last_generate_end[0]
        text = transcribe(lane.buffer).strip()
        took = _time.time() - started
        last_generate_end[0] = _time.time()
        log(f"{lane.stream or 'office'} {kind} buf={secs:.1f}s took={took:.3f}s "
            f"({secs / max(took, 1e-6):.0f}x) wait={wait:.3f}s")
        return text

    def flush_lane(lane: "Lane") -> None:
        """FLUSH one lane: finalize its own utterance, reset its own state."""
        if lane.buffer.size > SR // 4:  # ignore < 250 ms blips
            emit("final", timed_transcribe(lane, "final"), lane.stream)
        lane.reset()

    def feed_lane(lane: "Lane", samples: np.ndarray) -> None:
        """Append to ONE lane and emit that lane's partial when it is due."""
        lane.append(samples)
        if not lane.wants_partial():
            return
        lane.samples_since_partial = 0
        # Don't spend a generate() on a silent lane. The counter is reset above
        # either way, so a quiet lane retries on the next cadence tick rather
        # than firing the moment it is asked again. Silence is judged on the
        # recent tail (last ~4 s), not the whole buffer, so a lane that has gone
        # quiet stops emitting even while its buffer still holds old speech.
        tail = lane.buffer[-4 * SR:]
        rms = float(np.sqrt(np.mean(tail * tail))) if tail.size else 0.0
        if rms < PARTIAL_SILENCE_RMS:
            return
        emit("partial", timed_transcribe(lane, "partial"), lane.stream)

    emit("status", "READY")

    # The remote lane exists from the start but stays empty — it costs a
    # zero-length array — so a single-stream session never allocates, never
    # transcribes and never emits anything for it.
    # BOTH lanes take the same cadence — one setting, one value, so office and
    # remote captions can never update at different rates.
    log(f"partial cadence {partial_ms} ms "
        f"(~{2 * 0.235 / (partial_ms / 1000) * 100:.0f}% duty with two active lanes "
        f"at a 30 s buffer)")
    office = Lane(partial_every=partial_every)
    remote = Lane("remote", partial_every=partial_every)
    stdin = sys.stdin.buffer

    def read_samples(count: int):
        """Read `count` float32 samples, or None if stdin ended mid-frame."""
        data = stdin.read(count * 4)
        if not data or len(data) < count * 4:
            return None
        return np.frombuffer(data, dtype=np.float32)

    while True:
        header = stdin.read(4)
        if not header or len(header) < 4:
            break
        (n,) = struct.unpack("<i", header)

        if n == -1:
            break
        if n == -3:  # FLUSH remote
            flush_lane(remote)
            continue
        if n == -2:  # REMOTE audio — its own length prefix follows
            hdr2 = stdin.read(4)
            if not hdr2 or len(hdr2) < 4:
                break
            (m,) = struct.unpack("<i", hdr2)
            if m <= 0:
                continue
            samples = read_samples(m)
            if samples is None:
                break
            feed_lane(remote, samples)
            continue
        if n == 0:  # FLUSH office
            flush_lane(office)
            continue

        samples = read_samples(n)
        if samples is None:
            break
        feed_lane(office, samples)


if __name__ == "__main__":
    main()
