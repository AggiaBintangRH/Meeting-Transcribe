#!/usr/bin/env python3
"""
Nemotron 3.5 ASR streaming sidecar for MeetingTranscriber.

Loads mlx-community/nemotron-3.5-asr-streaming-0.6b (multilingual, 40 locales)
via mlx-audio and transcribes audio streamed from the app.

Dual-stream (Office + Remote): ONE process, TWO independent audio LANES.
- The weights are loaded once and shared; a second process would cost another
  ~1 GB resident for nothing. Sharing is safe because this sidecar never uses
  cache-aware streaming ACROSS calls: every partial/final re-transcribes that
  lane's whole buffer with a fresh `model.generate()`, and mlx-audio builds its
  encoder/decoder caches inside that call (`stream_encode_chunks` allocates
  `attn_cache`/`conv_cache` locally, `_decode_prompted_chunks` starts from
  `decoder_hidden = None`). Nothing survives a call, so nothing can leak.
- The AUDIO is never shared. Each lane owns its own buffer, its own
  `samples_since_partial` counter, its own FLUSH and its own MAX_BUFFER trim.
  Office and Remote samples are never mixed, concatenated or summed — that
  separation IS the point of dual-stream capture (multi-mic is the only thing
  that ever solved overlapping speech here; it works precisely because the two
  waveforms never meet).

Protocol:
  stdout: JSON lines {"type": "status"|"partial"|"final", "text": ...}
          first line after model load contains READY
          remote-lane messages additionally carry "stream":"remote"; the field
          is OMITTED for office messages, so a single-stream session's output
          is byte-identical to what it was before the remote lane existed.
  stdin:  length-prefixed frames:
          [int32 n]
            n  > 0 : n float32 mono 16 kHz samples follow (n*4 bytes) — OFFICE
            n == 0 : FLUSH office — transcribe current utterance as final, reset
            n == -1: exit
            n == -2: REMOTE audio — [int32 m] then m float32 samples follow
            n == -3: FLUSH remote
          The negative opcodes are purely ADDITIVE (same shape as the chunked
          sidecar's `-2` file-transcribe frame): an app that never sends them
          produces exactly the byte stream this sidecar has always received.

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


def _emit_raw(kind: str, text: str, stream: str = None) -> None:
    # `stream` is appended only when set, so office lines keep their exact
    # historical bytes (same keys, same order) — see the protocol note above.
    message = {"type": kind, "text": text}
    if stream:
        message["stream"] = stream
    try:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)  # app closed the pipe (quit/restart) — exit quietly


def log(message: str) -> None:
    """Timing/diagnostic line → logs/nemotron-asr.log via stderr.

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

MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
SR = 16_000
MAX_BUFFER = 60 * SR          # cap utterance buffer at 60 s
PARTIAL_EVERY = int(1.5 * SR)  # emit a partial every 1.5 s of new audio

# A partial transcribes the WHOLE buffer, so the live caption shows the full
# growing utterance — not just its tail. An earlier version capped the partial
# to the last 10 s to bound cost; that made the caption freeze mid-sentence and
# drop older words, which is worse than the cost it saved. (This model runs at
# ~13x realtime the way the sidecar uses it, so a full re-transcribe of a long
# buffer is not free — a 30 s buffer takes ~2.3 s.)
#
# Cost is bounded instead by SLOWING the cadence as the buffer grows: emit a
# partial once per `max(PARTIAL_EVERY, buffer * PARTIAL_DUTY)` of new audio. At
# a short buffer that is the responsive 1.5 s floor; at 30 s it stretches to
# ~4.5 s, keeping per-second compute roughly constant (~1/(13·DUTY) per lane)
# regardless of utterance length, while the caption still updates and still
# shows the whole sentence. Finals always use the whole buffer, unchanged.
PARTIAL_DUTY = 0.15

# A lane whose trailing window is this quiet emits no partial at all. Same
# threshold as the chunked path's remote silence gate.
#
# The chunked gate skips silent REMOTE chunks, but nothing guarded the realtime
# lanes: an idle conferencing channel was grinding ~0.76 s of compute every
# 1.5 s to transcribe nothing, indefinitely. That is not hypothetical — the
# owner's own log shows the remote channel at rms 0.00000 for 47 minutes.
# With both lanes busy the budget is ~100% duty, so reclaiming the idle one
# is what actually buys headroom in the common case (people take turns).
# FLUSH is deliberately NOT gated: a final over a quiet buffer is one cheap
# call, and silence there is the correct thing to confirm.
PARTIAL_SILENCE_RMS = 0.004

# Chunk-size setting → trained attention look-ahead [left, right]
ATT_CONTEXT = {80: [56, 0], 160: [56, 3], 560: [56, 6], 1120: [56, 13]}


def emit(kind: str, text: str, stream: str = None) -> None:
    _emit_raw(kind, text, stream)


class Lane:
    """One audio stream's recognizer state — and nothing else's.

    Everything that was module-level mutable state before dual-stream lives
    here, per lane: the utterance buffer, the partial cadence counter and the
    tag that goes on this lane's output. Two Lane instances share the model
    object (stateless across calls) and share nothing else. There is
    deliberately no path by which one lane's samples can reach another's
    `buffer` — mixing the streams would defeat the whole reason both are
    captured separately.
    """

    def __init__(self, stream: str = None) -> None:
        self.stream = stream  # None = office; "remote" tags the output lines
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_partial = 0

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_partial = 0

    def append(self, samples: np.ndarray) -> None:
        self.buffer = np.concatenate([self.buffer, samples])
        if self.buffer.size > MAX_BUFFER:
            self.buffer = self.buffer[-MAX_BUFFER:]
        self.samples_since_partial += samples.size

    def wants_partial(self) -> bool:
        # Cadence stretches with buffer length so a full re-transcribe of a long
        # utterance does not fall behind realtime (see PARTIAL_DUTY).
        needed = max(PARTIAL_EVERY, int(self.buffer.size * PARTIAL_DUTY))
        return self.samples_since_partial >= needed and self.buffer.size > SR // 2


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

    # Timing instrumentation. The loop is single-threaded, so with two lanes
    # active they take turns: while one lane is inside generate(), the other's
    # frames sit in the pipe. `wait` below is the gap since the PREVIOUS
    # generate() finished, which is what makes that turn-taking visible —
    # a large wait on the remote lane means it queued behind office.
    #
    # This measures; it does not fix. Two processes would not help either,
    # because MLX work serializes on the one GPU regardless. If these numbers
    # turn out bad, the lever is that a partial re-transcribes the WHOLE
    # buffer every 1.5 s, so cost grows quadratically across an utterance.
    import time as _time
    last_generate_end = [_time.time()]

    def timed_transcribe(lane: "Lane", kind: str) -> str:
        # Both partial and final transcribe the whole buffer, so the live caption
        # shows the full utterance; cadence (wants_partial) is what bounds cost.
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
        # Don't spend a generate() on a silent lane. The counter is reset
        # above either way, so a quiet lane retries on the next cadence tick
        # rather than firing the moment it is asked again. Silence is judged on
        # the recent tail (last ~4 s), not the whole buffer, so a lane that has
        # gone quiet stops emitting even while its buffer still holds old speech.
        tail = lane.buffer[-4 * SR:]
        rms = float(np.sqrt(np.mean(tail * tail))) if tail.size else 0.0
        if rms < PARTIAL_SILENCE_RMS:
            return
        emit("partial", timed_transcribe(lane, "partial"), lane.stream)

    emit("status", "READY")

    # The remote lane exists from the start but stays empty — it costs a
    # zero-length array — so a single-stream session never allocates, never
    # transcribes and never emits anything for it.
    office = Lane()
    remote = Lane("remote")
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
