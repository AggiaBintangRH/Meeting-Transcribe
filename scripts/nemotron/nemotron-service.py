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

# This file lives at scripts/<service>/<file>.py (one folder per service,
# owner 2026-07-29), so the project root — which owns models/ — is THREE levels
# up, not two. Bundled, that root is Contents/Resources.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    """Timing/diagnostic line → logs/nemotron.log via stderr.

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

    def __init__(self, stream: str = None, partial_every: int = PARTIAL_EVERY) -> None:
        self.stream = stream  # None = office; "remote" tags the output lines
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_partial = 0
        # Per-lane, never a module global, so `--partial-ms` cannot leave office
        # and remote running to different cadences. Both lanes get the same
        # value at startup and nothing mutates it after.
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
        # Cadence stretches with buffer length so a full re-transcribe of a long
        # utterance does not fall behind realtime (see PARTIAL_DUTY).
        needed = max(self.partial_every, int(self.buffer.size * PARTIAL_DUTY))
        return self.samples_since_partial >= needed and self.buffer.size > SR // 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="auto")
    parser.add_argument("--chunk-ms", type=int, default=160)
    # THE CAPTION CADENCE — the same control, the same name and the same values
    # Parakeet has (owner, 2026-08-11: *"jangan dibedakan, soalnya sama tentang
    # waktu interval"*). It had never been exposed here; the cadence was the
    # `PARTIAL_EVERY` constant and nothing could change it.
    #
    # ONE DIFFERENCE FROM PARAKEET, and it is protective rather than cosmetic:
    # here the value is a FLOOR, not the exact cadence. `PARTIAL_DUTY` still
    # stretches it as the buffer grows (`max(partial_every, buffer * 0.15)`),
    # because a Nemotron partial costs 2.08 s on a 30 s buffer against
    # Parakeet's 0.235 s. Without that stretch a short interval would put the
    # lane past 100 % duty on a long utterance and it would never catch up. So
    # a short interval here buys responsiveness EARLY in an utterance, which is
    # where it is felt, and is overridden later, which is where it would hurt.
    parser.add_argument("--partial-ms", type=int,
                        default=int(PARTIAL_EVERY / SR * 1000))
    args = parser.parse_args()

    # Clamped, not trusted — same range as Parakeet's.
    partial_ms = max(250, min(args.partial_ms, 10_000))
    partial_every = int(partial_ms / 1000 * SR)
    if partial_ms != args.partial_ms:
        log(f"--partial-ms {args.partial_ms} out of range, clamped to {partial_ms}")

    att_context = ATT_CONTEXT.get(args.chunk_ms, [56, 3])
    language = None if args.language in ("", "auto") else args.language

    import traceback

    def brief_traceback() -> str:
        lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
        return " | ".join(lines[-3:])

    try:
        from mlx_audio.stt import load  # heavy import — keep inside main
        # mlx-audio 0.4.7's `_prepare_audio` wants an `mx.array`; a numpy buffer
        # raises TypeError and costs the chunk-size setting (see `transcribe`).
        # Imported HERE, beside its only consumer, and not at module scope —
        # mlx_audio pulls it in anyway, so this adds no load time.
        import mlx.core as mx
    except Exception:  # noqa: BLE001
        fail(f"mlx-audio import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    try:
        model = load(MODEL)
    except Exception:  # noqa: BLE001
        fail(f"Nemotron model load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    # WHICH BRANCH RAN — logged once, the first time each one is taken.
    #
    # This ladder degrades SILENTLY by design, and on mlx-audio 0.4.7 that
    # silence hides something the user can see in Settings: `model.generate()`
    # raises `TypeError` for a numpy array, so the first branch never runs, the
    # second raises too, and every call lands in the temp-WAV last resort —
    # which DROPS `att_context_size` entirely. The chunk-size picker therefore
    # does not currently change recognition at all.
    #
    # Deliberately NOT "fixed" here (2026-08-11): measured, the detour costs the
    # same (0.708/2.135/4.315 s at 10/30/60 s), and passing att_context_size
    # properly makes Nemotron SLOWER (5.46 s vs 2.12 s on a 30 s buffer) with
    # its text equivalence unmeasured. That is its own change and its own device
    # run. What is fixed is the EVIDENCE: the log now says which branch ran, so
    # the fact lives in logs/nemotron.log instead of only in a comment. The tab
    # says the same thing in words.
    branch_logged = set()

    def note_branch(which: str) -> None:
        if which not in branch_logged:
            branch_logged.add(which)
            log(f"transcribe branch: {which}")

    def transcribe(audio: np.ndarray) -> str:
        kwargs = {}
        if language:
            kwargs["language"] = language
        try:
            # `mx.array`, NOT the raw numpy buffer — FIXED 2026-08-11 (owner).
            #
            # This one conversion is the whole repair. mlx-audio 0.4.7 changed
            # `_prepare_audio` to call `audio.astype(mx.float32)`, which numpy
            # cannot interpret, so BOTH array branches raised `TypeError` and the
            # ladder fell through to the temp-WAV branch below — which takes a
            # PATH and therefore cannot carry `att_context_size`. The setting was
            # accepted, stored, shown in Settings, and silently discarded. It
            # broke when `download-best-models.sh` moved mlx-audio 0.4.5 → 0.4.7
            # unpinned on 2026-08-10; nothing failed loudly because the fallback
            # produced correct text at the same speed.
            #
            # Measured cost of restoring it, seconds per partial, best of 2, on
            # `recordings/Meeting5People.wav` — the numbers behind the duty
            # figures in `RealtimeModelTab.chunkHint`:
            #
            #   buffer   none     80    160    560   1120
            #      5 s   0.35   2.96   0.91   0.57   0.36
            #     10 s   0.68   5.98   1.82   1.13   0.67
            #     20 s   1.38  12.03   3.67   2.27   1.37
            #     30 s   2.08  18.04   5.49   3.42   2.08
            #
            # Two things that table settles. **1120 ms is exactly the old
            # behaviour** — same time, byte-identical text — so an install left
            # at the shipped default transcribes precisely as it did before this
            # fix, and the 160 ms default is the only value that changes anything
            # for an untouched user. **80 ms cannot be served here**: ~400 % duty
            # against its own cadence on ONE lane, so it falls permanently
            # behind. The tab states each option's duty rather than hiding the
            # choice — the owner sets settings, this file makes them true.
            text = model.generate(mx.array(audio),
                                  att_context_size=att_context, **kwargs).text
            note_branch(f"mx.array + att_context_size={att_context}")
            return text
        except TypeError:
            pass  # build without att_context_size / array support — degrade below
        try:
            text = model.generate(mx.array(audio), **kwargs).text
            note_branch("mx.array, NO att_context_size (chunk size has no effect)")
            return text
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
                text = model.generate(path, **kwargs).text
                note_branch("temp WAV path, NO att_context_size "
                            "(chunk size has no effect)")
                return text
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
    log(f"partial cadence floor {partial_ms} ms (stretched by PARTIAL_DUTY "
        f"on long buffers)")
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
