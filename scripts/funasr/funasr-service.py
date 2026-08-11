#!/usr/bin/env python3
"""
Fun-ASR MLT Nano 2512 realtime ASR sidecar for MeetingTranscriber.

The THIRD realtime engine (2026-08-11), beside scripts/nemotron/ and
scripts/parakeet/. Loads mlx-community/Fun-ASR-MLT-Nano-2512-fp16 (Apache 2.0,
1.97 GB, 1.0B params) via mlx-audio and transcribes audio streamed from the app.
It speaks EXACTLY the frame protocol the other two speak — one Swift client
(`RealtimeASRService`) drives all three, so the wire is a contract rather than a
coincidence (`realtime/funasr-protocol-matches-nemotron` in
scripts/sidecar-tests.py fails if they ever drift).

Architecture: SenseVoice-style SANM audio encoder → adaptor → Qwen3 0.6B text
decoder. It is therefore an autoregressive LLM-decoder ASR, like MOSS and unlike
Parakeet's TDT — which is why its cost tracks TOKENS GENERATED rather than audio
length, and why the two behaviours below (silence hallucination, a real language
parameter) look like MOSS's rather than Parakeet's.

Dual-stream (Office + Remote): ONE process, TWO independent audio LANES — same
design and same reasons as the other two sidecars. The weights load once and are
shared; the AUDIO is never shared. Each lane owns its buffer, its partial
counter, its FLUSH and its MAX_BUFFER trim. Nothing survives a call: every
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

Config via argv: --language <auto|en|zh|ja|...>, --partial-ms <250..10000>
  ⚠ UNLIKE Parakeet's, this --language is REAL and is HONOURED. See
  `resolve_language()` — and note it can RAISE, which is why it is validated
  here rather than trusted.

MEASURED ON THIS M4 (2026-08-11), best of 2 with the warm-up call discarded,
on recordings/Meeting5People.wav:

  buffer   fun-asr             parakeet            nemotron
   10 s    0.151 s ( 66x RT)   0.083 s (121x RT)   0.708 s (14.1x)
   30 s    0.609 s ( 49x RT)   0.235 s (128x)      2.135 s (14.1x)
   60 s    0.621 s ( 97x RT)   0.462 s (130x)      4.315 s (13.9x)

  Load ~0.7 s — by far the fastest of the three, because the weights are
  mmapped and there is no ONNX/graph build step.

  The 60 s row is NOT a typo and NOT a mistake: it costs the same as the 30 s
  row. Cost here tracks TOKENS GENERATED, not seconds of audio, and that 60 s
  window simply did not contain proportionally more speech. Judge this engine's
  budget by its worst row (30 s), never by its best.

FOUR decisions that follow from those numbers and behaviours, each of which a
future audit would otherwise be tempted to "fix":

1. `PARTIAL_DUTY` CADENCE STRETCHING IS KEPT, unlike the Parakeet sidecar.
   Two ACTIVE lanes at a flat 1.5 s cadence and a 30 s buffer measure
   2 x 0.609 / 1.5 = **81 % duty** — under 100 %, and far too thin to ship,
   the same "works with no margin" state the dual-stream notes flag for
   Nemotron. At DUTY 0.08 the cadence stretches to 2.4 s at a 30 s buffer
   (51 %) and 4.8 s at 60 s (26 %), which bounds the cost instead of letting
   it grow with the utterance. Parakeet needs none of this at 0.235 s; this
   engine is 2.6x more expensive and does.

2. HALLUCINATION GATES ARE REQUIRED HERE, AND THEY ARE MEASURED — the exact
   opposite of the Parakeet sidecar, whose docstring says none must be added.
   Driven on this M4 before the gates were written:

     30 s digital silence  -> '<gbg>'
     30 s of tiny noise    -> "I'm not sure if I can do that."
     3 s digital silence   -> 'Okay.'

   So it emits a FunASR garbage tag, a chatbot refusal, and a plausible-looking
   canned caption, over audio containing nothing. All three are guarded below.
   Do not port these gates to Parakeet, and do not delete them here: each one
   is written against a string that was actually observed.

3. THE LANGUAGE FLAG IS HONOURED, AND IT CAN RAISE. `_map_language()` inside
   mlx-audio rejects any ISO code outside its own table with a `ValueError`, so
   an unvalidated code does not degrade — it kills the call, and with it that
   lane's caption for the rest of the meeting. Validated at startup, once, and
   never passed through unchecked. See `resolve_language()`.

4. NO TOKEN MERGING, same as Parakeet and for the same reason: incremental
   hypothesis stitching imports a bug class this project has paid for twice
   (`TranscriptMerge` COMBINE duplication, DiCoW's overlapping spans). Recorded
   as the LEVER if duty ever tips, with a measurement first.

THREE LOAD SHIMS, ALL REQUIRED, ALL MEASURED — see `load_funasr()`. The
checkpoint was converted with `mlx-audio-plus` (a fork), and mainline mlx-audio
0.4.7 cannot load it correctly without them. The failure mode is SILENT: with
mlx-audio's own default (`strict=False`) the model loads, runs, and emits
`!!!!!!!!` forever.

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
    """Timing/diagnostic line → logs/funasr.log via stderr.

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

MODEL = "mlx-community/Fun-ASR-MLT-Nano-2512-fp16"
SR = 16_000
MAX_BUFFER = 60 * SR          # cap utterance buffer at 60 s
PARTIAL_EVERY = int(1.5 * SR)  # FLOOR for a partial: 1.5 s of new audio

# Cadence stretch — see decision 1. Derived from the measurement, not copied:
# a 30 s partial costs 0.609 s, and holding two ACTIVE lanes near 50 % duty
# needs a ~2.4 s cadence there, i.e. 2.4 / 30 = 0.08. Nemotron uses 0.15
# because its partial costs 2.135 s; the constants differ because the models do.
PARTIAL_DUTY = 0.08

# A lane whose trailing window is this quiet emits no partial at all. Same
# threshold and same reasoning as the other two realtime sidecars.
#
# HERE IT DOES A SECOND JOB, and that is why FLUSH is gated too (below) while
# Parakeet's is not: this model does not return "" over silence, it INVENTS
# text. So the audio-level gate is the primary hallucination defence — it
# judges the SIGNAL, where "there are no words here" is a fact, rather than the
# TEXT, where it is a guess.
PARTIAL_SILENCE_RMS = 0.004

# The ISO codes mlx-audio's `_map_language()` will accept. Anything else RAISES
# (decision 3). This is deliberately the roster of the IMPLEMENTATION, not the
# roster of the model card: FunAudioLLM advertises 30 languages for this
# checkpoint, and mainline mlx-audio can express eleven of them. The Granite
# lesson, one turn further on — the card is not the authority, the code that
# runs is.
SUPPORTED_LANGUAGES = frozenset(
    {"cjy", "cmn", "en", "gan", "hak", "hsn", "ja", "nan", "wuu", "yue", "zh"}
)

# FunASR's own "this segment is garbage" tag. It is not English, not a caption
# and never something a person said, so it is dropped outright rather than
# being subjected to the density test the canned captions get.
GARBAGE_TAGS = ("<gbg>", "<unk>", "<sil>")

# MEASURED refusals — the model answering as the Qwen assistant underneath it
# instead of transcribing. Kept DELIBERATELY NARROW and anchored, exactly like
# the MOSS refusal gate: a human says "I'm sorry"; no human being recorded in a
# meeting says "I'm not sure if I can do that." as their entire utterance. The
# generic CANNED_HALLUCINATIONS set from the mlx-audio chunked services is NOT
# reused here — over-deletion is the direction that leaves no trace.
REFUSAL_PREFIXES = (
    "i'm not sure if i can do that",
    "i'm sorry, i can't assist",
    "i cannot assist with that",
    "as an ai language model",
    "i'm a qwen",
)

# Canned captions this model produced over SILENCE. Unlike the refusals these
# are things a person really might say, so they are dropped ONLY when the audio
# was also implausibly sparse — the two-condition rule the Granite gate uses.
# `Okay.` after real speech survives; `Okay.` over 3 s of digital silence does
# not.
CANNED_OVER_SILENCE = ("okay.", "ok.", "thank you.", "thanks.", "yeah.", "bye.")
CANNED_MIN_WORDS_PER_SEC = 0.5


def emit(kind: str, text: str, stream: str = None) -> None:
    _emit_raw(kind, text, stream)


def rms(samples: "np.ndarray") -> float:
    """Root-mean-square of a buffer, 0.0 when empty."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def drop_reason(text: str, seconds: float, level: float) -> str:
    """Why this text must NOT be shown, or "" to keep it.

    Every drop is logged with its reason by the caller. That log is the only
    trace a deleted sentence leaves, and it is what identified both directions
    of the Whisper gate bug — so a gate without a logged reason is not
    acceptable in this codebase.
    """
    stripped = text.strip()
    if not stripped:
        return ""  # nothing to show, nothing to explain

    bare = stripped
    for tag in GARBAGE_TAGS:
        bare = bare.replace(tag, "")
    if not bare.strip():
        return f"garbage tag only ({stripped!r})"

    lowered = bare.strip().lower()
    for prefix in REFUSAL_PREFIXES:
        if lowered.startswith(prefix):
            return f"model refusal ({stripped[:60]!r})"

    if lowered in CANNED_OVER_SILENCE:
        words = len(bare.split())
        density = words / seconds if seconds > 0 else 0.0
        if density < CANNED_MIN_WORDS_PER_SEC:
            return (f"canned caption over sparse audio ({stripped!r}, "
                    f"{density:.2f} w/s, rms {level:.5f})")
    return ""


def clean(text: str) -> str:
    """Strip FunASR's special tags from text that survived the gates."""
    out = text
    for tag in GARBAGE_TAGS:
        out = out.replace(tag, "")
    return out.strip()


def resolve_language(requested: str) -> str:
    """The code to hand `generate()`, or None for auto-detect.

    DEFENSIVE ON PURPOSE (decision 3). The Swift side already clamps unsupported
    codes to "auto" at the read boundary, so in a correct build nothing invalid
    arrives — but `_map_language()` raises rather than ignores, and a raise here
    is not a degraded caption, it is NO caption for the rest of the meeting.
    Two independent guards for a failure that is total and silent is the same
    trade the language pickers already make elsewhere in this project.
    """
    if requested in ("", "auto", None):
        return None
    normalized = requested.lower().replace("_", "-").split("-")[0]
    if normalized not in SUPPORTED_LANGUAGES:
        log(f"--language {requested!r} is not one of "
            f"{sorted(SUPPORTED_LANGUAGES)} — falling back to auto-detect "
            f"(passing it would raise inside mlx-audio and kill the lane)")
        return None
    return normalized


def load_funasr():
    """Build the model, apply the three load shims, return it ready to call.

    ALL THREE ARE REQUIRED AND ALL THREE WERE MEASURED. The checkpoint was
    produced by `mlx-audio-plus` (a fork) and mainline mlx-audio 0.4.7 disagrees
    with it in three places. None of them raises on its own — with mlx-audio's
    default `strict=False` the model loads happily and then emits `!!!!!!!!`
    forever, which is how long this took to find.

    SHIM 1 — CONFIG KEY NAMES. The checkpoint's config.json names its
    sub-configs `encoder`, `adaptor`, `llm`; `FunASRNanoConfig.from_dict()`
    understands only `audio_encoder_conf`, `audio_adaptor_conf` and
    `text_config`/`llm_config`. So EVERY sub-config silently fell back to its
    dataclass defaults — hidden_size 2048 against the checkpoint's 1024 — and
    `strict=False` then skipped every mismatched tensor without a word. With
    the rename, shape mismatches go to zero and `strict=True` passes.

    SHIM 2 — THE UNTIED LM HEAD. `Qwen3CausalLM` has no `lm_head` and always
    computes logits as `embed_tokens.as_linear(h)`, i.e. it assumes tied
    embeddings. This checkpoint sets `tie_word_embeddings: false` and ships a
    real `llm.lm_head.weight`, which mlx-audio would leave unused. Logits from
    the wrong matrix are not slightly wrong, they are garbage.

    SHIM 3 — FLOAT32. The repo is named `-fp16` and the weights ARE fp16, but
    running the forward pass in fp16 produces non-finite logits: measured,
    `mx.all(mx.isfinite(logits))` is False and every argmax lands on token 0
    (`!`). Upcasting the weights to float32 at load fixes it completely — the
    same "verified-identical output, keep float32" position the MOSS notes
    reach, arrived at from the opposite direction. Cost is RAM, not speed.

    (A fourth, benign difference: the tokenizer sits at the repo root while
    `qwen_tokenizer_path` defaults to a `Qwen3-0.6B` subfolder that this repo
    does not have. Handled below by looking for the subfolder and falling back
    to the root, so a future repo laid out either way still loads.)
    """
    import glob
    import json as _json

    import mlx.core as mx
    import mlx.nn as nn
    import mlx_audio.stt.models.fun_asr_nano.fun_asr_nano as F
    from mlx_audio.stt.models.fun_asr_nano import Model, ModelConfig
    from transformers import AutoTokenizer

    # SHIM 2 — give the LM the head the checkpoint actually carries, but ONLY
    # when the config says the embeddings are untied. A checkpoint that really
    # is tied keeps mlx-audio's own path, so this cannot break other repos.
    original_init = F.Qwen3CausalLM.__init__

    def patched_init(self, config):
        original_init(self, config)
        if not getattr(config, "tie_word_embeddings", True):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                     bias=False)

    F.Qwen3CausalLM.__init__ = patched_init

    def patched_call(self, input_ids, input_embeddings=None, cache=None):
        if input_embeddings is None:
            input_embeddings = self.llm.model.embed_tokens(input_ids)
        hidden = self.llm.model(inputs_embeds=input_embeddings, cache=cache)
        if hasattr(self.llm, "lm_head"):
            return self.llm.lm_head(hidden)
        return self.llm.model.embed_tokens.as_linear(hidden)

    F.FunASRNano.__call__ = patched_call

    # Resolve the snapshot on disk. HF_HUB_OFFLINE is already set, so this must
    # not reach the network: the folder is found by glob, never downloaded.
    pattern = os.path.join(
        os.environ["HF_HOME"], "hub",
        "models--" + MODEL.replace("/", "--"), "snapshots", "*")
    snapshots = sorted(glob.glob(pattern))
    if not snapshots:
        raise FileNotFoundError(
            f"{MODEL} is not in {os.environ['HF_HOME']}/hub — "
            "run download-best-models.sh")
    snapshot = snapshots[-1]

    # SHIM 1 — translate the fork's config key names into the ones mainline
    # mlx-audio reads. Copied, never mutated in place: the snapshot on disk is
    # left exactly as downloaded so its hash still matches the hub.
    raw = _json.load(open(os.path.join(snapshot, "config.json")))
    cfg = dict(raw)
    cfg["audio_encoder_conf"] = raw["encoder"]
    cfg["audio_adaptor_conf"] = raw["adaptor"]
    cfg["text_config"] = raw["llm"]
    cfg["frontend_conf"] = {"fs": raw["sample_rate"], "n_mels": raw["n_mels"],
                            "lfr_m": raw["lfr_m"], "lfr_n": raw["lfr_n"]}

    model = Model(ModelConfig.from_dict(cfg))
    weights = model.sanitize(mx.load(os.path.join(snapshot, "model.safetensors")))
    # SHIM 3 — float32, and `strict=True`. Strictness is the point: it is what
    # converts "silently emits !!!!" into a startup error the app can show.
    model.load_weights([(k, v.astype(mx.float32)) for k, v in weights.items()],
                       strict=True)
    mx.eval(model.parameters())

    tokenizer_dir = os.path.join(snapshot, model.config.qwen_tokenizer_path)
    if not os.path.isdir(tokenizer_dir):
        tokenizer_dir = snapshot

    # `fix_mistral_regex=True` — transformers warns without it that this
    # tokenizer's regex is wrong and "will lead to incorrect tokenization". It
    # is upstream's own flag (the MOSS sidecars pass it for the same reason),
    # not an invention here.
    #
    # MEASURED before adopting, because a warning is not evidence: with and
    # without it, a 30 s English chunk decoded BYTE-IDENTICALLY. So it is set
    # for the languages nobody here has measured, not for English, and the
    # honest claim is "inert where we tested, and free".
    #
    # The fallback is not defensive habit: `download-best-models.sh` upgrades
    # transformers UNPINNED on every run (it moved 5.12.1 → 5.14.1 on
    # 2026-08-10), so a kwarg that disappears upstream would otherwise turn a
    # silenced warning into a sidecar that cannot start at all.
    try:
        model._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        log("this transformers has no fix_mistral_regex — loading without it")
        model._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True)
    return model


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
        # Cadence stretches with buffer length so a full re-transcribe of a long
        # utterance does not fall behind realtime (see PARTIAL_DUTY, decision 1).
        needed = max(self.partial_every, int(self.buffer.size * PARTIAL_DUTY))
        return self.samples_since_partial >= needed and self.buffer.size > SR // 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="auto")
    # THE CAPTION CADENCE — how much NEW audio a lane must collect before it
    # spends a partial. As on the Nemotron sidecar this value is a FLOOR, not
    # the exact cadence: `PARTIAL_DUTY` stretches it as the buffer grows, so a
    # short interval buys responsiveness EARLY in an utterance, which is where
    # it is felt, and is overridden later, which is where it would hurt.
    #
    # The DEFAULT is 1500 ms = `PARTIAL_EVERY`, i.e. exactly the behaviour the
    # other two engines ship with — the house rule that a new setting must
    # change nothing until it is deliberately moved.
    parser.add_argument("--partial-ms", type=int,
                        default=int(PARTIAL_EVERY / SR * 1000))
    args = parser.parse_args()

    # Clamped, not trusted — same bounds as the other two sidecars.
    partial_ms = max(250, min(args.partial_ms, 10_000))
    partial_every = int(partial_ms / 1000 * SR)
    if partial_ms != args.partial_ms:
        log(f"--partial-ms {args.partial_ms} out of range, clamped to {partial_ms}")

    language = resolve_language(args.language)

    import traceback

    def brief_traceback() -> str:
        lines = [l for l in traceback.format_exc().splitlines() if l.strip()]
        return " | ".join(lines[-3:])

    try:
        import mlx.core as mx  # noqa: F401  (kept for symmetry / transcribe)
    except Exception:  # noqa: BLE001
        fail(f"mlx import failed: {brief_traceback()} — "
             "run download-best-models.sh")

    try:
        model = load_funasr()
    except Exception:  # noqa: BLE001
        fail(f"Fun-ASR model load failed: {brief_traceback()} — "
             "run download-best-models.sh")

    log(f"loaded {MODEL} · language={args.language} -> "
        f"{language or 'auto-detect'} (HONOURED — unlike the Parakeet sidecar, "
        f"this model has a real language parameter)")

    import mlx.core as mx

    def transcribe(audio: np.ndarray) -> str:
        # `mx.array` is REQUIRED, not a style choice: handed a numpy array,
        # mlx-audio raises
        # `TypeError: Cannot interpret 'mlx.core.float32' as a data type`.
        return model.generate(mx.array(audio), language=language).text

    # Timing instrumentation, same shape as the other two sidecars' so one eye
    # can read any of the three logs. The loop is single-threaded, so with two
    # lanes active they take turns: `wait` is the gap since the PREVIOUS
    # generate() finished, which is what makes that turn-taking visible.
    import time as _time
    last_generate_end = [_time.time()]

    def timed_transcribe(lane: "Lane", kind: str) -> str:
        secs = lane.buffer.size / SR
        level = rms(lane.buffer)
        started = _time.time()
        wait = started - last_generate_end[0]
        raw_text = transcribe(lane.buffer).strip()
        took = _time.time() - started
        last_generate_end[0] = _time.time()
        log(f"{lane.stream or 'office'} {kind} buf={secs:.1f}s took={took:.3f}s "
            f"({secs / max(took, 1e-6):.0f}x) wait={wait:.3f}s")

        reason = drop_reason(raw_text, secs, level)
        if reason:
            log(f"{lane.stream or 'office'} {kind} DROPPED — {reason}")
            return ""
        return clean(raw_text)

    def flush_lane(lane: "Lane") -> None:
        """FLUSH one lane: finalize its own utterance, reset its own state."""
        if lane.buffer.size > SR // 4:  # ignore < 250 ms blips
            # GATED, unlike the Parakeet sidecar's flush. That one is ungated
            # because silence there returns ""; here it returns 'Okay.'. A
            # final is the text that reaches the transcript, so the cheapest
            # correct answer over a silent buffer is to say nothing.
            level = rms(lane.buffer)
            if level < PARTIAL_SILENCE_RMS:
                log(f"{lane.stream or 'office'} final SKIPPED — silent buffer "
                    f"(rms {level:.5f} < {PARTIAL_SILENCE_RMS})")
            else:
                text = timed_transcribe(lane, "final")
                if text:
                    emit("final", text, lane.stream)
        lane.reset()

    def feed_lane(lane: "Lane", samples: np.ndarray) -> None:
        """Append to ONE lane and emit that lane's partial when it is due."""
        lane.append(samples)
        if not lane.wants_partial():
            return
        lane.samples_since_partial = 0
        # Don't spend a generate() on a silent lane. Silence is judged on the
        # recent tail (last ~4 s), not the whole buffer, so a lane that has gone
        # quiet stops emitting even while its buffer still holds old speech.
        tail = lane.buffer[-4 * SR:]
        if rms(tail) < PARTIAL_SILENCE_RMS:
            return
        text = timed_transcribe(lane, "partial")
        if text:
            emit("partial", text, lane.stream)

    emit("status", "READY")

    # The remote lane exists from the start but stays empty — it costs a
    # zero-length array — so a single-stream session never allocates, never
    # transcribes and never emits anything for it.
    # BOTH lanes take the same cadence — one setting, one value, so office and
    # remote captions can never update at different rates.
    log(f"partial cadence floor {partial_ms} ms (stretched by PARTIAL_DUTY "
        f"on long buffers; ~{2 * 0.609 / max(partial_ms / 1000, 0.25) * 100:.0f}% "
        f"duty with two active lanes at a 30 s buffer before the stretch)")
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
