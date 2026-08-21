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


def hub_dir() -> str:
    """Where the HF snapshots actually are.

    ⚠ NOT `HF_HOME/hub`. That is only the default; `HF_HUB_CACHE` overrides it,
    and the packaged app SETS THEM TO DIFFERENT PLACES on purpose
    (`PythonRuntime.sidecarEnvironment`):

        HF_HOME      = Application Support/.../hf-home   (mutable, no models)
        HF_HUB_CACHE = <app bundle>/models/hub           (the 35 GB of weights)

    So globbing `HF_HOME/hub` finds nothing in the bundle while working perfectly
    in dev, where HF_HOME defaults to the project `models/` and its `hub` is the
    real one. That is the dev-vs-bundle divergence this project has paid for
    repeatedly — and it is why hand-driving the bundled sidecar did not catch it:
    a hand drive inherits the DEFAULT env, not the app's.

    Reported from a client Mac 2026-08-18: "Wespeaker/wespeaker-voxceleb-campplus-LM
    is not in /Users/ess/Library/Application Support/Meeting Transcriber/hf-home/hub"
    — the weights were in the bundle the whole time.
    """
    return os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
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

# The FLUSH gate is now its own, far lower number — a FINAL is transcript, a
# partial is a caption that the chunked pass replaces, and the two deserve
# different answers.
#
# MEASURED 2026-08-20, this model, gate bypassed: real speech attenuated to
# -40 / -52 / -60 / -70 dBFS transcribed IDENTICALLY at every level. At 0.004
# (= -48.0 dBFS) every one of the bottom three was discarded instead — and the
# client Mac's capture chain averages -47 dBFS, one dB above that cliff, so a
# chunk quieter than average lost its text with no trace outside this log.
#
# ⚠ THIS COULD ONLY BE LOWERED ONCE speech_peak_ratio EXISTED, and the order
# matters. Lowering it alone was tried first and REVERTED within the hour: this
# model does not go quiet on noise the way MOSS does, it invents fluent
# sentences, so the level gate really was its primary hallucination defence.
# What changed is that the defence moved to a BETTER place — the signal's
# dynamics rather than its loudness — and that one is strictly stronger:
#
#   old level gate    caught noise only BELOW -48 dBFS. The measured
#                     hallucinations at rms 0.02 and 0.08 sailed past it.
#   speech_peak_ratio catches them at EVERY level, and lets quiet speech
#                     through at every level.
#
# So this threshold no longer has a hallucination job at all. It is left only to
# refuse a genuinely dead input — true digital silence (rms 0.00000, which the
# owner's dead BlackHole capture measured over 4027 s) and a dead channel.
#
# 0.0001 = -80 dBFS. Measured with the gate bypassed, real speech attenuated to
# -40 / -52 / -60 / -70 dBFS transcribed IDENTICALLY at every level, while 0.004
# (= -48.0 dBFS) discarded the bottom three — and the client Mac's chain
# averages -47 dBFS, one dB above that cliff.
FINAL_SILENCE_RMS = 0.0001

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

# ⚠ THE EXACT LIST ABOVE MISSED BY ONE WORD, and that is the whole reason this
# second family exists. Measured 2026-08-20 on white noise: the model returned
#
#     "I'm not sure if I can do it."      <- 'it', where the list says 'that'
#
# at every level tested INCLUDING levels above the old audio gate, so this
# escape predates the gate change entirely. The owner separately reported
# seeing "I'm not gonna do that" — a THIRD wording. Prose is not a protocol:
# an assistant paraphrases, and a list of exact sentences will always be one
# rewording behind it.
#
# So these are STEMS, and the price of the looser match is paid by a second
# condition: they fire ONLY when the chunk is also implausibly sparse, the same
# two-condition shape as CANNED_OVER_SILENCE. "I'm not sure if I can make it on
# Friday" is a thing a person really says in a meeting — and a chunk carrying it
# carries the rest of the meeting's words too, so its density clears the bar and
# it survives. A refusal is the ENTIRE content of a chunk that had no speech in
# it, which is what makes the pair discriminating.
REFUSAL_STEMS = (
    "i'm not sure if i can",
    "i am not sure if i can",
    "i'm not gonna",
    "i'm not going to",
    "i am not going to",
    "i can't help with",
    "i cannot help with",
)

# Canned captions this model produced over SILENCE. Unlike the refusals these
# are things a person really might say, so they are dropped ONLY when the audio
# was also implausibly sparse — the two-condition rule the Granite gate uses.
# `Okay.` after real speech survives; `Okay.` over 3 s of digital silence does
# not.
# ⚠ THE LAST ENTRY IS KOREAN, AND IT IS THE ONE THAT MATTERS. Measured
# 2026-08-20 with the FLUSH level gate bypassed: digital silence and room noise
# at -52, -60 and -70 dBFS ALL returned exactly '음악을 들으면서.' ("while
# listening to music"), identically, and NOTHING in this gate caught it — the
# level gate was the only thing standing in front of it. That is why the gate
# below could not simply be lowered the way MOSS's was; this had to come first.
# It is a whole-string match under the same two-condition rule as the rest, so a
# person really saying it in a dense chunk still survives.
CANNED_OVER_SILENCE = ("okay.", "ok.", "thank you.", "thanks.", "yeah.", "bye.",
                       "음악을 들으면서.")
CANNED_MIN_WORDS_PER_SEC = 0.5
# ⚠ AN ABSOLUTE CEILING FOR THE STEM FAMILY ONLY, and only that family needs it.
# The canned set above is a WHOLE-STRING match on short phrases, so its text is
# short by construction; the stems match a PREFIX, so the rest of the sentence
# is unbounded. On a long buffer the density bar stops being a bar at all —
# 0.5 w/s over a 120 s chunk-boundary flush means "fewer than 60 words", and a
# quiet 120 s stretch of a real meeting is routinely under 60. Measured against
# this rule on 2026-08-21:
#
#   "I'm not going to be there on Thursday, but Sam can cover it and we should
#    still ship the release by Friday afternoon."   22 words / 120 s -> DROPPED
#
# Both refusals ever OBSERVED here are far under the ceiling — "I'm not sure if
# I can do it." is 8 words — so 10 keeps every measured case and spares a real
# utterance that carries a clause after the opening. What it cannot spare is a
# real, SHORT "I'm not gonna do that." alone in a long buffer: the text is
# identical to the hallucination, and this gate has only the text. Logged.
REFUSAL_STEM_MAX_WORDS = 10


def emit(kind: str, text: str, stream: str = None) -> None:
    _emit_raw(kind, text, stream)


# --- "is there speech in this buffer at all?", judged on the SIGNAL ----------
#
# WHY THIS EXISTS, and why an RMS threshold could never have done it. Measured
# 2026-08-20 over 78 speech-free inputs, this model does not go quiet on noise
# the way MOSS does — it invents fluent sentences, in at least three languages:
#
#     "I'm not sure if I can do it."
#     '음악을 들으면서.'                                    (Korean)
#     'Các bạn có thể xem video và đăng ký kênh của mình…'  (Vietnamese)
#
# and it did so at rms 0.02 and 0.08 — ABOVE the level gate — so those reached
# the transcript whatever that threshold was set to. Enumerating the phrases is
# a losing game: the model paraphrases, and it does so in languages nobody here
# has a list for (the exact-sentence refusal list was already caught missing by
# a single word, 'do it' where it said 'do that').
#
# THE DISCRIMINATOR IS DYNAMICS, NOT LEVEL. Speech is bursty — syllables,
# pauses, breaths. Noise, hum and tones are flat. So: split the buffer into
# 50 ms frames, take each frame's RMS, and compare the LOUDEST frame to a QUIET
# one. The ratio is dimensionless, which is the whole point — it is IDENTICAL
# at -40 dBFS and at -70 dBFS, so it cannot reintroduce the level cliff that
# this same day's work removed from MOSS.
#
# ── 🔴 THE DENOMINATOR WAS THE MEDIAN FOR ONE DAY, AND IT DELETED REAL SPEECH ─
#
# Shipped 2026-08-20 as max/MEDIAN at a bar of 2.0, calibrated on 30 s and
# 120 s buffers. Audited 2026-08-21 and the calibration did not survive the
# audit, because THE SIDECAR DOES NOT SEE 30 s BUFFERS. `flush_lane` is driven
# by the app's VAD speech→silence edge (AudioRecorder.swift), so a FINAL buffer
# is ONE UTTERANCE — a second or two — and `feed_lane` judges a 4 s tail.
#
# max/median asks "how much of this buffer is silence?". On a VAD-trimmed
# utterance the answer is "almost none", so the median frame IS speech and the
# ratio collapses to the crest factor. Measured on six real recordings, windows
# the pre-existing rms gate already admits, and every offender below confirmed
# to be speech by transcribing it with mlx_whisper, which has no audio gate:
#
#     2 s windows   8-13% read BELOW 2.0     3 s   1.6%     4 s   2.0%
#
#     t= 26.0s ratio 1.86  "I think that's a great idea."
#     t= 57.0s ratio 1.76  "Does anyone have an idea what's going on?"
#     t= 34.0s ratio 1.77  'because humans have labored for thousands of years'
#
# Those are not edge cases; the last one is the general shape of the failure —
# DENSE CONTINUOUS SPEECH, where there is no silence to set the median. It
# persists to 8 s and only disappears near 10 s, i.e. above any real utterance.
#
# ── THE FIX, MEASURED IN BOTH DIRECTIONS ────────────────────────────────────
#
# Two changes, and neither alone is enough:
#
#   1. the denominator is a LOW PERCENTILE, not the median. An inter-syllable
#      dip exists even in dense speech; in noise, hum and tones the tenth
#      percentile is the same flat floor as the peak.
#   2. the rule ABSTAINS below MIN_SPEECH_JUDGE_SEC. Swept at 50 and 20 ms
#      frames against p50/p25/p10/p5, NOTHING separates below ~4 s: the best
#      separation at 1 s is 1.21x and at 2 s is 1.40x, i.e. the two populations
#      genuinely overlap. A rule that cannot tell must not answer.
#
# Measured with 50 ms frames and p10 — speech is every window of six real
# recordings that the rms gate admits; noise is 18 variants (white/pink/rumble/
# 50 Hz hum/1 kHz tone/DC at three levels each):
#
#     window     noise max     speech min     gap
#       4 s         2.19           4.71      2.16x
#       5 s         2.19           5.11      2.33x
#       6 s         2.05           5.42      2.64x
#       8 s         2.27           6.71      2.96x
#      15 s         2.22           8.78      3.96x
#      30 s         2.46          21.38      8.68x
#      60 s         2.31          26.21     11.37x
#
# Noise never reaches 2.46 and speech never falls below 4.71, at every length
# this rule is allowed to judge. 3.4 is the geometric middle of that pair, not
# an edge — 1.38x of margin to noise and 1.38x to speech.
#
# ⚠ WHAT ABSTAINING COSTS, stated rather than discovered later: a hallucination
# over a SHORT buffer is no longer caught here. It is still caught by the text
# gates in drop_reason (REFUSAL_PREFIXES unconditionally, CANNED_OVER_SILENCE
# and REFUSAL_STEMS over sparse audio) — which is what caught the measured
# 3 s case, 'Okay.' over digital silence, before this rule existed at all. Every
# hallucination this rule was written for was measured on a 30 s or 120 s
# buffer, so none of them moves out of reach.
#
# ⚠ AND A BUFFER THAT IS MOSTLY DIGITAL ZEROS reads enormous rather than flat,
# because the p10 frame lands on the 1e-12 floor. That direction is FAIL-OPEN
# (text is kept), and true silence is refused by FINAL_SILENCE_RMS a line later,
# so it is left alone deliberately. It was equally true of the median form.
SPEECH_FRAME_SEC = 0.05
SPEECH_FLOOR_PERCENTILE = 10
MIN_SPEECH_PEAK_RATIO = 3.4
MIN_SPEECH_JUDGE_SEC = 4.0


def speech_peak_ratio(samples: "np.ndarray"):
    """Loudest 50 ms frame / a quiet one, or None when the buffer cannot say.

    None FAILS OPEN at both call sites: a buffer this rule cannot judge must
    never become a reason to delete a caption or a final.
    """
    if samples.size < MIN_SPEECH_JUDGE_SEC * SR:
        return None
    frame = int(SPEECH_FRAME_SEC * SR)
    count = samples.size // frame
    if count < 8:
        return None
    block = samples[:count * frame].reshape(count, frame)
    levels = np.sqrt((block * block).mean(axis=1)) + 1e-12
    return float(levels.max() / np.percentile(levels, SPEECH_FLOOR_PERCENTILE))


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

    words = len(bare.split())
    density = words / seconds if seconds > 0 else 0.0
    sparse = density < CANNED_MIN_WORDS_PER_SEC

    if lowered in CANNED_OVER_SILENCE and sparse:
        return (f"canned caption over sparse audio ({stripped!r}, "
                f"{density:.2f} w/s, rms {level:.5f})")

    # The loose family: a refusal OPENING, but only over audio that carried no
    # speech. See REFUSAL_STEMS for why the exact list above is not enough.
    if sparse and words <= REFUSAL_STEM_MAX_WORDS:
        for stem in REFUSAL_STEMS:
            if lowered.startswith(stem):
                return (f"refusal opening over sparse audio ({stripped[:60]!r}, "
                        f"{words}w, {density:.2f} w/s, rms {level:.5f})")
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
        hub_dir(), 
        "models--" + MODEL.replace("/", "--"), "snapshots", "*")
    snapshots = sorted(glob.glob(pattern))
    if not snapshots:
        raise FileNotFoundError(
            f"{MODEL} is not in {hub_dir()} — "
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
            shape = speech_peak_ratio(lane.buffer)
            if shape is not None and shape < MIN_SPEECH_PEAK_RATIO:
                log(f"{lane.stream or 'office'} final SKIPPED — no speech in the "
                    f"signal (peak/p{SPEECH_FLOOR_PERCENTILE} {shape:.2f} < "
                    f"{MIN_SPEECH_PEAK_RATIO}, "
                    f"rms {level:.5f})")
            elif level < FINAL_SILENCE_RMS:
                log(f"{lane.stream or 'office'} final SKIPPED — no signal "
                    f"(rms {level:.5f} < {FINAL_SILENCE_RMS})")
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
        # Same signal test as the final path. It can only ADD skips, so lane duty
        # moves down, never up — and a hallucinated caption is worth skipping
        # even though the chunked pass would eventually replace it.
        shape = speech_peak_ratio(tail)
        if shape is not None and shape < MIN_SPEECH_PEAK_RATIO:
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
