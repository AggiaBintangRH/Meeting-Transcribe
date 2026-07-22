# Meeting Transcribe

Offline meeting transcription for macOS: live captions while you talk, an
accurate transcript afterwards, and speaker labels for who said what.

**Nothing leaves the machine.** Models are downloaded once, then the app runs
fully air-gapped (`HF_HUB_OFFLINE=1`). There are no cloud APIs anywhere in the
recording path — that is the project's central constraint, not a feature.

---

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon Mac. Developed on an M4 with 32 GB; 16 GB works with the smaller models |
| macOS | 14 or newer |
| Disk | ~25 GB for model weights |
| Hugging Face token | Needed **once**, for setup only — pyannote is a gated repo |

The token is your credential and is never stored in the repo. Put it in `.env`
(gitignored) or export it for the session; the setup script asks if you do
neither.

```bash
echo 'HF_TOKEN=hf_xxx' > .env
```

You must also accept the terms at
`huggingface.co/pyannote/speaker-diarization-community-1` before setup can
fetch it.

## Setup

Double-click **`RUN-SETUP.command`**, or:

```bash
./download-best-models.sh
```

This creates a self-contained `.venv` inside the project and downloads every
model. It is idempotent — re-run it any time; it skips what is already there.
Output is saved to `logs/setup-run.log`.

## Run

Double-click **`RUN-APP.command`**, or:

```bash
cd MeetingTranscriber && swift run
```

To build a distributable, self-contained `.app` (bundled interpreter, scripts
and models):

```bash
./build.sh
```

---

## What it does

**Live captions** stream while you speak, from Nemotron 3.5 (0.6B, streaming).

**An accurate transcript** replaces them every chunk interval. Four chunked
ASR models are selectable so you can A/B on your own audio:

| Model | Notes |
|---|---|
| Qwen3-ASR 1.7B | Best accuracy/speed balance; the default |
| Whisper large-v3 | Strong punctuation; occasionally hallucinates tail text |
| Voxtral Mini 4B | Slowest by far (~1.1x realtime) |
| Granite Speech 4.1 2B | Lowercase output |

**Speaker labels** come from pyannote community-1 running on MPS, with voice
profiles that can persist across meetings so "Speaker 1" stays the same person.

**Word-level timestamps** (optional) from Qwen3-ForcedAligner let a speaker
change split the transcript at the exact word instead of at an estimated
character position. Works with all four chunked models — the aligner never
sees the recogniser, only audio and text.

**ATND1061 microphone array** (optional) is supported over IP control: beam
direction becomes speaker identity, so a person is identified by *where they
are* rather than only by voice. Includes TCP control, UDP multicast beam data,
and a command console.

**Dual-stream capture** (optional) records a room channel ("Office") and a
remote channel ("Remote", e.g. a conferencing app via a loopback device) from
one Aggregate Device, transcribing and diarizing each independently. The two
identity spaces never mix.

## Architecture

A SwiftUI app talking to long-lived Python sidecars over stdin/stdout
JSON-lines. The heavy models run in Python (MLX for ASR, PyTorch/MPS for
diarization); Swift owns capture, timing, the transcript model and the UI.

```
MeetingTranscriber/   SwiftPM app (not an Xcode project)
scripts/              Python sidecars, one per model family
models/               HF_HOME — downloaded weights, never committed
recordings/           your audio, never committed
logs/                 per-sidecar logs, never committed
```

## Tests

```bash
cd MeetingTranscriber && swift test      # Swift logic
.venv/bin/python3 scripts/sidecar-tests.py   # sidecar protocols, ~50s
```

The sidecar suite runs the real sidecars and checks the things that break
silently: that audio lanes never mix, that word indices survive punctuation,
that the two speaker-identity stores cannot contaminate each other. It points
`MT_PROFILE_DIR` at a temp directory and verifies your real speaker profiles
are byte-unchanged afterwards.

## Known limitations

**Overlapping speech on a single microphone is not solved**, and this is a
property of the problem rather than a bug in this app. When two people talk at
once into one mic, the audio is a single mixed waveform and standard ASR
cannot separate it. Three approaches were built, tested on real recordings and
removed — source separation (PixIT, MossFormer2) and target-speaker ASR
(DiCoW). Overlap is *detected* and tagged; the words spoken during true
overlap are approximate.

The reliable fix is one channel per source, which is what dual-stream capture
does for the room/remote split.

**Room echo in hybrid meetings.** If remote audio is played through room
speakers, the microphone array captures it too and the same speech is
transcribed twice. Use the array's own acoustic echo cancellation.

**Speaker attribution is sentence-level** unless the word aligner is enabled.

See `CLAUDE.md` for the full decision log, including what was tried and
rejected and why.

## License

Apache 2.0. See `LICENSE`.
