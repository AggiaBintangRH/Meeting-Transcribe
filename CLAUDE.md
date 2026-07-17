# Meeting Transcribe — Personalization & Decisions Log

Context file for this project. Read this first before making changes.

## Owner

- Aggia Bintang Ramadhan
- Hardware: Mac M4, 32GB RAM
- Likely meeting language: Indonesian (mixed possible) — always verify model language support

## Client constraints (hard requirements)

1. **100% offline** — no cloud APIs, no data leaves the machine. One-time model downloads OK; runtime must work air-gapped (`HF_HUB_OFFLINE=1`)
2. Transcription + diarization both required
3. Confidence scores wanted (transcript + speaker)

## Key decisions & rationale

| Decision | Choice | Why |
|---|---|---|
| Keep all 4 ASR models | Nemotron, Qwen3-ASR, Whisper large-v3, Voxtral Mini T2 | Owner wants selectable chunked ASR; A/B test per language |
| Realtime engine | Nemotron 3.5 0.6B | Fast (112x RT), cache-aware streaming, 40 locales |
| Primary chunked ASR | Qwen3-ASR 1.7B MLX | Best WER (1.32%), Indonesian support, 30–36x RT |
| Diarization | pyannote community-1 on PyTorch MPS | Best open DER (19.9% AMI SDM); tuned pipeline |
| ❌ Custom MLX diarization | Rejected | MLX ports = old 3.1 models, worse DER, high effort |
| ❌ FluidAudio / senko (CoreML) | Removed per owner | Not in scope |
| ❌ pyannoteAI precision-2 | Rejected | Cloud API — violates offline requirement |
| ❌ Parakeet | Rejected | No Indonesian |
| Confidence scores | DIY on community-1 | Segment frame probs + cluster distances (~100 lines) |

## Known limitations

- **Overlapping speech (two people talking at once) — hard limitation, not a bug.**
  With a single mic the audio is one mixed waveform; standard single-speaker ASR
  (Whisper, Qwen3, etc.) cannot separate simultaneous voices into two correct
  texts. pyannote can *detect* overlap (we keep it, `diarization.detectOverlap`,
  and tag overlapped rows), but the words during true overlap are approximate.
  Turn-taking transcribes fine; only genuinely simultaneous speech is affected.
- Text→speaker attribution is **sentence-level, estimated** (chunked ASR gives no
  word timestamps). Whisper *can* give real timestamps → future word-exact path.
- What would actually fix overlap (all outside the current offline-MLX-Mac stack):
  multi-mic (one channel per speaker, cleanest); source separation before ASR
  (SepFormer / TF-GridNet / MossFormer2); or multi-talker / speaker-attributed
  ASR (NVIDIA NeMo **Sortformer**, t-SOT/SOT, SURT — all PyTorch/CUDA, no MLX
  port yet). pyannoteAI precision handles overlap but is cloud (violates offline).
- Decision: rely on turn-taking + overlap tagging. An optional PixIT separation
  path was built, tested, and later removed — see the section below.
  (Historical gotcha if ever revisited: PixIT needs pyannote.audio 3.4.x, which
  conflicts with community-1's 4.x → it required its own separate venv/sidecar,
  plus `microsoft/wavlm-large`, speechbrain ecapa at BOTH the pinned commit and
  `main`, and `speechbrain==1.0.2`.)

### PixIT overlap separation — BUILT, TESTED, then REMOVED (2026-07-10)

The feature was fully built and wired (own Separation settings tab, Stop-time separation of
overlap regions → re-ASR → rows in the transcript; a MossFormer2/ClearerVoice engine variant
was also built and compared). After extensive on-device testing the owner had it **removed
entirely** (code, sidecars, `.venv-separation`, PixIT/WavLM/speechbrain model weights): on a
single microphone the separated output was duplicates/blends, never two clean voices, so it
did not add trustworthy text. The verdict below is kept because the evidence remains valid:
**PixIT (and single-mic separation generally) does NOT reliably separate overlapping speech
on a single microphone.**

Evidence (multiple real recordings, incl. a fair test with 3 people speaking genuinely different
sentences simultaneously):
- When two voices overlap, PixIT typically outputs **near-duplicate tracks** (same words) or a
  **single blended track** mixing both speakers — not two clean per-speaker tracks.
- It CAN surface *some* words that were buried/lost in the mix, but blended with the other voice,
  not cleanly isolated. So "complete, clean words per speaker" is **not achievable**.
- This is **not a bug and not a bad model choice** — fresh web research (2025–2026) confirms
  single-mic separation of **real, reverberant, far-field meeting audio remains largely unsolved
  field-wide**; models score well only on synthetic/anechoic mixtures and degrade badly on real
  data. PixIT (AMI-meeting-trained) is already a reasonable pick; swapping models won't change it.
- Swapping to SepFormer / TF-GridNet / MossFormer2 would not fundamentally help (same limit).
  Multi-talker / speaker-attributed ASR (NeMo Sortformer, SURT 2.0) is the promising *different*
  approach but is PyTorch/CUDA, heavy, and has no offline Mac/MLX port.

**Final stance:** feature REMOVED from the codebase (2026-07-10, owner request) — rely on the
turn-taking transcript + overlap tagging for day-to-day. **The only reliable fix for complete
per-speaker words during true overlap is MULTI-MIC** (one channel per person). Also evaluated
and rejected: word-exact in-sentence revision via Whisper word timestamps (choppy output),
MossFormer2/ClearerVoice (near-perfect on a synthetic TTS mix, blends on real recordings —
proving the tooling was right and the single-mic gap is fundamental), and ESPnet (same
synthetic-trained model class; its realistic meeting recipes are multi-channel).

### Overlap recovery v2 — DiCoW target-speaker ASR — BUILT, TESTED, then REMOVED (2026-07-14)

**DiCoW v3.3** (BUT-FIT, diarization-conditioned Whisper, target-speaker ASR) was built and
integrated end-to-end: `scripts/dicow-service.py` (own `.venv-dicow`, transformers 4.x) +
Settings → Models → Overlap (`overlap.recovery`, default OFF), missing word-runs word-aligned
against the main transcript and woven in as `【Speaker · recovered: …】`. On-device it correctly
recovered some genuinely lost speech (words Whisper dropped entirely), but a standalone quality
CLI (`dicow-test.sh` — since removed) surfaced real problems on the owner's actual recordings:
cross-speaker leakage (one speaker's words showing up under another speaker's target mask),
near-duplicate spans, and unreliable internal timestamps (one span spuriously covered 40+ words).
**Owner verdict (2026-07-14): not good enough — removed entirely.**

Everything DiCoW-related was deleted: `scripts/dicow-service.py`, `scripts/dicow-test.py`,
`dicow-test.sh`, `DicowService.swift`, `OverlapTab.swift`, the `.venv-dicow` venv, the
`BUT-FIT/DiCoW_v3_3_large` weights (~6GB) and its `transformers_modules` cache, the
`download-best-models.sh` setup section, and all `ModelCatalog`/`ModelLoader`/`SettingsView`/
`AudioRecorder`/`TranscriptView` wiring (recovery state, `configureRecovery`,
`maybeStartRecovery`, `applyRecoveredInsertions`). The app is back to: record → realtime +
chunked ASR → diarization → per-speaker rows, same as before the DiCoW experiment.

Combined with the PixIT/MossFormer2/ESPnet section above, this closes out the second
architectural approach tried for overlap. Current stance unchanged: turn-taking transcript +
overlap tagging is the honest day-to-day output; **multi-mic is the only approach not yet
tried that could fully solve true simultaneous-speech overlap.** Don't re-litigate either
approach (separation or target-speaker ASR) without materially new evidence.

### Overlap recovery v3 — MossFormer2 (librimix-2spk) standalone — BUILT 2026-07-14

Third architectural attempt at overlap recovery, owner-requested 2026-07-14. **This is
NOT the earlier-removed `clearvoice` / `MossFormer2_SS_16K` attempt** (that used the
`clearvoice` pip package and was removed with the PixIT section above). This build uses the
**STANDALONE `alibabasglab/MossFormer2` GitHub repo's vendored PyTorch code**
(`scripts/vendor/mossformer2/`) with the checkpoint **`alibabasglab/mossformer2-librimix-2spk`**
(8 kHz, 2-speaker, LibriMix-trained) at `models/mossformer2/mossformer2-librimix-2spk/`. Do not
conflate the two in code comments or docs.

How it works: at **Stop**, after the diarization final pass and the last chunk both complete, the
app finds 2-speaker overlap windows from the diarization turns, separates a short window around each
(`overlap.mossformer.windowSec`, default 10s) into two per-speaker tracks via the persistent
`scripts/mossformer2-service.py` sidecar (CPU, runs in the main `.venv`), re-transcribes each track
with the already-loaded chunked ASR model (new additive `-2` file-transcribe frame in
`chunked-asr-service.py`), and **overwrites** (REPLACE semantics, not an appended marker) the
overlapped text with two per-speaker rows — **only if quality gates pass**:
1. both re-ASR texts non-empty, 2. **near-duplicate guard** (Jaccard on word sets > 0.72 → skip),
3. **speaker-attribution check** (both tracks mapping to the same speaker → skip).
Windows touching 3+ speakers are skipped (checkpoint is 2-speaker only). Every decision
(OVERWRITE / SKIP + reason + before/after text) is logged to `logs/overlap-repair-decisions.log`
(the sidecar's own stderr goes to a separate `logs/overlap-repair-sidecar.log` — split
2026-07-15 to stop the two writers colliding on one file).

Off by default (`overlap.repair.enabled` — renamed from `overlap.mossformer.enabled` on
2026-07-16, see v4 below; Settings → Models → Overlap). Key files:
`scripts/mossformer2-service.py`, `OverlapRepairService.swift`, `OverlapTab.swift`,
`AudioRecorder.swift` (repair orchestration), `ModelCatalog`/`ModelLoader` wiring.

**Expected behaviour, not a bug:** on real single-mic audio the near-duplicate guard is expected to
skip **many/most** windows — consistent with all prior evidence that single-mic separation blends
real audio. Skipping = keeping the original honest text, which is the safe outcome. New UserDefaults
keys are `overlap.mossformer.*`; the retired `overlap.recovery` / `separation.enabled` keys are NOT
reused.

### Overlap recovery v4 — DiCoW re-integration (2026-07-16)

Owner-requested **re-attempt of the DiCoW engine removed in v2** (2026-07-14). This does not
overturn v2's evidence — the failures listed in the v2 section were real and are still expected.
What changed is that DiCoW is now **one of two selectable engines** (`overlap.engine`:
`mossformer2` | `dicow`) behind **quality gates written specifically against v2's failure modes**,
so bad output is rejected instead of spliced in. Still **OFF by default**.

Why v2 was worth retrying: v2 *did* recover genuinely lost speech; it was removed because nothing
filtered the bad spans out. The gates below are that filter.

| v2 failure mode | Gate now guarding it (`AudioRecorder.processDicowRepair`) |
|---|---|
| Cross-speaker leakage | Anchor cross-check: `jaccard(text, otherAnchor) > jaccard(text, ownAnchor) + 0.15` → SKIP that speaker |
| Near-duplicate spans | `jaccard(textA, textB) > 0.72` → SKIP the whole window |
| Unreliable timestamps / "40+ word span" | Two-part: the sidecar **rejects windows > 30s** (so `generate()` never enters its long-form seek loop — the actual source of the bug) and **never returns DiCoW's internal timestamps at all**; plus a word-density ceiling of **6 words/sec** of that speaker's own in-window turns → SKIP |

Survivors go through the same `TranscriptMerge.merge` → `applyRepair` path as MossFormer2. No 2×2
attribution is needed (unlike MossFormer2) — DiCoW is already conditioned per speaker.

**Runtime gotchas — all four cost real debugging, do not "clean them up":**
1. **transformers 4.55.0 ONLY.** On 5.x, `generate()` dies with
   `AttributeError: '_get_initial_cache_position'`. The main `.venv` needs 5.x for the MLX stack →
   hard conflict → DiCoW gets **its own `.venv-dicow`** (the only sidecar that is not in `.venv`).
2. **`pandas` is a required, undocumented import** of DiCoW's remote code (`ImportError` on load).
3. **`model.tokenizer = processor.tokenizer` must be set BY HAND after `from_pretrained`** —
   `generation.py:333 _fix_timestamps_from_segmentation` reads `self.tokenizer`, which is `None`
   otherwise. The single most important gotcha.
4. The dynamic-module cache (`models/modules/transformers_modules`) goes **stale** if written by a
   different transformers version → confusing `FileNotFoundError: .../decoding.py`. The sidecar
   pins `HF_HOME`/`HF_HUB_CACHE` explicitly before importing transformers.

Device is **CPU float32** (MPS unverified for this custom remote code — not worth the risk).
STNO mask is `(1, 4, 1500)` = Silence/Target/Non-target/Overlap at 50 Hz over a 30s frame, built
from the diarization turns; frames past the real window stay Silence.

**Settings restructure (same change):** `overlap.mossformer.enabled` → **`overlap.repair.enabled`**
(shared on/off; one-time migration copies the old value at startup in `AppDelegate`). Window size
and debug rows are now **per-engine** (`overlap.mossformer.*` / `overlap.dicow.*`) and the tab only
shows the selected engine's blocks — previously MossFormer2's debug toggle showed while DiCoW was
selected (owner complaint). DiCoW's window picker caps at 14s so ±window ≤ 28s stays under the 30s
limit; longer merged windows are skipped and logged.

**Limitation: `.venv-dicow` makes DiCoW dev-mode-only.** `package-app.sh` ships only the single
bundled interpreter, so a packaged `.app` cannot start the DiCoW sidecar (`DicowService` throws a
clear "run download-best-models.sh" error). Extending `package-app.sh` to carry a second
interpreter is outstanding work; MossFormer2 is unaffected.

Key files: `scripts/dicow-service.py`, `DicowService.swift`, `PythonRuntime.command(forScript:venvName:)`,
`AudioRecorder.swift` (`runDicowRepair`/`processDicowRepair`), `OverlapTab.swift`,
`ModelCatalog`/`ModelLoader` wiring, `download-best-models.sh` (4c weights, 4d `.venv-dicow`).
Decisions log is shared with MossFormer2: `logs/overlap-repair-decisions.log` (sidecar stderr →
`logs/dicow-sidecar.log`).

## Model routing (owner-set, 2026-07-14)

Per-phase models via subagents in `.claude/agents/` — each is pinned by its
`model:` frontmatter, so no model check/switch is ever needed:

| Phase | Agent | Model |
|---|---|---|
| Planning (multi-step changes) | `planner` | Fable |
| Implementation of an approved plan | `executor` | Opus |
| Verification after implementation | `tester` | Sonnet |

For quick one-file fixes the main conversation may act directly (skipping the
executor) — the subagent round-trip isn't worth it for trivial edits.

## Owner preferences

- Wants bigger/more-parameter models when they improve accuracy (chose MLX path over CoreML-only)
- Prefers evidence-based choices: benchmark on AMI SDM DER for diarization, WER for ASR
- Concise communication; tables preferred
- Verify with fresh web research before finalizing model choices — models update fast

## Evaluation metrics to use

- ASR: WER (word error rate), test on real client audio
- Diarization: DER on AMI SDM as reference benchmark; validate on client meetings
- Speed: real-time factor (RTF) on the M4
