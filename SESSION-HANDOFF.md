# Session Handoff — Meeting Transcribe

Read this first, then `CLAUDE.md` (decisions log).

## ⚠️ Pick up here — MossFormer2 overlap-repair is an OPEN investigation, not done

The most recent session (2026-07-14) built a full MossFormer2 overlap-repair pipeline (see
"Overlap recovery v3" below), then a `tester` verification pass found real bugs AND a
dangerous quality-gate gap, then the owner pushed back on the negative verdict and a deeper
investigation followed that is **still unresolved**. Do not assume this feature is finished,
safe, or abandoned — it's mid-investigation. Full detail in the v3 section below; TL;DR:

- **Bug fixed (2026-07-15):** `AudioRecorder.swift`'s `onChunkError` and the chunk watchdog
  timeout path now drain `pendingChunkWindows` (mirroring the success path), so a failed or
  timed-out LAST chunk no longer permanently blocks `lastChunkDone`/overlap repair, and a
  mid-recording failure no longer misaligns the next chunk's window. `swift build` verified
  clean; not yet exercised against a live failure injection.
- **Fixed (2026-07-15):** `OverlapRepairService`'s sidecar stderr log and `AudioRecorder`'s
  decision log used to collide on the same `logs/overlap-repair.log`. Split into
  `logs/overlap-repair-sidecar.log` (sidecar stderr) and `logs/overlap-repair-decisions.log`
  (decision log) — also updated the matching user-facing string in `OverlapTab.swift` and the
  timeout error message in `OverlapRepairService.swift`. `swift build` verified clean.
- **Duration experiment RUN + redesign BUILT & VERIFIED (2026-07-15):** the not-yet-run
  duration sweep was done on `recordings/meeting-2026-07-15T08-04-39Z.wav` (86.7s, real
  2-speaker overlap at ~[75.5,79.1]). Separation stayed **clean and distinct at 10s/15s/20s/24s**
  (track-vs-track Jaccard 0.06–0.09) — it did NOT reproduce the Overlap123 degradation, so
  "quality degrades with duration" is recording-dependent, not universal. Off the back of that,
  the overlap-repair windowing + attribution was redesigned and implemented (planner→executor→
  tester, all pass):
  - **Windowing:** now centers the separation window on the OVERLAP-REGION MIDPOINT ± `windowSec`
    (default 10 → 20s), instead of padding around a turn boundary. (`repairWindows` in
    `AudioRecorder.swift`.)
  - **Attribution:** now TEXT-BASED — re-ASR each separated track, then attribute track→speaker by
    word-overlap (Jaccard) against the existing display-row text (`anchorText` + new Gate 3 in
    `processRepair`), replacing the old diarization/active-span attribution. New gates: empty-anchor
    skip, `attributionFloor=0.15` no-match skip (hallucination guard), same-speaker skip (blend);
    kept the `nearDuplicateJaccard=0.72` track-vs-track guard. `OverlapTab.swift` caption updated.
  - **Verified end-to-end** on the above file: window→[67.3,86.7], tracks distinct (0.068),
    track1→Speaker1 (j=0.57), track2→Speaker2 (j=0.97) → correct OVERWRITE.
- **Still needed before enabling:** stress-test on HARDER cases (sustained overlap, similar-sounding
  content) — e.g. Overlap123.wav (owner holding it for later) and other real recordings. The
  validated case has only ~3.6s of overlap with very different content per speaker (an easy case).
- **Feature stays OFF by default** (`overlap.mossformer.enabled`) — do NOT enable it until the
  harder cases are tested.

## Overlap recovery v4 — DiCoW re-integrated as a second engine (2026-07-16) — BUILT, NOT YET TESTED ON REAL MEETINGS

Owner-requested re-attempt of the DiCoW engine removed in v2 (see below — that section's
evidence still stands and should be read first). DiCoW is now a **selectable engine**
alongside MossFormer2 (`overlap.engine`: `mossformer2` | `dicow`), behind quality gates
written specifically against v2's three failure modes. **Still OFF by default**
(`overlap.repair.enabled`). Full detail in `CLAUDE.md` → "Overlap recovery v4".

- **Why retry:** v2 *did* recover genuinely lost speech; it was removed because nothing filtered
  its bad spans. The gates are that filter: anchor cross-check (leakage), near-duplicate Jaccard
  (duplicate spans), word-density ceiling + a hard 30s window limit + never using DiCoW's internal
  timestamps (the runaway-span bug).
- **`.venv-dicow` is mandatory** — DiCoW needs transformers 4.55 (5.x breaks `generate()`), which
  conflicts with the main `.venv`'s MLX stack. It is the only sidecar outside `.venv`.
  **Consequence: DiCoW is dev-mode-only** until `package-app.sh` learns to ship a second
  interpreter. MossFormer2 is unaffected.
- **Verified this session:** sidecar loads + returns two distinct per-speaker transcripts on a real
  recording (~15s for a 2-target 20s window, ~4s warm startup); >30s and unknown-cmd guards fire;
  STNO mask is one-hot with correct T/O/N/S regions and silent padding; `swift build` clean;
  `py_compile` + `bash -n` clean.
- **NOT verified:** the app end-to-end (record → stop → repair), the quality gates against real
  overlap (no gate has ever fired on live data), and whether DiCoW beats MossFormer2 in practice.
  **Do not enable for the owner's real meetings until that runs.**
- Settings restructure shipped with it: `overlap.mossformer.enabled` → `overlap.repair.enabled`
  (one-time migration in `AppDelegate.migrateSettings`); window size + debug rows are now per-engine
  and the tab only shows the selected engine's blocks (fixes the owner's complaint that
  MossFormer2's debug toggle showed while DiCoW was selected).
- Key files: `scripts/dicow-service.py`, `DicowService.swift`, `AudioRecorder.swift`
  (`runDicowRepair`/`processDicowRepair`), `OverlapTab.swift`, `download-best-models.sh` (4c/4d).

## What the project is

100% offline meeting transcriber (macOS app, SwiftUI + Python sidecars). Realtime +
chunked ASR, speaker diarization, confidence scores. Owner: Aggia. Client language:
Indonesian (owner currently testing in English). All models run locally on MPS; no cloud.
See `CLAUDE.md` for the full model decisions.

## What already works

- Realtime ASR (Nemotron) + chunked ASR (selectable: Qwen3 / Whisper large-v3 / Voxtral).
- Diarization: pyannote **community-1** (main venv `.venv`, pyannote.audio **4.0.7**), live
  chunked + final refine, speaker profiles (rename, reset-on-start), overlap detect toggle.
- Transcript UI: one row per speaker turn (speaker · mm:ss–mm:ss · text), SPEAKER UNKNOWN
  placeholder before diarization, sentence-level text attribution, overlap tag.
- Owner currently on **Whisper large-v3 + English** for accuracy.

## Overlap separation — BUILT, TESTED, then REMOVED (2026-07-10)

A full overlap-separation feature (pyannote **PixIT** sidecar in its own `.venv-separation`,
Stop-time separation of overlap regions → re-ASR → rows in the transcript, own Settings tab;
plus a **MossFormer2/ClearerVoice** engine variant) was built, tested extensively on real
recordings, and **removed at the owner's request** — on a single mic, separation output was
duplicates/blends, not two clean voices, so it never added trustworthy text.

What testing established (full write-up in `CLAUDE.md` → PixIT section):
- Separation models (PixIT, MossFormer2, and by extension ESPnet's TF-GridNet class) separate
  **synthetic** mixtures near-perfectly (verified on-device with a TTS mix) but **blend on real
  single-mic audio** — a field-wide synthetic-vs-real gap confirmed by 2025–2026 research.
- The only reliable fix for complete per-speaker words during true overlap is **MULTI-MIC**.
- Day-to-day: turn-taking transcript + orange overlap tag (still present) is the honest output.

Everything separation-related was deleted: `SeparationService.swift`, `SeparationTab.swift`,
`scripts/separation-service.py`, `.venv-separation`, the PixIT/WavLM/speechbrain models in
`models/hub`, the setup-script section, the `-2` file-transcribe frame in
`chunked-asr-service.py`, and all `AudioRecorder`/`TranscriptView` orchestration. The app is
back to: record → realtime + chunked ASR → diarization → per-speaker rows.

Note: a `separation.enabled` / `separation.model` UserDefaults key may linger on the owner's
machine from testing; nothing reads them anymore.

## Overlap recovery v2 — DiCoW — BUILT, TESTED, then REMOVED (2026-07-14)

> **Superseded 2026-07-16 (see the v4 section at the top):** DiCoW was re-added as a
> selectable engine, so the "everything was deleted" paragraph below is now historical —
> `scripts/dicow-service.py`, `DicowService.swift`, `.venv-dicow`, and the weights all exist
> again. **The failure evidence in this section is still valid and is exactly what v4's
> quality gates are built against — read it before touching the DiCoW path.**

**DiCoW v3.3** (BUT-FIT, Diarization-Conditioned Whisper, target-speaker ASR) was built and fully
wired end-to-end (sidecar, Swift client, Settings tab, word-level merge into the main transcript).
It correctly recovered *some* genuinely lost speech on-device, but a standalone quality CLI
(`dicow-test.sh`, since removed — ran DiCoW per-speaker per-window with no merge logic, raw
output to terminal) exposed real problems on the owner's actual recordings: cross-speaker word
leakage, near-duplicate spans, and unreliable internal timestamps. **Owner verdict: not good
enough — removed entirely** (owner request, 2026-07-14, same pattern as the PixIT/MossFormer2
removal above).

Everything DiCoW-related was deleted: `scripts/dicow-service.py`, `scripts/dicow-test.py`,
`dicow-test.sh`, `DicowService.swift`, `OverlapTab.swift`, the `.venv-dicow` venv, the
`BUT-FIT/DiCoW_v3_3_large` weights + `transformers_modules` cache, the `download-best-models.sh`
setup section, and all `AudioRecorder`/`ModelLoader`/`ModelCatalog`/`SettingsView`/
`TranscriptView` wiring (recovery state, `configureRecovery`, `maybeStartRecovery`,
`applyRecoveredInsertions`). Back to: record → realtime + chunked ASR → diarization → per-speaker
rows — same shape as after the separation removal. Don't re-litigate target-speaker ASR either
without materially new evidence (e.g. a newer/better DiCoW checkpoint).

## Overlap recovery v3 — MossFormer2 (librimix-2spk) standalone — BUILT, VERIFIED, UNDER RECONSIDERATION (2026-07-14)

Overlap attempt #3, owner-requested 2026-07-14. **Distinct from the earlier-removed
`clearvoice` / `MossFormer2_SS_16K` attempt** (in the PixIT section of CLAUDE.md) — this one uses the
**standalone `alibabasglab/MossFormer2` repo's vendored PyTorch code** (`scripts/vendor/mossformer2/`)
+ the `alibabasglab/mossformer2-librimix-2spk` checkpoint (8 kHz, 2-speaker LibriMix). Do not conflate.

### Architecture as built

Runs at **Stop** (after diarization final pass + last chunk): separates a window around each detected
2-speaker overlap → re-ASRs each track with the loaded chunked model → **overwrites** the overlapped
text (REPLACE, not a marker) only when quality gates pass (near-duplicate Jaccard guard >0.72 → skip;
same-speaker attribution → skip; 3+ speakers → skip). Off by default (`overlap.mossformer.enabled`,
Settings → Models → Overlap; window size `overlap.mossformer.windowSec`, default 10s = a ~20s slice,
10s padding on each side of the overlap boundary). All decisions logged to `logs/overlap-repair.log`.

New/edited files: `scripts/mossformer2-service.py` (sidecar), `scripts/chunked-asr-service.py`
(additive `-2` file-transcribe frame), `OverlapRepairService.swift`, `OverlapTab.swift`,
`ChunkedASRService.transcribeFile`, `AudioRecorder.swift` (repair orchestration + splice),
`ModelCatalog`/`ModelLoader`/`SettingsView`/`TranscriptView` wiring, `download-best-models.sh`
(section 4b). New keys are `overlap.mossformer.*`; the retired `overlap.recovery` /
`separation.enabled` keys are NOT reused (stale values may linger — ignored).

### `tester` Stage 7 verification (2026-07-14) — build is clean, but NOT safe to ship

Build/protocol/sidecar all verified working (swift build clean, `-2` frame doesn't regress
streaming, sidecar's `separate` protocol returns clean JSON — the vendored
`Mossformer2Wrapper.__init__`'s stray `print()` is correctly redirected to stderr). But:

1. **Bug:** `AudioRecorder.swift` `onChunkError` (~line 274-282) and the chunk-flush watchdog
   timeout (~line 367-377) never call `pendingChunkWindows.removeFirst()` — only the success
   path does. If the LAST chunk at Stop errors/times out, `checkLastChunkDone()`'s
   `pendingChunkWindows.isEmpty` guard is permanently false → `lastChunkDone` never true →
   `maybeStartOverlapRepair()` silently never fires, no error surfaced to the user.
2. **Bug:** `OverlapRepairService.swift:66` (`process.standardError = PythonRuntime.logHandle(name:
   "overlap-repair")`) and `AudioRecorder.swift:947`'s decision-logging (`overlapLog()`) both
   write to `logs/overlap-repair.log` with no coordination — should be two separate files.
3. **Critical finding — the quality gates don't catch the real failure mode:** on
   `recordings/Overlap123.wav`, window `[36.86s,50.80s]` (speakers 3 & 9) passed ALL THREE
   quality gates (non-empty, Jaccard <0.72, different speaker attribution) — yet track B's
   re-ASR text was a fabricated sentence not present anywhere in the source audio
   ("...I can't do this, this is too hard"), which would have been spliced into the
   transcript as if speaker 9 genuinely said it. This is worse than "near-duplicate" (which
   the guard does catch) — it's **confident hallucination attributed to a real speaker**,
   and the current gates are blind to it (they measure word-overlap and energy-timing, not
   content plausibility).

On `recordings/meeting-2026-07-14T06-42-58Z.wav` (95.5s), the plan's own windowing logic
happened to chain-merge every overlap into one 4-speaker window, which is unconditionally
skipped (2-speaker-only checkpoint) — so the feature was a safe no-op there, but this also
meant the gates were never really exercised end-to-end on that file.

### Owner pushback + deeper investigation (2026-07-14, same session, after the tester report)

The owner didn't accept "single-mic separation is fundamentally broken" as the final word and
pushed for more direct evidence, which materially changed the picture:

1. **The owner ran the *official* upstream `MossFormer2_standalone/inference.py` themselves**
   (not our sidecar) on `recordings/meeting.wav` (31.7s) and got `recordings/index1.wav` /
   `index2.wav`. Transcribed with real timestamps, **both tracks are clean, coherent,
   non-repeating, complete monologues** — e.g. index1 is a continuous "one-minute speech /
   communication framework" narrative from 0:00 to 0:31.7 with no loops or garbage.
2. An earlier same-session critique (comparing a separated track's content against what's
   *audible in the still-mixed original* at the same timestamp) was **wrong methodology** —
   of course a cleanly separated track sounds different from the garbled mix; that's the
   whole point of separation. The valid test is whether each track is internally coherent on
   its own, not whether it resembles the noisy mix.
3. **Confirmed our sidecar has no bug:** feeding `scripts/mossformer2-service.py` the *entire*
   `meeting.wav` (start=0.0, end=31.7 — i.e. NOT windowed/cut) reproduced the official
   script's result byte-for-byte in transcript terms — same two clean, coherent tracks. This
   proves `mossformer2-service.py`'s separation logic itself matches the reference
   implementation; quality differences are about WHAT audio span you feed it, not a bug in
   our code.
4. **Owner's hypothesis:** cutting a short arbitrary window (our ~20s slice) hurts
   separation quality vs. feeding a complete, uncut span. **Partially confirmed, partially
   not:** re-ran the sidecar on the FULL (uncut) `recordings/Overlap123.wav` (59.02s, no
   windowing at all) — the **first ~25s came out clean and coherent** (matching the
   `meeting.wav` result), but **the last ~30s (30s-58s) converged into near-duplicate content
   again** in both tracks (both end up narrating the same "comparing myself to everyone /
   my parents said look" passage). So full-file processing is better than short windows, but
   quality still seems to **degrade with duration** — the `meeting.wav` success (31.7s) may
   partly be *because* it's short, not just because it's uncut.
5. **Not yet run:** the natural next experiment — separate the same overlap-containing
   recording at several different full-span durations (e.g. 15s, 20s, 30s, 45s chunks
   instead of 10s+10s padding) to find where quality starts to degrade, which would inform
   a redesigned windowing strategy (e.g. "process each detected overlap as its own natural
   turn-to-turn span, capped at N seconds" rather than the current fixed padding).

### How to quickly re-run separation tests next session

The sidecar's JSON-lines protocol (see `scripts/mossformer2-service.py` docstring) can be driven
directly without going through the app — this is the fastest way to keep investigating:

```python
import subprocess, json
p = subprocess.Popen(['.venv/bin/python3','scripts/mossformer2-service.py'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open('/tmp/sidecar.log','w'),
    text=True, bufsize=1)
p.stdin.write(json.dumps({'cmd':'separate','id':1,'audio':'recordings/Overlap123.wav',
    'start':0.0,'end':59.02,'out_dir':'/tmp/mf2out'})+'\n')
p.stdin.flush()
print(p.stdout.readline())   # {"type":"result","tracks":[...]}
p.stdin.write(json.dumps({'cmd':'exit'})+'\n'); p.stdin.flush()
```

Then transcribe each track with timestamps to judge coherence (not against the noisy mix —
against itself):
```
.venv/bin/mlx_whisper /tmp/mf2out/track1.wav --model mlx-community/whisper-large-v3-mlx \
    --output-format srt --output-dir /tmp/mf2out --verbose False
```

Get real diarization turns for any recording to find overlap windows first:
```
printf '{"cmd":"final","audio":"recordings/<file>.wav","num_speakers":0}\n' \
    | .venv/bin/python3 scripts/diarize-service.py
```

Test recordings available: `recordings/meeting.wav` (31.7s, clean full-file result — the
positive reference case), `recordings/Overlap123.wav` (59s, good first ~25s then degrades),
`recordings/meeting-2026-07-14T06-42-58Z.wav` (95.5s, chain-merges to one 4-speaker skip
under the current windowing logic — retest with different window sizes once that logic
changes).

### Bottom line for the next session

Not concluded. Don't re-enable the toggle or call this "done." Either (a) fix the two bugs
above and redesign the windowing to use short, uncut, natural spans based on the duration
experiment, then re-verify with `tester`, or (b) if the duration experiment shows no safe
operating window exists for real multi-speaker meetings (vs. this session's two clean-ish
ASR-narration-style test files), that's the new evidence needed to close this out like PixIT/DiCoW.

## Key files

- `scripts/diarize-service.py` — community-1 diarization sidecar.
- `scripts/chunked-asr-service.py` — Whisper/Qwen3/Voxtral chunked ASR sidecar.
- `scripts/nemotron-asr-service.py` / `scripts/silero-vad-service.py` — realtime ASR / VAD.
- `download-best-models.sh` — one-time model download + `.venv` setup (self-contained runtime).
- `MeetingTranscriber/Sources/.../Audio/AudioRecorder.swift` — recorder + display-row builder.
- `MeetingTranscriber/Sources/.../Transcription/DiarizationService.swift` — diarization client.
- `MeetingTranscriber/Sources/.../Views/Settings/DiarizationTab.swift` — settings toggles.

## Open work (from PLAN-diarization-next.md)

- Tune `SIM_THRESHOLD` on real client (Indonesian) audio.
- Confidence scores in the UI (client requirement).
- Profile management UI polish; live→final speaker-id continuity.

## Reminders

- Syntax-check Python with `python3 -m py_compile`; build Swift with `swift build`.
- Overlap words are inherently approximate on a single mic — this is a hard limitation
  (see CLAUDE.md); don't re-litigate separation without new evidence (e.g. multi-mic input).
