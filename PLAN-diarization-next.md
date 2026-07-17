# Diarization — Plan for Tomorrow (2026-07-08)

Continuation of the chunked/hybrid diarization + speaker-profile work.

## Where we left off (2026-07-07)

- Hybrid pipeline wired: live chunked diarization on its own interval + final refine at stop.
- Independent diarization interval setting (15/30/45/60s), separate from chunked ASR.
- Speaker rename UI in the transcript (saves to voice profile).
- **Fixed:** final pass collapsing every speaker onto one profile — matcher now does
  batch, mutually-exclusive assignment (distinct voices → distinct profiles).
- **Not yet verified on device.** All fixes are static-checked only.

## Known open issues (observed)

- Profile store never resets: a 2-person test left **6 junk profiles** in
  `models/speaker-profiles/`. Stale voices pollute matching across sessions.
- Live profiles (1,2) not reused by the final pass → final can mint new ids (3,4),
  so names/renames don't carry through the meeting.
- `SIM_THRESHOLD = 0.5` is a guess — not tuned on real Indonesian meeting audio.

## Tomorrow — priority order

| # | Task | Why / detail |
|---|---|---|
| 1 | **Verify the collapse fix** on a real 2-person recording | Confirm live + final both show 2 distinct speakers. Blocks everything else. |
| 2 | **Profile management + reset** | UI in Diarization settings: list saved speakers, rename, delete, "Reset all". Clear the 6 junk profiles. |
| 3 | **Live→final continuity** | Final should reuse the session's live profiles, not mint new ids. Options: seed final matching with session ids / carry a session-scoped id map. |
| 4 | **Tune `SIM_THRESHOLD`** | Sweep 0.4–0.7 on client audio; log per-match sims to pick a value. Confirm distinct vs same-speaker separation. |
| 5 | **Confidence scores** (client req) | Surface speaker confidence (cosine to matched centroid) + transcript confidence in the UI. Check what `diarize-service.py` already emits. |
| 6 | **Auto vs fixed speaker count** | Sanity-check behavior; decide default. Document guidance in the tab copy. |

## Test checklist for task 1

- Record ~2 min, two people alternating.
- During rec: labels switch between two speakers.
- After stop: still two speakers, not merged.
- Rename one → persists and re-labels its segments.
- Check `logs/diarize.log` for match sims and new-profile events.

## Files in play

- `scripts/diarize-service.py` — matcher (`ProfileStore.assign`), embeddings, final/chunk.
- `Audio/AudioRecorder.swift` — interval timers, overlap assignment, rename.
- `Views/Settings/DiarizationTab.swift` — toggles + interval; **needs profile-management section (task 2)**.
- `Transcription/SpeakerProfileStore.swift` — Swift read/rename/delete (has `delete`, needs "reset all").
- `Views/Main/TranscriptView.swift` — rename UI; **confidence display (task 5)**.
