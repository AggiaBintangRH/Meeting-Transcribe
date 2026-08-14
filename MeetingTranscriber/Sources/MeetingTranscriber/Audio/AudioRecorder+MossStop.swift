import Foundation

// What the MOSS DIARIZATION engine does at Stop (`moss.finalPass`,
// `moss.continueOnStop`) — the pair the owner asked for after the same pair was
// built for the chunked ASR pass.
//
// SCOPE: this drives the SECOND MOSS process only, i.e. "another model does the
// ASR, MOSS labels the speakers". In MOSS+MOSS mode `mossDiarService` is nil —
// one process carries both jobs and its tail is the CHUNKED tail, already
// governed by `chunked.finalPass` (the tail/full choice was retired 2026-08-06 —
// the chunked stop pass is always tail-only now). Two settings pairs
// for one process would be two answers to one question.
//
// WHY THE FULL PASS USES A LONGER WINDOW, AND WHY THAT IS THE WHOLE POINT
// ----------------------------------------------------------------------
// MOSS is byte-deterministic on MPS with `do_sample=False` — measured during the
// 2026-07-31 service split, where old and new sidecars produced identical md5s
// across separate runs. So re-running the SAME windows at Stop would return the
// SAME labels, exactly. A "re-diarize" button that cannot change its answer is
// the dead control this project has already refused three times (Granite's
// language picker, MOSS's, Voxtral's).
//
// What DOES change the answer is the window length, and that is measured too. On
// a 3-speaker recording (78.9 s), 30 s chunks returned ['S01'], ['S01'],
// ['S01','S02'] — the first two chunks are DIFFERENT PEOPLE both numbered S01,
// because the chunk boundary fell on the speaker change. One pass over the same
// audio returned ['S01','S02','S03'], all three separated. MOSS clusters within
// whatever it is shown; chunking is what breaks it.
//
// So the full pass re-runs at `mossFullPassWindowSec` (120 s) rather than the ASR
// cadence. 120 s and not more: audio costs ~13.2 context tokens per second, so
// 120 s uses ~1 638 of the model's 2 048-token budget, while 360 s was measured
// to hit the cap exactly and silently drop the truncated tail.
//
// It is fed with `feed()` + `flush()`, NOT a file frame: `moss-diar-service.py`
// deliberately has no `-2` FILE-TRANSCRIBE branch (removed in phase 2 of the
// service split and pinned by `moss/diar-has-no-file-branch`), and restoring one
// to serve this would undo that.
extension AudioRecorder {

    /// Window length for the stop-time re-diarization, in seconds. See the file
    /// comment: the LENGTH is the feature, and 120 s is the measured ceiling.
    nonisolated static let mossFullPassWindowSec: Double = 120

    /// What the MOSS diarization engine does at Stop.
    enum MossStopMode: Hashable {
        case none   // no MOSS pass at stop; the last live chunk ends the labels
        case tail   // flush the sidecar's live buffer (stop pass OFF)
        case full   // re-diarize the whole recording in 120 s windows
    }

    /// Pure so the decision is testable without a recorder or a sidecar, the same
    /// shape as `chunkedStopMode`.
    ///
    /// `hasDiarService` false covers BOTH the pyannote engine and MOSS+MOSS: in
    /// neither case is there a second process for these settings to govern, and
    /// `.none` there is not a degradation — it is what already happens.
    nonisolated static func mossStopMode(finalPass: Bool,
                                         continueOnStop: Bool,
                                         hasDiarService: Bool,
                                         hasRecording: Bool) -> MossStopMode {
        guard hasDiarService else { return .none }
        // ⚠ THE TWO SETTINGS NO LONGER OVERLAP (owner, 2026-08-14) — see
        // `remoteStopMode` for the failure that prompted it. `finalPass` ON is
        // ALWAYS a full pass; `continueOnStop` is read only where its toggle is
        // visible, i.e. with the stop pass OFF.
        //
        // ⚠ THIS CHANGES MOSS'S SHIPPED DEFAULT, and that is the point rather than
        // a side effect: `moss.continueOnStop` defaults to TRUE, so a default MOSS
        // session used to end in a tail and now ends in a full re-diarization. The
        // owner's rule was stated for the control, and one control governs both
        // pairs — a MOSS exception would be the same setting meaning two things.
        if !finalPass { return continueOnStop ? .tail : .none }
        // No recording on disk ⇒ nothing to re-read. Degrade to the tail rather
        // than to nothing, so a missing file never costs the user the tail.
        return hasRecording ? .full : .tail
    }

    /// Exactly what `stop()` does for a given mode.
    struct MossStopPlan: Equatable {
        /// Flush the live buffer for the tail window. NO LONGER THE DEFAULT —
        /// both keys absent gives `.full` since 2026-08-14; see `mossStopMode`.
        var flushesTail = false
        /// Settle `mossLastChunkDone` right here, because nothing async will.
        var settlesImmediately = false
        /// Drive `startMossFullPass`.
        var runsFullPass = false
    }

    nonisolated static func mossStopPlan(_ mode: MossStopMode) -> MossStopPlan {
        switch mode {
        case .tail:
            return MossStopPlan(flushesTail: true, settlesImmediately: false, runsFullPass: false)
        case .full:
            return MossStopPlan(flushesTail: false, settlesImmediately: false, runsFullPass: true)
        case .none:
            // `configureMoss` already sets `mossLastChunkDone = mossDiarService == nil`,
            // so with no service this is a no-op; with a service but the pass
            // switched off, THIS is what completes the leg.
            return MossStopPlan(flushesTail: false, settlesImmediately: true, runsFullPass: false)
        }
    }

    /// Re-diarize the whole recording in `mossFullPassWindowSec` windows.
    ///
    /// SEQUENTIAL, one window in flight at a time, and that is load-bearing
    /// rather than tidy: the sidecar caps its own buffer at `MAX_BUFFER_SEC`
    /// (300 s), so feeding an hour of audio ahead of the flushes would silently
    /// TRIM the front of it. Each window is fed, flushed, and awaited before the
    /// next is read.
    func startMossFullPass(recording: URL) {
        guard let service = mossDiarService else { checkMossChunkDone(); return }
        // Cut at silence, same as the chunked full pass and same as the live
        // path. It matters here too: MOSS transcribes as well as labels, so a
        // boundary through a word costs both the words and the speaker span.
        let energies = Self.fullPassEnergyProfile(from: recording)
        let windows = energies.map {
            Self.fullPassWindowsAtSilence(recordingLength: recordingElapsed,
                                          intervalSec: Self.mossFullPassWindowSec,
                                          energies: $0)
        } ?? Self.fullPassWindows(recordingLength: recordingElapsed,
                                  intervalSec: Self.mossFullPassWindowSec)
        guard !windows.isEmpty else {
            mossLog("FULL PASS skipped — only \(fmt(recordingElapsed))s recorded, nothing to re-label")
            checkMossChunkDone()
            return
        }

        // The live labels are about to be replaced wholesale. Cleared up front so
        // a stale set is not shown underneath the transcript for the whole pass.
        //
        // Anything ALREADY in flight belongs to the live pass and describes audio
        // this pass is about to re-label. Counted here, before a single window is
        // queued, so the replies can be dropped as they land instead of being
        // merged into the set just cleared. See `AudioRecorder.mossStaleReplies`
        // for the log that shows what this cost: 36 turns applied twice.
        mossStaleReplies = mossPendingWindows.count
        if mossStaleReplies > 0 {
            mossLog("FULL PASS will discard \(mossStaleReplies) live reply(ies) "
                    + "still in flight")
        }
        mossTurns = []
        rebuildDisplayRows()
        mossLog("FULL PASS start — \(windows.count) window(s) of "
                + "\(Int(Self.mossFullPassWindowSec))s over \(fmt(recordingElapsed))s")

        mossFullPassTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for (index, window) in windows.enumerated() {
                if Task.isCancelled { break }
                self.setStopStepName("moss-diarize",
                                     "Identifying speakers (\(index + 1)/\(windows.count))")
                let samples = await Task.detached(priority: .utility) {
                    Self.loadWindow16k(from: recording,
                                       start: window.lowerBound, end: window.upperBound)
                }.value
                guard let samples, !samples.isEmpty else {
                    self.mossLog("FULL PASS [\(self.fmt(window.lowerBound))-"
                                 + "\(self.fmt(window.upperBound))] unreadable — skipped")
                    continue
                }
                service.feed(samples)
                self.flushMossDiarChunk(window: window)
                // Wait for THIS window to settle before reading the next. The
                // callbacks installed by `installMossDiarizationCallbacks` pop
                // `mossPendingWindows`, and `flushMossDiarChunk`'s own 180 s
                // watchdog pops it on a hang — so this cannot wait forever.
                while !self.mossPendingWindows.isEmpty, !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(100))
                }
            }
            self.mossFullPassTask = nil
            self.mossLog("FULL PASS done — \(self.mossTurns.count) turns")
            await self.identifyMossTurns(recording: recording)
            self.setStopStepName("moss-diarize", "Labelling speakers")
            self.checkMossChunkDone()
        }
    }

    // MARK: - Identity: what turns RE-LABELLING into RECOGNISING

    /// `identify` ONE WINDOW AT A TIME, in order — not once over the meeting.
    ///
    /// WHY THIS EXISTS. A MOSS label is anonymous per model CALL: `S01` in window
    /// 1 is not `S01` in window 2, and the model never claimed it was. Measured on
    /// a real 4.3-minute recording, window 1's `S01` turned out to be the same
    /// person as window 0's **`S02`** (cosine 0.90) and its `S04` the same as
    /// window 0's `S01` (0.89). Without identity those read as four people; with
    /// it, 7 per-window labels collapse to 5. That permutation is exactly what
    /// makes a MOSS transcript look incoherent at every seam.
    ///
    /// **ONE CALL PER WINDOW, and the first version of this got it backwards.**
    /// It sent the whole meeting in a single `identify` and stitched NOTHING —
    /// 7 labels became 7 profiles, every one brand new. The reason is in
    /// `ProfileStore.assign`: it snapshots `self.centroids` BEFORE scoring and
    /// enforces a one-to-one mapping, so two labels in the SAME call can never
    /// land on one profile — by design, since within one diarization run two
    /// distinct labels really are two distinct voices. That guarantee is what
    /// makes a per-run call correct for pyannote, where a run is one chunk whose
    /// labels are already internally consistent. **A MOSS WINDOW IS THE
    /// EQUIVALENT OF A PYANNOTE CHUNK**, so windows must be separate calls: only
    /// then do window 0's profiles exist in the store when window 1 is scored.
    ///
    /// Grouped by the chunk index already embedded in the MOSS wire id
    /// (`mossIDBase + chunkIndex * 100 + local`), so no second bookkeeping
    /// structure is needed and the order is the recording's own.
    ///
    /// No temp WAV: the full pass already runs over the recording and its turns
    /// are in recording time, so every call reads the same file.
    ///
    /// **An unidentified turn KEEPS its per-window label.** `composeTurns`
    /// `compactMap`s unmatched turns away, which is right for pyannote and would
    /// be over-deletion here: a MOSS turn under `MIN_EMBED_SEC` carries real
    /// transcribed words, and dropping it would delete them with no trace.
    /// A failed call is likewise non-fatal — that window keeps the labels it had.
    /// MOSS+MOSS has NO full pass of its own, so identify had no caller there.
    ///
    /// THE REGRESSION THIS FIXES, found by the owner's recording on 2026-08-06.
    /// `identifyMossTurns` is driven by `startMossFullPass`, and `mossStopMode`
    /// returns `.none` when `hasDiarService` is false — which is exactly the
    /// MOSS+MOSS case, because one sidecar serves both roles and `mossDiarService`
    /// is nil by design. So identity ran only in the OTHER mode: MOSS diarizing
    /// beside a different ASR.
    ///
    /// That was survivable until the same day's MOSS⟺MOSS rule made MOSS+MOSS the
    /// ONLY reachable pairing — at which point stitching, saved profiles, renaming
    /// and `spk` all became unreachable under MOSS, silently. The log said so:
    /// `chunk #0 [0.0-48.3] 2 rows` with speaker ids still at 1 000 001, and no
    /// `IDENTIFY` line at all.
    ///
    /// Hooked to the CHUNKED stop pass instead, because in this mode that pass IS
    /// the last thing to add turns — `applyMossChunk` appends on every chunk
    /// including the tail. Returns true when it took over, so `checkLastChunkDone`
    /// knows to let the completion handler finish the gate rather than doing it
    /// twice.
    @discardableResult
    func startMossIdentifyForOwnASR() -> Bool {
        guard mossDiarizationActive, mossIsChunkedModel,
              !mossIdentifyStarted, !mossTurns.isEmpty,
              let recording = lastRecordingURL else { return false }
        mossIdentifyStarted = true
        setStopStepName("chunk", "Labelling speakers")
        Task { @MainActor [weak self] in
            guard let self else { return }
            await self.identifyMossTurns(recording: recording)
            self.rebuildDisplayRows()
            // Repair runs AFTER identity on purpose: its windows are attributed by
            // speaker id, and after this those ids are profile ids that mean the
            // same person across windows.
            self.maybeStartOverlapRepair()
            self.checkStopProcessingDone()
        }
        return true
    }

    func identifyMossTurns(recording: URL) async {
        guard mossDiarizationActive, !mossTurns.isEmpty else { return }
        guard let embedding = modelLoader.embedding else {
            mossLog("IDENTIFY skipped — no embedder in this session; speaker numbers "
                    + "stay per-window")
            return
        }
        let before = Set(mossTurns.map(\.id)).count
        // Chunk index out of the wire id, so the grouping cannot drift from the
        // ids the turns already carry.
        let windows = Dictionary(grouping: mossTurns.indices) {
            (mossTurns[$0].id - Self.mossIDBase) / 100
        }
        var identified = 0
        for key in windows.keys.sorted() {
            let idxs = windows[key] ?? []
            let locals = idxs.map {
                PyannoteService.LocalTurn(start: mossTurns[$0].start,
                                          end: mossTurns[$0].end,
                                          label: "M\(mossTurns[$0].id)")
            }
            do {
                let identity = try await embedding.identify(audio: recording.path,
                                                            turns: locals, stream: .office)
                for (offset, i) in idxs.enumerated() {
                    guard let who = identity[locals[offset].label] ?? nil else { continue }
                    identified += 1
                    mossTurns[i] = SpeakerTurn(start: mossTurns[i].start,
                                               end: mossTurns[i].end,
                                               id: who.id, name: who.name, conf: who.conf)
                }
            } catch {
                mossLog("IDENTIFY window \(key) FAILED — \(error.localizedDescription); "
                        + "its speaker numbers stay per-window")
            }
        }
        let after = Set(mossTurns.map(\.id)).count
        mossLog("IDENTIFY done — \(identified)/\(mossTurns.count) turns matched across "
                + "\(windows.count) window(s), \(before) per-window labels collapsed to "
                + "\(after) people")
        rebuildDisplayRows()
    }

}
