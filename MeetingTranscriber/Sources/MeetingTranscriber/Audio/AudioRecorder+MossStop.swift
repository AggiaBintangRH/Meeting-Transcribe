import Foundation

// What the MOSS DIARIZATION engine does at Stop (`moss.finalPass`,
// `moss.continueOnStop`) — the pair the owner asked for after the same pair was
// built for the chunked ASR pass.
//
// SCOPE: this drives the SECOND MOSS process only, i.e. "another model does the
// ASR, MOSS labels the speakers". In MOSS+MOSS mode `mossDiarService` is nil —
// one process carries both jobs and its tail is the CHUNKED tail, already
// governed by `chunked.finalPass` / `chunked.continueOnStop`. Two settings pairs
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
        case tail   // flush the sidecar's live buffer — today's behaviour
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
        guard finalPass else { return .none }
        if continueOnStop { return .tail }
        // No recording on disk ⇒ nothing to re-read. Degrade to the tail rather
        // than to nothing, so a missing file never costs the user the tail.
        return hasRecording ? .full : .tail
    }

    /// Exactly what `stop()` does for a given mode.
    struct MossStopPlan: Equatable {
        /// Flush the live buffer for the tail window (today's behaviour).
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
        mossTurns = []
        rebuildDisplayRows()
        mossLog("FULL PASS start — \(windows.count) window(s) of "
                + "\(Int(Self.mossFullPassWindowSec))s over \(fmt(recordingElapsed))s")

        mossFullPassTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for (index, window) in windows.enumerated() {
                if Task.isCancelled { break }
                self.setStopStepName("moss-diarize",
                                     "Re-labelling speakers (\(index + 1)/\(windows.count))")
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
            self.setStopStepName("moss-diarize", "Labelling speakers")
            self.checkMossChunkDone()
        }
    }
}
