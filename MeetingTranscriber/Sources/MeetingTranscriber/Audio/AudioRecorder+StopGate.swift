import Foundation

// The stop-processing gate: the overlay's step list, the five completion flags
// it waits on, the last-resort watchdog and the escape hatch.
//
// `checkStopProcessingDone()` drops the overlay only once ALL FIVE are settled,
// and every path that settles one ends by calling it. Where each flag is
// written, by file:
//
//   lastChunkDone         +StopGate           checkLastChunkDone (the only
//                                             settle; the chunk watchdog routes
//                                             through it)
//                         AudioRecorder       beginCapture (session reset);
//                                             stop() when there is no chunked model
//   remoteLastChunkDone   +RemoteStream       checkRemoteChunksDone;
//                                             startRemoteStopWatchdog
//                         AudioRecorder       declared true; beginCapture (reset to
//                                             !remoteStreamActive, failed-start path)
//   finalDiarDone         +OfficeDiarization  configureDiarization's onFinalResult
//                                             and onError; completeStopDiarization;
//                                             startDiarization's watchdog
//                         AudioRecorder       beginCapture (reset); stop() when no
//                                             stop pass runs
//   mossLastChunkDone     +Moss               checkMossChunkDone (the only settle;
//                                             the MOSS chunk watchdog routes
//                                             through it)
//                         AudioRecorder       declared true; configureMoss (reset
//                                             to `mossDiarService == nil`, so a
//                                             session without a SECOND MOSS
//                                             process is complete from the start)
//   remoteFinalDiarDone   +RemoteDiarization  startRemoteDiarization TAKES the gate;
//                                             completeRemoteDiarization is the ONLY
//                                             settle — result, error, dispatch
//                                             failure and startRemoteDiarWatchdog
//                                             all route through it
//                         AudioRecorder       declared true; beginCapture (reset)
//   repair flag           +OverlapRepair      `overlapRepairing` + `repairTask`:
//                                             taken by maybeStartOverlapRepair,
//                                             cleared at the end of runOverlapRepair
//                                             and runDicowRepair; finishRepairStep
//                                             covers every bail-out before a driver
//                                             started
//                         AudioRecorder       beginCapture (reset)
//
// `startStopWatchdog` is the last-resort backstop if any of that ever fails.
//
// Moved verbatim from the core file; the comment above is the only new text.
extension AudioRecorder {

    // MARK: - Stop processing gate (the blocking overlay)

    /// The legs the overlay lists for this stop, in the order they finish.
    /// `repair` only appears when the feature is on AND its engine loaded —
    /// otherwise there is nothing to wait for.
    func buildStopSteps(willRunStopPass: Bool, willRunRemoteDiar: Bool = false,
                        willRunMossDiar: Bool = false) {
        var steps = [StopStep(id: "chunk", name: "Transcribing final audio",
                              state: lastChunkDone ? .done : .loading)]
        // Remote only appears for a dual-stream session. Its state is read the
        // same way the chunk step's is, because the remote tail may already have
        // completed (or been gated as silent) before this list is built.
        if remoteStreamActive {
            steps.append(StopStep(id: "remote", name: "Transcribing remote audio",
                                  state: remoteLastChunkDone
                                      ? (remoteChunkError.map { .failed($0) } ?? .done)
                                      : .loading))
        }
        if willRunStopPass {
            steps.append(StopStep(id: "diarize", name: "Identifying speakers", state: .loading))
        }
        // Its own row: Office and Remote are separate identity spaces, so their
        // progress is separate too — and a failed remote pass must read as a
        // remote failure, not as "speakers could not be identified".
        if willRunRemoteDiar {
            steps.append(StopStep(id: "remote-diarize",
                                  name: "Identifying remote speakers", state: .loading))
        }
        // Only in the diarization-only mode (another model does the ASR, a second
        // MOSS process labels). With MOSS in BOTH roles there is one process and
        // one flush, already covered by the "chunk" row above — a second row for
        // the same work would say two things are happening when one is.
        if willRunMossDiar {
            steps.append(StopStep(id: "moss-diarize",
                                  name: "Identifying speakers (MOSS)", state: .loading))
        }
        if overlapRepairWillRun {
            steps.append(StopStep(id: "repair", name: "Repairing overlapping speech",
                                  state: .pending))
        }
        stopSteps = steps
    }

    func setStopStep(_ id: String, _ state: ModelLoader.ItemState) {
        guard let i = stopSteps.firstIndex(where: { $0.id == id }) else { return }
        stopSteps[i].state = state
    }

    /// Re-word a step while it runs. `ItemState` carries no progress payload, so
    /// a long-running leg reports where it is through its NAME — the stop-time
    /// full re-transcription counts its windows here ("Re-transcribing the
    /// recording (7/120)") and restores the resting name when it finishes.
    func setStopStepName(_ id: String, _ name: String) {
        guard let i = stopSteps.firstIndex(where: { $0.id == id }) else { return }
        guard stopSteps[i].name != name else { return }
        stopSteps[i].name = name
    }

    /// Whether a repair leg is worth listing — mirrors `maybeStartOverlapRepair`'s
    /// feature + engine checks, without the gates that are still pending at stop.
    var overlapRepairWillRun: Bool {
        let d = UserDefaults.standard
        guard d.object(forKey: "overlap.repair.enabled") as? Bool ?? false else { return false }
        // Both engines find their windows in pyannote turns, of which there are
        // none under the MOSS engine — `ModelLoader.buildSteps` does not even
        // load one. Checked here too because the services outlive a session: one
        // left over from a previous pyannote meeting would otherwise put a
        // "Repairing overlapping speech" row on the overlay for work that cannot
        // happen.
        guard !mossDiarizationActive else { return false }
        let engine = d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id
        return engine == ModelCatalog.overlapDicow.id
            ? modelLoader.dicowRepair != nil
            : modelLoader.overlapRepair != nil
    }

    /// All post-stop work landed → drop the overlay and re-enable Start.
    /// Idempotent; every leg's completion path calls it.
    func checkStopProcessingDone() {
        guard state == .processing, lastChunkDone, remoteLastChunkDone,
              finalDiarDone, remoteFinalDiarDone, mossLastChunkDone,
              !overlapRepairing, repairTask == nil else { return }
        stopWatchdog?.cancel()
        stopWatchdog = nil
        state = .idle
    }

    /// Last resort: never hold the controls hostage. The background work keeps
    /// running and still updates the transcript if it lands.
    /// `seconds` is scaled up by `stop()` when a full re-transcription runs: that
    /// pass legitimately takes minutes, and a 600 s backstop firing in the middle
    /// of it would mark healthy work "timed out" while it carried on running.
    func startStopWatchdog(seconds: Double = 600) {
        stopWatchdog?.cancel()
        stopWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.state == .processing else { return }
                for i in self.stopSteps.indices
                where self.stopSteps[i].state == .pending || self.stopSteps[i].state == .loading {
                    self.stopSteps[i].state = .failed("timed out")
                }
                self.stopWatchdog = nil
                self.state = .idle
            }
        }
    }

    /// Escape hatch offered by the overlay after a while: stop blocking, cancel
    /// nothing. Degrades to the pre-overlay behaviour — work continues silently.
    func continueInBackground() {
        guard state == .processing else { return }
        stopWatchdog?.cancel()
        stopWatchdog = nil
        state = .idle
    }

    /// Last chunk finished after stop → maybe kick off overlap repair.
    func checkLastChunkDone() {
        guard stopped, pendingChunkWindows.isEmpty, !chunkedBusy, !lastChunkDone else { return }
        lastChunkDone = true
        setStopStep("chunk", chunkedError.map { .failed($0) } ?? .done)
        // Recording is fully processed — drop any leftover realtime (unconfirmed)
        // segments so a fragment from just before Stop can't survive as an orphan
        // "SPEAKER UNKNOWN" row (e.g. if the last chunk errored/timed out before its
        // own cleanup could run). The chunked pass is authoritative for all audio.
        //
        // …but ONLY when a pass actually ran. With `chunked.finalPass` off there
        // is no authoritative text for the tail, and sweeping would silently
        // delete up to 1.5× the chunk interval of real speech — 45 s at the
        // default, 180 s at a 120 s interval — with no trace in the transcript.
        // `chunkedSweepsUnconfirmed` is written once per session in `stop()` from
        // `chunkedStopPlan(_:)`; see AudioRecorder+ChunkedStop.swift.
        if chunkedSweepsUnconfirmed, segments.contains(where: { !$0.confirmed }) {
            segments.removeAll { !$0.confirmed }
            rebuildDisplayRows()
        }
        maybeStartOverlapRepair()
        checkStopProcessingDone()
    }
}
