import Foundation

// Remote-stream diarization — the second identity space (ids >= remoteIDBase).
// Live chunks, the stop-pass decision and its two modes, their watchdog and the
// remote leg of the stop gate. Moved verbatim from the core file.
extension AudioRecorder {

    // MARK: - Remote diarization (the second identity space)

    /// Live: hand the remote audio accumulated since the last diarization
    /// boundary to the sidecar as a `remote` job. Mirrors `diarizeLiveChunk`,
    /// with no position/ATND involvement of any kind.
    func diarizeRemoteLiveChunk(windowStart: Double) {
        // THE SPECTRAL ENGINE HAS NO LIVE PATH AT ALL — its sidecar serves only
        // `cmd: "final"` and refuses anything else. Structurally it can never be
        // reached from here (`modelLoader.pyannote` is nil under that engine and
        // `SpectralService` has no chunk API), but the early-out below returns
        // WITHOUT clearing the buffer, so falling through would accumulate the
        // whole remote stream at ~230 MB/hour for a pass that reads the Remote WAV
        // and never looks at it. Same shape as the `chunkAudio` leak found on
        // 2026-07-31, caught here before it could ship.
        // NEMO takes the same guard for the same reason — see `diarizeLiveChunk`.
        // CAM++ takes the same guard for the same reason — see `diarizeLiveChunk`.
        guard !spectralDiarizationActive, !nemoDiarizationActive,
              !diarizenDiarizationActive, !camPlusDiarizationActive else {
            remoteDiarAudio = []; return
        }
        guard remoteStreamActive, let service = modelLoader.pyannote else { return }
        let liveOn = diarLiveEnabled
        guard liveOn else {
            // Exactly `diarizeLiveChunk`'s rule, and for the same reason: with live
            // labels off but "continue from live labels (tail only)" on, the whole
            // recording becomes the single tail diarized at stop, so the accumulated
            // audio must be KEPT. (Before the 2026-07-22 tail change this always
            // cleared, because the remote stop pass was always a full re-diarization
            // of the remote WAV — see `startRemoteDiarization`.)
            let continueOnStop = diarContinueOnStop
            if !(continueOnStop && !liveOn) { remoteDiarAudio = [] }
            return
        }
        let samples = remoteDiarAudio
        remoteDiarAudio = []
        guard samples.count > 16_000 else { return }   // skip chunks under 1s
        dispatchRemoteDiarChunk(samples: samples, windowStart: windowStart, service: service)
    }

    /// Shared: write a chunk of 16 kHz REMOTE samples to a temp WAV off-thread and
    /// hand it to the sidecar as a `remote` job. The remote twin of
    /// `dispatchDiarChunk` — live calls pass no failure handler (silent, as
    /// before); the stop-time tail passes one so the remote gate still settles.
    func dispatchRemoteDiarChunk(samples: [Float], windowStart: Double,
                                         service: PyannoteService,
                                         onDispatchFailure: (() -> Void)? = nil) {
        let detectOverlap = diarDetectOverlap
        Task.detached(priority: .utility) { [weak self] in
            guard let url = Self.writeTempWAV(samples: samples, prefix: "remote-diar") else {
                await MainActor.run { onDispatchFailure?() }
                return
            }
            await MainActor.run { [weak self] in
                self?.remoteChunkFileByWindow[windowStart] = url
            }
            service.diarizeChunk(audio: url, windowStart: windowStart,
                                 exclusive: !detectOverlap, stream: .remote)
        }
    }

    /// What the remote stop pass is for this session. Pure, so the branch itself
    /// is unit-testable without an engine, a sidecar or an Aggregate Device.
    enum RemoteStopMode: Hashable {
        case none   // no remote pass at all — the gate is never taken
        case tail   // diarize only the audio since the last remote live boundary
        case full   // re-diarize the whole Remote WAV
    }

    /// The stop-pass decision. Remote now honours `diarization.continueOnStop`
    /// exactly as Office does — see `startRemoteDiarization` for WHY this reverses
    /// the phase-4 rule.
    ///
    /// `supportsTail` is FALSE for the spectral engine, and it is a parameter
    /// rather than a branch at the call site so this stays the one function that
    /// enumerates the modes. Spectral has no chunk job to send and no live labels
    /// for a tail to continue from, so `continueOnStop` cannot be honoured there;
    /// it falls through to the full pass instead, which is the only thing that
    /// engine can do. It defaults to true so every pyannote call — and every
    /// existing test — is byte-for-byte unchanged.
    nonisolated static func remoteStopMode(finalPass: Bool,
                                           continueOnStop: Bool,
                                           remoteStreamActive: Bool,
                                           hasDiarizationService: Bool,
                                           hasRemoteRecording: Bool,
                                           tailSamples: Int,
                                           supportsTail: Bool = true) -> RemoteStopMode {
        guard finalPass, remoteStreamActive, hasDiarizationService else { return .none }
        if continueOnStop, supportsTail {
            // < 1 s of tail is not worth a job — the same early-out (and the same
            // 16 000-sample threshold) as `diarizeTailChunk`. Returning `.none`
            // rather than dispatching keeps the stop gate untaken, so there is
            // nothing left to settle.
            return tailSamples > 16_000 ? .tail : .none
        }
        return hasRemoteRecording ? .full : .none
    }

    /// At stop: the remote twin of the office stop branch, and it now honours the
    /// SAME `diarization.continueOnStop` setting.
    ///
    /// Phase 4 made this pass ALWAYS a full re-diarization of the Remote WAV, on
    /// the reasoning that a clean separate waveform clusters better globally. The
    /// owner overruled that on 2026-07-22 for a stronger reason: LABEL STABILITY.
    /// A full pass re-embeds voices the live passes already enrolled, and one real
    /// session produced a second profile (R2) for a voice that had matched profile
    /// 1 at sim=0.89 four seconds earlier — one person shown as two speakers, with
    /// the transcript rendering the final pass's mapping. Why that embedding
    /// collapsed below SIM_THRESHOLD=0.5 against a centroid it had just matched at
    /// 0.89 is UNEXPLAINED — nobody has accounted for it, so do not assume it was
    /// understood. Tail mode sidesteps it structurally: the tail never re-embeds
    /// already-enrolled voices, so it cannot mint a duplicate profile for them.
    ///
    /// Returns whether a pass was actually dispatched, so `stop()` can decide
    /// whether the overlay gets a remote-diarization row. This MUST stay decidable
    /// synchronously — `stop()` calls it before `buildStopSteps`.
    @discardableResult
    func startRemoteDiarization() -> Bool {
        let d = UserDefaults.standard
        let finalOn = d.object(forKey: "diarization.finalPass") as? Bool ?? true
        let continueOnStop = diarContinueOnStop
        let mode = Self.remoteStopMode(finalPass: finalOn,
                                       continueOnStop: continueOnStop,
                                       remoteStreamActive: remoteStreamActive,
                                       hasDiarizationService: modelLoader.pyannote != nil,
                                       hasRemoteRecording: remoteRecordingURL != nil,
                                       tailSamples: remoteDiarAudio.count)
        guard let service = modelLoader.pyannote, mode != .none else {
            // Nothing dispatched → the gate was never taken (`remoteFinalDiarDone`
            // is still true) and no overlay row is added. Drop any pending tail
            // audio so it cannot outlive the session.
            remoteDiarAudio = []
            return false
        }
        remoteFinalDiarDone = false
        switch mode {
        case .tail:  startRemoteTailDiarization(service: service)
        case .full:  startRemoteFullDiarization(service: service)
        case .none:  break   // unreachable, guarded above
        }
        return true
    }

    /// `continueOnStop == true`: diarize only the remote audio accumulated since
    /// the last remote live boundary, as a `chunk` job on the remote stream, so
    /// every label the live passes assigned survives Stop untouched. Mirrors
    /// `diarizeTailChunk`, including its window-start reasoning.
    func startRemoteTailDiarization(service: PyannoteService) {
        let samples = remoteDiarAudio
        remoteDiarAudio = []
        // Same rule as the office tail: with live labels on, the pending audio
        // began at the last live diarization boundary — which is the SAME
        // `lastDiarBoundary`, because remote rides the office diarization cadence
        // (see `diarizeRemoteLiveChunk`'s call site). With live off (+
        // continueOnStop) nothing was ever cleared, so this is the whole recording
        // and begins at 0.
        let liveOn = diarLiveEnabled
        let windowStart = liveOn ? lastDiarBoundary : 0
        awaitingRemoteTailWindowStart = windowStart
        // The tail is a chunk job, not a full pass, so it is bounded by the office
        // tail's limit rather than the recording length — but it still queues
        // BEHIND the office stop job on one stdin, hence the doubling.
        startRemoteDiarWatchdog(seconds: 240, message: "Remote tail diarization timed out")
        dispatchRemoteDiarChunk(samples: samples, windowStart: windowStart, service: service,
                                onDispatchFailure: { [weak self] in
                                    self?.dualStreamLog("could not write remote tail audio for diarization")
                                    self?.completeRemoteDiarization(
                                        error: "Could not write remote tail audio for diarization")
                                })
    }

    /// `continueOnStop == false`: one batch pass over the whole Remote WAV — the
    /// remote twin of `startDiarization`, and phase 4's original behaviour.
    func startRemoteFullDiarization(service: PyannoteService) {
        guard let recording = remoteRecordingURL else {
            // `remoteStopMode` already proved this non-nil; belt-and-braces so the
            // gate can never be left open by a future edit.
            completeRemoteDiarization(error: "Remote recording is missing")
            return
        }
        // AUTO, never the room's count — see `AudioRecorder.remoteNumSpeakers`.
        let numSpeakers = Self.remoteNumSpeakers
        let detectOverlap = diarDetectOverlap
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers,
                             exclusive: !detectOverlap, stream: .remote)
        // Same scale rule as the office final pass: the remote job is queued BEHIND
        // the office one on a single stdin, so it can legitimately wait out the
        // office pass before it even starts.
        startRemoteDiarWatchdog(seconds: max(180, recordingElapsed) * 2,
                                message: "Remote diarization timed out")
    }

    /// One watchdog for both remote stop modes — whichever path stalls, the gate
    /// still settles and the overlay still drops.
    func startRemoteDiarWatchdog(seconds: Double, message: String) {
        remoteFinalDiarWatchdog?.cancel()
        remoteFinalDiarWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.remoteFinalDiarDone else { return }
                self.completeRemoteDiarization(error: message)
            }
        }
    }

    /// The remote final result replaces the running remote set, exactly as
    /// `applyFinalSpeakers` does for Office — but into `remoteLiveTurns`, so the
    /// two spaces never share a collection.
    func applyRemoteFinalSpeakers(_ turns: [SpeakerTurn]) {
        let turns = Self.remoteTurnsOnly(turns, "applyRemoteFinalSpeakers")
        remoteSpeakerCount = Set(turns.map(\.id)).count
        remoteLiveTurns = turns
        rebuildDisplayRows()
    }

    /// Settle the remote leg of the stop gate exactly once. Idempotent; every
    /// remote diarization exit path (result, error, timeout) calls it.
    func completeRemoteDiarization(error: String? = nil) {
        remoteFinalDiarWatchdog?.cancel()
        remoteFinalDiarWatchdog = nil
        awaitingRemoteTailWindowStart = nil
        guard !remoteFinalDiarDone else { return }
        remoteFinalDiarDone = true
        setStopStep("remote-diarize", error.map { .failed($0) } ?? .done)
        checkStopProcessingDone()
    }
}
