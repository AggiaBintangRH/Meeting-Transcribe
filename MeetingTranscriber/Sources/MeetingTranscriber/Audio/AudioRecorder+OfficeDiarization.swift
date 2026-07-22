import AVFoundation
import Foundation

// Office-stream diarization: the session's service callbacks, the live/tail
// chunk dispatch, the stop-time passes and their completion gate. Also the
// shared `writeTempWAV` helper, used by the remote stream and remote
// diarization too. Moved verbatim from the core file.
extension AudioRecorder {

    // MARK: - Diarization

    /// Wire the diarization service callbacks for this session.
    func configureDiarization() {
        guard let service = modelLoader.diarization else { return }

        service.onChunkResult = { [weak self] windowStart, turns, stream in
            Task { @MainActor in
                guard let self else { return }
                // Turn times are chunk-local — offset to absolute recording time.
                let absolute = turns.map {
                    DiarizationService.Turn(start: $0.start + windowStart,
                                            end: $0.end + windowStart,
                                            id: $0.id, name: $0.name)
                }
                // The Remote branch returns early ON PURPOSE: it must not touch
                // liveTurns, sessionSpeakerIDs, speakerCount or the office tail
                // gate. Its ids are >= remoteIDBase and belong to the other space.
                guard stream == .office else {
                    if let file = self.remoteChunkFileByWindow.removeValue(forKey: windowStart) {
                        try? FileManager.default.removeItem(at: file)
                    }
                    let remote = Self.remoteTurnsOnly(absolute, "remote onChunkResult")
                    self.remoteLiveTurns.append(contentsOf: remote)
                    for turn in remote { self.remoteSessionSpeakerIDs.insert(turn.id) }
                    self.remoteSpeakerCount = self.remoteSessionSpeakerIDs.count
                    self.rebuildDisplayRows()
                    // Remote tail-mode stop: THIS chunk result is the remote tail —
                    // settle the remote leg of the gate. Matched on the remote
                    // window start only, so an office chunk sharing the same window
                    // can never settle it (and vice versa).
                    if let expected = self.awaitingRemoteTailWindowStart,
                       abs(windowStart - expected) < 0.001 {
                        self.completeRemoteDiarization()
                    }
                    return
                }
                // Clean up the temp chunk file
                if let file = self.chunkFileByWindow.removeValue(forKey: windowStart) {
                    try? FileManager.default.removeItem(at: file)
                }
                // Raw pyannote turns — pyannote is authoritative. Position labels
                // are folded in only at display time (derivedRows), never here.
                let office = Self.officeTurnsOnly(absolute, "office onChunkResult")
                self.liveTurns.append(contentsOf: office)
                for turn in office { self.sessionSpeakerIDs.insert(turn.id) }
                self.speakerCount = self.sessionSpeakerIDs.count
                self.rebuildDisplayRows()
                // Tail-only stop mode: this chunk result IS the tail — complete the gate.
                if let expected = self.awaitingTailWindowStart, abs(windowStart - expected) < 0.001 {
                    self.completeStopDiarization()
                }
            }
        }

        service.onFinalResult = { [weak self] turns, stream in
            Task { @MainActor in
                guard let self else { return }
                guard stream == .office else {
                    self.applyRemoteFinalSpeakers(turns)
                    self.completeRemoteDiarization()
                    return
                }
                self.applyFinalSpeakers(turns)
                self.diarizing = false
                self.finalDiarDone = true
                self.finalDiarWatchdog?.cancel()
                self.finalDiarWatchdog = nil
                self.setStopStep("diarize", .done)
                self.maybeStartOverlapRepair()
                self.checkStopProcessingDone()
            }
        }

        service.onError = { [weak self] message, stream in
            Task { @MainActor in
                guard let self else { return }
                // A remote job's failure settles only the remote leg. Remote
                // trouble must never cost the user their office transcript —
                // the same rule the remote chunk path already follows.
                guard stream == .office else {
                    self.dualStreamLog("remote diarization failed: \(message)")
                    self.completeRemoteDiarization(error: message)
                    return
                }
                self.diarizing = false
                self.diarizationError = message
                self.finalDiarDone = true
                self.awaitingTailWindowStart = nil
                self.diarTailWatchdog?.cancel()
                self.diarTailWatchdog = nil
                self.finalDiarWatchdog?.cancel()
                self.finalDiarWatchdog = nil
                self.setStopStep("diarize", .failed(message))
                self.maybeStartOverlapRepair()
                self.checkStopProcessingDone()
            }
        }
    }

    /// Live: write the current chunk's audio to a temp WAV and diarize it.
    func diarizeLiveChunk(windowStart: Double) {
        let liveOn = UserDefaults.standard.object(forKey: "diarization.live") as? Bool ?? true
        guard liveOn, modelLoader.diarization != nil else {
            // When live labels are off but "continue from live labels (tail only)"
            // is on, DON'T clear the accumulated audio: the whole recording then
            // becomes the single tail diarized at stop (matches the UI caption).
            let continueOnStop = UserDefaults.standard.object(forKey: "diarization.continueOnStop") as? Bool ?? true
            if !(continueOnStop && !liveOn) { chunkAudio = [] }
            return
        }
        let samples = chunkAudio
        chunkAudio = []
        guard samples.count > 16_000 else { return } // skip chunks under 1s
        dispatchDiarChunk(samples: samples, windowStart: windowStart)
    }

    /// Shared: write a chunk of 16 kHz samples to a temp WAV off-thread and hand
    /// it to the diarization sidecar. Live calls pass no failure handler (silent,
    /// as before); the stop-time tail passes one so the gate can still complete.
    func dispatchDiarChunk(samples: [Float], windowStart: Double,
                                   onDispatchFailure: (() -> Void)? = nil) {
        guard let service = modelLoader.diarization else { onDispatchFailure?(); return }
        Task.detached(priority: .utility) { [weak self] in
            guard let url = Self.writeTempWAV(samples: samples) else {
                await MainActor.run { onDispatchFailure?() }
                return
            }
            await MainActor.run { [weak self] in
                self?.chunkFileByWindow[windowStart] = url
            }
            let detectOverlap = UserDefaults.standard.object(forKey: "diarization.detectOverlap") as? Bool ?? true
            service.diarizeChunk(audio: url, windowStart: windowStart, exclusive: !detectOverlap)
        }
    }

    /// At stop (tail-only mode): diarize just the audio accumulated since the last
    /// live chunk and append it, keeping live speaker numbering stable. Routes the
    /// completion gate through `completeStopDiarization()`.
    func diarizeTailChunk() {
        guard modelLoader.diarization != nil else { completeStopDiarization(); return }
        let samples = chunkAudio
        chunkAudio = []
        // With live labels on, the pending audio began at the last live-chunk
        // boundary. With live off (+ continueOnStop), nothing was ever cleared so
        // chunkAudio is the whole recording — it begins at 0, and lastDiarBoundary
        // (which still advances in the tap) is stale, so don't use it.
        let liveOn = UserDefaults.standard.object(forKey: "diarization.live") as? Bool ?? true
        let windowStart = liveOn ? lastDiarBoundary : 0
        guard samples.count > 16_000 else { completeStopDiarization(); return } // <1s tail
        diarizing = true
        diarizationError = nil
        awaitingTailWindowStart = windowStart
        diarTailWatchdog?.cancel()
        diarTailWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(120))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.awaitingTailWindowStart != nil else { return }
                self.diarizationError = self.diarizationError ?? "Tail diarization timed out"
                self.completeStopDiarization()
            }
        }
        dispatchDiarChunk(samples: samples, windowStart: windowStart,
                          onDispatchFailure: { [weak self] in
                              self?.diarizationError = "Could not write tail audio for diarization"
                              self?.completeStopDiarization()
                          })
    }

    /// Finish the stop-time (tail) diarization gate exactly once, then let overlap
    /// repair proceed. Safe to call from any of the tail exit paths.
    func completeStopDiarization() {
        diarTailWatchdog?.cancel()
        diarTailWatchdog = nil
        awaitingTailWindowStart = nil
        diarizing = false
        finalDiarDone = true
        setStopStep("diarize", diarizationError.map { .failed($0) } ?? .done)
        maybeStartOverlapRepair()
        checkStopProcessingDone()
    }

    /// At stop: batch refinement over the full recording (best accuracy).
    func startDiarization(_ recording: URL) {
        let finalOn = UserDefaults.standard.object(forKey: "diarization.finalPass") as? Bool ?? true
        guard finalOn, let service = modelLoader.diarization else { return }
        diarizing = true
        diarizationError = nil
        let numSpeakers = UserDefaults.standard.integer(forKey: "diarization.numSpeakers")
        let detectOverlap = UserDefaults.standard.object(forKey: "diarization.detectOverlap") as? Bool ?? true
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers, exclusive: !detectOverlap)
        // A final pass over a long meeting legitimately takes a while, so scale
        // the limit with the recording — never below 3 minutes.
        let limit = max(180, recordingElapsed)
        finalDiarWatchdog?.cancel()
        finalDiarWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(limit))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.finalDiarDone else { return }
                self.diarizing = false
                self.diarizationError = "Final diarization timed out"
                self.finalDiarDone = true
                self.setStopStep("diarize", .failed("Final diarization timed out"))
                self.maybeStartOverlapRepair()
                self.checkStopProcessingDone()
            }
        }
    }

    /// Final pass: the globally-clustered pyannote turns replace the running live
    /// set. Raw pyannote (authoritative); position labels are folded in only at
    /// display time (derivedRows), so liveTurns stays pure pyannote.
    func applyFinalSpeakers(_ turns: [DiarizationService.Turn]) {
        let turns = Self.officeTurnsOnly(turns, "applyFinalSpeakers")
        speakerCount = Set(turns.map(\.id)).count
        liveTurns = turns
        rebuildDisplayRows()
    }

    /// Write 16 kHz mono float samples to a temp WAV (chunk diarization, and the
    /// remote chunk handed to the sidecar's file-transcribe frame — `prefix` only
    /// names the file, so the two are told apart in the temp dir).
    /// Pure/self-contained, so it runs off the main actor from the detached task.
    nonisolated static func writeTempWAV(samples: [Float],
                                                 prefix: String = "diar-chunk") -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString).wav")
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: 16_000, channels: 1,
                                         interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(samples.count))
        else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            buffer.floatChannelData?[0].update(from: source.baseAddress!, count: samples.count)
        }
        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
            return url
        } catch {
            return nil
        }
    }
}
