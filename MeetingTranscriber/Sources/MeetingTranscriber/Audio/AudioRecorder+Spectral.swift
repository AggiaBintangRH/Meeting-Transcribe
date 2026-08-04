import Foundation

// The SPECTRAL diarization engine (`diarization.engine == "spectral"`) — session
// configuration, the two whole-file stop passes (office and remote), their
// watchdogs, the startup refusal, and this domain's own log writer.
//
// WHAT MAKES THIS ENGINE DIFFERENT FROM THE OTHER TWO
// ---------------------------------------------------
// pyannote diarizes live 30 s windows AND does a stop pass. MOSS labels as it
// transcribes, per chunk. Spectral does NEITHER: it counts speakers globally
// (GMM-BIC over every embedding in the file) and clusters globally, so a 30 s
// window would be counted and clustered on its own and its labels would mean
// nothing across windows — the exact failure MOSS has when it is called per
// chunk (two different people both numbered `S01` because the boundary fell on
// the speaker change). The sidecar therefore serves ONLY `cmd: "final"` and
// refuses anything else loudly; `SpectralService` has no chunk API at all.
//
// So the whole session is: record with no speaker labels, then one pass over the
// recording at Stop. That is a real degradation against pyannote and it is stated
// rather than hidden — see `spectralRefusalMessage` for the one configuration
// where it becomes a total absence and is refused instead.
//
// WHAT IT KEEPS, which is the payoff of the 2026-07-30 pyannote/wespeaker split:
// its labels are run-LOCAL (`SPEAKER_00`), so they go through the SAME
// `identify` → `composeTurns` path pyannote's do. Saved voice profiles, renaming
// and `spk` confidence all work under this engine with no new identity code.
extension AudioRecorder {

    // MARK: - Session configuration

    /// Read the engine choice for this session and wire the spectral callbacks.
    /// Called from `beginCapture` alongside `configureDiarization` and
    /// `configureMoss`, so the engine is fixed for the whole recording like every
    /// other setting — half a transcript labelled by each engine is not a state
    /// any display path can render honestly.
    func configureSpectral() {
        let d = UserDefaults.standard
        let engine = d.string(forKey: "diarization.engine") ?? ModelLoader.pyannoteEngineID
        let diarOn = d.object(forKey: "diarization.enabled") as? Bool ?? true

        spectralDiarizationActive = diarOn && engine == ModelLoader.spectralEngineID
        guard spectralDiarizationActive, let service = modelLoader.spectral else { return }

        spectralLog("engine=spectral — no live labels this session; "
                    + "one whole-file pass per stream at Stop")

        // The SAME two handlers pyannote's final replies use, because the wire
        // shape is the same and the identity stage is shared. `identifyFinalTurns`
        // and `handleDiarizationFailure` already settle every gate correctly for
        // both streams; duplicating either here is how the two would drift.
        service.onFinalResult = { [weak self] audioPath, localTurns, stream in
            Task { @MainActor in
                guard let self else { return }
                await self.identifyFinalTurns(audio: audioPath, localTurns: localTurns,
                                              stream: stream)
            }
        }
        service.onError = { [weak self] message, stream in
            Task { @MainActor in
                self?.spectralLog("FAILED (\(stream == .office ? "office" : "remote")): \(message)")
                self?.handleDiarizationFailure(message, stream: stream)
            }
        }
    }

    // MARK: - Refusal (refuse loudly, never substitute)

    /// Why this spectral configuration cannot work, or nil to start normally.
    ///
    /// **THE ONE REFUSAL: `diarization.finalPass` OFF.**
    ///
    /// For pyannote and MOSS, turning the stop pass off is a *degradation* — the
    /// live labels are kept and the transcript still names its speakers. Spectral
    /// has no live path, so the stop pass is not a refinement of the labels, it IS
    /// the labels. With it off the user asks for speaker diarization, waits out a
    /// model load, records a whole meeting, and gets a transcript with no speakers
    /// at all — and nothing anywhere says why.
    ///
    /// Refusal rather than silently forcing the pass on (which would ignore a
    /// setting the user deliberately set) and rather than silently substituting
    /// pyannote (the house rule this project has followed for Voxtral + dual
    /// stream, Voxtral + MOSS and the chunked full pass). Checked in
    /// `prepareAndCapture` beside those three, so it is said BEFORE the meeting
    /// rather than discovered after it.
    ///
    /// `diarizationEnabled` gates the whole rule: turning diarization off entirely
    /// is a coherent choice and must keep working whatever engine is selected.
    nonisolated static func spectralRefusalMessage(diarizationEngine: String,
                                                   diarizationEnabled: Bool,
                                                   finalPass: Bool) -> String? {
        guard diarizationEnabled, diarizationEngine == ModelLoader.spectralEngineID else {
            return nil
        }
        guard !finalPass else { return nil }
        return "The spectral engine only diarizes at the end of a recording — it clusters "
             + "speakers over the whole file at once and has no live pass, so with "
             + "\"Run a diarization pass at stop\" switched off this meeting would produce no "
             + "speaker labels at all. Turn that pass back on in Settings → Models → "
             + "Diarization, pick the pyannote engine (which does label live), or turn speaker "
             + "diarization off entirely."
    }

    // MARK: - The office whole-file pass

    /// Whether a spectral session dispatches its office whole-file pass at Stop.
    ///
    /// PURE and static, like `remoteStopMode` and `mossStopMode`, so the rule can
    /// be tested without a sidecar — and so `stop()` reads ONE condition rather
    /// than growing a second one that disagrees with `spectralRefusalMessage`.
    ///
    /// **`continueOnStop` IS DELIBERATELY NOT A PARAMETER.** Taking it and then
    /// ignoring it would imply the tail was considered and rejected per session;
    /// it is not available at all. This engine has no chunk job to send and, since
    /// it has no live path, no live labels for a tail to continue from — so the
    /// full pass is the only mode, and the setting simply does not reach here.
    /// The remote twin says the same thing in its own vocabulary, by passing
    /// `supportsTail: false` to `remoteStopMode`.
    ///
    /// `finalPass` is still asked even though `spectralRefusalMessage` refuses that
    /// combination at start: the settings are re-read at Stop, and a rule that
    /// trusted an earlier check would dispatch a pass the user had since switched
    /// off. Asking twice costs nothing; disagreeing would hang a stop leg.
    nonisolated static func runsSpectralOfficePass(spectralActive: Bool,
                                                   finalPass: Bool,
                                                   hasService: Bool,
                                                   hasRecording: Bool) -> Bool {
        spectralActive && finalPass && hasService && hasRecording
    }

    /// Seconds to allow one spectral whole-file pass over `recordingLength`
    /// seconds of audio.
    ///
    /// Same floor and shape as `startDiarization`'s `max(180, recordingElapsed)`
    /// and `tailDiarWatchdogSeconds`: a generous constant, then at least 1×
    /// realtime for whatever is actually being processed. The multiplier is above
    /// 1 because this engine runs entirely on **CPU** (the vendored embedder pins
    /// `torch.device("cpu")`) while pyannote's pass runs on MPS, so 1× realtime is
    /// a far tighter bound here than it is there. It has NOT been measured on a
    /// real meeting on this hardware — the number is a deliberately loose backstop,
    /// not a prediction, and the whole point of a watchdog is that being wrong in
    /// this direction only costs patience.
    nonisolated static func spectralWatchdogSeconds(recordingLength: Double) -> Double {
        max(180, recordingLength * 2)
    }

    /// At Stop: one batch pass over the whole office recording.
    ///
    /// Reuses the pyannote final pass's gate exactly — `finalDiarDone`,
    /// `finalDiarWatchdog`, the `"diarize"` step, and `applyFinalTurns` as the
    /// settle — so there is one "Identifying speakers" leg however it is served.
    /// A second parallel gate for a second engine is how a stop overlay gets stuck.
    func startSpectralDiarization(_ recording: URL) {
        guard let service = modelLoader.spectral else {
            // Unreachable through `stop()` (which tests the service first), kept
            // so the leg can never be left un-settleable by a future edit.
            diarizationError = "The spectral diarization sidecar is not running"
            finalDiarDone = true
            setStopStep("diarize", .failed(diarizationError ?? ""))
            checkStopProcessingDone()
            return
        }
        diarizing = true
        diarizationError = nil
        let numSpeakers = UserDefaults.standard.integer(forKey: "diarization.numSpeakers")
        // `diarization.detectOverlap` is NOT sent, and that is not an omission.
        // This engine's Viterbi smoothing assigns exactly one label per frame, so
        // it cannot report two speakers at one instant whatever it is asked. See
        // `ModelLoader.wantedOverlapEngine` for the consequence.
        spectralLog("FINAL PASS start — \(fmt(recordingElapsed))s of office audio, "
                    + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers)

        let limit = Self.spectralWatchdogSeconds(recordingLength: recordingElapsed)
        finalDiarWatchdog?.cancel()
        finalDiarWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(limit))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.finalDiarDone else { return }
                self.spectralLog("FINAL PASS TIMEOUT after \(Int(limit))s")
                self.diarizing = false
                self.diarizationError = "Spectral diarization timed out"
                self.finalDiarDone = true
                self.setStopStep("diarize", .failed("Spectral diarization timed out"))
                self.maybeStartOverlapRepair()
                self.checkStopProcessingDone()
            }
        }
    }

    // MARK: - The remote whole-file pass

    /// At Stop: the remote twin, over the Remote WAV.
    ///
    /// **ALWAYS A FULL PASS, never a tail**, whatever `diarization.continueOnStop`
    /// says — there is no chunk job to send, and no live labels for a tail to
    /// continue from. `remoteStopMode` is told that with `supportsTail: false`
    /// rather than being second-guessed here, so the branch stays in the one pure
    /// function that enumerates it.
    ///
    /// Returns whether a pass was dispatched, so `stop()` can decide whether the
    /// overlay gets a remote-diarization row — the same synchronous contract
    /// `startRemoteDiarization` has.
    @discardableResult
    func startRemoteSpectralDiarization() -> Bool {
        let d = UserDefaults.standard
        let finalOn = d.object(forKey: "diarization.finalPass") as? Bool ?? true
        let continueOnStop = d.object(forKey: "diarization.continueOnStop") as? Bool ?? false
        let mode = Self.remoteStopMode(finalPass: finalOn,
                                       continueOnStop: continueOnStop,
                                       remoteStreamActive: remoteStreamActive,
                                       hasDiarizationService: modelLoader.spectral != nil,
                                       hasRemoteRecording: remoteRecordingURL != nil,
                                       tailSamples: remoteDiarAudio.count,
                                       supportsTail: false)
        guard let service = modelLoader.spectral, mode == .full,
              let recording = remoteRecordingURL else {
            // Nothing dispatched → the gate was never taken and no overlay row is
            // added. Drop any pending audio so it cannot outlive the session.
            remoteDiarAudio = []
            return false
        }
        remoteDiarAudio = []
        remoteFinalDiarDone = false
        let numSpeakers = d.integer(forKey: "diarization.numSpeakers")
        spectralLog("REMOTE FINAL PASS start — whole Remote WAV, "
                    + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers, stream: .remote)
        // Queued BEHIND the office pass on one stdin, so it can legitimately wait
        // out the office pass before it even starts — the doubling is the same
        // rule `startRemoteFullDiarization` uses for the same reason.
        startRemoteDiarWatchdog(
            seconds: Self.spectralWatchdogSeconds(recordingLength: recordingElapsed) * 2,
            message: "Remote spectral diarization timed out")
        return true
    }

    // MARK: - Log

    /// Append a line to `logs/spectral-diarization.log`.
    ///
    /// SWIFT-OWNED and single-writer, like `moss-diarization.log`,
    /// `position-diarization.log` and `dual-stream.log`. The sidecar naming rule
    /// (`scripts/<name>/<name>-service.py` → `logs/<name>.log`) covers SIDECAR
    /// logs only: `logs/spectral.log` is the sidecar's stderr and this file is the
    /// app's own decisions. Two writers on one file is the 2026-07-15 mistake.
    func spectralLog(_ message: String) {
        let dir = PythonRuntime.dataDir.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("spectral-diarization.log")
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        let stamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let line = "[\(stamp)] \(message)\n"
        guard let handle = try? FileHandle(forWritingTo: file) else { return }
        handle.seekToEndOfFile()
        if let data = line.data(using: .utf8) { handle.write(data) }
        try? handle.close()
    }
}
