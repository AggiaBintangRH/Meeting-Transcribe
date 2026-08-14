import Foundation

// The NEMO diarization engine (`diarization.engine == "nemo"`) — session
// configuration, the two whole-file stop passes (office and remote), and this
// domain's own log writer.
//
// WHAT SHAPE THIS ENGINE HAS
// --------------------------
// pyannote diarizes live 30 s windows AND does a stop pass. MOSS labels as it
// transcribes, per chunk. NeMo does NEITHER — it is the SPECTRAL shape exactly:
// NME-SC counts speakers globally (an eigengap analysis over the affinity matrix
// of the whole file) and clusters globally, so a 30 s window would be counted and
// clustered on its own and its labels would mean nothing across windows — the
// failure MOSS has when it is called per chunk. The sidecar therefore serves ONLY
// `cmd: "final"` and refuses anything else loudly; `NemoService` has no chunk API
// at all.
//
// So the whole session is: record with no speaker labels, then one pass over the
// recording at Stop. That is a real degradation against pyannote and the
// Diarization tab states it rather than letting the user discover it.
//
// WHAT IT KEEPS, which is the payoff of the 2026-07-30 pyannote/wespeaker split:
// its labels are run-LOCAL (`speaker_0`, `speaker_1`, …), so they go through the
// SAME `identify` → `composeTurns` path pyannote's and spectral's do. Saved voice
// profiles, renaming and `spk` confidence all work under this engine with NO new
// identity code and NO new id space — the labels end up as office profile ids
// (< 10 000) like every other pipeline engine's.
//
// THIS FILE ADDS NO STOP-GATE LEG. The office pass settles `finalDiarDone` and
// the `"diarize"` step through the shared `identifyFinalTurns` → `applyFinalTurns`
// path, and the remote pass settles the shared remote leg. A second parallel gate
// for a second engine is how a stop overlay gets stuck forever.
extension AudioRecorder {

    // MARK: - Session configuration

    /// Read the engine choice for this session and wire the NeMo callbacks.
    /// Called from `beginCapture` alongside `configureDiarization`,
    /// `configureMoss` and `configureSpectral`, so the engine is fixed for the
    /// whole recording like every other setting — half a transcript labelled by
    /// each engine is not a state any display path can render honestly.
    func configureNemo() {
        let d = UserDefaults.standard
        let engine = d.string(forKey: "diarization.engine") ?? ModelLoader.pyannoteEngineID
        let diarOn = d.object(forKey: "diarization.enabled") as? Bool ?? true

        nemoDiarizationActive = diarOn && engine == ModelLoader.nemoEngineID
        guard nemoDiarizationActive, let service = modelLoader.nemo else { return }

        nemoLog("engine=nemo — no live labels this session; "
                + "one whole-file pass per stream at Stop")

        // The SAME two handlers pyannote's and spectral's final replies use,
        // because the wire shape is the same and the identity stage is shared.
        // `identifyFinalTurns` and `handleDiarizationFailure` already settle every
        // gate correctly for both streams; duplicating either here is how the two
        // would drift.
        service.onFinalResult = { [weak self] audioPath, localTurns, stream in
            Task { @MainActor in
                guard let self else { return }
                // LIVE WINDOW? Routed first, and by the temp-WAV PATH the reply
                // echoes, so a window can never be mistaken for the stop pass or
                // the other way round. Returns false for the stop pass, which is
                // everything below — untouched.
                if self.handleBatchLiveResult(audio: audioPath, localTurns: localTurns,
                                              stream: stream) { return }
                // A DONE line, added by the 2026-08-10 audit. This log recorded
                // only starts and failures, so a pass that WORKED left the file
                // ending mid-sentence — indistinguishable at a glance from one that
                // hung. Reading a real session, that absence was the first thing
                // that looked like a bug, and nothing was wrong.
                //
                // Deliberately NOT reporting the identity outcome: that is
                // `wespeaker.log`'s (`identify done — N/M voices identified`), which
                // already says it well and is that service's single-writer file.
                // Two logs narrating one event is how they come to disagree.
                self.nemoLog("FINAL PASS done (\(stream == .office ? "office" : "remote")) "
                             + "— \(localTurns.count) turn(s), "
                             + "\(Set(localTurns.map(\.label)).count) local label(s); "
                             + "identity next, see logs/\(WeSpeakerService.Config.logName).log")
                await self.identifyFinalTurns(audio: audioPath, localTurns: localTurns,
                                              stream: stream)
            }
        }
        service.onError = { [weak self] message, stream in
            Task { @MainActor in
                self?.nemoLog("FAILED (\(stream == .office ? "office" : "remote")): \(message)")
                self?.handleDiarizationFailure(message, stream: stream)
            }
        }
        // The one reply that comes down the error channel without being an error.
        // Same settling as a failure — deliberately, see `handleDiarizationFailure`
        // — but logged and reported as the verdict it is. NO SPEECH is not FAILED,
        // and the log has to say so too: this file is the evidence a later audit
        // reads, and it called a correct result a failure until 2026-08-12.
        service.onNoSpeech = { [weak self] message, stream in
            Task { @MainActor in
                self?.nemoLog("NO SPEECH (\(stream == .office ? "office" : "remote")): "
                              + "\(message)")
                self?.handleDiarizationFailure(message, stream: stream, noSpeech: true)
            }
        }
    }

    // MARK: - The office whole-file pass

    /// At Stop: one batch pass over the whole office recording.
    ///
    /// Whether it runs at all is `AudioRecorder.runsBatchOfficePass` — the SAME
    /// pure rule spectral uses, called with this engine's flag. `finalPass` is not
    /// one of its inputs: the pass IS the labels here, so a value pyannote left in
    /// that key must not be able to strand a NeMo session with none at all.
    ///
    /// Reuses the pyannote final pass's gate exactly — `finalDiarDone`,
    /// `finalDiarWatchdog`, the `"diarize"` step, and `applyFinalTurns` as the
    /// settle — so there is one "Identifying speakers" leg however it is served.
    func startNemoDiarization(_ recording: URL) {
        guard let service = modelLoader.nemo else {
            // Unreachable through `stop()` (which tests the service first), kept
            // so the leg can never be left un-settleable by a future edit.
            diarizationError = "The NeMo diarization sidecar is not running"
            finalDiarDone = true
            setStopStep("diarize", .failed(diarizationError ?? ""))
            checkStopProcessingDone()
            return
        }
        diarizing = true
        diarizationError = nil
        // ALWAYS 0 = auto under this engine, and there is no Settings control for
        // it (owner, 2026-08-07). NME-SC estimates the count itself and
        // `max_num_speakers` is pinned at 20 inside the sidecar; a picker here
        // would be a control the engine is never asked to honour.
        let numSpeakers = Self.diarNumSpeakers
        // `diarization.detectOverlap` is NOT sent, and that is not an omission.
        // NME-SC assigns exactly one label per instant, so it cannot report two
        // speakers at one time whatever it is asked. See
        // `ModelLoader.wantedOverlapEngine` for the consequence — and note that the
        // standalone overlap DETECTOR removes it, which is why repair is available
        // under this engine once that is switched on.
        nemoLog("FINAL PASS start — \(fmt(recordingElapsed))s of office audio, "
                + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers)

        let limit = Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed)
        finalDiarWatchdog?.cancel()
        finalDiarWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(limit))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.finalDiarDone else { return }
                self.nemoLog("FINAL PASS TIMEOUT after \(Int(limit))s")
                self.diarizing = false
                self.diarizationError = "NeMo diarization timed out"
                self.finalDiarDone = true
                self.setStopStep("diarize", .failed("NeMo diarization timed out"))
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
    /// `startRemoteDiarization` and `startRemoteSpectralDiarization` have.
    @discardableResult
    func startRemoteNemoDiarization() -> Bool {
        // `finalPass` IS PASSED AS `true`, NOT READ — see the twin in
        // `AudioRecorder+Spectral.startRemoteSpectralDiarization` for the whole
        // reason. In one line: the office pass ignores this key, `DiarizationTab`
        // hides its toggle under batch engines, and reading it here let a value an
        // earlier pyannote session left `false` delete every remote label with no
        // message and no control able to undo it (2026-08-10 audit).
        let continueOnStop = diarContinueOnStop
        // HONOURED NOW, not forced true. It was pinned because the toggle
        // was hidden for this engine, so a value left by a pyannote session
        // deleted every remote label with nothing able to undo it. The
        // toggle is visible for every engine since 2026-08-13, and off means
        // the live per-interval path carries the labels instead.
        let mode = Self.remoteStopMode(
            finalPass: UserDefaults.standard.object(forKey: "diarization.finalPass")
                as? Bool ?? true,
                                       continueOnStop: continueOnStop,
                                       remoteStreamActive: remoteStreamActive,
                                       hasDiarizationService: modelLoader.nemo != nil,
                                       hasRemoteRecording: remoteRecordingURL != nil,
                                       tailSamples: remoteDiarAudio.count,
                                       supportsTail: false)
        guard let service = modelLoader.nemo, mode == .full,
              let recording = remoteRecordingURL else {
            // Nothing dispatched → the gate was never taken and no overlay row is
            // added. Drop any pending audio so it cannot outlive the session.
            remoteDiarAudio = []
            return false
        }
        remoteDiarAudio = []
        remoteFinalDiarDone = false
        // THE FAR END'S OWN count, never the room's — see
        // `AudioRecorder.remoteNumSpeakers`. It was a pinned 0 until the owner
        // asked for a Remote picker (2026-08-13); what has never changed is that
        // it must not be `diarNumSpeakers`, which counts the ROOM.
        // Here a pinned count would arrive as `oracle_num_speakers`, i.e. as a
        // certainty rather than a hint.
        let numSpeakers = Self.remoteNumSpeakers
        nemoLog("REMOTE FINAL PASS start — whole Remote WAV, "
                + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers, stream: .remote)
        // Queued BEHIND the office pass on one stdin, so it can legitimately wait
        // out the office pass before it even starts — the doubling is the same
        // rule `startRemoteFullDiarization` and the spectral twin use.
        startRemoteDiarWatchdog(
            seconds: Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed) * 2,
            message: "Remote NeMo diarization timed out")
        return true
    }

    // MARK: - Log

    /// Append a line to `logs/nemo-diarization.log`.
    ///
    /// SWIFT-OWNED and single-writer, like `spectral-diarization.log`,
    /// `moss-diarization.log`, `position-diarization.log` and `dual-stream.log`.
    /// The sidecar naming rule (`scripts/<name>/<name>-service.py` →
    /// `logs/<name>.log`) covers SIDECAR logs only: `logs/nemo.log` is the
    /// sidecar's stderr — and for this engine that stderr is busy, since NeMo's own
    /// logger is redirected there wholesale — while this file is the app's own
    /// decisions. Two writers on one file is the 2026-07-15 mistake.
    func nemoLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "nemo-diarization", message: message)
    }
}
