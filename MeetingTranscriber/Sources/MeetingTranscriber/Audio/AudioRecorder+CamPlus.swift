import Foundation

// The CAM++ diarization engine (`diarization.engine == "campplus"`) — session
// configuration, the two whole-file stop passes (office and remote), their
// watchdogs, and this domain's own log writer.
//
// WHAT IT IS
// ----------
// A pipeline built around a speaker EMBEDDING model, because that is all CAM++
// is: Silero VAD → CAM++ vectors over 2 s sliding sub-windows → spectral
// clustering with an eigengap speaker count. Whole-file only, so the session
// shape is identical to spectral's, NeMo's and DiariZen's: record with no
// speaker labels, then one pass per stream at Stop.
//
// WHY IT IS NOT JUST "SPECTRAL WITH ANOTHER EMBEDDER", which is what it looks
// like: the `spectral` engine's speaker COUNTING is the stage this project has
// measured failing — GMM-BIC returning 20 speakers on a 67-minute meeting and
// 13 on a 3-person clip — while its clustering is fine. This engine replaces
// exactly that stage (eigengap on the normalized Laplacian, the NME-SC family
// NeMo also belongs to) and keeps the rest. On the same 67-minute recording it
// returns 3 speakers / 99 turns where spectral returns 20 / 1869.
//
// WHAT IT KEEPS, which is the 2026-07-30 pyannote/wespeaker split paying out for
// the fourth time: its labels are run-LOCAL (`SPEAKER_00`), so they go through
// the SAME `identify` → `composeTurns` path pyannote's do. Saved voice profiles,
// renaming and `spk` confidence all work here with no new identity code and no
// new id space.
//
// WHAT IT CANNOT DO, stated rather than hidden: no live labels (the engine
// counts and clusters globally, so a 30 s window's labels would mean nothing
// across windows), no tail pass (there are no live labels to continue from),
// and no overlap marking of its own — clustering assigns exactly one label per
// window, so its turns never intersect. That last point is why it sits on the
// same side of `usesDetectedRegionsForRepair` as spectral and NeMo rather than
// with pyannote and DiariZen.
extension AudioRecorder {

    // MARK: - Session configuration

    /// Read the engine choice for this session and wire the CAM++ callbacks.
    /// Called from `beginCapture` alongside the other engines' configure calls,
    /// so the engine is fixed for the whole recording like every other setting —
    /// half a transcript labelled by each engine is not a state any display path
    /// can render honestly.
    func configureCamPlus() {
        let d = UserDefaults.standard
        let engine = d.string(forKey: "diarization.engine") ?? ModelLoader.pyannoteEngineID
        let diarOn = d.object(forKey: "diarization.enabled") as? Bool ?? true

        camPlusDiarizationActive = diarOn && engine == ModelLoader.camPlusEngineID
        guard camPlusDiarizationActive, let service = modelLoader.camPlus else { return }

        // NAMES ITS OWN ENGINE. The 2026-08-10 DiariZen audit found
        // `configureDiarizen()` logging `engine=nemo` into the DiariZen log — and
        // the log is the evidence, so a log that names the wrong engine is worse
        // than no log.
        camPlusLog("engine=campplus — no live labels this session; "
                   + "one whole-file pass per stream at Stop")

        // The SAME two handlers pyannote's final replies use, because the wire
        // shape is the same and the identity stage is shared. `identifyFinalTurns`
        // and `handleDiarizationFailure` already settle every gate correctly for
        // both streams; duplicating either here is how the two would drift.
        service.onFinalResult = { [weak self] audioPath, localTurns, stream in
            Task { @MainActor in
                guard let self else { return }
                // The DONE line both batch engines gained in the 2026-08-10 audit:
                // without it the log recorded only starts and failures, so a pass
                // that SUCCEEDED ended mid-sentence and looked like a hang.
                self.camPlusLog("FINAL PASS done (\(stream == .office ? "office" : "remote")) "
                                + "— \(localTurns.count) turn(s), "
                                + "\(Set(localTurns.map(\.label)).count) local label(s); "
                                + "identity next, see logs/\(WeSpeakerService.Config.logName).log")
                await self.identifyFinalTurns(audio: audioPath, localTurns: localTurns,
                                              stream: stream)
            }
        }
        service.onError = { [weak self] message, stream in
            Task { @MainActor in
                self?.camPlusLog("FAILED (\(stream == .office ? "office" : "remote")): \(message)")
                self?.handleDiarizationFailure(message, stream: stream)
            }
        }
    }

    // MARK: - The office whole-file pass

    /// At Stop: one batch pass over the whole office recording.
    ///
    /// Reuses the pyannote final pass's gate exactly — `finalDiarDone`,
    /// `finalDiarWatchdog`, the `"diarize"` step, and `applyFinalTurns` as the
    /// settle — so there is one "Identifying speakers" leg however it is served.
    /// A second parallel gate for a second engine is how a stop overlay gets stuck.
    ///
    /// `runsBatchOfficePass` and `batchPassWatchdogSeconds` are shared with the
    /// other batch engines rather than re-derived here: this engine has the
    /// identical shape (no live path, one pass at Stop, no tail), and two rules
    /// that could drift is what that sharing exists to prevent.
    func startCamPlusDiarization(_ recording: URL) {
        guard let service = modelLoader.camPlus else {
            // Unreachable through `stop()` (which tests the service first), kept
            // so the leg can never be left un-settleable by a future edit.
            diarizationError = "The CAM++ diarization sidecar is not running"
            finalDiarDone = true
            setStopStep("diarize", .failed(diarizationError ?? ""))
            checkStopProcessingDone()
            return
        }
        diarizing = true
        diarizationError = nil
        let numSpeakers = Self.diarNumSpeakers
        // `diarization.detectOverlap` is NOT sent, and that is not an omission:
        // this engine assigns exactly one label per window, so it cannot report
        // two speakers at one instant whatever it is asked. See
        // `ModelLoader.wantedOverlapEngine` for the consequence.
        camPlusLog("FINAL PASS start — \(fmt(recordingElapsed))s of office audio, "
                   + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers)

        let limit = Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed)
        finalDiarWatchdog?.cancel()
        finalDiarWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(limit))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.finalDiarDone else { return }
                self.camPlusLog("FINAL PASS TIMEOUT after \(Int(limit))s")
                self.diarizing = false
                self.diarizationError = "CAM++ diarization timed out"
                self.finalDiarDone = true
                self.setStopStep("diarize", .failed("CAM++ diarization timed out"))
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
    /// overlay gets a remote-diarization row.
    @discardableResult
    func startRemoteCamPlusDiarization() -> Bool {
        // `finalPass` IS PASSED AS `true`, NOT READ — the fix for the silent data
        // loss the 2026-08-10 audit found in the other batch engines. The toggle
        // is HIDDEN under a batch engine, so a `false` left by an earlier pyannote
        // session would delete every remote label with nothing in the UI able to
        // put it back: a value outliving its control, which the 2026-08-06
        // settings pass exists to forbid. Passed rather than dropped from the
        // signature because `remoteStopMode` is shared with pyannote, where the
        // toggle IS visible and must still be honoured.
        let continueOnStop = diarContinueOnStop
        let mode = Self.remoteStopMode(finalPass: true,
                                       continueOnStop: continueOnStop,
                                       remoteStreamActive: remoteStreamActive,
                                       hasDiarizationService: modelLoader.camPlus != nil,
                                       hasRemoteRecording: remoteRecordingURL != nil,
                                       tailSamples: remoteDiarAudio.count,
                                       supportsTail: false)
        guard let service = modelLoader.camPlus, mode == .full,
              let recording = remoteRecordingURL else {
            // Nothing dispatched → the gate was never taken and no overlay row is
            // added. Drop any pending audio so it cannot outlive the session.
            remoteDiarAudio = []
            return false
        }
        remoteDiarAudio = []
        remoteFinalDiarDone = false
        // AUTO, never the room's count — see `AudioRecorder.remoteNumSpeakers`.
        // The SPK chip reports the ROOM; nothing anywhere asks how many people
        // are on the far end, so sending that number would force the room's
        // headcount onto the callers.
        let numSpeakers = Self.remoteNumSpeakers
        camPlusLog("REMOTE FINAL PASS start — whole Remote WAV, "
                   + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers, stream: .remote)
        // Queued BEHIND the office pass on one stdin, so it can legitimately wait
        // out the office pass before it even starts — hence the doubling.
        startRemoteDiarWatchdog(
            seconds: Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed) * 2,
            message: "Remote CAM++ diarization timed out")
        return true
    }

    // MARK: - Log

    /// Append a line to `logs/campplus-diarization.log`.
    ///
    /// SWIFT-OWNED and single-writer, like `spectral-diarization.log` and
    /// `moss-diarization.log`. The sidecar naming rule
    /// (`scripts/<name>/<name>-service.py` → `logs/<name>.log`) covers SIDECAR
    /// logs only: `logs/campplus.log` is the sidecar's stderr and this file is
    /// the app's own decisions. Two writers on one file is the 2026-07-15
    /// mistake.
    func camPlusLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "campplus-diarization", message: message)
    }
}
