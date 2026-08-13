import Foundation

// The DIARIZEN diarization engine (`diarization.engine == "diarizen"`) — session
// configuration, the two whole-file stop passes (office and remote), and this
// domain's own log writer.
//
// WHAT SHAPE THIS ENGINE HAS
// --------------------------
// pyannote diarizes live 30 s windows AND does a stop pass. MOSS labels as it
// transcribes, per chunk. DiariZen does NEITHER — it is the SPECTRAL and NEMO
// shape: a WavLM+Conformer network segments the WHOLE recording end-to-end, then
// WeSpeaker embeddings are clustered agglomeratively across all of it. A 30 s
// window would be clustered on its own and its labels would mean nothing across
// windows — the failure MOSS has when it is called per chunk. The sidecar
// therefore serves ONLY `cmd: "final"` and refuses anything else loudly;
// `DiarizenService` has no chunk API at all.
//
// So the whole session is: record with no speaker labels, then one pass over the
// recording at Stop. That is a real degradation against pyannote and the
// Diarization tab states it rather than letting the user discover it.
//
// WHY IT IS HERE AT ALL, given four engines already exist: it is the only one
// TRAINED ON FAR-FIELD SINGLE-CHANNEL MEETINGS (AMI, AISHELL-4, AliMeeting) —
// the owner's ATND array's exact conditions, and where the other four struggle.
// Its segmentation is also the only neural end-to-end one here, so when it agrees
// with them it is genuinely independent evidence rather than a fourth method
// sharing the same blind spot.
//
// MEASURED on this M4: 12.4-15.2x realtime on MPS, 1.7-3.7 GB, 4.8 s from launch
// to LOADED. Output is IDENTICAL to CPU, which is what makes MPS verified here
// rather than assumed (the MOSS precedent, not DiCoW's).
//
// The counts in an older draft of this comment (4 speakers, 11 turns on
// `Meeting5People.wav`) were taken BEFORE `min_cluster_size` was tuned; that
// file's truth is 5 and the engine now returns 5. See the constant's declaration
// in the sidecar for the measurement table.
//
// WHAT IT KEEPS, which is the payoff of the 2026-07-30 pyannote/wespeaker split:
// its labels are run-LOCAL, so they go through the SAME `identify` →
// `composeTurns` path pyannote's, spectral's and NeMo's do. Saved voice profiles
// and renaming work with NO new identity code and NO new id space — the labels
// end up as office profile ids (< 10 000) like every other pipeline engine's.
// (`spk` confidence is a different matter — see `AudioRecorder.meetingFinished`'s
// neighbours and the batch-engine note in `DiarizationTab`.)
//
// THIS FILE ADDS NO STOP-GATE LEG. The office pass settles `finalDiarDone` and
// the `"diarize"` step through the shared `identifyFinalTurns` → `applyFinalTurns`
// path, and the remote pass settles the shared remote leg. A second parallel gate
// for a second engine is how a stop overlay gets stuck forever.
extension AudioRecorder {

    // MARK: - Session configuration

    /// Read the engine choice for this session and wire the DiariZen callbacks.
    /// Called from `beginCapture` alongside `configureDiarization`,
    /// `configureMoss` and `configureSpectral`, so the engine is fixed for the
    /// whole recording like every other setting — half a transcript labelled by
    /// each engine is not a state any display path can render honestly.
    func configureDiarizen() {
        let d = UserDefaults.standard
        let engine = d.string(forKey: "diarization.engine") ?? ModelLoader.pyannoteEngineID
        let diarOn = d.object(forKey: "diarization.enabled") as? Bool ?? true

        diarizenDiarizationActive = diarOn && engine == ModelLoader.diarizenEngineID
        guard diarizenDiarizationActive, let service = modelLoader.diarizen else { return }

        diarizenLog("engine=diarizen — no live labels this session; "
                + "one whole-file pass per stream at Stop")

        // The SAME two handlers pyannote's and spectral's final replies use,
        // because the wire shape is the same and the identity stage is shared.
        // `identifyFinalTurns` and `handleDiarizationFailure` already settle every
        // gate correctly for both streams; duplicating either here is how the two
        // would drift.
        service.onFinalResult = { [weak self] audioPath, localTurns, stream in
            Task { @MainActor in
                guard let self else { return }
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
                self.diarizenLog("FINAL PASS done (\(stream == .office ? "office" : "remote")) "
                             + "— \(localTurns.count) turn(s), "
                             + "\(Set(localTurns.map(\.label)).count) local label(s); "
                             + "identity next, see logs/\(WeSpeakerService.Config.logName).log")
                await self.identifyFinalTurns(audio: audioPath, localTurns: localTurns,
                                              stream: stream)
            }
        }
        service.onError = { [weak self] message, stream in
            Task { @MainActor in
                self?.diarizenLog("FAILED (\(stream == .office ? "office" : "remote")): \(message)")
                self?.handleDiarizationFailure(message, stream: stream)
            }
        }
    }

    // MARK: - The office whole-file pass

    /// At Stop: one batch pass over the whole office recording.
    ///
    /// Whether it runs at all is `AudioRecorder.runsBatchOfficePass` — the SAME
    /// pure rule spectral uses, called with this engine's flag. `finalPass` is not
    /// one of its inputs: the pass IS the labels here, so a value pyannote left in
    /// that key must not be able to strand a DiariZen session with none at all.
    ///
    /// Reuses the pyannote final pass's gate exactly — `finalDiarDone`,
    /// `finalDiarWatchdog`, the `"diarize"` step, and `applyFinalTurns` as the
    /// settle — so there is one "Identifying speakers" leg however it is served.
    func startDiarizenDiarization(_ recording: URL) {
        guard let service = modelLoader.diarizen else {
            // Unreachable through `stop()` (which tests the service first), kept
            // so the leg can never be left un-settleable by a future edit.
            diarizationError = "The DiariZen diarization sidecar is not running"
            finalDiarDone = true
            setStopStep("diarize", .failed(diarizationError ?? ""))
            checkStopProcessingDone()
            return
        }
        diarizing = true
        diarizationError = nil
        // ALWAYS 0 = auto under this engine, and there is no Settings control for
        // it (owner, 2026-08-10: the count must be automatic, never written). The
        // checkpoint's own `min_speakers` 2 / `max_speakers` 8 and its
        // `min_cluster_size` decide it; the sidecar does not even read this field.
        // A picker here would be a control the engine is never asked to honour.
        let numSpeakers = Self.diarNumSpeakers
        // `diarization.detectOverlap` is NOT sent because this sidecar has no such
        // knob — NOT because the engine cannot mark overlap. It CAN: EEND predicts
        // per-speaker activity, and its turns really do intersect (measured on
        // `recordings/Overlap123.wav` — 11 intersecting pairs, 13.30 s), unlike
        // spectral's and NeMo's, which assign one label per instant structurally.
        // Overlap TAGGING and overlap REPAIR both take their regions from those
        // turns — see `AudioRecorder.usesDetectedRegionsForRepair`, which puts this
        // engine on pyannote's side of that line since 2026-08-10.
        diarizenLog("FINAL PASS start — \(fmt(recordingElapsed))s of office audio, "
                + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers)

        let limit = Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed)
        finalDiarWatchdog?.cancel()
        finalDiarWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(limit))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.finalDiarDone else { return }
                self.diarizenLog("FINAL PASS TIMEOUT after \(Int(limit))s")
                self.diarizing = false
                self.diarizationError = "DiariZen diarization timed out"
                self.finalDiarDone = true
                self.setStopStep("diarize", .failed("DiariZen diarization timed out"))
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
    func startRemoteDiarizenDiarization() -> Bool {
        // `finalPass` IS PASSED AS `true`, NOT READ — see the twin in
        // `AudioRecorder+Spectral.startRemoteSpectralDiarization` for the whole
        // reason. In one line: the office pass ignores this key, `DiarizationTab`
        // hides its toggle under batch engines, and reading it here let a value an
        // earlier pyannote session left `false` delete every remote label with no
        // message and no control able to undo it (2026-08-10 audit).
        let continueOnStop = diarContinueOnStop
        let mode = Self.remoteStopMode(finalPass: true,
                                       continueOnStop: continueOnStop,
                                       remoteStreamActive: remoteStreamActive,
                                       hasDiarizationService: modelLoader.diarizen != nil,
                                       hasRemoteRecording: remoteRecordingURL != nil,
                                       tailSamples: remoteDiarAudio.count,
                                       supportsTail: false)
        guard let service = modelLoader.diarizen, mode == .full,
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
        // Inert on this engine, which reads no count at all, and passed anyway so
        // the four remote passes state ONE rule rather than three plus a special
        // case that would silently become wrong if the sidecar ever grew the knob.
        let numSpeakers = Self.remoteNumSpeakers
        diarizenLog("REMOTE FINAL PASS start — whole Remote WAV, "
                + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers, stream: .remote)
        // Queued BEHIND the office pass on one stdin, so it can legitimately wait
        // out the office pass before it even starts — the doubling is the same
        // rule `startRemoteFullDiarization` and the spectral twin use.
        startRemoteDiarWatchdog(
            seconds: Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed) * 2,
            message: "Remote DiariZen diarization timed out")
        return true
    }

    // MARK: - Log

    /// Append a line to `logs/diarizen-diarization.log`.
    ///
    /// SWIFT-OWNED and single-writer, like `spectral-diarization.log`,
    /// `moss-diarization.log`, `position-diarization.log` and `dual-stream.log`.
    /// The sidecar naming rule (`scripts/<name>/<name>-service.py` →
    /// `logs/<name>.log`) covers SIDECAR logs only: `logs/diarizen.log` is that
    /// sidecar's stderr — and for this engine that file is busy, since
    /// `install_wire()` redirects DiariZen's own stdout logger there wholesale —
    /// while THIS one is the app's own decisions. Two writers on one file is the
    /// 2026-07-15 mistake.
    func diarizenLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "diarizen-diarization", message: message)
    }
}
