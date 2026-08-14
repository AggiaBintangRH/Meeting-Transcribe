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
// rather than hidden. It used to be refusable — `diarization.finalPass` off left
// this engine with no labels at all — but since 2026-08-06 that setting is not
// shown under spectral and not read by it, so the configuration cannot be
// expressed and there is nothing left to refuse.
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
                // LIVE WINDOW? Routed first, and by the temp-WAV PATH the reply
                // echoes, so a window can never be mistaken for the stop pass or
                // the other way round. Returns false for the stop pass, which is
                // everything below — untouched.
                if self.handleBatchLiveResult(audio: audioPath, localTurns: localTurns,
                                              stream: stream) { return }
                // The DONE line the 2026-08-10 audit added to both batch engines —
                // see `AudioRecorder+Nemo.configureNemo` for the whole reason. In
                // short: this log recorded only starts and failures, so a pass that
                // succeeded was indistinguishable from one that hung.
                self.spectralLog("FINAL PASS done (\(stream == .office ? "office" : "remote")) "
                                 + "— \(localTurns.count) turn(s), "
                                 + "\(Set(localTurns.map(\.label)).count) local label(s); "
                                 + "identity next, see logs/\(WeSpeakerService.Config.logName).log")
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

    // MARK: - The office whole-file pass

    /// Whether a WHOLE-FILE BATCH session dispatches its office pass at Stop.
    ///
    /// ENGINE-NEUTRAL SINCE 2026-08-07, and shared by spectral and NeMo. It was
    /// `runsSpectralOfficePass(spectralActive:…)`; NeMo has the identical shape (no
    /// live path, one pass over the recording at Stop, no tail), so it is ONE
    /// function called with each engine's own flag rather than two rules that
    /// could drift. The rename is the whole change — the body is untouched.
    ///
    /// PURE and static, like `remoteStopMode` and `mossStopMode`, so the rule can
    /// be tested without a sidecar — and so `stop()` reads ONE condition per engine
    /// rather than growing a second one that could disagree with it.
    ///
    /// **NEITHER `continueOnStop` NOR `finalPass` IS A PARAMETER**, and for the
    /// same reason: taking a setting and then ignoring it would imply it was
    /// considered and rejected per session, when it is not available at all.
    ///
    /// There is no tail because there are no live labels to continue from, and
    /// since 2026-08-06 there is no stop-pass switch either — the pass IS the
    /// labels under these engines, so the Diarization tab shows neither control
    /// while one is selected and nothing here reads them. That also retired
    /// `spectralRefusalMessage`: the configuration it refused (stop pass off, no
    /// labels at all) can no longer be expressed.
    ///
    /// The stored `diarization.finalPass` deliberately keeps whatever pyannote
    /// left in it — these engines simply never ask.
    /// ⚠ `finalPass` JOINED THIS RULE ON 2026-08-13, reversing a deliberate
    /// omission. It was left out because the toggle was HIDDEN for these engines,
    /// and a value a pyannote session had stored would then have silently turned
    /// off the only pass that produces labels — the 🔴 defect the remote half of
    /// this hit for real. The toggle is now shown for every engine and they have a
    /// live path to fall back to, so honouring it is what the control means.
    nonisolated static func runsBatchOfficePass(batchActive: Bool,
                                                hasService: Bool,
                                                hasRecording: Bool,
                                                finalPass: Bool) -> Bool {
        batchActive && hasService && hasRecording && finalPass
    }

    /// Seconds to allow one WHOLE-FILE batch pass over `recordingLength` seconds
    /// of audio — spectral's and NeMo's, from one rule.
    ///
    /// Same floor and shape as `startDiarization`'s `max(180, recordingElapsed)`
    /// and `tailDiarWatchdogSeconds`: a generous constant, then at least 1×
    /// realtime for whatever is actually being processed. The multiplier is above
    /// 1 because spectral runs entirely on **CPU** (the vendored embedder pins
    /// `torch.device("cpu")`) while pyannote's pass runs on MPS, so 1× realtime is
    /// a far tighter bound there than it is for pyannote.
    ///
    /// NeMo shares it with room to spare rather than needing its own number: it
    /// was measured at **16–23× realtime on MPS** (48.2 min of audio in 170 s), so
    /// 2× realtime is roughly 35× its measured cost. Neither engine's figure has
    /// been taken on a *worst case* though — this is a deliberately loose backstop,
    /// not a prediction, and the whole point of a watchdog is that being wrong in
    /// this direction only costs patience.
    nonisolated static func batchPassWatchdogSeconds(recordingLength: Double) -> Double {
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
        let numSpeakers = Self.diarNumSpeakers
        // `diarization.detectOverlap` is NOT sent, and that is not an omission.
        // This engine's Viterbi smoothing assigns exactly one label per frame, so
        // it cannot report two speakers at one instant whatever it is asked. See
        // `ModelLoader.wantedOverlapEngine` for the consequence.
        spectralLog("FINAL PASS start — \(fmt(recordingElapsed))s of office audio, "
                    + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers)

        let limit = Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed)
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
        // `finalPass` IS PASSED AS `true`, NOT READ — and that is the fix for a
        // silent data loss found in the 2026-08-10 audit.
        //
        // The office pass already ignores this key (`runsBatchOfficePass` does not
        // take it) for the reason in that function's doc: under a batch engine the
        // stop pass IS the labels, so a stored value must not be able to leave a
        // session with none. The remote pass read it anyway, and `DiarizationTab`
        // HIDES the toggle under batch engines — so a value left `false` by an
        // earlier pyannote session removed every remote label, silently, with
        // nothing in the UI able to put it back. That is exactly the failure the
        // 2026-08-06 settings pass exists to forbid: a value outliving its control.
        //
        // Passed rather than deleted from the signature because `remoteStopMode` is
        // shared with pyannote, where the toggle is visible and must still be
        // honoured. `supportsTail:` already establishes this shape — the caller
        // states its engine's truth, the one pure function keeps enumerating.
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
        // THE FAR END'S OWN count, never the room's — see
        // `AudioRecorder.remoteNumSpeakers`. It was a pinned 0 until the owner
        // asked for a Remote picker (2026-08-13); what has never changed is that
        // it must not be `diarNumSpeakers`, which counts the ROOM.
        // This engine is the one a pinned count measurably moves, so it is also
        // the one a WRONG pinned count moves the furthest.
        let numSpeakers = Self.remoteNumSpeakers
        spectralLog("REMOTE FINAL PASS start — whole Remote WAV, "
                    + "num_speakers=\(numSpeakers == 0 ? "auto" : String(numSpeakers))")
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers, stream: .remote)
        // Queued BEHIND the office pass on one stdin, so it can legitimately wait
        // out the office pass before it even starts — the doubling is the same
        // rule `startRemoteFullDiarization` uses for the same reason.
        startRemoteDiarWatchdog(
            seconds: Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed) * 2,
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
        PythonRuntime.appendAppLog(name: "spectral-diarization", message: message)
    }
}

extension AudioRecorder {

    /// Wire the overlap DETECTOR for this session, and run it at Stop.
    ///
    /// Deliberately NOT part of the stop GATE. The detector adds a mark to rows
    /// whose text already exists; holding the blocking overlay for it would make a
    /// hint cost the user their transcript's arrival. It lands late, exactly as
    /// aligner words do, and `rebuildDisplayRows` picks it up — the same
    /// late-arrival pattern diarization turns have always used.
    ///
    /// Failure is silent in the transcript and loud in the log: a missing mark is
    /// a missing hint, never wrong text.
    func configureOverlapDetect() {
        guard let service = modelLoader.overlapDetect else { return }
        service.onResult = { [weak self] regions, audio in
            Task { @MainActor in
                guard let self else { return }
                // ROUTED BY THE ECHOED PATH, which is why the sidecar echoes it —
                // the same design pyannote uses to say which file a reply is
                // about. No protocol change was needed to add the second stream.
                let isRemote = self.isRemoteRecordingPath(audio)
                if isRemote { self.remoteDetectedOverlapRegions = regions }
                else        { self.detectedOverlapRegions = regions }
                let total = regions.reduce(0.0) { $0 + ($1.end - $1.start) }
                self.overlapDetectLog("DETECT done (\(isRemote ? "remote" : "office")) "
                                      + "— \(regions.count) region(s), "
                                      + "\(String(format: "%.1f", total))s marked")
                self.rebuildDisplayRows()
                // Repair waits for this under MOSS and spectral, where these
                // regions are its ONLY input. Released after the rows are rebuilt
                // so the mark is on screen either way, even if repair then skips.
                self.finishOverlapDetectJob()
            }
        }
        service.onError = { [weak self] message in
            Task { @MainActor in
                guard let self else { return }
                self.overlapDetectLog("DETECT failed — \(message)")
                // A failed detection must RELEASE repair, not hold it: repair will
                // find no regions and settle its own leg, which is the difference
                // between a missing mark and a stop overlay stuck for ten minutes.
                //
                // The error carries no stream, so it cannot be attributed — which
                // does not matter, because the counter only needs to know that ONE
                // job ended. Guessing a stream here would be the thing that breaks
                // it: guess wrong twice and the count never reaches zero.
                self.finishOverlapDetectJob()
            }
        }
    }

    /// One detection job ended — result or error, office or remote. Releases
    /// repair only when the last one has landed. Idempotent below zero.
    func finishOverlapDetectJob() {
        overlapDetectPending = max(0, overlapDetectPending - 1)
        guard overlapDetectPending == 0 else { return }
        overlapDetectDone = true
        maybeStartOverlapRepair()
    }

    /// Is this reply about the Remote WAV? Compared against the recorder's own
    /// remote URL rather than by looking for "-remote" in the string: the suffix
    /// is `AudioRecorder.remoteURL(for:)`'s business, and a path test that
    /// re-derives a naming rule is a second place for it to change.
    func isRemoteRecordingPath(_ path: String) -> Bool {
        guard !path.isEmpty, let office = lastRecordingURL else { return false }
        return path == Self.remoteURL(forOffice: office).path
    }

    /// At Stop: one pass over the whole recording. ~160x realtime, so a 43-minute
    /// meeting costs about 16 seconds — measured, not estimated.
    func startOverlapDetection(_ recording: URL) {
        // No detector this session: `overlapDetectDone` STAYS true, so repair
        // never waits for a pass that will not happen.
        guard let service = modelLoader.overlapDetect else { return }
        detectedOverlapRegions = []
        remoteDetectedOverlapRegions = []
        overlapDetectDone = false

        // The REMOTE WAV gets its own pass (owner, 2026-08-13). Two remote
        // participants talking over each other is real overlap and went unmarked
        // entirely until now; the detector reads audio and needs no turns, so it
        // is the only source that works under all six engines.
        //
        // COUNTED BEFORE ANYTHING IS DISPATCHED. Incrementing as each job is sent
        // would let the first reply arrive while the count stood at 1, release the
        // gate, and leave the second stream's regions landing after repair had
        // already decided there was nothing to do.
        let remote = remoteStreamActive ? Self.remoteURL(forOffice: recording) : nil
        let hasRemote = remote.map { FileManager.default.fileExists(atPath: $0.path) } ?? false
        overlapDetectPending = hasRemote ? 2 : 1

        overlapDetectLog("DETECT start — \(fmt(recordingElapsed))s of audio"
                         + (hasRemote ? " (office + remote)" : ""))
        service.detect(audio: recording)
        if hasRemote, let remote { service.detect(audio: remote) }
    }

    /// `logs/overlap-detect-decisions.log` — SWIFT-owned, one writer, and named
    /// apart from the sidecar's own `logs/overlap-detect.log`. Two writers on one
    /// file is the 2026-07-15 mistake.
    func overlapDetectLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "overlap-detect-decisions", message: message)
    }
}
