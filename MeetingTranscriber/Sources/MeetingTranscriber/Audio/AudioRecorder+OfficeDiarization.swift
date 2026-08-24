import AVFoundation
import Foundation

// Office-stream diarization: the session's service callbacks, the live/tail
// chunk dispatch, the stop-time passes and their completion gate. Also the
// shared `writeTempWAV` helper, used by the remote stream and remote
// diarization too. Moved verbatim from the core file.
extension AudioRecorder {

    // MARK: - Diarization

    /// Wire the diarization service callbacks for this session.
    ///
    /// TWO SIDECARS SINCE 2026-07-30. A pyannote reply is no longer a finished
    /// answer: it carries run-local labels (`SPEAKER_00`), and the identity stage
    /// turns those into profile ids and names. So every callback here dispatches
    /// an `identify` and does the real work in the COMPOSED handler afterwards.
    ///
    /// ORDERING INVARIANTS — preserved, not newly argued:
    ///  * pyannote's single stdin is FIFO, exactly as the merged sidecar's was.
    ///  * `WeSpeakerService`'s serial gate makes identify FIFO too.
    ///  * therefore PROFILE-STORE mutation order and `liveTurns` append order are
    ///    identical to the old single-stdin order, INCLUDING the office/remote
    ///    interleaving: both streams still share one queue, just a different one.
    ///  * the stop gates settle only in the composed handlers, never on the
    ///    pyannote reply, so no leg can complete with turns not yet appended.
    ///    `buildStopSteps`, `checkStopProcessingDone` and every watchdog are
    ///    untouched — they already span the combined latency.
    func configureDiarization() {
        guard let service = modelLoader.pyannote else { return }

        service.onChunkResult = { [weak self] windowStart, audioPath, localTurns, stream in
            Task { @MainActor in
                guard let self else { return }
                await self.identifyChunkTurns(windowStart: windowStart, audio: audioPath,
                                              localTurns: localTurns, stream: stream)
            }
        }

        service.onFinalResult = { [weak self] audioPath, localTurns, stream in
            Task { @MainActor in
                guard let self else { return }
                await self.identifyFinalTurns(audio: audioPath, localTurns: localTurns,
                                              stream: stream)
            }
        }

        service.onError = { [weak self] message, stream in
            Task { @MainActor in
                self?.handleDiarizationFailure(message, stream: stream)
            }
        }
    }

    // MARK: - The identity stage

    /// Ask who spoke in one live chunk, then run the existing chunk handler.
    ///
    /// The temp WAV is released at the SETTLE of this stage — success or failure —
    /// not on the pyannote reply, because the identity stage still has to read it.
    /// Only what the window map owns is deleted, which is unchanged: a final pass's
    /// audio is the recording itself and was never in the map.
    func identifyChunkTurns(windowStart: Double, audio: String,
                            localTurns: [PyannoteService.LocalTurn],
                            stream: PyannoteService.Stream) async {
        guard let embedding = modelLoader.embedding else {
            handleDiarizationFailure(Self.identityMissingMessage, stream: stream)
            releaseDiarChunkFile(windowStart: windowStart, stream: stream)
            return
        }
        do {
            let identity = try await embedding.identify(audio: audio, turns: localTurns,
                                                        stream: stream)
            applyChunkTurns(Self.composeTurns(localTurns, identity: identity),
                            windowStart: windowStart, stream: stream)
        } catch {
            handleDiarizationFailure(
                "Speaker identification failed: \(error.localizedDescription) — "
                + "see logs/\(WeSpeakerService.Config.logName).log", stream: stream)
        }
        releaseDiarChunkFile(windowStart: windowStart, stream: stream)
    }

    /// The same for a stop-time batch pass. No file to release: the audio is the
    /// recording, which the session owns.
    func identifyFinalTurns(audio: String, localTurns: [PyannoteService.LocalTurn],
                            stream: PyannoteService.Stream) async {
        guard let embedding = modelLoader.embedding else {
            handleDiarizationFailure(Self.identityMissingMessage, stream: stream)
            return
        }
        do {
            let identity = try await embedding.identify(audio: audio, turns: localTurns,
                                                        stream: stream)
            applyFinalTurns(Self.composeTurns(localTurns, identity: identity),
                            stream: stream)
        } catch {
            handleDiarizationFailure(
                "Speaker identification failed: \(error.localizedDescription) — "
                + "see logs/\(WeSpeakerService.Config.logName).log", stream: stream)
        }
    }

    /// Both sidecars load or neither does (`ModelLoader.buildSteps`), so this is
    /// only reachable if the identity process died mid-session. It routes through
    /// the ordinary diarization-failure path, so the gate still settles.
    nonisolated static let identityMissingMessage =
        "The speaker-identity sidecar is not running — restart the recording session."

    /// Join pipeline-local turns to their identities. PURE and `static` so the
    /// three encodings below can be unit-tested without a sidecar.
    ///
    /// Three cases, and conflating any two of them would be a silent lie:
    ///  * an identity with a `conf` — the cosine that matched this voice, carried
    ///    through verbatim;
    ///  * an identity WITHOUT one — a brand-new profile, `conf: nil`. NEVER 0: a
    ///    first appearance was not scored against anything, and 0.00 in front of a
    ///    correctly-named speaker reads as "definitely the wrong person";
    ///  * a `null` identity, or a label the map does not mention at all — the voice
    ///    could not be identified, and the turn is DROPPED. That is exactly what
    ///    the merged sidecar's `labeled_segments` did before it emitted; the drop
    ///    just moved to the side that owns the spans.
    ///
    /// Times are rounded to 3 dp HERE because that is where the merged sidecar
    /// rounded them, so app-visible turn times are unchanged. The wire carries
    /// full precision on purpose — the identity stage slices the waveform with
    /// those numbers (see `PyannoteService.LocalTurn`).
    nonisolated static func composeTurns(_ turns: [PyannoteService.LocalTurn],
                                         identity: WeSpeakerService.IdentityMap)
        -> [SpeakerTurn] {
        turns.compactMap { turn in
            // `identity[label]` is doubly optional: the outer level is "no such
            // label", the inner is the sidecar's explicit null. Both mean the
            // same thing to us, hence the flatten.
            guard let who = identity[turn.label] ?? nil else { return nil }
            return SpeakerTurn(start: Self.round3(turn.start), end: Self.round3(turn.end),
                               id: who.id, name: who.name, conf: who.conf)
        }
    }

    /// 3 decimal places, matching the merged sidecar's `round(x, 3)`. Exact ties
    /// resolve differently from Python's round-half-to-even, which is unreachable
    /// in practice: these are arbitrary pipeline floats, and a tie would move a
    /// boundary by 1 ms against turn boundaries already wrong by a median 263 ms.
    nonisolated static func round3(_ value: Double) -> Double {
        (value * 1000).rounded() / 1000
    }

    /// Release the temp WAV this window owns, if the map owns one.
    func releaseDiarChunkFile(windowStart: Double, stream: PyannoteService.Stream) {
        let file = stream == .office
            ? chunkFileByWindow.removeValue(forKey: windowStart)
            : remoteChunkFileByWindow.removeValue(forKey: windowStart)
        if let file { try? FileManager.default.removeItem(at: file) }
    }

    // MARK: - The composed handlers (the former callback bodies, verbatim)

    func applyChunkTurns(_ turns: [SpeakerTurn], windowStart: Double,
                         stream: PyannoteService.Stream) {
        // Turn times are chunk-local — offset to absolute recording time.
        let absolute = Self.offsetTurns(turns, by: windowStart)
        // The Remote branch returns early ON PURPOSE: it must not touch
        // liveTurns, sessionSpeakerIDs, speakerCount or the office tail
        // gate. Its ids are >= remoteIDBase and belong to the other space.
        guard stream == .office else {
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

    func applyFinalTurns(_ turns: [SpeakerTurn], stream: PyannoteService.Stream) {
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

    /// One failure path for BOTH stages: a pyannote job error and a failed
    /// identify are the same thing to the user — this stream produced no
    /// speakers — and settling one gate differently from the other is how a
    /// stop overlay gets stuck.
    /// Delete every temp WAV still owned by a window map, for one stream.
    ///
    /// The per-window release (`releaseDiarChunkFile`) needs a `windowStart`, and
    /// `PyannoteService.onError` does not carry one — it reports that the stream
    /// failed, not which window. So a failure leaked that window's file: the map
    /// entry survived until the next session's `chunkFileByWindow = [:]`, which
    /// drops the URL WITHOUT deleting anything. 1.9 MB per failed chunk, in a
    /// directory macOS clears only on its own schedule.
    func releaseAllDiarChunkFiles(stream: PyannoteService.Stream) {
        let files: [URL]
        if stream == .office {
            files = Array(chunkFileByWindow.values); chunkFileByWindow = [:]
        } else {
            files = Array(remoteChunkFileByWindow.values); remoteChunkFileByWindow = [:]
        }
        for file in files { try? FileManager.default.removeItem(at: file) }
    }

    /// `noSpeech` marks the ONE outcome that arrives down this path without
    /// being a failure: the engine ran and correctly found nothing to label.
    ///
    /// It takes this path because the settling is identical — the same six
    /// watchdogs, flags and gates have to be released either way, and a second
    /// near-copy of that sequence is how one of them comes to be forgotten. Only
    /// the two REPORTING lines differ, and both matter:
    ///
    ///   * the step is `.skipped`, not `.failed`, so the panel neither turns red
    ///     nor demands an acknowledgement for a correct result;
    ///   * `diarizationError` stays nil. `TranscriptView` renders it as a red
    ///     banner over the transcript, so setting it would put the alarm back on
    ///     screen after the panel had carefully avoided raising one.
    func handleDiarizationFailure(_ message: String, stream: PyannoteService.Stream,
                                  noSpeech: Bool = false) {
        // Nothing will ask for these windows again on this stream.
        releaseAllDiarChunkFiles(stream: stream)
        // A remote job's failure settles only the remote leg. Remote
        // trouble must never cost the user their office transcript —
        // the same rule the remote chunk path already follows.
        guard stream == .office else {
            self.dualStreamLog(noSpeech ? "remote diarization found no speech: \(message)"
                                        : "remote diarization failed: \(message)")
            self.completeRemoteDiarization(error: message, noSpeech: noSpeech)
            return
        }
        self.diarizing = false
        if !noSpeech { self.diarizationError = message }
        self.finalDiarDone = true
        self.awaitingTailWindowStart = nil
        self.diarTailWatchdog?.cancel()
        self.diarTailWatchdog = nil
        self.finalDiarWatchdog?.cancel()
        self.finalDiarWatchdog = nil
        self.setStopStep("diarize", noSpeech ? .skipped(message) : .failed(message))
        self.maybeStartOverlapRepair()
        self.checkStopProcessingDone()
    }

    /// Shift chunk-local turn times onto the recording clock, carrying every
    /// other field through untouched — id, name, and the match confidence.
    ///
    /// Pure and `static` so the carry-through can be unit-tested: rebuilding a
    /// `Turn` field-by-field is exactly where a newly added field gets silently
    /// dropped, and a dropped `conf` would show up only as a number quietly
    /// missing from the transcript.
    nonisolated static func offsetTurns(_ turns: [SpeakerTurn],
                                        by offset: Double) -> [SpeakerTurn] {
        turns.map {
            SpeakerTurn(start: $0.start + offset, end: $0.end + offset,
                                    id: $0.id, name: $0.name, conf: $0.conf)
        }
    }

    /// Live: write the current chunk's audio to a temp WAV and diarize it.
    func diarizeLiveChunk(windowStart: Double) {
        // THE SPECTRAL ENGINE HAS NO LIVE PATH AT ALL. Its sidecar serves only
        // `cmd: "final"` and refuses anything else loudly, because it counts and
        // clusters speakers over the WHOLE file — a 30 s window's labels would
        // look continuous across windows while meaning nothing (the failure MOSS
        // has when it is called per chunk). `SpectralService` has no chunk API and
        // `modelLoader.pyannote` is nil under this engine, so a live chunk cannot
        // be dispatched even by accident; the guard is here anyway because it also
        // has to CLEAR the buffer. Without it, `willBeConsumed` below is false, so
        // the audio would be dropped correctly — but stating the rule at the top
        // keeps a future edit to that condition from resurrecting the ~230 MB/hour
        // accumulation for a pass that reads the recording file instead.
        // NEMO IS THE SAME SHAPE and takes the same guard: NME-SC counts and
        // clusters globally, `NemoService` has no chunk API, and its sidecar refuses
        // any `cmd` but `final`. Written as one condition rather than a second guard
        // so a future edit cannot fix one engine's buffer leak and miss the other's.
        // CAM++ IS THE SAME SHAPE AGAIN and joins the same condition: it counts
        // and clusters globally, `CamPlusService` has no chunk API, and its
        // sidecar refuses any `cmd` but `final`. One condition, not a fourth
        // guard, so a future edit cannot fix one engine's buffer leak and miss
        // the others'.
        // ⚠ THE PARAGRAPH ABOVE IS NOW HALF TRUE (owner, 2026-08-13). Those four
        // engines still have no `cmd: "chunk"` and their sidecars still refuse
        // anything but `final` — the two pinned checks that say so are untouched.
        // What changed is that a WINDOW IS JUST A SHORT RECORDING, so when the
        // stop pass is switched off they are now sent one `final` per interval on
        // a temp WAV, and their run-local labels are stitched across windows by
        // `identify` exactly as pyannote's are. See
        // `AudioRecorder+BatchLiveDiarization` for why that carries continuity and
        // for the measured cost of leaving the speaker count on Auto.
        //
        // With the stop pass ON these four behave exactly as before: the buffer is
        // cleared here and the whole file is read at Stop.
        guard !spectralDiarizationActive, !nemoDiarizationActive,
              !diarizenDiarizationActive, !camPlusDiarizationActive else {
            if batchLiveDiarizationActive {
                dispatchBatchLiveWindow(windowStart: chunkAudioStart, samples: chunkAudio)
            }
            chunkAudio = []; chunkAudioStart = lastDiarBoundary; return
        }
        let liveOn = diarLiveEnabled
        guard liveOn, modelLoader.pyannote != nil else {
            // When live labels are off but "continue from live labels (tail only)"
            // is on, DON'T clear the accumulated audio: the whole recording then
            // becomes the single tail diarized at stop (matches the UI caption).
            let continueOnStop = diarContinueOnStop
            // Keep the audio ONLY when something will actually consume it. The
            // condition used to be just `continueOnStop && !liveOn`, which is the
            // live-off tail mode — but it never asked whether a diarizer exists,
            // so with diarization switched OFF entirely (`pyannote == nil`, the
            // guard above) it accumulated the whole recording at ~230 MB/hour for
            // a stop pass that can never run. `diarizeTailChunk` bails on a nil
            // pyannote too, so nothing ever drained it.
            let willBeConsumed = continueOnStop && !liveOn && modelLoader.pyannote != nil
            if !willBeConsumed { chunkAudio = []; chunkAudioStart = lastDiarBoundary }
            return
        }
        let samples = chunkAudio
        chunkAudio = []
        chunkAudioStart = lastDiarBoundary
        guard samples.count > 16_000 else { return } // skip chunks under 1s
        dispatchDiarChunk(samples: samples, windowStart: windowStart)
    }

    /// Shared: write a chunk of 16 kHz samples to a temp WAV off-thread and hand
    /// it to the diarization sidecar. Live calls pass no failure handler (silent,
    /// as before); the stop-time tail passes one so the gate can still complete.
    func dispatchDiarChunk(samples: [Float], windowStart: Double,
                                   onDispatchFailure: (() -> Void)? = nil) {
        guard let service = modelLoader.pyannote else { onDispatchFailure?(); return }
        // Captured on the MainActor BEFORE the detached task: the value is the
        // session's locked one, and reading it inside the closure would be both a
        // concurrency error and a second read of a thing that must not be read
        // twice. See `AudioRecorder.lockDiarizationSettings`.
        let detectOverlap = diarDetectOverlap
        Task.detached(priority: .utility) { [weak self] in
            guard let url = Self.writeTempWAV(samples: samples) else {
                await MainActor.run { onDispatchFailure?() }
                return
            }
            await MainActor.run { [weak self] in
                self?.chunkFileByWindow[windowStart] = url
            }
            service.diarizeChunk(audio: url, windowStart: windowStart, exclusive: !detectOverlap)
        }
    }

    /// At stop (tail-only mode): diarize just the audio accumulated since the last
    /// live chunk and append it, keeping live speaker numbering stable. Routes the
    /// completion gate through `completeStopDiarization()`.
    /// How long to allow the stop-time TAIL diarization, in seconds.
    ///
    /// Scaled by the tail's own length, not by a constant. The tail is normally
    /// one diarization interval (~30 s of audio) and a flat 120 s was generous —
    /// but there is one combination where the "tail" is the ENTIRE recording:
    /// live labels OFF with "Continue from live labels (tail only)" ON. Nothing
    /// clears `chunkAudio` in that mode, so it accumulates from 0 to the end and
    /// is submitted as a single job. A 60-minute meeting cannot be diarized in
    /// 120 s, so the watchdog fired, the transcript was marked
    /// "Tail diarization timed out" and the speaker labels were lost — while the
    /// sidecar was still working normally.
    ///
    /// The floor and the shape match `startDiarization`'s own
    /// `max(180, recordingElapsed)` one screen below: at least a generous
    /// constant, then at least 1x realtime for whatever is actually being
    /// processed. Measured against it: the final pass over the owner's 67-minute
    /// recording loads and resamples in 1.81 s, so 1x realtime is not tight.
    nonisolated static func tailDiarWatchdogSeconds(sampleCount: Int,
                                                    sampleRate: Double = 16_000) -> Double {
        max(120, Double(sampleCount) / sampleRate)
    }

    func diarizeTailChunk() {
        guard modelLoader.pyannote != nil else { completeStopDiarization(); return }
        let samples = chunkAudio
        chunkAudio = []
        // Where this buffer actually STARTED, recorded at every clear — not
        // derived from `diarization.live` read here at stop time. See
        // `chunkAudioStart`: that read produced turns shifted to 0 whenever the
        // setting was toggled mid-meeting, because the buffer then began at the
        // toggle while the setting claimed it began at the start of the meeting.
        // With live labels on this equals the last live-chunk boundary; with live
        // off plus continue-on-stop, nothing was ever cleared and it is still 0,
        // i.e. both original cases are preserved exactly.
        let windowStart = chunkAudioStart
        guard samples.count > 16_000 else { completeStopDiarization(); return } // <1s tail
        diarizing = true
        diarizationError = nil
        awaitingTailWindowStart = windowStart
        diarTailWatchdog?.cancel()
        let tailBudget = Self.tailDiarWatchdogSeconds(sampleCount: samples.count)
        diarTailWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(tailBudget))
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
        guard finalOn, let service = modelLoader.pyannote else { return }
        diarizing = true
        diarizationError = nil
        let numSpeakers = Self.diarNumSpeakers
        let detectOverlap = diarDetectOverlap
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
    func applyFinalSpeakers(_ turns: [SpeakerTurn]) {
        let turns = Self.officeTurnsOnly(turns, "applyFinalSpeakers")
        speakerCount = Set(turns.map(\.id)).count
        liveTurns = turns
        noteImplausibleSpeakerCount(turns, stream: .office)
        rebuildDisplayRows()
    }

    /// Does this speaker count look like clustering fragmentation rather than a
    /// room? Returns the caution to show, or nil.
    ///
    /// **The rule: four or more speakers, of whom the MEDIAN spoke exactly once.**
    /// A real participant takes more than one turn; a set of speakers who each
    /// appear once is the signature of one voice split many ways. Both halves are
    /// load-bearing — the count alone would flag a genuine 5-person meeting, and
    /// the median alone would flag a correct 2- or 3-speaker clip where each
    /// person legitimately has a single turn.
    ///
    /// **MEASURED on 2026-08-13 against nine known-answer cases across three
    /// recordings, and it separates them exactly** — the three wrong results are
    /// flagged and the six correct ones are not:
    ///
    /// | engine | file | truth | got | median turns | flagged |
    /// |---|---|---|---|---|---|
    /// | CAM++ | 25 s remote | 2 | **9** | 1.0 | yes |
    /// | spectral | 25 s remote | 2 | **12** | 1.0 | yes |
    /// | NeMo | 25 s remote | 2 | **12** | 1.0 | yes |
    /// | pyannote | 25 s remote | 2 | 2 | 1.0 | no (under 4) |
    /// | DiariZen | 25 s remote | 2 | 2 | 1.5 | no |
    /// | pyannote | Meeting5People | 5 | 5 | 3.0 | no |
    /// | CAM++ | Meeting5People | 5 | 5 | 3.0 | no |
    /// | pyannote | Overlap123 | 3 | 3 | 1.0 | no (under 4) |
    /// | DiariZen | Overlap123 | 3 | 3 | 3.0 | no |
    ///
    /// ⚠ **What it deliberately does NOT catch**, so nobody reads more into it:
    /// spectral's 20 speakers on the 67-minute recording, where each cluster holds
    /// ~93 turns. That case is not fragmentation-by-this-signature and is still an
    /// open question in this project — a wider rule would be guessing at it.
    ///
    /// It CAUTIONS and never corrects. The count came from the engine, and
    /// silently overriding a model's answer is the substitution this project
    /// refuses everywhere else.
    nonisolated static func implausibleSpeakerCount(_ turns: [SpeakerTurn]) -> String? {
        var turnsPerSpeaker: [Int: Int] = [:]
        for turn in turns { turnsPerSpeaker[turn.id, default: 0] += 1 }
        let speakers = turnsPerSpeaker.count
        guard speakers >= 4 else { return nil }
        let counts = turnsPerSpeaker.values.sorted()
        // Lower median on an even count — the conservative side, so a borderline
        // set is left alone rather than accused.
        let median = counts[(counts.count - 1) / 2]
        guard median <= 1 else { return nil }
        return "\(speakers) speakers were found, and most of them speak only once — "
             + "that usually means one voice was split rather than a room this full. "
             + "Set the speaker count beside the record button, or switch the "
             + "diarization engine in Settings → Models → Diarization."
    }

    /// Log the caution. **LOG ONLY — it deliberately does NOT raise the banner
    /// under the transcript** (owner, 2026-08-24: "please remove this").
    ///
    /// The rule itself is unchanged and still measured — see
    /// `implausibleSpeakerCount` above — so the evidence keeps landing in
    /// `logs/position-diarization.log` as `SPEAKER COUNT CAUTION`. What was
    /// removed is the amber row, which fired on every live window under the
    /// shipped Auto count and so was in front of the user constantly.
    ///
    /// ⚠ `diarizationCaution` is SHARED with the dead-engine warning in
    /// `handleBatchLiveFailure` (three consecutive failed windows). That one must
    /// keep raising it — do not "tidy up" the channel because this writer stopped
    /// using it.
    func noteImplausibleSpeakerCount(_ turns: [SpeakerTurn],
                                     stream: PyannoteService.Stream) {
        guard let caution = Self.implausibleSpeakerCount(turns) else { return }
        let where_ = stream == .office ? "office" : "remote"
        positionLog("SPEAKER COUNT CAUTION (\(where_)) — \(caution)")
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
