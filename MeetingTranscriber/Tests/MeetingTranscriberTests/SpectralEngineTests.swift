import XCTest
@testable import MeetingTranscriber

/// The SPECTRAL diarization engine (2026-08-04) — the THIRD diarizer, and the
/// first one added since the 2026-07-30 pyannote/wespeaker split.
///
/// WHAT THESE PIN, and why each is a real failure rather than tidiness:
///
///  1. **The three-way stack rule.** `wantedDiarizationStack` feeds BOTH the load
///     step and the teardown. The 2026-07-31 audit found the overlap engines
///     started by one rule and never stopped because there was no second one —
///     ~6 GB idle. A third engine is a third chance to make that mistake, and the
///     one that would bite hardest is `usesSpeakerIdentity`: two independent
///     `!= .pyannote` tests would each have torn down the shared embedder for the
///     other engine's session, leaving a stack with no identity at all.
///  2. **Spectral never selects an overlap engine.** Its Viterbi smoothing assigns
///     exactly one label per frame, so its turns never intersect and
///     `overlapRegions()` is empty by construction. Loading a ~6 GB DiCoW to
///     discover that once per meeting is the waste the MOSS branch already avoids.
///  3. **The live-dispatch guard.** The sidecar has no chunk branch at all
///     (`spectral/no-live-chunk-branch`), so a live window must not be dispatched
///     — and, the part that is silent, its buffer must not accumulate at
///     ~230 MB/hour for a pass that reads the recording FILE and never looks at it.
///  4. **Literal script and log names, differing per engine.** `Config.mossDiarization()`
///     once DERIVED its `logName` from a model type; a rename then repointed one
///     process's stderr into another's log while it still ran the other script —
///     invisible, because both processes keep working.
final class SpectralEngineTests: XCTestCase {

    private let pyannote = ModelLoader.pyannoteEngineID
    private let moss = ModelLoader.mossEngineID
    private let spectral = ModelLoader.spectralEngineID

    // MARK: - 1. The three-way stack rule

    private func stack(enabled: Bool = true, engine: String, chunked: String = "qwen3")
        -> ModelLoader.DiarizationStack? {
        ModelLoader.wantedDiarizationStack(diarizationEnabled: enabled,
                                           engine: engine, chunkedID: chunked)
    }

    /// Each engine selects ITS OWN stack, and — the direction that makes this able
    /// to fail — never another engine's. Asserted across chunked models too,
    /// because the MOSS branch is the only one that reads `chunkedID` and a
    /// mis-placed spectral case could inherit that sensitivity.
    func testEachEngineSelectsItsOwnStack() {
        for chunked in ["qwen3", "whisper", "moss"] {
            XCTAssertEqual(stack(engine: spectral, chunked: chunked), .spectral,
                           "chunked=\(chunked): spectral must select the spectral stack")
            XCTAssertEqual(stack(engine: pyannote, chunked: chunked), .pyannote,
                           "chunked=\(chunked): the spectral case must not capture pyannote")
            XCTAssertNotEqual(stack(engine: moss, chunked: chunked), .spectral,
                              "chunked=\(chunked): MOSS must never select the spectral stack")
        }
    }

    /// The master switch still wins over all THREE engines. This is the 2026-07-31
    /// bug's step 4 — the teardown that never fired — extended to the new engine.
    func testDisablingDiarizationKeepsNothingAliveUnderSpectralToo() {
        for engine in [pyannote, spectral, moss] {
            XCTAssertNil(stack(enabled: false, engine: engine),
                         "engine=\(engine): the off switch was ignored")
        }
    }

    /// An unknown/stale stored value still resolves to pyannote, exactly as
    /// `diarizationEngine(forEngine:)` shows it and as the recorder's
    /// `?? pyannoteEngineID` defaults produce. Adding a third case must not have
    /// turned the fall-through into a hole that loads nothing.
    func testAnUnknownEngineStillResolvesToPyannote() {
        XCTAssertEqual(stack(engine: "spectralish"), .pyannote)
        XCTAssertEqual(stack(engine: ""), .pyannote)
    }

    /// THE SHARED EMBEDDER — now shared by ALL stacks (2026-08-05).
    ///
    /// Both pipeline engines emit run-local labels and need WeSpeaker to say who
    /// those people are. MOSS used to be the exception, "names its own speakers
    /// and never reaches the stores" — and that exception WAS the bug: its `S01`
    /// is run-local too, just per model call rather than per pipeline run, so
    /// without the embedder its numbering restarted at every window seam and it
    /// had no profiles, no renaming and no `spk`.
    func testEveryStackUsesSpeakerIdentity() {
        for stack: ModelLoader.DiarizationStack in [.pyannote, .spectral,
                                                    .mossSecondProcess, .mossOwnASR] {
            XCTAssertTrue(stack.usesSpeakerIdentity,
                          "\(stack) has no way to name anyone without the embedder")
        }
    }

    /// The catalog id and the setting value must round-trip. This is the MOSS trap
    /// (`moss-diar` the id, `moss` the setting) generalised: binding a card
    /// directly to `model.id` stored a value nothing ever matched, so the card
    /// looked selected while the engine never switched.
    func testEveryEngineCardRoundTripsThroughItsSettingValue() {
        for card in ModelCatalog.diarizationEngines {
            let value = ModelCatalog.diarizationEngineValue(card)
            XCTAssertEqual(ModelCatalog.diarizationEngine(forEngine: value).id, card.id,
                           "\(card.id) does not round-trip through \"\(value)\"")
        }
        // And the values really are the three the loader tests against — a card
        // whose value fell through to pyannote would round-trip only by accident.
        XCTAssertEqual(ModelCatalog.diarizationEngineValue(ModelCatalog.spectralDiarization),
                       spectral)
        XCTAssertEqual(ModelCatalog.diarizationEngine(forEngine: spectral).id,
                       ModelCatalog.spectralDiarization.id)
        XCTAssertNotEqual(ModelCatalog.diarizationEngineValue(ModelCatalog.mossDiarization),
                          spectral)
    }

    // MARK: - 2. Spectral never selects an overlap engine

    /// Neither repair engine is kept alive under spectral — and, the other
    /// direction, both still are under pyannote, so this cannot pass by refusing
    /// everything.
    func testSpectralKeepsNeitherOverlapEngine() {
        for engineID in [ModelCatalog.overlapSeparation.id, ModelCatalog.overlapDicow.id] {
            XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                         engineID: engineID,
                                                         diarEngine: spectral),
                         "\(engineID) was kept alive for an engine that cannot produce "
                         + "overlap regions at all")
            XCTAssertEqual(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                           engineID: engineID,
                                                           diarEngine: pyannote),
                           engineID,
                           "\(engineID) must still load under pyannote — the spectral "
                           + "exclusion must not have widened into every engine")
        }
    }

    // MARK: - 3. The live-dispatch guard

    /// A live office window under spectral is dropped, not queued: no dispatch,
    /// no temp WAV registered, and — the silent half — no buffer left to grow.
    @MainActor
    func testSpectralNeverDispatchesAnOfficeLiveChunk() {
        let recorder = AudioRecorder()
        recorder.spectralDiarizationActive = true
        recorder.chunkAudio = [Float](repeating: 0.2, count: 480_000)  // 30 s at 16 kHz

        recorder.diarizeLiveChunk(windowStart: 0)

        XCTAssertTrue(recorder.chunkAudio.isEmpty,
                      "the office buffer must be dropped, not accumulated for a pass that "
                      + "reads the recording file instead (~230 MB/hour otherwise)")
        XCTAssertTrue(recorder.chunkFileByWindow.isEmpty,
                      "nothing may be handed to a sidecar that has no chunk branch")
    }

    /// The remote twin, and it discriminates in BOTH directions: under spectral
    /// the buffer is dropped, while the ordinary early-out (no remote stream)
    /// deliberately RETAINS it — so a guard that had been left out, or one that
    /// swallowed every session, fails here.
    @MainActor
    func testSpectralDropsTheRemoteDiarizationBufferButOtherEnginesKeepIt() {
        let underSpectral = AudioRecorder()
        underSpectral.spectralDiarizationActive = true
        underSpectral.remoteDiarAudio = [Float](repeating: 0.2, count: 480_000)
        underSpectral.diarizeRemoteLiveChunk(windowStart: 0)
        XCTAssertTrue(underSpectral.remoteDiarAudio.isEmpty)

        let underPyannote = AudioRecorder()
        underPyannote.spectralDiarizationActive = false
        underPyannote.remoteDiarAudio = [Float](repeating: 0.2, count: 480_000)
        underPyannote.diarizeRemoteLiveChunk(windowStart: 0)
        XCTAssertEqual(underPyannote.remoteDiarAudio.count, 480_000,
                       "the pre-existing early-out keeps its buffer — the spectral guard "
                       + "must not have changed behaviour for the other engines")
    }

    /// Overlap repair is skipped under spectral wherever the decision is made:
    /// the loader (above), the overlay's row list, and the run itself. All three,
    /// because the services outlive a session — one left over from a previous
    /// pyannote meeting would otherwise put a row on the overlay for work that
    /// cannot happen.
    @MainActor
    func testSpectralListsNoOverlapRepairRow() {
        let recorder = AudioRecorder()
        recorder.spectralDiarizationActive = true
        XCTAssertFalse(recorder.overlapRepairWillRun)
    }

    // MARK: - The stop pass, and the tail that does not exist

    /// The office rule, all four inputs both ways. `continueOnStop` is absent from
    /// the signature ON PURPOSE — there is no tail — so the only way this engine
    /// can fail to label a meeting is one of these four being false.
    func testTheOfficePassNeedsTheEngineTheSettingTheServiceAndTheRecording() {
        func runs(active: Bool = true, finalPass: Bool = true,
                  service: Bool = true, recording: Bool = true) -> Bool {
            AudioRecorder.runsSpectralOfficePass(spectralActive: active, finalPass: finalPass,
                                                 hasService: service, hasRecording: recording)
        }
        XCTAssertTrue(runs())
        XCTAssertFalse(runs(active: false), "a pyannote session must not take this branch")
        XCTAssertFalse(runs(finalPass: false),
                       "the settings are re-read at Stop — a pass the user has since "
                       + "switched off must not be dispatched")
        XCTAssertFalse(runs(service: false))
        XCTAssertFalse(runs(recording: false))
    }

    /// `continueOnStop` cannot be honoured by an engine with no chunk job and no
    /// live labels, so the remote pass is ALWAYS full. Both directions: the same
    /// inputs with `supportsTail` left at its default still give `.tail`, which is
    /// what proves the parameter is what changed the answer.
    func testRemoteTailIsUnavailableToSpectralButUnchangedForPyannote() {
        func mode(supportsTail: Bool) -> AudioRecorder.RemoteStopMode {
            AudioRecorder.remoteStopMode(finalPass: true, continueOnStop: true,
                                         remoteStreamActive: true, hasDiarizationService: true,
                                         hasRemoteRecording: true, tailSamples: 480_000,
                                         supportsTail: supportsTail)
        }
        XCTAssertEqual(mode(supportsTail: false), .full,
                       "with no tail available the pass must fall through to the full one, "
                       + "not to .none — .none would leave the meeting unlabelled")
        XCTAssertEqual(mode(supportsTail: true), .tail,
                       "pyannote's behaviour must be byte-for-byte what it was")
    }

    /// The startup refusal: this engine with its stop pass off produces NO speaker
    /// labels whatsoever, so it is refused before the meeting rather than
    /// discovered as an unlabelled transcript after it. Every other combination
    /// must start normally — a refusal that fired too widely would block sessions
    /// that work.
    func testTheStopPassIsRefusedOnlyWhereItWouldLoseEveryLabel() {
        XCTAssertNotNil(AudioRecorder.spectralRefusalMessage(diarizationEngine: spectral,
                                                             diarizationEnabled: true,
                                                             finalPass: false))
        XCTAssertNil(AudioRecorder.spectralRefusalMessage(diarizationEngine: spectral,
                                                          diarizationEnabled: true,
                                                          finalPass: true))
        // Diarization off entirely is a coherent choice under any engine.
        XCTAssertNil(AudioRecorder.spectralRefusalMessage(diarizationEngine: spectral,
                                                          diarizationEnabled: false,
                                                          finalPass: false))
        // And the other engines keep their stop pass optional — for them it is a
        // degradation, not a total absence.
        for engine in [pyannote, moss] {
            XCTAssertNil(AudioRecorder.spectralRefusalMessage(diarizationEngine: engine,
                                                              diarizationEnabled: true,
                                                              finalPass: false),
                         "engine=\(engine) was refused by a spectral-only rule")
        }
    }

    /// The refusal's TEXT, not just its existence. A refusal the user cannot act on
    /// is the same dead end as no refusal at all: it lands in the loading overlay
    /// with no other context, so it has to name the engine, the control, and where
    /// that control lives. Asserted here because the message is the entire
    /// user-visible product of this rule — the Bool half is pinned above.
    func testTheRefusalNamesTheEngineAndTheControlTheUserMustChange() {
        guard let message = AudioRecorder.spectralRefusalMessage(diarizationEngine: spectral,
                                                                  diarizationEnabled: true,
                                                                  finalPass: false) else {
            return XCTFail("the one configuration that loses every label was not refused")
        }
        XCTAssertTrue(message.lowercased().contains("spectral"),
                      "the overlay shows this alone — it must say which engine refused")
        XCTAssertTrue(message.contains("Run a diarization pass at stop"),
                      "the setting to change must be named exactly as the tab labels it, or "
                      + "the user has to guess which toggle is meant")
        XCTAssertTrue(message.contains("Diarization"),
                      "the message must point at the Settings tab holding that toggle")
        // And it must offer the two ways out this project always offers instead of
        // silently substituting an engine: use the one that CAN label live, or turn
        // the feature off deliberately.
        XCTAssertTrue(message.contains("pyannote"))
    }

    /// **THE DUAL-STREAM DECISION, pinned as the absence it is.**
    ///
    /// Voxtral is refused with a Remote channel because its ~27 s per 30 s chunk
    /// exceeds 100 % duty for one stream, so a second stream falls permanently
    /// behind the recording. That arithmetic does not reach this engine: it runs
    /// ONCE, after the recording has stopped, so a second job cannot fall behind
    /// anything — it can only be waited for.
    ///
    /// MEASURED 2026-08-04 by driving `scripts/spectral/spectral-service.py` with
    /// both jobs written back to back on ONE stdin, as `stop()` dispatches them:
    /// a 4027.4 s recording answered 169.3 s after dispatch and a 3630.7 s one
    /// queued behind it a further 151.1 s — 320.4 s for 2.1 hours of audio, at
    /// 23.8x and 24.0x realtime. `spectralWatchdogSeconds` allows 2x REALTIME per
    /// leg and doubles it for the remote one, so the measured need is ~2 % of the
    /// budget already in place. A refusal would be an unnecessary one, and an
    /// unnecessary refusal is its own defect — so there is none, and this test is
    /// what stops one being added by symmetry with Voxtral's.
    ///
    /// Both directions: Voxtral + Remote must STILL refuse, which is what proves
    /// this is checking the real rules rather than three functions that return nil.
    func testDualStreamIsAllowedWithSpectralWhileVoxtralIsStillRefused() {
        for chunked in ["qwen3", "whisper", "granite", "moss"] {
            XCTAssertNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                                chunkedModelID: chunked),
                         "chunked=\(chunked): a dual-stream spectral session was refused")
            XCTAssertNil(AudioRecorder.mossRefusalMessage(chunkedModelID: chunked,
                                                          diarizationEngine: spectral,
                                                          remoteChannel: 1),
                         "chunked=\(chunked): the MOSS-engine rule captured spectral")
            // The engine's own rule is about the stop pass and nothing else — a
            // Remote channel is not one of its inputs and must never become one.
            XCTAssertNil(AudioRecorder.spectralRefusalMessage(diarizationEngine: spectral,
                                                              diarizationEnabled: true,
                                                              finalPass: true),
                         "chunked=\(chunked): a working spectral session was refused")
        }
        XCTAssertNotNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                               chunkedModelID: "voxtral"),
                        "Voxtral + Remote must still be refused — its duty cycle is the "
                        + "reason that rule exists and spectral does not repeal it")
    }

    /// The stop overlay must outlast the leg it is waiting on. A spectral pass has
    /// no windows to count, so it contributes SECONDS — and every pre-existing
    /// budget must be unchanged when it contributes none.
    func testTheStopBudgetCoversASpectralPassAndIsOtherwiseUnchanged() {
        let baseline = AudioRecorder.stopWatchdogSeconds(chunkedFullPassWindows: 0,
                                                         mossFullPassWindows: 0)
        XCTAssertEqual(AudioRecorder.stopWatchdogSeconds(chunkedFullPassWindows: 0,
                                                         mossFullPassWindows: 0,
                                                         spectralPassSeconds: 0),
                       baseline, "a session with no spectral pass must budget exactly as before")

        // A pass long enough to matter must move the budget past it. 3600 s of
        // audio → a 7200 s leg watchdog, which no floor covers.
        let long = AudioRecorder.spectralWatchdogSeconds(recordingLength: 3600)
        let budget = AudioRecorder.stopWatchdogSeconds(chunkedFullPassWindows: 0,
                                                       mossFullPassWindows: 0,
                                                       spectralPassSeconds: long)
        XCTAssertGreaterThan(budget, long,
                             "the overlay would give up before the leg it is waiting on")
        // And it still composes with the other passes rather than replacing them.
        XCTAssertGreaterThan(
            AudioRecorder.stopWatchdogSeconds(chunkedFullPassWindows: 40,
                                              mossFullPassWindows: 0,
                                              spectralPassSeconds: long),
            AudioRecorder.stopWatchdogSeconds(chunkedFullPassWindows: 40,
                                              mossFullPassWindows: 0))
    }

    /// The leg watchdog itself: never below the floor, and at least real-time for
    /// what is actually being processed. CPU-only, so 1x realtime is a far tighter
    /// bound here than it is for pyannote's MPS pass.
    func testTheLegWatchdogHasAFloorAndScalesWithTheRecording() {
        XCTAssertEqual(AudioRecorder.spectralWatchdogSeconds(recordingLength: 0), 180)
        XCTAssertEqual(AudioRecorder.spectralWatchdogSeconds(recordingLength: 10), 180)
        XCTAssertGreaterThan(AudioRecorder.spectralWatchdogSeconds(recordingLength: 3600), 3600)
    }

    // MARK: - 4. Literal script and log names, differing per engine

    /// The repo's `scripts/` directory, from this file's own path — the app's
    /// `PythonRuntime.scriptsDir` resolves against the bundle or the launch
    /// directory, neither of which a unit test can rely on.
    private var scriptsDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // …/Tests/MeetingTranscriberTests
            .deletingLastPathComponent()   // …/Tests
            .deletingLastPathComponent()   // …/MeetingTranscriber
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("scripts")
    }

    /// Both names are LITERALS and both differ from every other diarizer's. The
    /// failure this forbids is the `Config.mossDiarization()` one: a derived
    /// `logName` silently repointing one process's stderr into another's log while
    /// it still runs the other script — a drifted script/log pair, invisible
    /// because both processes keep working.
    func testSpectralConfigNamesItsOwnSidecarAndLogAsLiterals() {
        XCTAssertEqual(SpectralService.Config.scriptName, "spectral/spectral-service.py")
        XCTAssertEqual(SpectralService.Config.logName, "spectral")

        let others: [(engine: String, script: String, log: String)] = [
            ("pyannote", PyannoteService.scriptName, PyannoteService.logName),
            ("wespeaker", WeSpeakerService.Config.scriptName, WeSpeakerService.Config.logName),
            ("moss-diar", ChunkedASRService.Config.mossDiarization().scriptName,
                          ChunkedASRService.Config.mossDiarization().logName),
        ]
        for other in others {
            XCTAssertNotEqual(SpectralService.Config.scriptName, other.script,
                              "spectral must not run \(other.engine)'s script")
            XCTAssertNotEqual(SpectralService.Config.logName, other.log,
                              "spectral must not write \(other.engine)'s log — two writers "
                              + "on one file is the 2026-07-15 mistake")
        }

        // The named script really exists. This is the half that catches a rename
        // applied to disk but not to Swift; for a diarizer it surfaces loudly as
        // `ServiceError.scriptMissing` at session start, in front of the user.
        for script in [SpectralService.Config.scriptName] + others.map(\.script) {
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: scriptsDir.appendingPathComponent(script).path),
                          "\(script) does not exist")
        }
    }

    /// The log basename obeys the exceptionless rule
    /// `scripts/<name>/<name>-service.py` → `logs/<name>.log`, derived here rather
    /// than restated, so a rename of either half fails.
    func testTheLogBasenameIsDerivedFromTheServiceFolder() {
        let folder = SpectralService.Config.scriptName.split(separator: "/").first.map(String.init)
        XCTAssertEqual(folder, SpectralService.Config.logName)
        XCTAssertEqual(SpectralService.Config.scriptName, "\(folder ?? "")/\(folder ?? "")-service.py")
    }
}
