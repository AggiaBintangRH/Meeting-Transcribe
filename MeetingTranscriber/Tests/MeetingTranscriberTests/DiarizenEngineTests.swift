import XCTest
@testable import MeetingTranscriber

/// The DIARIZEN diarization engine (2026-08-10) — the FIFTH diarizer, and the
/// third whole-file batch one after spectral and NeMo.
///
/// These MIRROR `NemoEngineTests` and `SpectralEngineTests` deliberately rather
/// than sharing a base class: the point of each is that THIS engine's wiring is
/// right, and a shared harness would let a missing DiariZen case pass on another
/// engine's behalf. The engine arrived with **no tests of its own at all** — the
/// suite stayed at 511, exactly the count from before it landed — which is what
/// let the defects in the 2026-08-10 audit through.
///
/// What is pinned here, and why each is a real failure rather than tidiness:
///
///  1. **The five-way stack rule.** `wantedDiarizationStack` feeds BOTH the load
///     step and the teardown. Four separate leaks in this codebase (pyannote, the
///     two overlap engines, Nemotron) were each a load rule and a teardown rule
///     computed apart. DiariZen holds a WavLM encoder plus the pyannote 3.1 stack
///     at 1.7–3.7 GB.
///  2. **The live-dispatch guards**, including the silent half: neither buffer may
///     accumulate at ~230 MB/hour for a pass that reads the recording FILE.
///  3. **`diarization.finalPass` cannot reach either pass.** This is the exact
///     bug the 2026-08-10 audit found for spectral and NeMo — a value with no
///     control under a batch engine deleting every remote label — and DiariZen
///     inherited the same shape. The sidecar-side pin
///     (`layout/batch-engines-ignore-final-pass`) names only the other two files,
///     so this is the Swift half that covers the third.
///  4. **Literal script and log names, differing per engine**, and its OWN
///     interpreter. `Config.mossDiarization()` once DERIVED its `logName` from a
///     model type; a rename then repointed one process's stderr into another's log
///     while it still ran the other script — invisible, because both processes
///     keep working. THIS FILE SHIPPED NAMING `scripts/nemo/nemo-service.py` in a
///     user-facing error, which is the same class of defect caught by hand.
final class DiarizenEngineTests: XCTestCase {

    private let pyannote = ModelLoader.pyannoteEngineID
    private let moss = ModelLoader.mossEngineID
    private let spectral = ModelLoader.spectralEngineID
    private let nemo = ModelLoader.nemoEngineID
    private let diarizen = ModelLoader.diarizenEngineID

    // MARK: - 1. The five-way stack rule

    private func stack(enabled: Bool = true, engine: String, chunked: String = "qwen3",
                       chunkedOn: Bool = true) -> ModelLoader.DiarizationStack? {
        ModelLoader.wantedDiarizationStack(diarizationEnabled: enabled,
                                           engine: engine, chunkedID: chunked,
                                           chunkedEnabled: chunkedOn)
    }

    /// DiariZen selects ITS OWN stack, and — the direction that makes this able to
    /// fail — no other engine selects it. Swept across chunked models too, because
    /// the MOSS branch is the only one that reads `chunkedID` and a mis-placed case
    /// could inherit that sensitivity.
    func testDiarizenSelectsItsOwnStackAndNoOtherEngineDoes() {
        for chunked in ["qwen3", "whisper", "moss"] {
            XCTAssertEqual(stack(engine: diarizen, chunked: chunked), .diarizen,
                           "chunked=\(chunked)")
            for other in [pyannote, spectral, nemo, moss] {
                XCTAssertNotEqual(stack(engine: other, chunked: chunked), .diarizen,
                                  "engine=\(other) chunked=\(chunked) selected DiariZen's stack")
            }
        }
    }

    /// Switching diarization off keeps NOTHING alive, DiariZen included. This is
    /// the teardown half of the want/teardown pair.
    func testDisablingDiarizationKeepsNothingAliveUnderDiarizen() {
        XCTAssertNil(stack(enabled: false, engine: diarizen))
    }

    /// The chunked switch must not be able to move this engine's stack. Only the
    /// MOSS branch reads it, and a case added in the wrong place would inherit that.
    func testTheChunkedSwitchCannotChangeTheDiarizenStack() {
        for on in [true, false] {
            for chunked in ["qwen3", "whisper", "moss"] {
                XCTAssertEqual(stack(engine: diarizen, chunked: chunked, chunkedOn: on),
                               .diarizen, "chunkedEnabled=\(on) chunked=\(chunked)")
            }
        }
    }

    /// It is on the PIPELINE side of the 2026-07-30 split, so it needs the
    /// embedder. That is the whole reason profiles and renaming work here with no
    /// new identity code — if this ever returned false, identity would be torn
    /// down and every voice would come back nameless.
    func testTheDiarizenStackUsesSpeakerIdentity() {
        XCTAssertEqual(stack(engine: diarizen)?.usesSpeakerIdentity, true)
    }

    /// The card's catalog id and the stored setting value round-trip. `ModelCatalog`
    /// keeps them separable on purpose (MOSS's card is `moss-diar` while its value
    /// is `moss`), so binding them by assumption is how a card looks selected while
    /// the engine never switches.
    func testTheDiarizenCardRoundTripsThroughItsSettingValue() {
        let card = ModelCatalog.diarizenDiarization
        XCTAssertEqual(ModelCatalog.diarizationEngineValue(card), diarizen)
        XCTAssertEqual(ModelCatalog.diarizationEngine(forEngine: diarizen).id, card.id)
        XCTAssertTrue(ModelCatalog.diarizationEngines.contains { $0.id == card.id },
                      "the engine must actually be offered in Settings")
    }

    // MARK: - 2. The live-dispatch guards

    /// No office live chunk is ever dispatched, and — the silent half — the buffer
    /// is DROPPED rather than accumulated for a pass that reads the recording file.
    @MainActor
    func testDiarizenNeverDispatchesAnOfficeLiveChunk() {
        let recorder = AudioRecorder()
        recorder.diarizenDiarizationActive = true
        recorder.chunkAudio = [Float](repeating: 0.2, count: 480_000)  // 30 s at 16 kHz

        recorder.diarizeLiveChunk(windowStart: 0)

        XCTAssertTrue(recorder.chunkAudio.isEmpty,
                      "the office buffer must be dropped, not accumulated for a pass that "
                      + "reads the recording file instead (~230 MB/hour otherwise)")
        XCTAssertTrue(recorder.chunkFileByWindow.isEmpty,
                      "nothing may be handed to a sidecar that has no chunk branch")
    }

    /// The remote twin, discriminating in BOTH directions: under DiariZen the
    /// buffer is dropped, while the ordinary early-out (no remote stream)
    /// deliberately RETAINS it — so a guard left out, or one that swallowed every
    /// session, fails here.
    @MainActor
    func testDiarizenDropsTheRemoteDiarizationBufferButOtherEnginesKeepIt() {
        let underDiarizen = AudioRecorder()
        underDiarizen.diarizenDiarizationActive = true
        underDiarizen.remoteDiarAudio = [Float](repeating: 0.2, count: 480_000)
        underDiarizen.diarizeRemoteLiveChunk(windowStart: 0)
        XCTAssertTrue(underDiarizen.remoteDiarAudio.isEmpty)

        let underPyannote = AudioRecorder()
        underPyannote.remoteDiarAudio = [Float](repeating: 0.2, count: 480_000)
        underPyannote.diarizeRemoteLiveChunk(windowStart: 0)
        XCTAssertEqual(underPyannote.remoteDiarAudio.count, 480_000,
                       "the pre-existing early-out keeps its buffer — the DiariZen guard "
                       + "must not have changed behaviour for the other engines")
    }

    /// DIARIZEN SUPPLIES ITS OWN OVERLAP REGIONS, so it is on pyannote's side of
    /// `usesDetectedRegionsForRepair` — not with the batch engines it otherwise
    /// resembles.
    ///
    /// It shipped on the wrong side (2026-08-10), by analogy rather than by
    /// measurement. The measurement: its Conformer head is a POWERSET, and for this
    /// checkpoint that is **11 classes = 1 silence + 4 single-speaker + 6 PAIRS**
    /// (`powerset_max_classes = 2`). Two people at once is predicted directly, per
    /// 20 ms frame, so its turns intersect — 11 pairs over 13.30 s on
    /// `recordings/Overlap123.wav`, reproduced with the BUNDLED interpreter.
    ///
    /// Both directions are asserted, because the cost of the old grouping was
    /// invisible: repair silently required a redundant 32 MB detector, and the
    /// engine's own regions were discarded.
    @MainActor
    func testDiarizenSuppliesItsOwnOverlapRegionsLikePyannote() {
        let underDiarizen = AudioRecorder()
        underDiarizen.diarizenDiarizationActive = true
        XCTAssertFalse(underDiarizen.usesDetectedRegionsForRepair,
                       "DiariZen's turns intersect, so repair must take its regions "
                       + "from them — requiring the detector throws that away")

        // The three that genuinely cannot, so this test can fail in both directions.
        for (name, set) in [("spectral", { (r: AudioRecorder) in r.spectralDiarizationActive = true }),
                            ("nemo", { (r: AudioRecorder) in r.nemoDiarizationActive = true }),
                            ("moss", { (r: AudioRecorder) in r.mossDiarizationActive = true })] {
            let recorder = AudioRecorder()
            set(recorder)
            XCTAssertTrue(recorder.usesDetectedRegionsForRepair,
                          "\(name) assigns one label per instant and MUST still take "
                          + "its regions from the detector")
        }
    }

    /// THE DETECTOR IS NEVER LOADED UNDER DIARIZEN (owner, 2026-08-10), whatever
    /// the Detect overlap switch says.
    ///
    /// Under this engine overlap detection is DiariZen's own: its powerset head
    /// predicts two-speaker frames directly, so a second 32 MB segmentation network
    /// scanning the same audio answers a question already answered. The switch
    /// cannot override it — a stored `true` from a previous pyannote session must
    /// not quietly load a model this session has no use for, which is the
    /// value-outliving-its-control failure the 2026-08-06 settings pass forbids.
    ///
    /// **pyannote is asserted in the OTHER direction in the same test.** It also
    /// marks its own overlap, and it deliberately still gets the detector as a
    /// second opinion (2026-08-06). The two engines therefore give the same answer
    /// to "do you mark overlap yourself?" and different answers here — so this
    /// pins the asymmetry rather than letting a future edit "tidy" it by deriving
    /// one rule from the other and silently moving pyannote.
    func testTheDetectorIsNeverLoadedUnderDiarizenButPyannoteStillOffersIt() {
        for diarOn in [true, false] {
            XCTAssertFalse(ModelLoader.wantsOverlapDetect(detectEnabled: true,
                                                          diarizationEnabled: diarOn,
                                                          diarEngine: diarizen),
                           "the detector must never load under DiariZen "
                           + "(diarizationEnabled=\(diarOn))")
        }
        XCTAssertTrue(ModelLoader.wantsOverlapDetect(detectEnabled: true,
                                                     diarizationEnabled: true,
                                                     diarEngine: pyannote),
                      "pyannote keeps the detector as a second opinion — removing it "
                      + "would reverse the 2026-08-06 decision by accident")

        // The three that NEED it must be untouched, or this change would have
        // taken repair away from them.
        for engine in [spectral, nemo, moss] {
            XCTAssertTrue(ModelLoader.wantsOverlapDetect(detectEnabled: true,
                                                         diarizationEnabled: true,
                                                         diarEngine: engine),
                          "\(engine) depends on the detector and must still load it")
        }

        // And the pre-existing rule survives: no diarization, no rows to mark.
        XCTAssertFalse(ModelLoader.wantsOverlapDetect(detectEnabled: true,
                                                      diarizationEnabled: false,
                                                      diarEngine: pyannote))
        XCTAssertFalse(ModelLoader.wantsOverlapDetect(detectEnabled: false,
                                                      diarizationEnabled: true,
                                                      diarEngine: pyannote))
    }

    /// THE DETECT-OVERLAP TAB OFFERS DIARIZEN, AND ONLY DIARIZEN, UNDER THIS ENGINE
    /// (owner, 2026-08-10) — and pyannote's segmentation detector, and only that,
    /// under every other engine.
    ///
    /// A BICONDITIONAL, like MOSS⟺MOSS, asserted in both directions here because
    /// half a rule is how the two halves come to disagree: DiariZen's detection
    /// cannot be offered where DiariZen is not loaded, and pyannote's detector has
    /// nothing to add where DiariZen already predicts overlap directly.
    func testTheDetectOverlapTabOffersDiarizenOnlyUnderDiarizen() {
        let underDiarizen = ModelCatalog.overlapDetectors(forDiarEngine: diarizen)
        XCTAssertEqual(underDiarizen.map(\.id), [ModelCatalog.overlapDetectDiarizen.id],
                       "under DiariZen the only detector offered is DiariZen itself")

        for engine in [pyannote, spectral, nemo, moss] {
            XCTAssertEqual(ModelCatalog.overlapDetectors(forDiarEngine: engine).map(\.id),
                           [ModelCatalog.overlapDetectPyannote.id],
                           "\(engine) must be offered pyannote's detector and not "
                           + "DiariZen's, which is not even loaded there")
        }
    }

    /// A STORED DETECTOR THE CURRENT ENGINE DOES NOT OFFER IS CORRECTED, never
    /// honoured. This is the value-outliving-its-control failure the 2026-08-06
    /// settings pass exists to forbid: a `pyannote-segmentation` left behind by an
    /// earlier session must not name the detector for a DiariZen meeting, and vice
    /// versa. The tab surfaces the switch rather than making it silently.
    func testAStoredDetectorFromAnotherEngineIsCorrected() {
        let pyannoteID = ModelCatalog.overlapDetectPyannote.id
        let diarizenID = ModelCatalog.overlapDetectDiarizen.id

        XCTAssertEqual(ModelCatalog.overlapDetector(id: pyannoteID,
                                                    forDiarEngine: diarizen).id, diarizenID,
                       "a stored pyannote detector must not survive into a DiariZen session")
        XCTAssertEqual(ModelCatalog.overlapDetector(id: diarizenID,
                                                    forDiarEngine: pyannote).id, pyannoteID,
                       "and DiariZen's must not survive out of one")
        // The ordinary case still round-trips, so this cannot pass by always
        // returning the engine's default.
        XCTAssertEqual(ModelCatalog.overlapDetector(id: diarizenID,
                                                    forDiarEngine: diarizen).id, diarizenID)
        XCTAssertEqual(ModelCatalog.overlapDetector(id: pyannoteID,
                                                    forDiarEngine: pyannote).id, pyannoteID)
    }

    /// THE SWITCH REALLY SWITCHES. Under DiariZen, Detect overlap turns the marking
    /// on and off; with it off, `overlapRegions()` is empty — so no row is tagged
    /// AND repair has nowhere to work, which is one decision made at one point
    /// rather than two that can disagree.
    ///
    /// The pyannote case is the half that makes this able to fail: that engine must
    /// keep marking overlap from its own turns whatever the switch says, exactly as
    /// it always has.
    @MainActor
    func testTheDetectSwitchGatesDiarizenMarkingButNeverPyannotes() {
        func recorder(diarizen on: Bool, marking: Bool) -> AudioRecorder {
            let r = AudioRecorder()
            r.diarizenDiarizationActive = on
            r.diarizenOverlapMarking = marking
            // Two speakers genuinely overlapping for 2 s — well past the 0.4 s bar.
            r.liveTurns = [SpeakerTurn(start: 0, end: 5, id: 1, name: "A", conf: nil),
                           SpeakerTurn(start: 3, end: 8, id: 2, name: "B", conf: nil)]
            return r
        }

        XCTAssertFalse(recorder(diarizen: true, marking: true).overlapRegions().isEmpty,
                       "with marking on, DiariZen's intersecting turns must produce regions")
        XCTAssertTrue(recorder(diarizen: true, marking: false).overlapRegions().isEmpty,
                      "with marking off, nothing may be tagged and repair gets no regions")

        // pyannote: the flag is inert, in BOTH positions.
        for marking in [true, false] {
            XCTAssertFalse(recorder(diarizen: false, marking: marking).overlapRegions().isEmpty,
                           "pyannote must keep marking overlap (marking flag = \(marking)) — "
                           + "this switch is not allowed to reach it")
        }
    }

    /// The skip log names the engine that is actually running. DiariZen has no case
    /// there ON PURPOSE — it never reaches that log now — so the fallback must
    /// still be spectral's, and a new engine added above it would silently steal
    /// every other engine's name.
    @MainActor
    func testTheSkipLogStillNamesTheBatchEnginesCorrectly() {
        let spectralRecorder = AudioRecorder()
        spectralRecorder.spectralDiarizationActive = true
        XCTAssertEqual(spectralRecorder.batchEngineNameForLog, "SPECTRAL")

        let nemoRecorder = AudioRecorder()
        nemoRecorder.nemoDiarizationActive = true
        XCTAssertEqual(nemoRecorder.batchEngineNameForLog, "NEMO")
    }

    // MARK: - 3. The stop passes, and the setting that must not reach them

    /// The office pass needs the engine, the service AND the recording. Its own
    /// call, not NeMo's: sharing one would ask whether NeMo's sidecar was up for a
    /// DiariZen session.
    func testTheOfficePassNeedsTheEngineTheServiceAndTheRecording() {
        func runs(active: Bool = true, service: Bool = true, recording: Bool = true) -> Bool {
            AudioRecorder.runsBatchOfficePass(batchActive: active,
                                              hasService: service, hasRecording: recording)
        }
        XCTAssertTrue(runs())
        XCTAssertFalse(runs(active: false), "a pyannote session must not take this branch")
        XCTAssertFalse(runs(service: false))
        XCTAssertFalse(runs(recording: false))
    }

    /// `diarization.finalPass` is SHARED with pyannote, and a DiariZen session must
    /// be unaffected by whatever pyannote left in it.
    ///
    /// THIS IS THE 2026-08-10 DATA-LOSS BUG'S SHAPE. Under a batch engine
    /// `DiarizationTab` HIDES that toggle, so a `false` left by an earlier pyannote
    /// session is unreachable — and the remote pass reading it returned `.none`,
    /// deleting every remote label with no message and no control able to undo it.
    /// It was fixed for spectral and NeMo and pinned by
    /// `layout/batch-engines-ignore-final-pass`, which names only those two files.
    /// This is the third engine's half.
    func testAStoredFinalPassCannotReachEitherDiarizenPass() {
        let d = UserDefaults.standard
        let saved = d.object(forKey: "diarization.finalPass")
        defer {
            if let saved { d.set(saved, forKey: "diarization.finalPass") }
            else { d.removeObject(forKey: "diarization.finalPass") }
        }
        for stored in [true, false] {
            d.set(stored, forKey: "diarization.finalPass")
            XCTAssertTrue(AudioRecorder.runsBatchOfficePass(batchActive: true,
                                                            hasService: true,
                                                            hasRecording: true),
                          "a stored finalPass=\(stored) reached the DiariZen office pass")
            // The remote half is what actually lost data: the caller must STATE
            // its engine's truth rather than read the key.
            XCTAssertEqual(AudioRecorder.remoteStopMode(finalPass: true,
                                                        continueOnStop: false,
                                                        remoteStreamActive: true,
                                                        hasDiarizationService: true,
                                                        hasRemoteRecording: true,
                                                        tailSamples: 0,
                                                        supportsTail: false),
                           .full,
                           "the remote pass must run whatever finalPass=\(stored) says")
        }
    }

    /// `continueOnStop` cannot be honoured by an engine with no chunk job and no
    /// live labels, so the remote pass is ALWAYS full — `supportsTail: false`, as
    /// spectral and NeMo pass it. Both directions: the same inputs with the default
    /// still give `.tail`, which is what proves the parameter changed the answer.
    func testRemoteTailIsUnavailableToDiarizenButUnchangedForPyannote() {
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

    /// DiariZen adds NO startup refusal, and that is a decision rather than an
    /// omission: it runs once after Stop, so a second stream can only be waited
    /// for — the same evidence that allows spectral and NeMo with dual-stream.
    /// Voxtral's refusal is about duty cycle DURING recording and must survive.
    func testDualStreamIsAllowedWithDiarizenWhileVoxtralIsStillRefused() {
        for chunked in ["qwen3", "whisper", "granite", "moss"] {
            XCTAssertNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                                chunkedModelID: chunked),
                         "chunked=\(chunked): a dual-stream DiariZen session was refused")
        }
        XCTAssertNotNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                               chunkedModelID: "voxtral"),
                        "Voxtral + dual-stream must still be refused at startup")
    }

    // MARK: - 4. Literal names, and its OWN interpreter

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

    /// Both names are LITERALS and both differ from every other diarizer's.
    ///
    /// The failure this forbids is the `Config.mossDiarization()` one: a derived
    /// `logName` silently repointing one process's stderr into another's log while
    /// it still runs the other script — invisible, because both keep working.
    func testDiarizenConfigNamesItsOwnSidecarAndLogAsLiterals() {
        XCTAssertEqual(DiarizenService.Config.scriptName, "diarizen/diarizen-service.py")
        XCTAssertEqual(DiarizenService.Config.logName, "diarizen")

        let others: [(engine: String, script: String, log: String)] = [
            ("pyannote", PyannoteService.scriptName, PyannoteService.logName),
            ("spectral", SpectralService.Config.scriptName, SpectralService.Config.logName),
            ("nemo", NemoService.Config.scriptName, NemoService.Config.logName),
            ("wespeaker", WeSpeakerService.Config.scriptName, WeSpeakerService.Config.logName),
            ("moss-diar", ChunkedASRService.Config.mossDiarization().scriptName,
                          ChunkedASRService.Config.mossDiarization().logName),
        ]
        for other in others {
            XCTAssertNotEqual(DiarizenService.Config.scriptName, other.script,
                              "diarizen must not run \(other.engine)'s script")
            XCTAssertNotEqual(DiarizenService.Config.logName, other.log,
                              "diarizen must not write \(other.engine)'s log — two writers "
                              + "on one file is the 2026-07-15 mistake")
        }

        // The named script really exists. This is the half that catches a rename
        // applied to disk but not to Swift; for a diarizer it surfaces loudly as
        // `ServiceError.scriptMissing` at session start, in front of the user.
        for script in [DiarizenService.Config.scriptName] + others.map(\.script) {
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: scriptsDir.appendingPathComponent(script).path),
                          "\(script) does not exist")
        }
    }

    /// The log basename obeys the exceptionless rule
    /// `scripts/<name>/<name>-service.py` → `logs/<name>.log`, derived here rather
    /// than restated, so a rename of either half fails.
    func testTheLogBasenameIsDerivedFromTheServiceFolder() {
        let folder = DiarizenService.Config.scriptName.split(separator: "/").first.map(String.init)
        XCTAssertEqual(folder, DiarizenService.Config.logName)
        XCTAssertEqual(DiarizenService.Config.scriptName,
                       "\(folder ?? "")/\(folder ?? "")-service.py")
    }

    /// The app's own decision log is a DIFFERENT file from the sidecar's stderr,
    /// and for this engine that matters as much as for NeMo: DiariZen logs to
    /// stdout, so its sidecar's `install_wire()` redirects every non-protocol
    /// writer into `logs/diarizen.log`, which is therefore busy.
    func testTheSwiftDecisionLogIsNotTheSidecarLog() {
        XCTAssertNotEqual("diarizen-diarization", DiarizenService.Config.logName)
    }

    /// It runs in its OWN interpreter, and the name is a literal on the Swift side
    /// because `PythonRuntime.command(forScript:venvName:)` returns nil for a
    /// missing one — deliberately, with NO fallback to the main `.venv`.
    ///
    /// It is also the ONLY interpreter here on Python 3.11: upstream pins
    /// torch 2.1.1, which has no 3.12 wheels.
    func testDiarizenDeclaresItsOwnVenvAndSaysSoWhenItIsMissing() {
        XCTAssertEqual(DiarizenService.Config.venvName, ".venv-diarizen")
        XCTAssertNotEqual(DiarizenService.Config.venvName, NemoService.Config.venvName)

        let message = DiarizenService.ServiceError.venvMissing.errorDescription ?? ""
        XCTAssertTrue(message.contains(".venv-diarizen"), message)
        XCTAssertTrue(message.contains("download-best-models.sh"),
                      "a setup error must name the thing that fixes it: \(message)")
    }

    /// EVERY user-facing error names THIS engine's own files.
    ///
    /// Not tidiness — the shipped `scriptMissing` said
    /// `scripts/nemo/nemo-service.py not found`, so a user hitting it would look
    /// for the wrong file, in the wrong folder, for an engine they were not
    /// running. Caught by hand on 2026-08-10; this is what makes it stay caught.
    func testNoUserFacingErrorNamesAnotherEnginesFiles() {
        let messages = [
            DiarizenService.ServiceError.scriptMissing,
            DiarizenService.ServiceError.venvMissing,
            DiarizenService.ServiceError.launchFailed("x"),
        ].map { $0.errorDescription ?? "" }

        for message in messages {
            XCTAssertFalse(message.lowercased().contains("nemo"),
                           "a DiariZen error names NeMo: \(message)")
            XCTAssertFalse(message.contains("spectral") || message.contains("pyannote"),
                           "a DiariZen error names another engine: \(message)")
        }
        XCTAssertTrue(messages[0].contains("diarizen/diarizen-service.py"),
                      "scriptMissing must name its own script: \(messages[0])")
    }

    /// The install check requires BOTH halves — the HF-hub checkpoint AND the
    /// vendored source tree — and it must agree with what is on disk. A check that
    /// passes with one missing fails deep inside the sidecar at Stop instead, which
    /// is the worst moment: under this engine the stop pass IS the labels.
    func testTheInstallCheckLooksForTheCheckpointAndTheVendoredSource() {
        let fm = FileManager.default
        let hub = ModelCatalog.modelsDir
            .appendingPathComponent("hub/models--BUT-FIT--diarizen-meeting-base")
        let vendor = scriptsDir.appendingPathComponent("diarizen/vendor/diarizen")
        let both = fm.fileExists(atPath: hub.path) && fm.fileExists(atPath: vendor.path)

        XCTAssertEqual(ModelCatalog.isInstalled(ModelCatalog.diarizenDiarization), both,
                       "the install check disagrees with the files on disk")
    }

    /// The checkpoint is the MIT one, and the card says so.
    ///
    /// A HARD CLIENT REQUIREMENT, not a label: five of the six DiariZen
    /// checkpoints are CC BY-NC 4.0 and cannot ship in a paid product. Swapping
    /// `hfRepo` for a "better" one is a licence violation, and this is the only
    /// thing in the Swift layer that would notice.
    func testTheCheckpointIsTheMitOne() {
        XCTAssertEqual(ModelCatalog.diarizenDiarization.hfRepo, "BUT-FIT/diarizen-meeting-base")
        XCTAssertTrue(ModelCatalog.diarizenDiarization.badges.contains("MIT"),
                      "the licence must be visible on the card: "
                      + "\(ModelCatalog.diarizenDiarization.badges)")
    }
}
